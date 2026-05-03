"""Evaluation runner: loop dataset × methods, score, summarise.

Usage:
    python -m ttt.eval.runner \\
        --dataset elsa \\
        --methods baseline,icl,rag,ttt \\
        --metrics emotion_acc,bleu,rouge_l \\
        --device auto \\
        --max-examples 50

What it does, in plain pseudocode:

    dataset = ttt.datasets.get(<name>)
    methods = [ttt.methods.get(<name>, model) for ... ]
    metrics = [ttt.eval.metrics.get(<name>) for ... ]

    for ex in dataset.test_examples():
        for method in methods:
            method.prepare(ex)
            pred = method.predict(ex)
            method.cleanup()
            for metric in metrics:
                score = metric(pred, ex.task_output, ex.metadata)
                results[method.name][metric.name].append(score)

    print_table(results)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from typing import Iterable

import torch

from .. import datasets as datasets_pkg
from .. import methods as methods_pkg
from ..mam_model import TTTGPT2
from ..mam_outer import resolve_device
from . import metrics as metrics_pkg


def _parse_kv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_dataset_kwargs(s: str) -> dict:
    """--dataset-args 'profile_size=20,coarse_only=true' -> dict."""
    if not s:
        return {}
    out: dict = {}
    for kv in _parse_kv_list(s):
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        v_stripped = v.strip()
        if v_stripped.lower() in ("true", "false"):
            out[k.strip()] = v_stripped.lower() == "true"
        else:
            try:
                out[k.strip()] = int(v_stripped)
            except ValueError:
                try:
                    out[k.strip()] = float(v_stripped)
                except ValueError:
                    out[k.strip()] = v_stripped
    return out


def _take(it: Iterable, n: int):
    if n is None or n <= 0:
        yield from it
        return
    for i, x in enumerate(it):
        if i >= n:
            return
        yield x


def main() -> None:
    ap = argparse.ArgumentParser(description="Run methods × dataset × metrics.")
    ap.add_argument("--dataset", required=True, choices=datasets_pkg.available())
    ap.add_argument("--dataset-args", default="",
                    help="comma-separated k=v pairs forwarded to the dataset adapter.")
    ap.add_argument("--methods", default="baseline,icl,rag,ttt",
                    help=f"comma-separated subset of {methods_pkg.available()}")
    ap.add_argument("--metrics", default="bleu,rouge_l",
                    help=f"comma-separated subset of {metrics_pkg.available()}")
    ap.add_argument("--checkpoint", default=None,
                    help="optional meta-trained checkpoint; omit for raw GPT-2.")
    ap.add_argument("--max-examples", type=int, default=20,
                    help="cap test set size for quick runs; 0 means use all.")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--output-dir", default=None,
                    help="if set, write per-example results.csv and summary.json here.")
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[runner] device={device}", flush=True)

    # -- dataset --------------------------------------------------------------
    ds_kwargs = _parse_dataset_kwargs(args.dataset_args)
    ds = datasets_pkg.get(args.dataset, **ds_kwargs)
    print(f"[runner] dataset={args.dataset} info={ds.info()}", flush=True)

    # -- model + methods ------------------------------------------------------
    print("[runner] loading model...", flush=True)
    model = TTTGPT2("gpt2")
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model = model.to(device)
    model.eval()

    method_names = _parse_kv_list(args.methods)
    methods = [methods_pkg.get(name, model) for name in method_names]
    print(f"[runner] methods={method_names}", flush=True)

    metric_names = _parse_kv_list(args.metrics)
    metrics_list = [metrics_pkg.get(n) for n in metric_names]
    print(f"[runner] metrics={metric_names}", flush=True)

    # -- run loop -------------------------------------------------------------
    rows: list[dict] = []
    per_method_scores: dict[str, dict[str, list[float]]] = {
        m.name: {n: [] for n in metric_names} for m in methods
    }

    t0 = time.time()
    examples = list(_take(ds.test_examples(), args.max_examples))
    print(f"[runner] running {len(examples)} examples × {len(methods)} methods...\n", flush=True)

    for i, ex in enumerate(examples):
        for method in methods:
            method.prepare(ex)
            try:
                pred = method.predict(ex, max_new_tokens=args.max_new_tokens)
            finally:
                method.cleanup()

            row = {
                "example_idx": i,
                "user_id": ex.user_id,
                "method": method.name,
                "task_input": ex.task_input,
                "gold": ex.task_output,
                "prediction": pred,
                **ex.metadata,
            }
            for metric in metrics_list:
                score = metric(pred, ex.task_output, ex.metadata)
                row[metric.name] = score
                per_method_scores[method.name][metric.name].append(score)
            rows.append(row)

            print(f"  ex{i:>3} {method.name:<10} -> {pred[:80]!r}", flush=True)

    elapsed = time.time() - t0

    # -- summary --------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print(f"Summary  ({len(examples)} examples, {elapsed:.1f}s)", flush=True)
    print("=" * 70, flush=True)
    header = f"  {'method':<12} " + "  ".join(f"{n:>14}" for n in metric_names)
    print(header, flush=True)
    summary: dict = {"dataset": args.dataset, "n_examples": len(examples), "methods": {}}
    for m in methods:
        cells = []
        m_summary: dict = {}
        for n in metric_names:
            vals = per_method_scores[m.name][n]
            mean = statistics.mean(vals) if vals else float("nan")
            cells.append(f"{mean:>14.4f}")
            m_summary[n] = {"mean": mean, "n": len(vals)}
        print(f"  {m.name:<12} " + "  ".join(cells), flush=True)
        summary["methods"][m.name] = m_summary

    # -- optional persistence -------------------------------------------------
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, "results.csv")
        json_path = os.path.join(args.output_dir, "summary.json")
        if rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[runner] wrote {csv_path} and {json_path}", flush=True)


if __name__ == "__main__":
    main()
