"""Personalization via TTT: write `--prompt` in the style of `--style-file`.

Pipeline:
  1. Load meta-trained TTT-GPT-2.
  2. Read a corpus of one person's writing (`--style-file`).
  3. Run the inner loop on that corpus -- this is the personalization step.
     The fast weights now encode "what writing in this person's voice looks
     like." This is exactly what TTT was designed for: per-prompt adaptation.
  4. Generate from `--prompt` using the adapted weights.
  5. (Optional) Score the output against a held-out reference message in the
     same style, using a pluggable scorer (see ttt/scorers.py).
  6. Reset fast weights so the next call starts fresh.

The inner loop is single-pass; the strength of personalization scales with the
length of the style corpus, not the number of repeats.

Example:
  python -m ttt.personalize \\
      --checkpoint checkpoints/latest.pt \\
      --style-file styles/alice.txt \\
      --prompt "Subject: lunch tomorrow\\n\\nHi Bob," \\
      --reference styles/alice_held_out.txt \\
      --scorer perplexity --inner-lr 1e-3
"""

from __future__ import annotations

import argparse

import torch

from .mam_inner import inner_adapt_inplace
from .mam_model import TTTGPT2
from .mam_outer import resolve_device
from .scorers import SCORERS, get as get_scorer


def personalize(
    model: TTTGPT2,
    style_corpus: str,
    prompt: str,
    *,
    inner_lr: float,
    window: int,
    max_new_tokens: int,
    adapt: bool,
) -> tuple[str, str]:
    tok = model.tokenizer
    device = next(model.parameters()).device
    style_ids = tok.encode(style_corpus, return_tensors="pt").to(device)
    prompt_ids = tok.encode(prompt, return_tensors="pt").to(device)

    snap = model.snapshot_inner()
    try:
        if adapt:
            # The TTT step IS the personalization. We are training the model
            # on this person's text, at inference time, just before generation.
            inner_adapt_inplace(model, style_ids, lr=inner_lr, window=window)
        with torch.no_grad():
            out = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic for fair scoring
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        gen_only = tok.decode(out[0, prompt_ids.size(-1):], skip_special_tokens=True)
        return text, gen_only
    finally:
        # Reset: personalization is per-call, never sticks.
        model.restore_inner(snap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--style-file", type=str, required=True,
                    help="Corpus of the target person's writing.")
    ap.add_argument("--prompt", type=str, default=None,
                    help="What to write. Either this or --prompt-file.")
    ap.add_argument("--prompt-file", type=str, default=None)
    ap.add_argument("--reference", type=str, default=None,
                    help="Optional held-out message in the target style. "
                         "Required for scorers that need a reference (e.g. ngram).")
    ap.add_argument("--scorer", type=str, default=None,
                    choices=list(SCORERS),
                    help="Scoring metric. Omit to skip scoring.")
    ap.add_argument("--inner-lr", type=float, default=1e-3)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--baseline", action="store_true",
                    help="Also run with adaptation disabled and print both, "
                         "so you can see what TTT contributed.")
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="auto picks cuda if available, else cpu.",
    )
    args = ap.parse_args()

    if args.prompt is None and args.prompt_file is None:
        raise SystemExit("provide --prompt or --prompt-file")
    prompt = args.prompt or open(args.prompt_file, encoding="utf-8").read()
    style = open(args.style_file, encoding="utf-8").read()
    reference = open(args.reference, encoding="utf-8").read() if args.reference else None

    device = resolve_device(args.device)
    print(f"[personalize] device={device}")

    model = TTTGPT2("gpt2")
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model = model.to(device)

    runs: list[tuple[str, bool]] = []
    if args.baseline:
        runs.append(("no-TTT  ", False))
    runs.append(("TTT     ", True))

    scorer = get_scorer(args.scorer) if args.scorer else None

    for label, adapt in runs:
        full, gen_only = personalize(
            model, style, prompt,
            inner_lr=args.inner_lr,
            window=args.window,
            max_new_tokens=args.max_new_tokens,
            adapt=adapt,
        )
        print(f"\n=== {label} ===")
        print(full)
        if scorer is not None:
            # Re-adapt for scorers that need to look at text under the
            # personalized model (e.g. perplexity).
            if scorer.name == "perplexity":
                snap = model.snapshot_inner()
                if adapt:
                    style_ids = model.tokenizer.encode(style, return_tensors="pt").to(device)
                    inner_adapt_inplace(model, style_ids, lr=args.inner_lr, window=args.window)
                s = scorer(gen_only, reference or "", model=model)
                model.restore_inner(snap)
            else:
                s = scorer(gen_only, reference or "", model=model)
            direction = "↓" if scorer.lower_is_better else "↑"
            print(f"[score:{scorer.name} {direction}]  {s:.4f}")


if __name__ == "__main__":
    main()
