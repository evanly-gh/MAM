"""LaMP-5 / LaMP-7 dataset adapters.

Thin port of lamp/data/data_io.py focused on producing
PersonalizedExample objects -- nothing more. The full lamp repo also
handles LoRA-train merging and BLEU/ROUGE/METEOR scoring; we keep
those concerns separate (see ttt/eval/metrics.py).

LaMP file format (https://lamp-benchmark.github.io):
  * <split>_questions.json: list of {"id", "input", "profile"}
      profile is a list of past items (papers / tweets / etc.) belonging
      to this user.
  * <split>_outputs.json: gold answers, either
      [{"id", "output"}, ...]   or
      {"task": "LaMP_5", "golds": [{"id", "output"}, ...]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator

from .base import DatasetAdapter, PersonalizedExample


def _read_questions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_outputs(path: str) -> dict[str, str]:
    """Returns id -> gold output, handling both layouts."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "golds" in data:
        data = data["golds"]
    return {str(o["id"]): o["output"] for o in data}


def _flatten_profile(task: str, profile: list[dict]) -> list[str]:
    """Return one history-item-string per profile entry."""
    out: list[str] = []
    if task == "LaMP-5":
        for p in profile:
            t = (p.get("title") or "").strip()
            a = (p.get("abstract") or "").strip()
            if t and a:
                out.append(f"[title] {t}\n[abstract] {a}")
            elif a:
                out.append(f"[abstract] {a}")
            elif t:
                out.append(f"[title] {t}")
    elif task == "LaMP-7":
        for p in profile:
            tx = (p.get("text") or "").strip()
            if tx:
                out.append(tx)
    else:
        raise ValueError(task)
    return out


@dataclass
class LampConfig:
    task: str  # "LaMP-5" or "LaMP-7"
    train_questions_json: str
    train_outputs_json: str
    test_questions_json: str
    test_outputs_json: str


class LampAdapter(DatasetAdapter):
    name = "lamp"  # set explicitly per instance via .name = "lamp-5" etc.
    task_type = "generation"
    default_metric = "rouge_l"

    def __init__(self, config: LampConfig):
        self.cfg = config
        self.name = config.task.lower().replace("_", "-")  # "lamp-5"

    def _load(self, q_path: str, o_path: str) -> list[PersonalizedExample]:
        qs = _read_questions(q_path)
        outs = _read_outputs(o_path)
        out: list[PersonalizedExample] = []
        for q in qs:
            qid = str(q["id"])
            if qid not in outs:
                continue
            profile_strs = _flatten_profile(self.cfg.task, q.get("profile") or [])
            if not profile_strs:
                continue
            out.append(
                PersonalizedExample(
                    user_id=qid,  # LaMP doesn't expose user_id; row id is unique-per-user enough for our purposes
                    profile=profile_strs,
                    task_input=q["input"],
                    task_output=outs[qid],
                    metadata={"task": self.cfg.task},
                )
            )
        return out

    def train_examples(self) -> Iterator[PersonalizedExample]:
        yield from self._load(self.cfg.train_questions_json, self.cfg.train_outputs_json)

    def test_examples(self) -> Iterator[PersonalizedExample]:
        yield from self._load(self.cfg.test_questions_json, self.cfg.test_outputs_json)
