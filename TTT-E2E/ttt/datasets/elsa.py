"""ELSA dataset adapter.

ELSA = Emotion and Language Style Alignment Dataset
  (Gandhi & Gandhi 2025, https://aircconline.com/csit/papers/vol15/csit152201.pdf)
  HuggingFace: joyspace-ai/ELSA-Emotion-and-Language-Style-Alignment-Dataset

The raw dataset has one split, ``train``, with 10,434 rows. Each row:

  * id, dair_ai_id           -- bookkeeping
  * original_text            -- a neutral-ish source sentence
  * original_emotion         -- one of 6 coarse: sadness, anger, love,
                                surprise, fear, joy
  * emotion_type             -- one of 21 fine-grained GoEmotions
                                sub-labels (e.g. "grief", "remorse",
                                "excitement"). Some have stray emoji
                                characters; we strip those.
  * conversational, poetic,  -- 4 stylistic rewrites of original_text,
    formal, narrative           all preserving emotion_type.

Personalization framing:
  * user_id   = (fine_emotion, style)        -- up to 84 personas
  * profile   = N rewrites with that (fine_emotion, style)
  * task_in   = a held-out original_text
  * task_out  = that original_text rewritten in (fine_emotion, style)

This makes personalization very visible: (joy, poetic) vs (anger, formal)
produce structurally different outputs, unlike LaMP-7 where most users
sound similar.

Train/test split: per-persona. We sort each persona's rows by id (stable),
take the last ``test_per_persona`` rows as the test set; the rest go into
the persona's profile pool. This guarantees:
  * test rows never appear in any profile,
  * every test example has at least ``profile_size`` profile items
    available (provided the persona has enough rows).

Personas with fewer than ``min_persona_size`` rows are skipped.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator

from .base import DatasetAdapter, PersonalizedExample


# Stray emoji garbage in some emotion_type values. Strip and lowercase.
_EMOJI_RE = re.compile(r"[^\w\s-]", flags=re.UNICODE)


def _clean_emotion(s: str) -> str:
    return _EMOJI_RE.sub("", s).strip().lower()


STYLES = ("conversational", "poetic", "formal", "narrative")

# profile_size — how big each example's profile is (default 30)
# test_per_persona — held-out rows per (emotion, style) pair (default 10)
# min_persona_size — drop tiny personas (default 50)
# coarse_only — collapse to 6×4=24 personas with bigger profiles, vs 84 fine-grained
@dataclass
class ElsaConfig:
    profile_size: int = 30
    """How many profile items each test example carries."""

    test_per_persona: int = 10
    """How many test examples to draw per (emotion, style) persona."""

    min_persona_size: int = 50
    """Drop personas with fewer rows than this (avoid trivial profiles)."""

    coarse_only: bool = False
    """If True, treat all rows with the same coarse ``original_emotion``
    as one persona (6 emotions × 4 styles = 24 personas, much bigger
    profiles). If False, split by fine ``emotion_type`` (~84 personas)."""

    seed: int = 0
    """Controls profile sampling per test example (deterministic)."""


class ElsaAdapter(DatasetAdapter):
    name = "elsa"
    task_type = "generation"
    default_metric = "emotion_style_acc"  # see ttt/eval/metrics.py

    def __init__(self, config: ElsaConfig | None = None, hf_dataset=None):
        """
        config:     hyperparameters controlling profile/test sizes.
        hf_dataset: optional preloaded HuggingFace Dataset; if None we
                    load on demand. Keeping this injectable makes the
                    adapter easy to test with a tiny dummy.
        """
        self.cfg = config or ElsaConfig()
        self._hf_ds = hf_dataset
        self._train: list[PersonalizedExample] | None = None
        self._test: list[PersonalizedExample] | None = None

    # -- lazy load + index --------------------------------------------------

    def _load_raw(self):
        if self._hf_ds is None:
            from datasets import load_dataset

            self._hf_ds = load_dataset(
                "joyspace-ai/ELSA-Emotion-and-Language-Style-Alignment-Dataset",
                split="train",
            )
        return self._hf_ds

    def _persona_key(self, row) -> str:
        emo = (
            row["original_emotion"]
            if self.cfg.coarse_only
            else _clean_emotion(row["emotion_type"])
        )
        return emo

    def _build(self):
        """Group rows by (emotion, style), split, materialize examples."""
        if self._train is not None and self._test is not None:
            return

        ds = self._load_raw()
        # Group rows by (emotion, style). One ELSA row contributes one
        # (input, output) pair PER style column, so we explode here.
        by_persona: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in ds:
            emo = self._persona_key(row)
            if not emo:
                continue
            for style in STYLES:
                rewrite = (row[style] or "").strip()
                if not rewrite:
                    continue
                by_persona[(emo, style)].append(
                    {
                        "id": int(row["id"]),
                        "input": (row["original_text"] or "").strip(),
                        "output": rewrite,
                    }
                )

        train: list[PersonalizedExample] = []
        test: list[PersonalizedExample] = []
        rng = random.Random(self.cfg.seed)

        for (emo, style), rows in by_persona.items():
            if len(rows) < self.cfg.min_persona_size:
                continue
            # Stable order so the split is reproducible across runs.
            rows.sort(key=lambda r: r["id"])

            # Last test_per_persona go to test; the rest is the profile pool.
            test_rows = rows[-self.cfg.test_per_persona :]
            pool_rows = rows[: -self.cfg.test_per_persona] if self.cfg.test_per_persona else rows
            pool_outputs = [r["output"] for r in pool_rows]

            uid = f"{emo}-{style}"
            meta = {"emotion": emo, "style": style}

            # Train: every (input, output) pair in the pool. Profile = the
            # full pool minus the current item (so a method that just
            # parrots back from profile can't trivially win).
            for i, r in enumerate(pool_rows):
                profile = pool_outputs[:i] + pool_outputs[i + 1 :]
                if len(profile) > self.cfg.profile_size:
                    profile = rng.sample(profile, self.cfg.profile_size)
                train.append(
                    PersonalizedExample(
                        user_id=uid,
                        profile=profile,
                        task_input=r["input"],
                        task_output=r["output"],
                        metadata=meta,
                    )
                )

            # Test: each test row gets a profile sampled from the pool.
            for r in test_rows:
                if len(pool_outputs) > self.cfg.profile_size:
                    profile = rng.sample(pool_outputs, self.cfg.profile_size)
                else:
                    profile = list(pool_outputs)
                test.append(
                    PersonalizedExample(
                        user_id=uid,
                        profile=profile,
                        task_input=r["input"],
                        task_output=r["output"],
                        metadata=meta,
                    )
                )

        self._train = train
        self._test = test

    # -- DatasetAdapter API -------------------------------------------------

    def train_examples(self) -> Iterator[PersonalizedExample]:
        self._build()
        yield from self._train

    def test_examples(self) -> Iterator[PersonalizedExample]:
        self._build()
        yield from self._test

    def info(self) -> dict:
        self._build()
        personas = sorted({(e.metadata["emotion"], e.metadata["style"]) for e in self._test})
        return {
            **super().info(),
            "n_train": len(self._train),
            "n_test": len(self._test),
            "n_personas": len(personas),
            "config": vars(self.cfg),
        }
