"""Pluggable scorers for the personalization task.

A scorer answers: "how well does `generated` match the style of `reference`?"
Lower-is-better OR higher-is-better is encoded in the `lower_is_better`
attribute so callers can compare two configurations correctly.

Adding a new scorer:
  1. Define a function `score(generated: str, reference: str, model=None) -> float`
  2. Wrap it in `Scorer(...)` and register it in SCORERS below.
  3. Use it from the CLI via `--scorer your_name`.

The model is passed in optionally so scorers that need a language model
(e.g. perplexity) can use the same one we just adapted, rather than loading
their own.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from .model import TTTGPT2


@dataclass
class Scorer:
    name: str
    fn: Callable[..., float]
    lower_is_better: bool
    description: str

    def __call__(self, *args, **kwargs) -> float:
        return self.fn(*args, **kwargs)


# ---- individual scorers -----------------------------------------------------


def perplexity_under_model(generated: str, reference: str, model: TTTGPT2 = None) -> float:
    """Perplexity of `generated` under the (assumed already TTT-adapted) model.

    Interpretation: if the model was just adapted on the person's writing,
    text "in their style" should look LOW-PERPLEXITY to it. Generated text
    that wanders off-style will look high-perplexity.

    `reference` is unused here -- kept in signature for interface uniformity.
    """
    assert model is not None, "perplexity_under_model needs a model"
    ids = model.tokenizer.encode(generated, return_tensors="pt")
    if ids.size(-1) < 2:
        return float("inf")
    with torch.no_grad():
        logits = model(ids).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = ids[:, 1:]
    nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
    )
    return math.exp(nll.item())


_WORD_RE = re.compile(r"\w+")


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def ngram_overlap(generated: str, reference: str, model=None, n: int = 3) -> float:
    """Jaccard similarity of n-grams between `generated` and `reference`.

    A blunt lexical-style proxy: shared phrasing, characteristic word
    sequences. Robust but ignores semantics.
    """
    g = _WORD_RE.findall(generated.lower())
    r = _WORD_RE.findall(reference.lower())
    if len(g) < n or len(r) < n:
        return 0.0
    G, R = _ngrams(g, n), _ngrams(r, n)
    inter = sum((G & R).values())
    union = sum((G | R).values())
    return inter / union if union else 0.0


def style_stats_distance(generated: str, reference: str, model=None) -> float:
    """L1 distance between summary stats: mean sentence length, type-token
    ratio, mean word length. Crude but interpretable.

    Lower = more stylistically similar.
    """

    def stats(s: str) -> tuple[float, float, float]:
        words = _WORD_RE.findall(s)
        if not words:
            return (0.0, 0.0, 0.0)
        sents = re.split(r"[.!?]+", s)
        sents = [x for x in sents if x.strip()]
        msl = len(words) / max(1, len(sents))
        ttr = len(set(w.lower() for w in words)) / len(words)
        mwl = sum(len(w) for w in words) / len(words)
        return (msl, ttr, mwl)

    a, b = stats(generated), stats(reference)
    # Normalize each component to a comparable scale before summing.
    weights = (1 / 20.0, 1.0, 1 / 5.0)  # rough scale: msl~20, ttr~1, mwl~5
    return sum(abs(x - y) * w for x, y, w in zip(a, b, weights))


# ---- registry ---------------------------------------------------------------


SCORERS: dict[str, Scorer] = {
    "perplexity": Scorer(
        "perplexity",
        perplexity_under_model,
        lower_is_better=True,
        description="Perplexity of generated text under the TTT-adapted model. "
                    "Low = the model finds the generation likely under the "
                    "person's-style distribution it just absorbed.",
    ),
    "ngram": Scorer(
        "ngram",
        ngram_overlap,
        lower_is_better=False,
        description="Trigram Jaccard between generation and reference corpus. "
                    "Captures shared phrasing; ignores semantics.",
    ),
    "style-stats": Scorer(
        "style-stats",
        style_stats_distance,
        lower_is_better=True,
        description="L1 distance between (mean sent length, TTR, mean word "
                    "length). Crude surface-level style proxy.",
    ),
}


def get(name: str) -> Scorer:
    if name not in SCORERS:
        raise SystemExit(f"unknown scorer {name!r}; choices: {list(SCORERS)}")
    return SCORERS[name]
