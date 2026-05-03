"""Pluggable metrics for the evaluation runner.

Every metric is a callable: ``score(prediction, gold, metadata) -> float``.
``higher_is_better`` is captured per metric so the runner can rank rows.

Builtins:
  * exact_match           prediction == gold (case/space-normalised)
  * bleu                  4-gram BLEU vs gold (pure-python smoothed BLEU)
  * rouge_l               ROUGE-L F-measure vs gold (LCS-based)
  * emotion_acc           pretrained GoEmotions classifier; check predicted
                          coarse emotion vs metadata["emotion"] (or a
                          GoEmotions sub-label).
  * style_acc_zeroshot    zero-shot NLI classifier; check predicted style
                          vs metadata["style"]. Slower; use sparingly.
  * emotion_style_acc     average of emotion_acc and style_acc_zeroshot.

All HF models are lazy-loaded on first call so a runner that doesn't use
them never pays the download cost.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable


_WORD_RE = re.compile(r"\w+")


def _tokenise(s: str) -> list[str]:
    return _WORD_RE.findall(s.lower())


@dataclass
class Metric:
    name: str
    fn: Callable[..., float]
    higher_is_better: bool

    def __call__(self, prediction: str, gold: str, metadata: dict | None = None) -> float:
        return float(self.fn(prediction, gold, metadata or {}))


# -- exact match ---------------------------------------------------------------


def _exact_match(pred: str, gold: str, meta: dict) -> float:
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0


# -- BLEU (pure python, with add-one smoothing) --------------------------------


def _ngrams(toks: list[str], n: int) -> Counter:
    return Counter(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))


def _bleu(pred: str, gold: str, meta: dict, max_n: int = 4) -> float:
    p_tok = _tokenise(pred)
    g_tok = _tokenise(gold)
    if not p_tok or not g_tok:
        return 0.0
    weights = [1.0 / max_n] * max_n
    log_p = 0.0
    for n, w in enumerate(weights, start=1):
        p_ng = _ngrams(p_tok, n)
        g_ng = _ngrams(g_tok, n)
        # add-one smoothing avoids log(0).
        match = sum((p_ng & g_ng).values()) + 1
        total = sum(p_ng.values()) + 1
        log_p += w * math.log(match / total)
    bp = 1.0 if len(p_tok) > len(g_tok) else math.exp(1 - len(g_tok) / max(1, len(p_tok)))
    return bp * math.exp(log_p)


# -- ROUGE-L (LCS-based F1) ----------------------------------------------------


def _lcs_len(a: list[str], b: list[str]) -> int:
    """Standard DP LCS. Quadratic memory; fine for sentence-length inputs."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i, x in enumerate(a, start=1):
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def _rouge_l(pred: str, gold: str, meta: dict) -> float:
    p_tok = _tokenise(pred)
    g_tok = _tokenise(gold)
    if not p_tok or not g_tok:
        return 0.0
    lcs = _lcs_len(p_tok, g_tok)
    if lcs == 0:
        return 0.0
    prec = lcs / len(p_tok)
    rec = lcs / len(g_tok)
    return 2 * prec * rec / (prec + rec)


# -- emotion classifier (GoEmotions) -------------------------------------------


_EMO_PIPE = None  # lazy
_GOEMOTION_TO_COARSE = {
    "sadness": "sadness", "grief": "sadness", "remorse": "sadness", "disappointment": "sadness",
    "anger": "anger", "annoyance": "anger", "disapproval": "anger", "embarrassment": "anger",
    "love": "love", "admiration": "love", "caring": "love",
    "surprise": "surprise", "realization": "surprise",
    "fear": "fear", "nervousness": "fear",
    "joy": "joy", "excitement": "joy", "pride": "joy", "gratitude": "joy", "amusement": "joy",
}


def _get_emotion_pipe():
    global _EMO_PIPE
    if _EMO_PIPE is not None:
        return _EMO_PIPE
    from transformers import pipeline

    _EMO_PIPE = pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=1,
        truncation=True,
    )
    return _EMO_PIPE


def _emotion_acc(pred: str, gold: str, meta: dict) -> float:
    target = (meta.get("emotion") or "").strip().lower()
    if not target or not pred.strip():
        return 0.0
    coarse_target = _GOEMOTION_TO_COARSE.get(target, target)
    pipe = _get_emotion_pipe()
    res = pipe(pred[:512])
    # pipeline with top_k=1 returns [[{"label": ..., "score": ...}]]
    label = res[0][0]["label"].strip().lower() if isinstance(res[0], list) else res[0]["label"].strip().lower()
    coarse_pred = _GOEMOTION_TO_COARSE.get(label, label)
    return 1.0 if coarse_pred == coarse_target else 0.0


# -- style classifier (zero-shot NLI) ------------------------------------------


_STYLE_PIPE = None
_STYLES = ["formal", "conversational", "poetic", "narrative"]


def _get_style_pipe():
    global _STYLE_PIPE
    if _STYLE_PIPE is not None:
        return _STYLE_PIPE
    from transformers import pipeline

    _STYLE_PIPE = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        truncation=True,
    )
    return _STYLE_PIPE


def _style_acc_zeroshot(pred: str, gold: str, meta: dict) -> float:
    target = (meta.get("style") or "").strip().lower()
    if not target or not pred.strip():
        return 0.0
    pipe = _get_style_pipe()
    res = pipe(pred[:512], candidate_labels=_STYLES)
    return 1.0 if res["labels"][0].lower() == target else 0.0


def _emotion_style_acc(pred: str, gold: str, meta: dict) -> float:
    return 0.5 * (_emotion_acc(pred, gold, meta) + _style_acc_zeroshot(pred, gold, meta))


# -- registry ------------------------------------------------------------------


METRICS: dict[str, Metric] = {
    "exact_match":        Metric("exact_match",        _exact_match,        higher_is_better=True),
    "bleu":               Metric("bleu",               _bleu,               higher_is_better=True),
    "rouge_l":            Metric("rouge_l",            _rouge_l,            higher_is_better=True),
    "emotion_acc":        Metric("emotion_acc",        _emotion_acc,        higher_is_better=True),
    "style_acc_zeroshot": Metric("style_acc_zeroshot", _style_acc_zeroshot, higher_is_better=True),
    "emotion_style_acc":  Metric("emotion_style_acc",  _emotion_style_acc,  higher_is_better=True),
}


def get(name: str) -> Metric:
    if name not in METRICS:
        raise KeyError(f"unknown metric {name!r}; choices: {sorted(METRICS)}")
    return METRICS[name]


def available() -> list[str]:
    return sorted(METRICS)
