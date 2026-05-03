"""RAG: retrieve top-K profile items most relevant to task_input, then ICL.

Same prompt scaffold as ICLMethod, but the K reference snippets are
chosen by similarity to the query instead of randomly. Implementation is
deliberately dependency-free: bag-of-words cosine similarity over
hash-vectorised tokens. If the user has scikit-learn or sentence-
transformers installed, swap the retriever in without touching anything
else.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import torch

from ..datasets.base import PersonalizedExample
from .base import Method, build_prompt


_WORD_RE = re.compile(r"\w+")


def _tokenise(s: str) -> list[str]:
    return _WORD_RE.findall(s.lower())


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    inter = set(a) & set(b)
    num = sum(a[t] * b[t] for t in inter)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    return num / (da * db) if da and db else 0.0


class RAGMethod(Method):
    name = "rag"

    def __init__(
        self,
        model,
        tokenizer=None,
        device: torch.device | None = None,
        *,
        n_retrieved: int = 5,
    ):
        super().__init__(model, tokenizer=tokenizer, device=device)
        self.n_retrieved = n_retrieved
        self._references: list[str] = []

    def prepare(self, example: PersonalizedExample) -> None:
        if not example.profile:
            self._references = []
            return
        query_vec = Counter(_tokenise(example.task_input))
        scored = [(_cosine(query_vec, Counter(_tokenise(p))), p) for p in example.profile]
        scored.sort(key=lambda x: x[0], reverse=True)
        self._references = [p for _, p in scored[: self.n_retrieved]]

    def predict(self, example: PersonalizedExample, max_new_tokens: int = 80) -> str:
        prompt = build_prompt(example.task_input, references=self._references)
        return self._generate(prompt, max_new_tokens=max_new_tokens)

    def cleanup(self) -> None:
        self._references = []
