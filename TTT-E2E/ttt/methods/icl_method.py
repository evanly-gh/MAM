"""In-context learning: stuff K random profile items into the prompt.

No adaptation, no retrieval -- just take the first / random K snippets
from the profile and show them to the model as reference style.

This is the cheapest form of personalization and a strong baseline
because LLMs are generally good at imitating styles they've just seen.
"""

from __future__ import annotations

import random

import torch

from ..datasets.base import PersonalizedExample
from .base import Method, build_prompt


class ICLMethod(Method):
    name = "icl"

    def __init__(
        self,
        model,
        tokenizer=None,
        device: torch.device | None = None,
        *,
        n_examples: int = 5,
        seed: int = 0,
    ):
        super().__init__(model, tokenizer=tokenizer, device=device)
        self.n_examples = n_examples
        self.rng = random.Random(seed)
        self._references: list[str] = []

    def prepare(self, example: PersonalizedExample) -> None:
        if len(example.profile) <= self.n_examples:
            self._references = list(example.profile)
        else:
            self._references = self.rng.sample(example.profile, self.n_examples)

    def predict(self, example: PersonalizedExample, max_new_tokens: int = 80) -> str:
        prompt = build_prompt(example.task_input, references=self._references)
        return self._generate(prompt, max_new_tokens=max_new_tokens)

    def cleanup(self) -> None:
        self._references = []
