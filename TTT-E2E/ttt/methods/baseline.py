"""Baseline: same prompt scaffold, no profile, no adaptation.

Establishes the floor: what does the model produce when it has no idea
who the user is and hasn't been adapted on anything? Other methods'
gains are measured against this.
"""

from __future__ import annotations

from ..datasets.base import PersonalizedExample
from .base import Method, build_prompt


class BaselineMethod(Method):
    name = "baseline"

    def prepare(self, example: PersonalizedExample) -> None:
        return None

    def predict(self, example: PersonalizedExample, max_new_tokens: int = 80) -> str:
        prompt = build_prompt(example.task_input, references=None)
        return self._generate(prompt, max_new_tokens=max_new_tokens)
