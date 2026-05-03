"""Method interface: TTT, RAG, ICL, LoRA, ... all reduce to the same shape.

Every method:
  1. ``prepare(example)``  -- per-example setup. TTT runs the inner loop,
                              RAG indexes the profile, ICL formats a prompt
                              prefix, baseline does nothing.
  2. ``predict(example)``  -- generates the answer for example.task_input.
                              All methods end up calling model.generate(...).
  3. ``cleanup()``         -- undo per-example state. TTT restores fast
                              weights; RAG drops the index; ICL clears the
                              prefix.

Shared prompt scaffold (so methods are comparable):

    {reference_block}Rewrite the following text:
    {task_input}

    Rewrite:

`reference_block` is empty for baseline / TTT and populated with profile
snippets for ICL / RAG. Keeping the rest of the prompt identical makes
the diff between methods solely about *information access*, not about
prompt engineering.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from ..datasets.base import PersonalizedExample


REWRITE_TASK_TEMPLATE = (
    "{reference_block}Rewrite the following text:\n"
    "{task_input}\n\n"
    "Rewrite:"
)


def build_reference_block(snippets: list[str]) -> str:
    """Format profile snippets as a reference block, or empty string."""
    if not snippets:
        return ""
    bullets = "\n".join(f"- {s.strip()}" for s in snippets)
    return f"Reference style examples:\n{bullets}\n\n"


def build_prompt(task_input: str, references: list[str] | None = None) -> str:
    block = build_reference_block(references or [])
    return REWRITE_TASK_TEMPLATE.format(reference_block=block, task_input=task_input.strip())


class Method(ABC):
    """One per technique. Stateful across prepare/predict/cleanup."""

    name: str  # "ttt" / "rag" / "icl" / "baseline"

    def __init__(self, model, tokenizer=None, device: torch.device | None = None):
        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.device = device or next(model.parameters()).device

    @abstractmethod
    def prepare(self, example: PersonalizedExample) -> None:
        """Per-example setup. Default subclasses override."""

    @abstractmethod
    def predict(self, example: PersonalizedExample, max_new_tokens: int = 80) -> str:
        """Generate one answer for example.task_input."""

    def cleanup(self) -> None:
        """Reset per-example state. Default: no-op."""
        return None

    # -- shared helpers ----------------------------------------------------

    def _generate(self, prompt: str, max_new_tokens: int = 80) -> str:
        """Greedy decode `max_new_tokens` and return only the first new line.

        The prompt ends with "Rewrite:" so the model's first non-empty line
        is the answer. Anything after a blank line is the model trying to
        roll a new exchange; we drop it.
        """
        ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0, ids.size(-1):]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        # First non-empty line is the answer.
        for line in text.splitlines():
            if line.strip():
                return line.strip()
        return text.strip()
