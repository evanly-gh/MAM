"""TTT: adapt the model's inner weights on the user's profile, then generate.

prepare()  : snapshot inner params, then run the single-pass inner loop
             on a concatenation of all profile snippets. The model is
             now specialised to this user's style.
predict()  : standard prompt scaffold (no profile in prompt -- the model
             already absorbed it through weight updates).
cleanup()  : restore inner params from the snapshot. Per-prompt state,
             never sticks.
"""

from __future__ import annotations

import torch

from ..datasets.base import PersonalizedExample
from ..mam_inner import inner_adapt_inplace
from .base import Method, build_prompt


class TTTMethod(Method):
    name = "ttt"

    def __init__(
        self,
        model,
        tokenizer=None,
        device: torch.device | None = None,
        *,
        inner_lr: float = 1e-3,
        window: int = 256,
        max_profile_tokens: int = 1024,
    ):
        super().__init__(model, tokenizer=tokenizer, device=device)
        self.inner_lr = inner_lr
        self.window = window
        self.max_profile_tokens = max_profile_tokens
        self._snapshot: list[torch.Tensor] | None = None

    def prepare(self, example: PersonalizedExample) -> None:
        # Snapshot before any adaptation so cleanup() always has something
        # to restore, even if the model below decides to skip.
        self._snapshot = self.model.snapshot_inner()

        if not example.profile:
            return

        # Concatenate profile snippets with separators so windows have
        # natural boundaries.
        profile_text = "\n\n".join(s.strip() for s in example.profile if s.strip())
        if not profile_text:
            return

        ids = self.tokenizer.encode(profile_text, return_tensors="pt").to(self.device)
        if ids.size(-1) < 2:
            return
        if ids.size(-1) > self.max_profile_tokens:
            ids = ids[:, : self.max_profile_tokens]

        inner_adapt_inplace(self.model, ids, lr=self.inner_lr, window=self.window)

    def predict(self, example: PersonalizedExample, max_new_tokens: int = 80) -> str:
        prompt = build_prompt(example.task_input, references=None)
        return self._generate(prompt, max_new_tokens=max_new_tokens)

    def cleanup(self) -> None:
        if self._snapshot is not None:
            self.model.restore_inner(self._snapshot)
            self._snapshot = None
