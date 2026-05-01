"""TTT wrapper around pretrained GPT-2.

Design choices (cite paper sections from https://test-time-training.github.io/e2e.pdf):
  * Only the last quarter of transformer blocks have their MLP weights updated
    in the inner loop (paper §3.3: balances TTT "state capacity" vs compute).
    For GPT-2 124M with 12 blocks, that is blocks 9, 10, 11.
  * Attention, embeddings, and layernorms are NEVER touched by the inner loop
    (paper §3.2: inner-loop attention updates are empirically unstable).
  * Each late block gets a *parallel static MLP* — a frozen copy of the
    pretrained MLP whose output is added to the trainable MLP's output.
    (paper §3.4 / "preserve pretrained knowledge"). Without it, the trainable
    MLP can drift arbitrarily far from the pretrained behavior during inner
    updates. The static path anchors it.

This file was renamed from ``model.py`` to ``mam_model.py`` when the Flan-T5
implementation was merged in alongside (see ``flan_dual_mlp_model.py``).
"""

from __future__ import annotations

import copy
from typing import Iterator

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP


class DualMLP(nn.Module):
    """Runs a trainable MLP and a frozen-copy "static" MLP on the same input
    and sums their outputs.

    `trainable` is what the inner loop updates each test-time step; `static`
    is fixed throughout the inner loop (it IS updated by the outer loop, so
    meta-learning can still shape the pretrained-knowledge anchor).
    """

    def __init__(self, base_mlp: GPT2MLP):
        super().__init__()
        # Two independent deepcopies so their parameters are separate tensors.
        self.trainable = copy.deepcopy(base_mlp)
        self.static = copy.deepcopy(base_mlp)

    def forward(self, hidden_states):
        # Element-wise sum of two MLP branches. This doubles the MLP compute
        # in the last quarter of the model -- a modest cost for GPT-2.
        return self.trainable(hidden_states) + self.static(hidden_states)


class TTTGPT2(nn.Module):
    """GPT-2 LM head model with the last quarter of MLPs replaced by DualMLP.

    The HF `GPT2Block` reads its MLP as `.mlp`, so we can swap in place.
    """

    def __init__(self, model_name: str = "gpt2", ttt_fraction: float = 0.25):
        super().__init__()
        # `attn_implementation="eager"` avoids torch SDPA, whose CPU flash
        # kernel has no backward-of-backward -- which the outer loop needs.
        # On CUDA this is a no-op (eager attention works fine on GPU too),
        # so keeping it makes the code portable across both devices.
        self.lm = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # GPT-2 pad-less -- set pad = eos for generate() convenience.
        self.tokenizer.pad_token = self.tokenizer.eos_token

        blocks = self.lm.transformer.h
        n_blocks = len(blocks)
        # "Last quarter" -- paper §3.3.
        n_ttt = max(1, int(round(n_blocks * ttt_fraction)))
        self.ttt_block_indices = list(range(n_blocks - n_ttt, n_blocks))

        for idx in self.ttt_block_indices:
            blocks[idx].mlp = DualMLP(blocks[idx].mlp)

        self._mark_param_roles()

    # -- parameter grouping --------------------------------------------------

    def _mark_param_roles(self):
        """Tag every parameter with `._ttt_role` in {"inner", "outer"}.

        inner = updated at test time by the inner loop (fast weights φ).
        outer = updated only by the outer loop / meta-training (slow weights θ).

        Concretely: the `trainable` branches of DualMLPs in late blocks are
        "inner"; everything else (attention, embeddings, layernorms, early
        MLPs, and the "static" branches of DualMLPs) is "outer".
        """
        for name, p in self.named_parameters():
            # "mlp.trainable" is the branch inside DualMLP in a late block.
            if ".mlp.trainable." in name:
                p._ttt_role = "inner"
            else:
                p._ttt_role = "outer"

    def inner_params(self) -> Iterator[nn.Parameter]:
        for p in self.parameters():
            if getattr(p, "_ttt_role", "outer") == "inner":
                yield p

    def outer_params(self) -> Iterator[nn.Parameter]:
        for p in self.parameters():
            if getattr(p, "_ttt_role", "outer") == "outer":
                yield p

    def snapshot_inner(self) -> list[torch.Tensor]:
        """Detached clones of inner params -- used to reset between prompts
        so that TTT state is per-prompt, not a permanent finetune."""
        return [p.detach().clone() for p in self.inner_params()]

    def restore_inner(self, snapshot: list[torch.Tensor]) -> None:
        """Copy a snapshot back into the inner params, in-place.

        The .to(device, dtype) conversion lets a snapshot taken on CPU be
        restored onto a CUDA model and vice versa -- useful when moving
        checkpoints between machines.
        """
        with torch.no_grad():
            for p, s in zip(self.inner_params(), snapshot):
                p.copy_(s.to(device=p.device, dtype=p.dtype))

    # -- forward passthrough -------------------------------------------------

    def forward(self, input_ids, labels=None, **kw):
        return self.lm(input_ids=input_ids, labels=labels, **kw)

    def generate(self, *args, **kw):
        return self.lm.generate(*args, **kw)
