"""Inner loop: the actual "test-time training" step.

At inference, we take K gradient steps on the next-token CE loss of the given
context, updating ONLY the inner params (trainable MLPs in late blocks).

Two entry points:
  * `inner_adapt_inplace(model, ...)` -- modifies `model` in place with plain
    SGD. Used at inference time (generate.py). Cheap, single-order gradients.
  * `inner_adapt_functional(fmodel, diffopt, ...)` -- takes a functional model
    view from `higher.innerloop_ctx` and a differentiable optimizer, so the
    outer loop can backprop THROUGH the inner updates (paper: "end-to-end" in
    the meta-learning sense).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _ce_next_token(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Standard causal LM loss: predict token t from tokens <t."""
    # Shift: logits[:, :-1] predicts input_ids[:, 1:].
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def _iter_windows(ids: torch.Tensor, window: int, stride: int):
    """Yield sliding windows over a 1D sequence of token ids."""
    n = ids.size(-1)
    if n <= window:
        yield ids
        return
    start = 0
    while start < n:
        end = min(start + window, n)
        yield ids[..., start:end]
        if end == n:
            break
        start += stride


@torch.enable_grad()
def inner_adapt_inplace(
    model,
    context_ids: torch.Tensor,
    steps: int = 1,
    lr: float = 1e-3,
    window: int = 256,
    stride: int | None = None,
):
    """Adapt `model`'s inner params on `context_ids` via plain SGD, in place.

    This is what the paper calls the inner loop at test time. It is a real
    training step -- forward, loss, backward, parameter update -- but it
    happens AT INFERENCE, on the prompt the user just handed us. No labels
    exist; the "labels" are the next tokens of the prompt itself.
    """
    if stride is None:
        stride = window  # non-overlapping windows -- cheapest
    inner_params = list(model.inner_params())
    opt = torch.optim.SGD(inner_params, lr=lr)

    model.train()  # enable dropout? GPT-2 default dropout is 0.1; acceptable.
    for _ in range(steps):
        for window_ids in _iter_windows(context_ids, window, stride):
            if window_ids.size(-1) < 2:
                continue
            opt.zero_grad()
            out = model(window_ids)
            loss = _ce_next_token(out.logits, window_ids)
            loss.backward()
            opt.step()
    model.eval()
    return model


def inner_adapt_functional(
    fmodel,
    diffopt,
    context_ids: torch.Tensor,
    steps: int = 1,
    window: int = 256,
    stride: int | None = None,
):
    """Inner loop using `higher`'s differentiable optimizer.

    `diffopt.step(loss)` records the parameter update in the autograd graph,
    so when the outer loop later calls `.backward()` on the meta-loss, the
    gradient flows:

        meta-loss --> adapted inner params φ' --> the inner update rule
                  --> the starting inner params φ AND the outer params θ
                      (θ enters through the forward passes that computed the
                      inner losses).

    This is the "gradient of gradient" that makes TTT-E2E end-to-end.
    """
    if stride is None:
        stride = window
    for _ in range(steps):
        for window_ids in _iter_windows(context_ids, window, stride):
            if window_ids.size(-1) < 2:
                continue
            out = fmodel(window_ids)
            loss = _ce_next_token(out.logits, window_ids)
            diffopt.step(loss)
    return fmodel
