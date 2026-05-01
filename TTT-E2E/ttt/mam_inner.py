"""Inner loop: the actual "test-time training" step.

At inference, we make ONE left-to-right streaming pass over the context,
taking ONE SGD step per sliding window. The inner loop has no `steps`
parameter -- if you want more updates, use a longer context (more windows)
rather than re-reading the same windows.

This single-pass semantics matches the TTT-E2E paper (Sun et al.,
arXiv:2512.23675) more faithfully than the older K-step version: the
paper's picture is "the model streams the prompt once, updating as it
goes," not "fine-tune on the prompt as if it were a tiny dataset."

Why this matters:
  * The earlier K-step version meant the FIRST window got K updates before
    the LAST window was ever seen. That breaks the streaming narrative and
    biases adaptation toward early-context tokens.
  * For meta-training with `higher`, more inner steps = a deeper
    differentiable graph = more memory + noisier second-order gradients.
    Single-pass keeps the graph small and the meta-signal cleaner.

Two entry points:
  * `inner_adapt_inplace(model, ...)` -- modifies `model` in place with plain
    SGD. Used at inference time (generate.py / compare.py / bench.py).
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
    lr: float = 1e-3,
    window: int = 256,
    stride: int | None = None,
):
    """Adapt `model`'s inner params on `context_ids` via plain SGD, in place.

    One left-to-right pass: one SGD step per sliding window over the context.
    No labels exist; the "labels" are the next tokens of the prompt itself.

    This is what runs at inference time. It mutates the model's inner params
    in place; pair it with `model.snapshot_inner()` / `model.restore_inner()`
    if you want per-prompt state that resets between calls.
    """
    if stride is None:
        stride = window  # non-overlapping windows -- cheapest
    inner_params = list(model.inner_params())
    opt = torch.optim.SGD(inner_params, lr=lr)

    model.train()  # enable dropout? GPT-2 default dropout is 0.1; acceptable.
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
    window: int = 256,
    stride: int | None = None,
):
    """Inner loop using `higher`'s differentiable optimizer.

    Same single-pass semantics as `inner_adapt_inplace`, but each
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
    for window_ids in _iter_windows(context_ids, window, stride):
        if window_ids.size(-1) < 2:
            continue
        out = fmodel(window_ids)
        loss = _ce_next_token(out.logits, window_ids)
        diffopt.step(loss)
    return fmodel
