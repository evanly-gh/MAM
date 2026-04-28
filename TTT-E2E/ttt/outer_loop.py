"""Outer loop: meta-train the base weights so the model performs well
*after* the inner loop has run.

Pseudocode of one meta-step:

    sample (context, continuation) from corpus
    φ₀  = current inner params  (starting point for TTT at test time)
    θ   = current outer params  (everything else)

    with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False):
        # functional model whose inner params are tracked in autograd graph
        run inner_adapt_functional: φ' = φ₀ - η ∇φ L_ctx(φ₀; θ)   (x K steps)
        meta_loss = L_cont(φ'; θ)    # CE on held-out continuation
        meta_loss.backward()         # second-order: grad flows into φ₀ AND θ

    outer_opt.step()   # updates both φ₀ and θ

`copy_initial_weights=False` is what makes φ₀ itself a meta-learned parameter.
If it were True, the inner loop would start from a detached copy and gradient
would not flow back into the real φ₀ -- we'd only be meta-training θ.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import higher
import torch
from tqdm import tqdm

from .data import meta_example_stream
from .inner_loop import _ce_next_token, inner_adapt_functional
from .model import TTTGPT2


def run(
    meta_steps: int = 2000,
    inner_steps: int = 1,
    inner_lr: float = 1e-3,
    outer_lr_outer: float = 1e-5,
    outer_lr_inner_init: float = 1e-4,
    context_len: int = 256,
    continuation_len: int = 64,
    window: int = 256,
    ckpt_dir: str = "checkpoints",
    log_path: str = "logs/meta_loss.csv",
    ckpt_every: int = 200,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    model = TTTGPT2("gpt2")
    # CPU path -- the user's laptop has no GPU. fp32 throughout.
    model = model.to("cpu")

    # Two parameter groups: outer weights (θ) get a small lr, the initial
    # inner weights (φ₀) get a larger lr because they are specifically the
    # starting point that the inner loop perturbs -- we want them to move
    # faster so the inner loop has a good anchor quickly.
    outer_opt = torch.optim.Adam(
        [
            {"params": list(model.outer_params()), "lr": outer_lr_outer},
            {"params": list(model.inner_params()), "lr": outer_lr_inner_init},
        ]
    )
    # Plain SGD for the inner loop: `higher` supports this out of the box, and
    # a momentum-free optimizer keeps the differentiable graph small.
    inner_opt = torch.optim.SGD(model.inner_params(), lr=inner_lr)

    stream = meta_example_stream(
        model.tokenizer,
        context_len=context_len,
        continuation_len=continuation_len,
    )

    log_file = open(log_path, "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["meta_step", "meta_loss", "elapsed_s"])
    t0 = time.time()

    pbar = tqdm(range(meta_steps), desc="meta-train")
    for step in pbar:
        ctx, cont = next(stream)

        outer_opt.zero_grad()

        # `track_higher_grads=True` (default) keeps the inner-update graph so
        # we can call .backward() on a loss computed through it. This is the
        # machinery behind the paper's end-to-end training.
        with higher.innerloop_ctx(
            model,
            inner_opt,
            copy_initial_weights=False,
            track_higher_grads=True,
        ) as (fmodel, diffopt):
            inner_adapt_functional(
                fmodel, diffopt, ctx, steps=inner_steps, window=window
            )

            # Meta-loss: teacher-force the continuation AFTER the context,
            # and take CE on the continuation tokens only. We feed the full
            # (context + continuation) sequence so positional embeddings and
            # attention see continuation tokens in their proper position.
            full = torch.cat([ctx, cont], dim=-1)
            logits = fmodel(full).logits
            # Predict token t from tokens <t. Continuation tokens are the
            # last `continuation_len` positions of `full`.
            cl = cont.size(-1)
            pred_logits = logits[:, -cl - 1 : -1, :]  # aligns with cont tokens
            meta_loss = torch.nn.functional.cross_entropy(
                pred_logits.reshape(-1, pred_logits.size(-1)),
                cont.reshape(-1),
            )

            # Second-order backward: grad flows through inner_adapt_functional
            # into both φ₀ (model.inner_params()) and θ (model.outer_params()).
            meta_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(model.outer_params()) + list(model.inner_params()), 1.0
        )
        outer_opt.step()

        elapsed = time.time() - t0
        log.writerow([step, meta_loss.item(), f"{elapsed:.1f}"])
        log_file.flush()
        pbar.set_postfix(loss=f"{meta_loss.item():.3f}")

        if (step + 1) % ckpt_every == 0 or step == meta_steps - 1:
            path = os.path.join(ckpt_dir, f"ttt_gpt2_meta_{step + 1}.pt")
            torch.save(model.state_dict(), path)
            latest = os.path.join(ckpt_dir, "latest.pt")
            torch.save(model.state_dict(), latest)

    log_file.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-steps", type=int, default=2000)
    ap.add_argument("--inner-steps", type=int, default=1)
    ap.add_argument("--inner-lr", type=float, default=1e-3)
    ap.add_argument("--context-len", type=int, default=256)
    ap.add_argument("--continuation-len", type=int, default=64)
    ap.add_argument("--ckpt-every", type=int, default=200)
    args = ap.parse_args()
    run(
        meta_steps=args.meta_steps,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        context_len=args.context_len,
        continuation_len=args.continuation_len,
        ckpt_every=args.ckpt_every,
    )


if __name__ == "__main__":
    main()
