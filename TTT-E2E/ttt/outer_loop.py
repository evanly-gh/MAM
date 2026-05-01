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
    # exist_ok=True means "don't crash if the folder already exists"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    # "gpt2" downloads the 124M pretrained weights from Hugging Face on first
    # run (~500 MB). Subsequent runs load from a local cache automatically.
    # TTTGPT2 wraps GPT-2 and replaces the last quarter of MLP layers with
    # DualMLPs (trainable + frozen static branch) -- see model.py.
    model = TTTGPT2("gpt2")
    # CPU path -- the user's laptop has no GPU. fp32 throughout.
    model = model.to("cpu")

    # Two parameter groups: outer weights (θ) get a small lr, the initial
    # inner weights (φ₀) get a larger lr because they are specifically the
    # starting point that the inner loop perturbs -- we want them to move
    # faster so the inner loop has a good anchor quickly.
    #
    # Adam is used here because it adapts its step size per parameter using
    # a running average of past gradients. This makes it more stable and
    # faster to converge than plain SGD for the long outer training loop.
    outer_opt = torch.optim.Adam(
        [
            {"params": list(model.outer_params()), "lr": outer_lr_outer},
            {"params": list(model.inner_params()), "lr": outer_lr_inner_init},
        ]
    )
    # SGD (not Adam) for the inner loop for two reasons:
    #   1. Adam tracks extra state (momentum buffers) per parameter. Inside
    #      `higher`'s differentiable graph, that state would need to be
    #      differentiated through too -- doubling the graph size and memory.
    #   2. The inner loop only takes 1-3 steps, so Adam's adaptive momentum
    #      doesn't have time to help anyway. SGD is simpler, cheaper, and
    #      works just as well here.
    # Note: this optimizer is handed to `higher` below -- `higher` wraps it
    # into `diffopt`, a differentiable version that records each step.
    inner_opt = torch.optim.SGD(model.inner_params(), lr=inner_lr)

    # meta_example_stream yields an infinite sequence of (context, continuation)
    # token-ID pairs sampled randomly from WikiText-103 (a large collection of
    # Wikipedia articles). On first run, it downloads the dataset from Hugging
    # Face and saves a tokenized cache to .cache/wikitext103_train.pt so future
    # runs skip the download. The tokenizer converts raw text to integer IDs.
    stream = meta_example_stream(
        model.tokenizer,
        context_len=context_len,
        continuation_len=continuation_len,
    )

    # Open a CSV log file. Every meta-step will write one row:
    # [step number, loss value, seconds elapsed]. This produces logs/meta_loss.csv
    # which you can open in Excel or plot to watch the loss trend downward.
    log_file = open(log_path, "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["meta_step", "meta_loss", "elapsed_s"])
    t0 = time.time()

    # tqdm wraps range(meta_steps) to draw a live progress bar in the terminal.
    # Each iteration of this loop is one complete meta-training step.
    pbar = tqdm(range(meta_steps), desc="meta-train")
    for step in pbar:
        # Pull the next (context, continuation) pair from the data stream.
        # ctx  -- tensor of token IDs the inner loop will train on [1, context_len]
        # cont -- tensor of token IDs held back to evaluate the model after adapting [1, continuation_len]
        ctx, cont = next(stream)

        # Clear leftover gradients from the previous step. PyTorch accumulates
        # gradients by default (adds them on top of each other), so you must
        # reset them manually at the start of each step.
        outer_opt.zero_grad()

        # `track_higher_grads=True` (default) keeps the inner-update graph so
        # we can call .backward() on a loss computed through it. This is the
        # machinery behind the paper's end-to-end training.
        #
        # `copy_initial_weights=False` means the real model weights are the
        # starting point -- so when .backward() runs, gradients flow back into
        # them and the outer optimizer can improve them. If this were True,
        # `higher` would start from a detached copy and the outer loop could
        # never improve the inner loop's initialization.
        #
        # fmodel  -- a "functional" view of the model where weight updates are
        #            recorded as computation graph nodes instead of being applied
        #            in-place (i.e. permanently changing a number in memory)
        # diffopt -- a differentiable version of inner_opt; calling diffopt.step()
        #            records the SGD update in the graph so .backward() can
        #            differentiate through it later
        with higher.innerloop_ctx(
            model,
            inner_opt,
            copy_initial_weights=False,
            track_higher_grads=True,
        ) as (fmodel, diffopt):
            # Run the inner loop: take `inner_steps` SGD steps on the context,
            # updating only the trainable MLP branches. After this, fmodel has
            # adapted weights φ' specialised to the patterns in ctx.
            inner_adapt_functional(
                fmodel, diffopt, ctx, steps=inner_steps, window=window
            )

            # Meta-loss: teacher-force the continuation AFTER the context,
            # and take CE on the continuation tokens only. We feed the full
            # (context + continuation) sequence so positional embeddings and
            # attention see continuation tokens in their proper position.
            #
            # torch.cat joins the two tensors end-to-end along the token dimension.
            full = torch.cat([ctx, cont], dim=-1)
            # fmodel(full).logits -- forward pass through the adapted model.
            # logits is a 3D tensor [batch=1, sequence_length, vocab_size=50257].
            # Each position holds a score for every possible next token.
            logits = fmodel(full).logits
            # Predict token t from tokens <t. Continuation tokens are the
            # last `continuation_len` positions of `full`.
            cl = cont.size(-1)
            pred_logits = logits[:, -cl - 1 : -1, :]  # aligns with cont tokens
            # Cross-entropy loss: for each continuation token, penalise the model
            # based on how much probability it assigned to the correct next token.
            # Low loss = model was confident and right. High loss = wrong or uncertain.
            # .reshape(-1, ...) flattens the batch and sequence dimensions into one
            # so cross_entropy sees a flat list of predictions vs. correct answers.
            meta_loss = torch.nn.functional.cross_entropy(
                pred_logits.reshape(-1, pred_logits.size(-1)),
                cont.reshape(-1),
            )

            # Second-order backward: grad flows through inner_adapt_functional
            # into both φ₀ (model.inner_params()) and θ (model.outer_params()).
            # "Second-order" means we are computing the gradient of a gradient --
            # the inner loop itself used a gradient step, and we are now
            # differentiating through that step to update the weights that
            # determine where the inner loop starts.
            meta_loss.backward()

        # Gradient clipping: cap the gradient vector's magnitude to 1.0.
        # Second-order gradients can occasionally explode to huge values and
        # cause weights to jump wildly ("exploding gradients"). Clipping
        # rescales the gradient if it exceeds the threshold, keeping training stable.
        torch.nn.utils.clip_grad_norm_(
            list(model.outer_params()) + list(model.inner_params()), 1.0
        )
        # Apply the clipped gradients: update all outer weights θ and the
        # inner starting weights φ₀ based on what we just computed.
        outer_opt.step()

        elapsed = time.time() - t0
        # .item() converts a PyTorch tensor to a plain Python float for logging/printing.
        # flush() forces the log to disk immediately so you can watch it in real time.
        log.writerow([step, meta_loss.item(), f"{elapsed:.1f}"])
        log_file.flush()
        pbar.set_postfix(loss=f"{meta_loss.item():.3f}")

        # Save a checkpoint every `ckpt_every` steps and at the final step.
        # % is the modulo operator: (step+1) % 200 == 0 is True every 200 steps.
        # model.state_dict() is a dictionary of all weight tensors.
        # torch.save serialises it to a .pt file so training can resume if it crashes.
        if (step + 1) % ckpt_every == 0 or step == meta_steps - 1:
            path = os.path.join(ckpt_dir, f"ttt_gpt2_meta_{step + 1}.pt")
            torch.save(model.state_dict(), path)
            # latest.pt is always overwritten -- it points to the most recent checkpoint.
            latest = os.path.join(ckpt_dir, "latest.pt")
            torch.save(model.state_dict(), latest)

    log_file.close()


def main():
    # argparse lets you pass options from the command line, e.g.:
    #   python -m ttt.outer_loop --meta-steps 2000 --inner-steps 1
    # Each add_argument call defines one flag, its type, and its default value.
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
