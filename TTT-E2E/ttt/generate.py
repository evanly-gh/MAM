"""Inference with TTT: adapt on the prompt, generate, then reset.

The reset at the end is important: TTT weights are *per-prompt state*, not a
permanent finetune. They play a role analogous to a KV cache in full-attention
models -- fresh for each new prompt.
"""

from __future__ import annotations

import argparse

import torch

from .inner_loop import inner_adapt_inplace
from .model import TTTGPT2


def generate_with_ttt(
    model: TTTGPT2,
    prompt: str,
    inner_steps: int = 5,
    inner_lr: float = 1e-3,
    window: int = 256,
    max_new_tokens: int = 100,
) -> str:
    tok = model.tokenizer
    ids = tok.encode(prompt, return_tensors="pt")

    # Snapshot fast weights so we can restore after generation.
    snap = model.snapshot_inner()
    try:
        if inner_steps > 0:
            inner_adapt_inplace(
                model, ids, steps=inner_steps, lr=inner_lr, window=window
            )
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy -- deterministic for comparison
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0], skip_special_tokens=True)
    finally:
        # ALWAYS reset -- even if something raised -- so the next prompt sees
        # the meta-learned φ₀ again, not the adapted φ'.
        model.restore_inner(snap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Path to meta-trained checkpoint; omit for raw GPT-2.")
    ap.add_argument("--prompt-file", type=str, required=True)
    ap.add_argument("--inner-steps", type=int, default=5)
    ap.add_argument("--inner-lr", type=float, default=1e-3)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    args = ap.parse_args()

    model = TTTGPT2("gpt2")
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()

    text = generate_with_ttt(
        model,
        prompt,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        max_new_tokens=args.max_new_tokens,
    )
    print(text)


if __name__ == "__main__":
    main()
