"""Inference with TTT: adapt on the prompt, generate, then reset.

The reset at the end is important: TTT weights are *per-prompt state*, not a
permanent finetune. They play a role analogous to a KV cache in full-attention
models -- fresh for each new prompt.

The inner loop is single-pass over the context (paper-faithful semantics from
``mam_inner.py``). If you want extra adaptation, use a longer prompt rather
than re-streaming the same one.
"""

from __future__ import annotations

import argparse

import torch

from .mam_inner import inner_adapt_inplace
from .mam_model import TTTGPT2
from .mam_outer import resolve_device


def generate_with_ttt(
    model: TTTGPT2,
    prompt: str,
    inner_lr: float = 1e-3,
    window: int = 256,
    max_new_tokens: int = 100,
    adapt: bool = True,
) -> str:
    tok = model.tokenizer
    device = next(model.parameters()).device
    ids = tok.encode(prompt, return_tensors="pt").to(device)

    # Snapshot fast weights so we can restore after generation.
    snap = model.snapshot_inner()
    try:
        if adapt:
            inner_adapt_inplace(model, ids, lr=inner_lr, window=window)
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
    ap.add_argument("--no-adapt", action="store_true",
                    help="Skip the inner loop entirely (baseline).")
    ap.add_argument("--inner-lr", type=float, default=1e-3)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="auto picks cuda if available, else cpu.",
    )
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[generate] device={device}")

    model = TTTGPT2("gpt2")
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model = model.to(device)

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()

    text = generate_with_ttt(
        model,
        prompt,
        inner_lr=args.inner_lr,
        max_new_tokens=args.max_new_tokens,
        adapt=not args.no_adapt,
    )
    print(text)


if __name__ == "__main__":
    main()
