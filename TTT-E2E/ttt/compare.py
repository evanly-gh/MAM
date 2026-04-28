"""Four-way ablation on a single long document.

Splits the document into (context, continuation). Measures perplexity of the
continuation under four configurations:

  1. plain GPT-2        : no outer, no inner
  2. inner-only         : raw GPT-2 + TTT at test time
  3. outer-only         : meta-trained init, inner_steps=0
  4. full TTT-E2E       : meta-trained init + TTT at test time

Expected (on a long, stylistically coherent document): 4 < 2 ≈ 3 < 1.
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from .inner_loop import inner_adapt_inplace
from .model import TTTGPT2


def continuation_nll(model: TTTGPT2, ctx: torch.Tensor, cont: torch.Tensor) -> float:
    full = torch.cat([ctx, cont], dim=-1)
    with torch.no_grad():
        logits = model(full).logits
    cl = cont.size(-1)
    pred = logits[:, -cl - 1 : -1, :]
    loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)), cont.reshape(-1))
    return loss.item()


def eval_config(model: TTTGPT2, ctx, cont, inner_steps, inner_lr, window) -> float:
    snap = model.snapshot_inner()
    try:
        if inner_steps > 0:
            inner_adapt_inplace(model, ctx, steps=inner_steps, lr=inner_lr, window=window)
        nll = continuation_nll(model, ctx, cont)
        return math.exp(nll)
    finally:
        model.restore_inner(snap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--prompt-file", type=str, required=True)
    ap.add_argument("--context-len", type=int, default=512)
    ap.add_argument("--continuation-len", type=int, default=64)
    ap.add_argument("--inner-steps", type=int, default=5)
    ap.add_argument("--inner-lr", type=float, default=1e-3)
    ap.add_argument("--window", type=int, default=256)
    args = ap.parse_args()

    # Two model instances: one raw pretrained (for configs 1 and 2), one with
    # meta-trained weights loaded (for configs 3 and 4).
    raw = TTTGPT2("gpt2")
    meta = TTTGPT2("gpt2")
    meta.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    tok = raw.tokenizer
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        ids = tok.encode(f.read(), return_tensors="pt")
    total = args.context_len + args.continuation_len
    assert ids.size(-1) >= total, f"prompt too short: need {total} tokens"
    ctx = ids[:, :args.context_len]
    cont = ids[:, args.context_len:total]

    results = {
        "1. plain GPT-2       ": eval_config(raw,  ctx, cont, 0, args.inner_lr, args.window),
        "2. inner-only        ": eval_config(raw,  ctx, cont, args.inner_steps, args.inner_lr, args.window),
        "3. outer-only (meta) ": eval_config(meta, ctx, cont, 0, args.inner_lr, args.window),
        "4. full TTT-E2E      ": eval_config(meta, ctx, cont, args.inner_steps, args.inner_lr, args.window),
    }

    print("\nContinuation perplexity (lower is better):")
    for k, v in results.items():
        print(f"  {k}  ppl = {v:8.3f}")


if __name__ == "__main__":
    main()
