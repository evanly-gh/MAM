"""Benchmarks for TTT on GPT-2.

Subcommands (each prints a small table):

  latency       per-token generation throughput, prompt-processing throughput,
                ms/inner-step, peak RSS. The "llama-bench"-style numbers.
  ctx-scaling   continuation perplexity vs context length (TTT-E2E paper §4:
                does quality keep improving as context grows?).
  steps-scaling continuation perplexity vs --inner-steps. Diminishing returns
                + eventual overfitting expected.
  needle        toy needle-in-a-haystack: hide a fact in long context, ask the
                model to recall it. Paper §5: full attention beats TTT here
                because TTT is compression, not lossless recall.
  reset         sanity check that fast weights reset between prompts (TTT
                state must be per-prompt, not a permanent finetune).

All commands take an optional --checkpoint; omit to bench raw GPT-2.
"""

from __future__ import annotations

import argparse
import math
import time

import psutil
import torch
import torch.nn.functional as F

from .inner_loop import inner_adapt_inplace
from .model import TTTGPT2


# ---- helpers ----------------------------------------------------------------


def _load(checkpoint: str | None) -> TTTGPT2:
    m = TTTGPT2("gpt2")
    if checkpoint:
        m.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    m.eval()
    return m


def _peak_rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


def _random_ids(model: TTTGPT2, n: int) -> torch.Tensor:
    """Random-but-valid token ids — not English, but fine for latency tests."""
    vocab = model.lm.config.vocab_size
    return torch.randint(0, vocab, (1, n))


def _continuation_ppl(model: TTTGPT2, ctx: torch.Tensor, cont: torch.Tensor) -> float:
    full = torch.cat([ctx, cont], dim=-1)
    with torch.no_grad():
        logits = model(full).logits
    cl = cont.size(-1)
    pred = logits[:, -cl - 1 : -1, :]
    nll = F.cross_entropy(pred.reshape(-1, pred.size(-1)), cont.reshape(-1))
    return math.exp(nll.item())


# ---- subcommands ------------------------------------------------------------


def cmd_latency(args):
    """llama-bench-style throughput numbers.

    pp (prompt processing) = forward-pass tokens/sec on a fresh long prompt.
    tg (token generation)  = autoregressive greedy decoding tokens/sec.
    inner-step             = ms per SGD step on `--ctx-len` tokens.
    peak RSS               = process resident memory at end of the run.
    """
    model = _load(args.checkpoint)
    ctx = _random_ids(model, args.ctx_len)

    # Prompt-processing: time a single forward pass over `ctx_len` tokens.
    with torch.no_grad():
        t0 = time.perf_counter()
        model(ctx)
        pp_s = time.perf_counter() - t0
    pp_tok_per_s = args.ctx_len / pp_s

    # Token-generation: greedy decode `gen_tokens` new tokens.
    with torch.no_grad():
        t0 = time.perf_counter()
        model.generate(
            ctx,
            max_new_tokens=args.gen_tokens,
            do_sample=False,
            pad_token_id=model.tokenizer.eos_token_id,
        )
        tg_s = time.perf_counter() - t0
    tg_tok_per_s = args.gen_tokens / tg_s

    # Inner-step: time `inner_steps` SGD updates on the prompt.
    snap = model.snapshot_inner()
    t0 = time.perf_counter()
    inner_adapt_inplace(model, ctx, steps=args.inner_steps, lr=1e-3, window=args.ctx_len)
    inner_s = time.perf_counter() - t0
    model.restore_inner(snap)
    ms_per_step = 1000 * inner_s / max(1, args.inner_steps)

    print(f"\nLatency @ ctx_len={args.ctx_len}, gen={args.gen_tokens} tokens")
    print(f"  prompt processing : {pp_tok_per_s:8.1f} tok/s   ({pp_s*1000:7.1f} ms total)")
    print(f"  token generation  : {tg_tok_per_s:8.1f} tok/s   ({tg_s:7.2f} s total)")
    print(f"  inner-loop step   : {ms_per_step:8.1f} ms/step  ({args.inner_steps} steps)")
    print(f"  peak RSS          : {_peak_rss_mb():8.0f} MB")


def cmd_ctx_scaling(args):
    """Continuation ppl as a function of context length.

    Paper §4 claim: TTT-E2E quality keeps improving with longer context, like
    full attention; pure RNNs plateau. At GPT-2 scale we can only sweep up to
    1024 tokens (model max), which is small but the trend should be visible.
    """
    model = _load(args.checkpoint)
    tok = model.tokenizer
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        ids = tok.encode(f.read(), return_tensors="pt")

    cont_len = args.continuation_len
    print(f"\nContinuation ppl vs context length (lower is better, cont_len={cont_len})")
    print(f"  {'ctx_len':>8}  {'no-TTT ppl':>12}  {'with-TTT ppl':>14}  {'inner_ms':>10}")
    for ctx_len in args.ctx_lens:
        if ids.size(-1) < ctx_len + cont_len:
            print(f"  {ctx_len:>8}  (prompt too short, skipping)")
            continue
        ctx = ids[:, :ctx_len]
        cont = ids[:, ctx_len : ctx_len + cont_len]

        no_ttt = _continuation_ppl(model, ctx, cont)

        snap = model.snapshot_inner()
        t0 = time.perf_counter()
        inner_adapt_inplace(model, ctx, steps=args.inner_steps, lr=args.inner_lr, window=min(ctx_len, 512))
        inner_ms = (time.perf_counter() - t0) * 1000
        with_ttt = _continuation_ppl(model, ctx, cont)
        model.restore_inner(snap)

        print(f"  {ctx_len:>8}  {no_ttt:>12.3f}  {with_ttt:>14.3f}  {inner_ms:>10.0f}")


def cmd_steps_scaling(args):
    """Continuation ppl as a function of inner-step count."""
    model = _load(args.checkpoint)
    tok = model.tokenizer
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        ids = tok.encode(f.read(), return_tensors="pt")
    ctx = ids[:, : args.ctx_len]
    cont = ids[:, args.ctx_len : args.ctx_len + args.continuation_len]

    print(f"\nContinuation ppl vs --inner-steps (ctx={args.ctx_len}, lr={args.inner_lr})")
    print(f"  {'steps':>6}  {'ppl':>10}  {'cum_ms':>8}")
    for s in args.steps_list:
        snap = model.snapshot_inner()
        t0 = time.perf_counter()
        if s > 0:
            inner_adapt_inplace(model, ctx, steps=s, lr=args.inner_lr, window=min(args.ctx_len, 512))
        ms = (time.perf_counter() - t0) * 1000
        ppl = _continuation_ppl(model, ctx, cont)
        model.restore_inner(snap)
        print(f"  {s:>6}  {ppl:>10.3f}  {ms:>8.0f}")


def cmd_needle(args):
    """Toy needle-in-a-haystack.

    Construct a prompt of `--haystack-len` tokens of filler text from the
    prompt file with a single sentence inserted: "The secret code is XYZ123."
    Then ask: "The secret code is" and check whether the model's top-1 next
    tokens reproduce "XYZ123".

    Paper §5: full attention solves this trivially; TTT-E2E does not, because
    its memory is *compressive* — fine details are lost. Our toy version
    should reproduce the failure mode.
    """
    model = _load(args.checkpoint)
    tok = model.tokenizer

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        filler_ids = tok.encode(f.read())
    needle_text = " The secret code is XYZ123."
    needle_ids = tok.encode(needle_text)
    query_text = " The secret code is"
    query_ids = tok.encode(query_text)
    answer_ids = tok.encode(" XYZ123")

    # Pad/truncate filler to haystack_len, drop needle near the start.
    if len(filler_ids) < args.haystack_len:
        # repeat
        filler_ids = (filler_ids * (args.haystack_len // max(1, len(filler_ids)) + 2))
    haystack = filler_ids[: args.haystack_len]
    insert_at = args.haystack_len // 4  # near beginning
    haystack = haystack[:insert_at] + needle_ids + haystack[insert_at:]
    full = torch.tensor([haystack + query_ids], dtype=torch.long)

    snap = model.snapshot_inner()
    if args.inner_steps > 0:
        ctx = full[:, : full.size(-1) - len(query_ids)]
        inner_adapt_inplace(model, ctx, steps=args.inner_steps, lr=args.inner_lr, window=512)

    with torch.no_grad():
        out = model.generate(
            full,
            max_new_tokens=len(answer_ids),
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    model.restore_inner(snap)

    generated = out[0, -len(answer_ids) :].tolist()
    expected = answer_ids
    correct = generated == expected
    print(f"\nNeedle-in-a-haystack @ haystack_len={args.haystack_len}, inner_steps={args.inner_steps}")
    print(f"  expected : {tok.decode(expected)!r}")
    print(f"  got      : {tok.decode(generated)!r}")
    print(f"  recovered: {correct}")


def cmd_reset(args):
    """Confirm fast weights reset between prompts."""
    model = _load(args.checkpoint)
    tok = model.tokenizer

    p1 = "The chemistry of organic synthesis"
    p2 = "Lord of the Rings begins with"
    ids1 = tok.encode(p1, return_tensors="pt")
    ids2 = tok.encode(p2, return_tensors="pt")

    def gen():
        with torch.no_grad():
            return model.generate(
                ids2, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id
            )

    base = gen()  # baseline output for prompt 2
    # Now adapt on prompt 1, then reset, then generate prompt 2 again.
    snap = model.snapshot_inner()
    inner_adapt_inplace(model, ids1, steps=10, lr=1e-3, window=512)
    model.restore_inner(snap)
    after = gen()

    same = torch.equal(base, after)
    print(f"\nReset check: outputs identical = {same}")
    if not same:
        print("  WARNING: fast weights leaking between prompts.")


# ---- entrypoint -------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("latency")
    a.add_argument("--ctx-len", type=int, default=512)
    a.add_argument("--gen-tokens", type=int, default=50)
    a.add_argument("--inner-steps", type=int, default=5)
    a.set_defaults(func=cmd_latency)

    a = sub.add_parser("ctx-scaling")
    a.add_argument("--prompt-file", type=str, required=True)
    a.add_argument("--ctx-lens", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    a.add_argument("--continuation-len", type=int, default=64)
    a.add_argument("--inner-steps", type=int, default=5)
    a.add_argument("--inner-lr", type=float, default=1e-3)
    a.set_defaults(func=cmd_ctx_scaling)

    a = sub.add_parser("steps-scaling")
    a.add_argument("--prompt-file", type=str, required=True)
    a.add_argument("--ctx-len", type=int, default=512)
    a.add_argument("--continuation-len", type=int, default=64)
    a.add_argument("--steps-list", type=int, nargs="+", default=[0, 1, 2, 5, 10, 20])
    a.add_argument("--inner-lr", type=float, default=1e-3)
    a.set_defaults(func=cmd_steps_scaling)

    a = sub.add_parser("needle")
    a.add_argument("--prompt-file", type=str, required=True)
    a.add_argument("--haystack-len", type=int, default=512)
    a.add_argument("--inner-steps", type=int, default=5)
    a.add_argument("--inner-lr", type=float, default=1e-3)
    a.set_defaults(func=cmd_needle)

    a = sub.add_parser("reset")
    a.set_defaults(func=cmd_reset)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
