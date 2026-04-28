# MAM — Test-Time Training on GPT-2

A minimal, faithful implementation of the **inner + outer loop** of
TTT-E2E ([paper](https://test-time-training.github.io/e2e.pdf)) on top of
pretrained **GPT-2 (124M)**. Runs on CPU with 16 GB RAM.

## What it does

- **Inner loop** ([ttt/inner_loop.py](ttt/inner_loop.py)): at inference, take
  K gradient steps on next-token CE loss of the prompt, updating only the
  MLPs in the last ¼ of transformer blocks. This is the test-time training.
- **Outer loop** ([ttt/outer_loop.py](ttt/outer_loop.py)): meta-train the
  base weights so that *after* the inner loop has run, a held-out
  continuation is predicted well. Uses `higher` for differentiable
  inner-loop updates (second-order gradients).
- **Preserves pretrained knowledge** ([ttt/model.py](ttt/model.py)): each
  late-block MLP is split into a trainable branch (updated by both loops)
  and a static branch (updated only by the outer loop, frozen during
  inner-loop adaptation). Their outputs are summed.

## Install

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads GPT-2 weights (~500 MB) and a WikiText-103 shard.

You might also need Visual Studio build tools for C++
https://visualstudio.microsoft.com/downloads/?q=build+tools

## Hardware

Designed for **CPU-only, 16 GB RAM** laptops. No GPU required. Expect:
- Smoke test (20 meta-steps): a few minutes.
- Full meta-training (2000 steps): several hours to overnight.
- TTT inference (adapt + generate on a 512-token prompt): ~30–90 s.

## Usage

### 1. Smoke test — verify the second-order backprop is wired up

```
python -m ttt.outer_loop --meta-steps 20
```

Runs 20 meta-training steps. Each step:
1. Samples a `(context, continuation)` pair from WikiText-103 (context = 256
   tokens, continuation = 64 tokens held out).
2. Enters a `higher.innerloop_ctx` — a functional view of the model whose
   inner-loop parameter updates are recorded in PyTorch's autograd graph.
3. Runs the **inner loop**: one SGD step on next-token CE loss over the
   context, updating only the trainable MLP branches in blocks 9, 10, 11.
4. Computes **meta-loss**: next-token CE on the held-out continuation, using
   the inner-loop-adapted weights.
5. `meta_loss.backward()` — this is the second-order bit. Gradients flow
   from the meta-loss, *through* the inner-loop update, into both the
   starting inner params (φ₀) and the outer params (θ).
6. One Adam step on the outer optimizer; clip grads to norm 1.
7. Log loss to `logs/meta_loss.csv`; checkpoint every 200 steps to
   `checkpoints/ttt_gpt2_meta_*.pt` and `checkpoints/latest.pt`.

If the loss printed in the progress bar is finite and generally trending
down, the double-backward is correctly assembled and you can trust longer
runs.

### 2. Real meta-training run

```
python -m ttt.outer_loop --meta-steps 2000
```

Same thing, overnight. Produces `checkpoints/latest.pt` — this is your
meta-learned φ₀ and θ: a GPT-2 whose weights have been tuned specifically
so that TTT (the inner loop) produces useful adaptation, not just generic
LM pretraining.

Useful flags:
- `--inner-steps N` — how many SGD steps the inner loop takes per meta-step
  (default 1). Larger is closer to the paper but quadratically more memory
  and compute for the double-backward graph.
- `--inner-lr 1e-3` — inner-loop SGD step size.
- `--context-len 256 --continuation-len 64` — shrink for faster iteration,
  grow for more realistic long-context training.
- `--ckpt-every 200` — checkpoint cadence.

### 3. Generate with TTT

```
python -m ttt.generate --checkpoint checkpoints/latest.pt --prompt-file examples/long.txt
```

This is the payoff — inference with test-time training turned on:
1. Loads GPT-2 + your meta-trained weights.
2. Tokenizes the prompt file.
3. **Snapshots** the current inner params (so we can reset them afterward —
   TTT weights are per-prompt state, not a permanent finetune).
4. Runs the inner loop in place (`inner_adapt_inplace`): plain SGD, no
   second-order tracking. This is cheap — a normal training step.
5. Calls `model.generate(...)` greedily for `--max-new-tokens` tokens,
   using the **adapted** weights.
6. Prints the decoded output.
7. Restores the inner params to φ₀ so the next run starts fresh.

Omit `--checkpoint` to see raw GPT-2 + inner loop only (no meta-training).

Useful flags:
- `--inner-steps 5` — how many adaptation steps on the prompt (default 5).
  More steps = more specialization to the prompt, but also more risk of
  drifting away from general LM behavior; the static parallel MLP (see
  [ttt/model.py](ttt/model.py)) is what keeps this in check.
- `--inner-lr 1e-3` — adaptation step size.
- `--max-new-tokens 100` — length of generation.
Use `--continuation-len 100` instead

### 4. Four-way ablation

```
python -m ttt.compare --checkpoint checkpoints/latest.pt --prompt-file examples/long.txt
```

Isolates what each loop contributes. On the same `(context, continuation)`
split of the prompt file, measures **perplexity of the continuation** under:

1. **plain GPT-2**        — pretrained weights, no TTT. The baseline.
2. **inner-only**         — pretrained weights + inner loop at test time.
                            Shows what raw TTT does with no meta-learning.
3. **outer-only (meta)**  — meta-trained weights, inner steps = 0.
                            Shows whether meta-training alone helped
                            (note: meta-training's objective was "be good
                            *after* TTT," so this configuration is off-
                            distribution and may not win).
4. **full TTT-E2E**       — meta-trained weights + inner loop.
                            What the paper actually proposes.

Expected ordering on a coherent long prompt: **4 < 2 ≈ 3 < 1** (lower ppl
is better). If you see it, end-to-end TTT is doing its job.

### 5. Benchmarks ([ttt/bench.py](ttt/bench.py))

A `llama-bench`-style suite. Each subcommand prints a small results table.
All accept an optional `--checkpoint`; omit to bench raw GPT-2.

**Latency** — prompt-processing throughput (`pp` tok/s), generation
throughput (`tg` tok/s), inner-loop step latency (ms/step), peak RSS:

```
python -m ttt.bench --checkpoint checkpoints/latest.pt latency --ctx-len 512 --gen-tokens 50 --inner-steps 5
```

Terms:
- **prompt processing (pp)**: how fast the model consumes a fresh prompt
  via a single forward pass. Tokens/sec. Bigger = faster.
- **token generation (tg)**: autoregressive greedy decoding speed,
  tokens/sec. This is what the user perceives as "response time."
- **inner-loop step**: one SGD update on the prompt, in milliseconds.
  This is the *added* cost of TTT vs vanilla inference; total TTT
  overhead is `inner_steps × this`.
- **peak RSS**: process resident memory in MB. Useful sanity check that
  you're not OOM-bound.

**Context-length scaling** — paper §4 claim: TTT-E2E quality keeps
improving as context grows, like full attention. Sweeps `--ctx-lens`:

```
python -m ttt.bench --checkpoint checkpoints/latest.pt ctx-scaling --prompt-file examples/long.txt --ctx-lens 64 128 256 512 1024
```

Reports continuation perplexity with and without TTT at each context
length, plus inner-step latency. At GPT-2 scale you can only sweep up
to 1024 tokens (model max).

**Inner-step scaling** — diminishing returns + eventual overfit:

```
python -m ttt.bench --checkpoint checkpoints/latest.pt steps-scaling --prompt-file examples/long.txt --steps-list 0 1 2 5 10 20
```

**Needle-in-a-haystack** — paper §5: full attention solves recall
trivially, TTT-E2E does not (its memory is *compressive*, not lossless).
Inserts "The secret code is XYZ123." into filler text, then asks the
model to complete "The secret code is":

```
python -m ttt.bench --checkpoint checkpoints/latest.pt needle --prompt-file examples/long.txt --haystack-len 512 --inner-steps 5
```

A `recovered: False` confirms the expected failure mode for compressive
memory.

**Reset check** — sanity that fast weights don't leak across prompts:

```
python -m ttt.bench --checkpoint checkpoints/latest.pt reset
```

### 6. Personalization (style transfer) ([ttt/personalize.py](ttt/personalize.py))

The whole point of TTT is per-prompt adaptation, so style transfer is a
natural fit: feed the model a person's writing as the inner-loop context,
then generate something new in that voice. The fast weights forget the
style as soon as the call finishes (good — no cross-user leakage).

```
python -m ttt.personalize \
    --checkpoint checkpoints/latest.pt \
    --style-file styles/alice.txt \
    --prompt "Subject: lunch tomorrow\n\nHi Bob," \
    --reference styles/alice_held_out.txt \
    --scorer perplexity \
    --inner-steps 20 --inner-lr 1e-3 \
    --baseline
```

What happens:
1. Reads `--style-file` (a corpus of one person's prose — emails, essays,
   chat logs, anything ≥ a few hundred tokens).
2. Runs the inner loop on the style corpus to produce per-person fast
   weights — this IS the personalization.
3. Generates `--max-new-tokens` tokens of continuation from `--prompt`
   under those weights.
4. (If `--scorer` given) scores the generation against `--reference`
   using a pluggable metric.
5. (If `--baseline`) repeats the run with `--inner-steps 0` so you can
   compare adapted vs unadapted output side by side.
6. Resets fast weights.

**Pluggable accuracy metrics** ([ttt/scorers.py](ttt/scorers.py)). The
scoring step is decoupled from generation, so you can swap or add metrics
without touching the rest. Built in:

- `perplexity` (↓ lower better) — perplexity of the generated text under
  the just-adapted model. If TTT really did absorb the style, in-style
  generations should look low-perplexity to it.
- `ngram` (↑ higher better) — trigram Jaccard between generation and a
  held-out reference message in the same style. Captures shared phrasing.
- `style-stats` (↓) — L1 distance between (mean sentence length, type-
  token ratio, mean word length). Crude but interpretable surface proxy.

Add your own: write a `score(generated, reference, model=None) -> float`
function in `scorers.py`, wrap it in `Scorer(...)`, register it in the
`SCORERS` dict, and use `--scorer your_name`. No other file changes.

### 7. Replicating paper findings (toy scale)

The paper's headline claims and how to probe them with this repo:

| Paper claim | Command | Expected at toy scale |
|---|---|---|
| TTT-E2E ≈ full attention on language modeling | `ttt.compare` | Config 4 wins, possibly modestly |
| Quality scales with context length (like full attention, unlike RNNs) | `ttt.bench ctx-scaling` | `with-TTT ppl` should keep dropping as `ctx_len` grows; raw GPT-2 has no built-in mechanism for this |
| Constant inference latency vs context length | `ttt.bench latency` (sweep `--ctx-len`) | TTT step time grows linearly with context (we don't have the paper's optimized kernels), but **per-token gen** stays flat — that's the property |
| 2.7× faster gen than full attention at 128K | not reproducible | GPT-2 max ctx is 1024; needs the paper's 3B model |
| Loses on recall (needle-in-a-haystack) | `ttt.bench needle` | `recovered: False` |
| Outer loop matters | `ttt.compare` (configs 2 vs 4) | Config 4 should beat config 2 |

### Advantages of TTT over alternatives

- **vs RNNs (Mamba, RWKV)**: TTT keeps improving with longer context;
  RNNs plateau because their fixed-size hidden state saturates.
- **vs full attention**: per-token generation cost is constant in
  context length (the adapted weights are the only "memory"), so very
  long contexts don't quadratically blow up inference.
- **vs finetuning**: no permanent weight changes, so no cross-user
  leakage and no catastrophic forgetting. Adaptation is per-call.
- **vs RAG/long-context prompting**: TTT compresses the context into
  weight updates rather than re-attending to it every token, which is
  cheaper at generation time.
- **Tradeoff**: TTT is *compressive* — exact recall (names, numbers,
  quotes) degrades. RAG / full attention remain better when verbatim
  retrieval is what you need.

## Output files
Those came from the meta-training run you just did (python -m ttt.outer_loop ...). They're checkpoints saved by ttt/outer_loop.py:135-139:

checkpoints/ttt_gpt2_meta_<N>.pt — snapshot of the full model weights (φ₀ + θ) taken every --ckpt-every steps (default 200) and at the final step. Each one is the complete meta-trained GPT-2 at that meta-step, ~500 MB.
checkpoints/latest.pt — overwritten on each checkpoint; always points at the most recent one. This is what ttt.generate and ttt.compare load by default.
.cache/wikitext103_train.pt — tokenized WikiText-103 shard cached by ttt/data.py so future runs skip the download+tokenize step. Not a model checkpoint.

## What to expect

This is a **toy-scale** reproduction. Do not expect the paper's long-context
scaling curves — those required 3B parameters and 164B training tokens on
multi-GPU clusters. What you WILL see:
- Meta-loss decreasing during outer-loop training (confirms the
  gradient-of-gradient is wired up).
- Configuration 4 (full TTT-E2E) beating config 1 (plain GPT-2) on
  continuation perplexity for long, stylistically coherent prompts.
- TTT state is per-prompt: running two different prompts in a row, the
  second is not influenced by the first.
