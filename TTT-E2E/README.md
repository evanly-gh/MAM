# MAM — Test-Time Training (GPT-2 + Flan-T5)

A faithful implementation of the **inner + outer loop** of TTT-E2E
([paper](https://test-time-training.github.io/e2e.pdf)) on top of two
pretrained model families:

- **GPT-2 (124M)** — decoder-only causal LM. Runs on CPU with 16 GB RAM.
- **Flan-T5 base** — instruction-tuned encoder-decoder. Faster to converge
  during meta-training; benefits from a GPU but works on CPU.

Pick whichever fits your hardware and experiment.

## What it does

Both architectures share the same conceptual structure:

- **Inner loop** ([ttt/mam_inner.py](ttt/mam_inner.py),
  [ttt/flan_inner.py](ttt/flan_inner.py)): at inference, take a single
  left-to-right streaming pass over the prompt, applying one SGD step per
  sliding window. This is the test-time training itself. No labels needed —
  the prompt's own next-token prediction loss is the signal.
- **Outer loop** ([ttt/mam_outer.py](ttt/mam_outer.py),
  [ttt/flan_outer.py](ttt/flan_outer.py)): meta-train the base weights so
  that *after* the inner loop has run, a held-out continuation is predicted
  well. Uses [`higher`](https://github.com/facebookresearch/higher) for
  differentiable inner-loop updates (second-order gradients).
- **Preserves pretrained knowledge**
  ([ttt/mam_model.py](ttt/mam_model.py),
  [ttt/flan_dual_mlp_model.py](ttt/flan_dual_mlp_model.py)): each late-block
  MLP is split into a *trainable* branch (updated by both loops) and a
  *static* branch (updated only by the outer loop; frozen during inner-loop
  adaptation). Their outputs are summed. This anchors the trainable branch
  so it can't drift arbitrarily far from the pretrained behaviour during
  per-prompt adaptation.

## Install

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads GPT-2 (~500 MB) or Flan-T5-base (~990 MB) plus a
WikiText-103 shard. Both are cached locally.

You might also need Visual Studio C++ build tools:
https://visualstudio.microsoft.com/downloads/?q=build+tools

## Hardware & the `--device` flag

Every CLI accepts `--device {auto,cpu,cuda}`:

- `auto` (default) — CUDA if available, otherwise CPU
- `cpu`            — force CPU even on a CUDA box (useful for
                     reproducibility checks across machines)
- `cuda`           — require GPU; errors out if none is present

Rough timings:

| Stage                                  | CPU (16 GB)   | CUDA (consumer GPU) |
|----------------------------------------|---------------|---------------------|
| Smoke test (20 meta-steps, GPT-2)      | a few minutes | seconds             |
| Full meta-training (2000 steps, GPT-2) | overnight     | < 1 hour            |
| Inference (adapt + generate, 512 tok)  | 30–90 s       | a few seconds       |

Flan-T5 is roughly 8× the parameter count of GPT-2 and noticeably slower per
step on CPU; if you have a GPU, use it for Flan-T5.

## File map

```
ttt/
├── __init__.py                  -- package exports (TTTGPT2, TTTFlanT5, ...)
│
├── mam_model.py                 -- TTTGPT2 + DualMLP (GPT-2 path)
├── mam_inner.py                 -- single-pass inner loop, GPT-2
├── mam_outer.py                 -- meta-training driver, GPT-2 (run, run_lamp)
├── mam_data.py                  -- WikiText-103 + LaMP-5/LaMP-7 streams
│
├── flan_dual_mlp_model.py       -- TTTFlanT5 + DualMLP (Flan-T5 path)
├── flan_inner.py                -- single-pass inner loop, Flan-T5
├── flan_outer.py                -- meta-training driver, Flan-T5
│
├── e2e.py                       -- shared utilities (history streams, etc.)
├── outer_meta.py                -- higher-level orchestration
├── training.py                  -- LaMP-profile training-pair pipeline
│
├── bench.py                     -- llama-bench-style benchmarks (GPT-2)
├── compare.py                   -- 4-way ablation (GPT-2)
├── generate.py                  -- inference + generation (GPT-2)
├── personalize.py               -- per-user style transfer (GPT-2)
└── scorers.py                   -- pluggable accuracy metrics
```

The inference/benchmark scripts (`bench`, `compare`, `generate`,
`personalize`) currently target the GPT-2 path. The Flan-T5 path has its own
training driver but reuses the same data streams.

---

## A. GPT-2 workflow

### A.1. Smoke test — verify the second-order backprop is wired up

```powershell
python -m ttt.mam_outer --meta-steps 20
```

Runs 20 meta-training steps. Each step:

1. Samples a `(context, continuation)` pair from WikiText-103 (context = 256
   tokens, continuation = 64 tokens held out).
2. Enters a `higher.innerloop_ctx` — a functional view of the model whose
   inner-loop parameter updates are recorded in PyTorch's autograd graph.
3. Runs the **inner loop**: a single left-to-right pass over the context,
   one SGD step per 256-token window, updating only the trainable MLP
   branches in blocks 9, 10, 11.
4. Computes **meta-loss**: next-token CE on the held-out continuation, using
   the inner-loop-adapted weights.
5. `meta_loss.backward()` — second-order. Gradients flow from the meta-loss,
   *through* the inner-loop update, into both the starting inner params (φ₀)
   and the outer params (θ).
6. Adam step on the outer optimizer; clip grads to norm 1.
7. Logs to `logs/meta_loss.csv`; checkpoints every 200 steps to
   `checkpoints/ttt_gpt2_meta_*.pt` and `checkpoints/latest.pt`.

If the loss in the progress bar is finite and trending downward, the
double-backward is correctly wired up.

### A.2. Full meta-training run

```powershell
python -m ttt.mam_outer --meta-steps 2000 --device auto
```

Useful flags:

- `--meta-steps N`         — total training iterations (default 2000)
- `--inner-lr 1e-3`        — inner-loop SGD step size
- `--context-len 256`      — context window for meta-training
- `--continuation-len 64`  — held-out length for meta-loss
- `--ckpt-every 200`       — checkpoint cadence
- `--model-name gpt2`      — Hugging Face model id (e.g. `gpt2-medium`)
- `--device {auto,cpu,cuda}`

Result: `checkpoints/latest.pt` — your meta-trained φ₀ + θ. This is what the
inference scripts below load by default.

### A.3. Generate with TTT — [ttt/generate.py](ttt/generate.py)

```powershell
python -m ttt.generate --checkpoint checkpoints/latest.pt --prompt-file examples/long.txt
```

What happens:

1. Loads GPT-2 + meta-trained weights.
2. Tokenises the prompt file.
3. Snapshots inner params (so we can reset afterwards — TTT weights are
   per-prompt state, not a permanent finetune).
4. Runs the inner loop on the prompt in place (`inner_adapt_inplace`).
5. Greedy-decodes `--max-new-tokens` tokens.
6. Restores inner params so the next call starts fresh.

Useful flags:

- `--no-adapt`             — skip the inner loop entirely (baseline output)
- `--inner-lr 1e-3`        — adaptation step size
- `--max-new-tokens 100`   — generation length
- `--device {auto,cpu,cuda}`

Omit `--checkpoint` to use raw GPT-2 + inner loop only (no meta-training).

### A.4. Four-way ablation — [ttt/compare.py](ttt/compare.py)

Isolates what each loop contributes. Same `(context, continuation)` split,
four configurations, perplexity on the continuation:

```powershell
python -m ttt.compare --checkpoint checkpoints/latest.pt --prompt-file examples/long.txt
```

| Config             | Description                                   |
|--------------------|-----------------------------------------------|
| 1. plain GPT-2     | pretrained weights, no TTT (baseline)         |
| 2. inner-only      | pretrained weights + inner loop at test time  |
| 3. outer-only      | meta-trained weights, inner loop disabled     |
| 4. full TTT-E2E    | meta-trained weights + inner loop             |

Expected ordering on a coherent long prompt: **4 < 2 ≈ 3 < 1**.

### A.5. Benchmarks — [ttt/bench.py](ttt/bench.py)

A `llama-bench`-style suite. All subcommands accept `--checkpoint`
(omit for raw GPT-2) and `--device`.

#### Latency — `pp` / `tg` / inner-pass time / peak RSS

```powershell
python -m ttt.bench --checkpoint checkpoints/latest.pt latency --ctx-len 512 --gen-tokens 50 --passes 1
```

Terms:

- **prompt processing (`pp`)** — single-forward-pass throughput on a fresh
  prompt, tokens/sec.
- **token generation (`tg`)** — autoregressive greedy decoding speed,
  tokens/sec. This is what the user perceives as "response time".
- **inner-loop pass** — one full single-pass adaptation, in milliseconds.
  This is the TTT *added* cost vs vanilla inference.
- **peak RSS** — process resident memory in MB.

#### Context-length scaling

Paper §4 claim: TTT-E2E quality keeps improving with longer context, like
full attention. Sweeps context length:

```powershell
python -m ttt.bench --checkpoint checkpoints/latest.pt ctx-scaling `
  --prompt-file examples/long.txt `
  --ctx-lens 64 128 256 512 1024
```

Reports continuation perplexity with and without TTT at each context length,
plus inner-pass latency. At GPT-2 scale you can only sweep up to 1024 tokens
(model max).

#### Passes scaling — degraded-baseline sweep

The paper's inner loop is one pass. This subcommand re-streams the same
context multiple times so you can see what the older K-step regime looked
like:

```powershell
python -m ttt.bench --checkpoint checkpoints/latest.pt passes-scaling `
  --prompt-file examples/long.txt `
  --passes-list 0 1 2 5 10
```

Default `--passes 1` everywhere is paper-faithful; sweep higher only as a
diagnostic.

#### Needle-in-a-haystack

Paper §5: full attention solves recall trivially; TTT-E2E does not (its
memory is *compressive*, not lossless):

```powershell
python -m ttt.bench --checkpoint checkpoints/latest.pt needle `
  --prompt-file examples/long.txt --haystack-len 512 --passes 1
```

`recovered: False` confirms the expected failure mode.

#### Reset check

Sanity that fast weights don't leak across prompts:

```powershell
python -m ttt.bench --checkpoint checkpoints/latest.pt reset
```

### A.6. Personalization — [ttt/personalize.py](ttt/personalize.py)

The whole point of TTT is per-prompt adaptation, so style transfer is a
natural fit. Feed a person's writing as the inner-loop context, then
generate something new in that voice. The fast weights forget the style as
soon as the call finishes (good — no cross-user leakage).

```powershell
python -m ttt.personalize `
  --checkpoint checkpoints/latest.pt `
  --style-file styles/alice.txt `
  --prompt "Subject: lunch tomorrow\n\nHi Bob," `
  --reference styles/alice_held_out.txt `
  --scorer perplexity `
  --baseline
```

Pluggable accuracy scorers ([ttt/scorers.py](ttt/scorers.py)):

- `perplexity` (↓) — perplexity of the generated text under the
  just-adapted model. If TTT really did absorb the style, in-style
  generations look low-perplexity.
- `ngram` (↑) — trigram Jaccard between generation and a held-out reference.
- `style-stats` (↓) — L1 distance between (mean sentence length, type-token
  ratio, mean word length).

Add your own: write `score(generated, reference, model=None) -> float` in
`scorers.py`, wrap in `Scorer(...)`, register in `SCORERS`.

---

## B. Flan-T5 workflow

### B.1. Meta-train

```powershell
python -m ttt.flan_outer --help
python -m ttt.flan_outer --meta-steps 2000
```

Same conceptual setup as `mam_outer`: streams `(context, continuation)`
pairs from WikiText-103, runs a single-pass inner loop on the trainable
DualMLP branches in late encoder/decoder feed-forward layers, computes
meta-loss on the held-out continuation, second-order backwards.

The inner-loop loss for Flan-T5 is *text-copy* (encoder reads the context,
decoder is teacher-forced to reproduce it) rather than next-token CE,
because Flan-T5 is encoder-decoder. See
[ttt/flan_inner.py](ttt/flan_inner.py).

Checkpoints land in `checkpoints/`. Loading a Flan-T5 checkpoint requires
constructing `TTTFlanT5(...)` rather than `TTTGPT2(...)` — the inference
scripts above are GPT-2-specific. To use Flan-T5 at inference, import
directly:

```python
from ttt import TTTFlanT5, inner_adapt_t5_inplace
import torch

model = TTTFlanT5("google/flan-t5-base")
model.load_state_dict(torch.load("checkpoints/latest.pt", map_location="cpu"))
# adapt + generate using model.generate(...) like a normal HF Seq2Seq model
```

### B.2. Why Flan-T5 may train faster than GPT-2

GPT-2 was pretrained on raw web text with no instruction tuning. Flan-T5
was instruction-tuned on a large mixture of structured tasks, so it
already "knows how to follow a prompt." Meta-training therefore has a
much better starting point for "be good after adaptation," and the loss
typically descends faster and to a lower floor.

---

## C. LaMP profile training (both architectures)

Both `mam_outer.run_lamp` and the Flan-T5 outer loop accept LaMP-5 / LaMP-7
profile data instead of WikiText. Flatten each user's profile (papers for
LaMP-5, tweets for LaMP-7) into one long token stream and meta-train on
random `(context, continuation)` slices, just like WikiText. The result is
a meta-trained init that's specifically good at adapting to per-user style.

See [ttt/mam_data.py](ttt/mam_data.py) — `meta_example_stream_lamp`.

---

## D. Replicating paper findings (toy scale)

| Paper claim                                                  | Command                                                | Expected at toy scale                            |
|--------------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------|
| TTT-E2E ≈ full attention on language modelling               | `ttt.compare`                                          | Config 4 wins, possibly modestly                 |
| Quality scales with context length (unlike RNNs)             | `ttt.bench ctx-scaling`                                | with-TTT ppl drops as ctx_len grows              |
| Constant per-token gen latency vs context length             | `ttt.bench latency` (sweep `--ctx-len`)                | tg flat; pp grows linearly without paper kernels |
| 2.7× faster gen than full attention at 128K                  | not reproducible                                       | needs paper's 3B model and 128K context          |
| Loses on recall (needle-in-a-haystack)                       | `ttt.bench needle`                                     | `recovered: False`                               |
| Outer loop matters                                           | `ttt.compare` (configs 2 vs 4)                         | Config 4 should beat config 2                    |

---

## E. Advantages of TTT over alternatives

- **vs RNNs (Mamba, RWKV)** — TTT keeps improving with longer context;
  RNNs plateau because their fixed-size hidden state saturates.
- **vs full attention** — per-token generation cost is constant in context
  length (the adapted weights are the only "memory"), so very long contexts
  don't quadratically blow up inference. The cost is upfront in TTFT.
- **vs finetuning** — no permanent weight changes, so no cross-user leakage
  and no catastrophic forgetting. Adaptation is per-call.
- **vs RAG / long-context prompting** — TTT compresses the context into
  weight updates rather than re-attending to it every token, which is
  cheaper at generation time.
- **Tradeoff** — TTT is *compressive*. Exact recall (names, numbers, quotes)
  degrades. RAG / full attention remain better when verbatim retrieval is
  what you need.

---

## F. Output files

After a meta-training run:

- `checkpoints/ttt_gpt2_meta_<N>.pt` — full model snapshots taken every
  `--ckpt-every` steps (~500 MB each for GPT-2; ~990 MB for Flan-T5-base).
- `checkpoints/latest.pt` — overwritten on each checkpoint; what the
  inference scripts load by default.
- `logs/meta_loss.csv` — per-step loss curve (plot it to verify the
  outer loop is descending).
- `.cache/wikitext103_train.pt` — tokenised dataset cache; safe to delete
  if you change tokenisers (will be re-downloaded).

---

## G. What to expect

This is a **toy-scale** reproduction. Do not expect the paper's
long-context scaling curves — those required 3B parameters and 164B
training tokens on multi-GPU clusters.

What you will see:

- Meta-loss decreasing during outer-loop training (confirms the
  gradient-of-gradient is wired up).
- Configuration 4 (full TTT-E2E) beating config 1 (plain GPT-2) on
  continuation perplexity for long, stylistically coherent prompts.
- TTT state is per-prompt: running two different prompts in a row, the
  second is not influenced by the first.
- Flan-T5's loss curve descends visibly faster than GPT-2's at the same
  step budget (instruction-tuned starting point).
