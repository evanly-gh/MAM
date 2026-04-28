# Test-Time Training on GPT-2 (Full Inner + Outer Loop) — Setup Plan

## Context
You want a faithful implementation of the full TTT-E2E idea from https://test-time-training.github.io/e2e.pdf — **both** the inner loop (test-time gradient updates on the context) **and** the outer loop (meta-learning the base weights so that the model performs well *after* inner-loop adaptation) — on top of pretrained GPT-2 (124M), running on your CPU-only laptop with 16 GB RAM.

This is feasible at small scale. It will **not** reproduce the paper's 3B/164B-token results, but it will faithfully reproduce the mechanism end-to-end on a toy corpus, which is what you need to learn how TTT works.

## Do You Need a GPU?
**No — CPU is sufficient** for GPT-2 124M with 16 GB RAM, but you must keep the settings small. Expect:
- Inner-loop adapt + generate on a 512-token prompt: **~30–90 seconds** per run.
- Outer-loop meta-training on a toy corpus (WikiText-103 sample, a few thousand meta-steps): **several hours to overnight**.
- A GPU would make this 20–50× faster but is not required.

The reason CPU works: we scale everything down to fit — GPT-2 124M (~500 MB fp32 weights), short sequences (256–512 tokens), 1 meta-example per meta-step, only 1–3 inner steps, and **second-order gradients only on the last ¼ of MLP params** (the only params that are inner-loop-trainable). Peak RAM under these settings stays around 4–8 GB.

## Software to Install
```
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.44.* datasets tiktoken higher tqdm
```
- `torch` — autograd, including the double-backward needed for the outer loop.
- `transformers` — pretrained GPT-2 124M weights + tokenizer.
- `datasets` — WikiText-103 (or PG19) for meta-training and eval.
- `higher` — enables differentiable inner-loop optimization (clean MAML-style "gradients through gradients"). Without this we'd have to hand-roll functional parameter updates; `higher` does it correctly and is the standard tool for this pattern.
- `tqdm` — progress bars (meta-training is long; you'll want them).

## What the Inner and Outer Loops Actually Are
- **Inner loop (test time)**: Given a context, take K gradient steps on the next-token-prediction loss *of that context* using only the late-block MLP weights. This produces adapted weights φ' = φ − η∇φ L(context; φ). Generation then uses φ'.
- **Outer loop (train time, meta-learning)**: Sample a long document, split into "context" + "held-out continuation." Run the inner loop on the context to get φ'. Compute loss on the **held-out continuation** using φ'. Backpropagate through the inner-loop steps into the **base** parameters θ (and the starting φ). Update θ, φ with Adam. This is MAML applied to language modeling: we train θ so that "θ after a few inner steps on new context" predicts the continuation well.

The paper calls this "end-to-end" because both loops optimize the true next-token loss — no proxy reconstruction objective.

## Files to Create

### 1. [ttt/model.py](ttt/model.py)
`TTTGPT2`: wraps `GPT2LMHeadModel`. Identifies the last ¼ of transformer blocks (blocks 9–11 of GPT-2's 12). For each, adds a **parallel static MLP** (frozen copy of the pretrained MLP, output summed with the trainable MLP's output). Exposes:
- `inner_params()` — the only tensors updated in the inner loop (late-block trainable MLP weights/biases).
- `outer_params()` — everything else: embeddings, attention, early MLPs, static parallel MLPs, layernorms. These are updated only by the outer loop.
Inline comments will cite paper sections: §3.2 (don't update attention — unstable), §3.3 (last ¼ — capacity/compute tradeoff), §3.4 (static parallel MLP preserves pretrained knowledge).

### 2. [ttt/inner_loop.py](ttt/inner_loop.py)
`inner_adapt(fmodel, diffopt, context_ids, steps, window)`:
- Uses `higher.innerloop_ctx` to get a **functional, differentiable** view of the model and a differentiable SGD optimizer.
- For `steps` iterations: forward pass on context windows → next-token CE loss → `diffopt.step(loss)` (this records the update in the autograd graph so the outer loop can backprop through it).
- Returns `fmodel` with adapted inner params. Comments will tie each step back to the paper's Algorithm 1.

### 3. [ttt/outer_loop.py](ttt/outer_loop.py)
The meta-training loop:
```
for meta_step in range(N):
    doc = sample_document(min_len=768)
    context, continuation = doc[:512], doc[512:768]

    with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
        inner_adapt(fmodel, diffopt, context, steps=2)          # inner loop
        meta_loss = next_token_ce(fmodel, continuation)          # eval on held-out
        meta_loss.backward()                                     # second-order backprop
    outer_opt.step(); outer_opt.zero_grad()
```
- `copy_initial_weights=False` makes the inner optimizer's starting point itself a learnable parameter of the outer loop — this is what lets meta-learning shape the *initialization* φ₀, matching the paper.
- Checkpoints every N steps to `checkpoints/ttt_gpt2_meta_*.pt`.
- Heavy comments: what "gradient of gradient" means here, why we use 1–3 inner steps (compute budget), why we hold out a short continuation (the meta-objective).

### 4. [ttt/generate.py](ttt/generate.py)
Inference driver: load a meta-trained checkpoint → run `inner_adapt` on the user's prompt → `model.generate()` → reset inner params (they're per-prompt state, not a finetune) → done. CLI flags for inner steps, lr, window.

### 5. [ttt/compare.py](ttt/compare.py)
Four-way comparison on the same long prompt, so you can see each piece contribute:
1. **Plain GPT-2** (no TTT, no meta).
2. **Inner-loop only** on plain GPT-2 (no meta-training — the "beginner" TTT).
3. **Outer-loop only** (meta-trained init, but inner steps = 0 at test time — tests whether meta-training alone helped).
4. **Full TTT-E2E** (meta-trained + inner-adapted at test time).
Reports perplexity on a held-out continuation for each. Expected ordering: 4 < 2 ≈ 3 < 1.

### 6. [ttt/data.py](ttt/data.py)
Streams WikiText-103 (or PG19 — configurable). Yields `(context, continuation)` pairs of configurable length. Caches a tokenized shard to disk so meta-training doesn't re-tokenize every epoch.

### 7. [requirements.txt](requirements.txt) and [README.md](README.md) update
Install + run instructions, recommended CPU settings (context=256, inner_steps=1, meta_steps=2000 for a first run that finishes overnight), plus a "what to expect" section on realistic outcomes at this scale.

## Recommended CPU-Friendly Hyperparameters (first run)
- Context window: 256 tokens (GPT-2 handles up to 1024; we go smaller for speed).
- Inner steps: 1 (the paper uses few; 1 is enough to see the effect and keeps the double-backward graph small).
- Inner lr: 1e-3 (SGD, no momentum — momentum would need more state in the differentiable optimizer).
- Outer optimizer: Adam, lr 1e-5 on `outer_params`, lr 1e-4 on initial `inner_params`.
- Meta-batch size: 1 (CPU).
- Meta-steps: 2000 for a first pass (~4–8 hours); 10000+ for better results.
- Precision: fp32 (CPU doesn't benefit from fp16).

## Verification
1. **Smoke test**: `python -m ttt.outer_loop --meta-steps 20` — should run in a few minutes and print decreasing meta-loss. This proves the double-backward wiring is correct.
2. **Real run**: `python -m ttt.outer_loop --meta-steps 2000` — overnight; loss curve saved to `logs/meta_loss.csv`.
3. **Inference**: `python -m ttt.generate --checkpoint checkpoints/latest.pt --prompt-file examples/long.txt`.
4. **Ablation**: `python -m ttt.compare --checkpoint checkpoints/latest.pt --prompt-file examples/long.txt` — produces the 4-way perplexity table. Confirms the outer loop materially helped (configuration 4 beats 2) — this is the end-to-end validation that both loops are doing their job.
5. **Reset check**: run `generate.py` on two different prompts in sequence; confirm the second isn't influenced by the first (per-prompt state, not finetune).

## Files Modified / Created
- `ttt/__init__.py`, `ttt/model.py`, `ttt/inner_loop.py`, `ttt/outer_loop.py`, `ttt/generate.py`, `ttt/compare.py`, `ttt/data.py` — new
- `requirements.txt` — new
- `examples/long.txt` — a sample long prompt for demos
- [README.md](README.md) — append usage + expectations section
