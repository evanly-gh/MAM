"""Data stream for meta-training.

Each "meta-example" is a single long document split into:
  * context       -- what the inner loop trains on at test time
  * continuation  -- held out; the outer loop's meta-loss is next-token CE
                     on this, evaluated AFTER the inner loop has adapted.

This structure is what makes meta-training teach the model "how to be good
*after* having just TTT'd on some context", rather than just "be good at LM".

Two stream sources:
  * `meta_example_stream`       -- the original WikiText-103 stream (general
                                   prose, our default for plain TTT-E2E
                                   training).
  * `meta_example_stream_lamp`  -- LaMP-5 / LaMP-7 user-profile stream from
                                   the lamp repo. Each "document" is one
                                   user's flattened profile (their papers
                                   for LaMP-5, their tweets for LaMP-7).
                                   Use this when you want the meta-trained
                                   init to be specifically good at adapting
                                   to per-user style.
"""

from __future__ import annotations

import os
import random
from typing import Any, Iterator

import torch


def _tokenize_and_cache(
    tokenizer,
    cache_path: str,
    dataset_name: str = "wikitext",
    config: str = "wikitext-103-raw-v1",
    split: str = "train",
    max_docs: int = 2000,
) -> torch.Tensor:
    """Tokenize a slice of the dataset once, cache as a flat LongTensor."""
    # Lazy import: `datasets` is a heavy dep, only needed on first run when
    # the cache is missing. After the cache exists, this function returns
    # without ever importing it.
    from datasets import load_dataset

    if os.path.exists(cache_path):
        # weights_only=False is required on torch >= 2.6 to load arbitrary
        # tensors; older versions don't accept the kwarg, hence the fallback.
        try:
            return torch.load(cache_path, weights_only=False)
        except TypeError:
            return torch.load(cache_path)

    ds = load_dataset(dataset_name, config, split=split, streaming=True)
    buf: list[int] = []
    docs = 0
    for row in ds:
        text = row["text"].strip()
        if not text:
            continue
        ids = tokenizer.encode(text)
        if len(ids) < 64:
            continue
        buf.extend(ids)
        buf.append(tokenizer.eos_token_id)
        docs += 1
        if docs >= max_docs:
            break

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tensor = torch.tensor(buf, dtype=torch.long)
    torch.save(tensor, cache_path)
    return tensor


def meta_example_stream(
    tokenizer,
    context_len: int = 256,
    continuation_len: int = 64,
    cache_path: str = ".cache/wikitext103_train.pt",
    seed: int = 0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Infinite stream of (context, continuation) id tensors, shape [1, T]."""
    flat = _tokenize_and_cache(tokenizer, cache_path)
    total = context_len + continuation_len
    rng = random.Random(seed)
    n = flat.size(0)
    while True:
        # max(0, ...) guards against pathological cases where the cache is
        # smaller than `total` -- without it randint() would raise.
        start = rng.randint(0, max(0, n - total - 1))
        chunk = flat[start : start + total]
        ctx = chunk[:context_len].unsqueeze(0)
        cont = chunk[context_len:].unsqueeze(0)
        yield ctx, cont


# -- LaMP profile streams ------------------------------------------------------
#
# Below is the LaMP-flavoured meta-training data, ported from
# MasaNakura/lamp@9e65964. LaMP is a benchmark for personalised language
# tasks; each row contains a `profile` (a list of past items belonging to one
# user) plus a current task input. We meta-train on the profile *content* so
# the inner loop learns to adapt to per-user style.


def _lamp_profile_document(task: str, profile: list[dict[str, Any]]) -> str:
    """Flatten a user's profile into a single training document.

    LaMP-5 profiles are scientific papers (title + abstract per item).
    LaMP-7 profiles are tweets (text per item).
    """
    parts: list[str] = []
    if task == "LaMP-5":
        for p in profile:
            t = (p.get("title") or "").strip()
            a = (p.get("abstract") or "").strip()
            if t:
                parts.append(f"[title] {t}")
            if a:
                parts.append(f"[abstract] {a}")
    elif task == "LaMP-7":
        for p in profile:
            tx = (p.get("text") or "").strip()
            if tx:
                parts.append(f"[tweet] {tx}")
    else:
        raise ValueError(task)
    return "\n\n".join(parts)


def _lamp_train_token_cache(
    tokenizer,
    rows: list[dict[str, Any]],
    task: str,
    cache_path: str,
) -> torch.Tensor:
    """Tokenize and concatenate every user's profile into one flat buffer.

    Same shape as the WikiText cache so the random-window sampler below can
    reuse the exact same logic.
    """
    if os.path.exists(cache_path):
        try:
            return torch.load(cache_path, weights_only=False)
        except TypeError:
            return torch.load(cache_path)

    # LaMP-7 rows are short tweets per profile item; use lower floors than
    # LaMP-5 title+abstract documents.
    min_doc_chars = 24 if task == "LaMP-7" else 80
    min_row_tokens = 12 if task == "LaMP-7" else 32
    min_total_tokens = 256 if task == "LaMP-7" else 512

    buf: list[int] = []
    for row in rows:
        prof = row.get("profile") or []
        doc = _lamp_profile_document(task, prof if isinstance(prof, list) else [])
        if len(doc) < min_doc_chars:
            continue
        ids = tokenizer.encode(doc)
        if len(ids) < min_row_tokens:
            continue
        buf.extend(ids)
        buf.append(tokenizer.eos_token_id)

    if len(buf) < min_total_tokens:
        # Fail loudly: a tiny buffer means the train JSON probably has
        # placeholder profiles, not real data.
        raise RuntimeError(
            f"LaMP meta cache: too few tokens after flattening profiles "
            f"({len(buf)} < {min_total_tokens}). "
            "Check that train JSON has real profile text, not placeholders. "
            "LaMP-7 needs enough tweets across the train split to fill the buffer."
        )
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tensor = torch.tensor(buf, dtype=torch.long)
    torch.save(tensor, cache_path)
    return tensor


def meta_example_stream_lamp(
    tokenizer,
    train_rows: list[dict[str, Any]],
    task: str,
    *,
    context_len: int = 256,
    continuation_len: int = 64,
    cache_path: str = ".cache/lamp_train_profiles.pt",
    seed: int = 0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Infinite stream of (context, continuation) id tensors from LaMP profiles."""
    flat = _lamp_train_token_cache(tokenizer, train_rows, task, cache_path)
    total = context_len + continuation_len
    rng = random.Random(seed)
    n = flat.size(0)
    if n < total + 1:
        raise RuntimeError(f"LaMP token buffer too short ({n} < {total + 1}).")
    while True:
        start = rng.randint(0, n - total - 1)
        chunk = flat[start : start + total]
        ctx = chunk[:context_len].unsqueeze(0)
        cont = chunk[context_len:].unsqueeze(0)
        yield ctx, cont
