"""Data stream for meta-training.

Each "meta-example" is a single long document split into:
  * context       -- what the inner loop trains on at test time
  * continuation  -- held out; the outer loop's meta-loss is next-token CE
                     on this, evaluated AFTER the inner loop has adapted.

This structure is what makes meta-training teach the model "how to be good
*after* having just TTT'd on some context", rather than just "be good at LM".
"""

from __future__ import annotations

import os
import random
from typing import Iterator

import torch
from datasets import load_dataset


def _tokenize_and_cache(
    tokenizer,
    cache_path: str,
    dataset_name: str = "wikitext",
    config: str = "wikitext-103-raw-v1",
    split: str = "train",
    max_docs: int = 2000,
) -> torch.Tensor:
    """Tokenize a slice of the dataset once, cache as a flat LongTensor."""
    if os.path.exists(cache_path):
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
        start = rng.randint(0, n - total - 1)
        chunk = flat[start : start + total]
        ctx = chunk[:context_len].unsqueeze(0)
        cont = chunk[context_len:].unsqueeze(0)
        yield ctx, cont
