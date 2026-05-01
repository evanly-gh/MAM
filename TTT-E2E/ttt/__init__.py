"""Test-Time Training (paper: https://test-time-training.github.io/e2e.pdf).

Two loops:
  - inner loop: at inference time, adapt MLP weights on the given context via
    next-token CE. Single left-to-right streaming pass, one SGD step per
    sliding window.
  - outer loop: at training time, meta-learn the base weights so that *after*
    the inner loop has run, the model predicts a held-out continuation well.
    Requires backprop-through-backprop, handled via the `higher` library.

Two model families (pick whichever fits your hardware / experiment):
  - TTTGPT2     -- GPT-2 124M with DualMLP on the last quarter of blocks.
                   Defined in mam_model.py. Inner/outer drivers in
                   mam_inner.py / mam_outer.py.
  - TTTFlanT5   -- google/flan-t5-base with DualMLP on the late
                   encoder/decoder feed-forward layers. Defined in
                   flan_dual_mlp_model.py. Inner/outer drivers in
                   flan_inner.py / flan_outer.py.

Pick a device with --device {auto,cpu,cuda} on any CLI driver.
"""

from . import e2e, outer_meta
from .flan_dual_mlp_model import TTTFlanT5
from .flan_inner import inner_adapt_t5_functional, inner_adapt_t5_inplace
from .mam_inner import inner_adapt_functional, inner_adapt_inplace
from .mam_model import TTTGPT2
from .training import build_profile_training_pairs, run_ttt_steps

__all__ = [
    "build_profile_training_pairs",
    "run_ttt_steps",
    "e2e",
    "outer_meta",
    "TTTGPT2",
    "TTTFlanT5",
    "inner_adapt_inplace",
    "inner_adapt_functional",
    "inner_adapt_t5_inplace",
    "inner_adapt_t5_functional",
]
