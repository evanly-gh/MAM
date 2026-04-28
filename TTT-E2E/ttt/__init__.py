"""Test-Time Training on GPT-2 (paper: https://test-time-training.github.io/e2e.pdf).

Two loops:
  - inner_loop: at inference time, adapt MLP weights on the given context via
    next-token CE. This is the "test-time training" step itself.
  - outer_loop: at training time, meta-learn the base weights so that *after*
    the inner loop has run, the model predicts a held-out continuation well.
    Requires backprop-through-backprop, handled via the `higher` library.
"""
