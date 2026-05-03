"""Method registry. Add new methods (LoRA, your-future-thing) here.

Construction signatures vary, so we register *factories* that take
(model, **kwargs). The runner reads the registry to know which methods
are available and how to build them.
"""

from __future__ import annotations

from typing import Callable

from .base import Method
from .baseline import BaselineMethod
from .icl_method import ICLMethod
from .rag_method import RAGMethod
from .ttt_method import TTTMethod


_REGISTRY: dict[str, Callable[..., Method]] = {
    "baseline": BaselineMethod,
    "icl": ICLMethod,
    "rag": RAGMethod,
    "ttt": TTTMethod,
}


def register(name: str, factory: Callable[..., Method]) -> None:
    _REGISTRY[name] = factory


def get(name: str, model, **kwargs) -> Method:
    if name not in _REGISTRY:
        raise KeyError(f"unknown method {name!r}; choices: {sorted(_REGISTRY)}")
    return _REGISTRY[name](model, **kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["Method", "register", "get", "available"]
