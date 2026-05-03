"""Dataset adapter registry.

Add a new dataset:

    1. Write a subclass of `DatasetAdapter` somewhere in this package.
    2. Register it via `register(name, factory)` below.
    3. Use it from any CLI: ``--dataset <name>`` -> the runner builds
       it via `get(name, **kwargs)`.

Methods (TTT, RAG, LoRA, ICL) and metrics never import anything from a
specific adapter -- they only depend on `PersonalizedExample`.
"""

from __future__ import annotations

from typing import Callable

from .base import DatasetAdapter, PersonalizedExample


_REGISTRY: dict[str, Callable[..., DatasetAdapter]] = {}


def register(name: str, factory: Callable[..., DatasetAdapter]) -> None:
    _REGISTRY[name] = factory


def get(name: str, **kwargs) -> DatasetAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; choices: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)


# -- builtin registrations -----------------------------------------------------


def _make_elsa(**kw) -> DatasetAdapter:
    from .elsa import ElsaAdapter, ElsaConfig

    return ElsaAdapter(ElsaConfig(**kw))


def _make_lamp5(**kw) -> DatasetAdapter:
    from .lamp import LampAdapter, LampConfig

    return LampAdapter(LampConfig(task="LaMP-5", **kw))


def _make_lamp7(**kw) -> DatasetAdapter:
    from .lamp import LampAdapter, LampConfig

    return LampAdapter(LampConfig(task="LaMP-7", **kw))


register("elsa", _make_elsa)
register("lamp-5", _make_lamp5)
register("lamp-7", _make_lamp7)


__all__ = ["DatasetAdapter", "PersonalizedExample", "register", "get", "available"]
