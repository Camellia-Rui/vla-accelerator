"""Registry: map method name -> pruner class, and build from config."""

from __future__ import annotations

from typing import Any, Dict, Type

from pruners.base import BasePruner

_REGISTRY: Dict[str, Type[BasePruner]] = {}


def register(cls: Type[BasePruner]) -> Type[BasePruner]:
    if not getattr(cls, "name", None) or cls.name == "base":
        raise ValueError(f"{cls.__name__} must define a unique 'name'")
    if cls.name in _REGISTRY:
        raise ValueError(f"duplicate pruner name: {cls.name}")
    _REGISTRY[cls.name] = cls
    return cls


def available() -> list[str]:
    return sorted(_REGISTRY)


def build(name: str, config: Dict[str, Any] | None = None) -> BasePruner:
    if name not in _REGISTRY:
        raise KeyError(f"unknown pruner '{name}'. available: {available()}")
    return _REGISTRY[name](config=config)