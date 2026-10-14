"""Pruners package: importing registers all built-in adapters."""

from pruners.registry import available, build, register
from pruners import vla_pruner, specprune  # noqa: F401

__all__ = ["available", "build", "register"]
