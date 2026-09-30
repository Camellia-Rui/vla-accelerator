"""BasePruner: the single interface all four methods must expose."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from core.hook import HookManager


class BasePruner(ABC):
    """Unified interface. Adapters translate each method's native API into this."""

    #: set by subclasses, used by the registry
    name: str = "base"
    #: "stateless" or "stateful"
    state_type: str = "stateless"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}
        self.hooks = HookManager()
        self._attached = False
        self._stats: Dict[str, Any] = {}

    # --- lifecycle -------------------------------------------------
    def attach(self, model: Any) -> None:
        """Install hooks. Must be reversible via detach()."""
        if self._attached:
            raise RuntimeError(f"{self.name}: already attached")
        self._do_attach(model)
        self._attached = True

    def detach(self, model: Any) -> None:
        """Remove every hook this pruner installed."""
        self.hooks.clear()
        self._do_detach(model)
        self._attached = False

    def reset(self) -> None:
        """Clear cross-frame state. No-op for stateless pruners."""
        self._do_reset()

    # --- per-step --------------------------------------------------
    def step(self, **kwargs: Any) -> None:
        """Called before each inference step. May update pruning decisions."""
        self._do_step(**kwargs)

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    # --- subclass hooks --------------------------------------------
    @abstractmethod
    def _do_attach(self, model: Any) -> None: ...

    def _do_detach(self, model: Any) -> None:
        return None

    def _do_reset(self) -> None:
        return None

    def _do_step(self, **kwargs: Any) -> None:
        return None