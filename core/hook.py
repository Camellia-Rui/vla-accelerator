"""Hook registration and exact removal, so pruners stay reversible."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List


@dataclass
class HookHandle:
    module: Any
    hook: Any
    kind: str  # "forward_pre", "forward", "forward_hook"


@dataclass
class HookManager:
    """Tracks every hook a pruner installs, so detach() is exact."""

    handles: List[HookHandle] = field(default_factory=list)

    def register_forward_pre(self, module: Any, fn: Callable) -> HookHandle:
        h = module.register_forward_pre_hook(fn)
        handle = HookHandle(module=module, hook=h, kind="forward_pre")
        self.handles.append(handle)
        return handle

    def register_forward(self, module: Any, fn: Callable) -> HookHandle:
        h = module.register_forward_hook(fn)
        handle = HookHandle(module=module, hook=h, kind="forward_hook")
        self.handles.append(handle)
        return handle

    def clear(self) -> int:
        """Remove all registered hooks. Returns how many were removed."""
        n = 0
        for handle in self.handles:
            try:
                handle.hook.remove()
                n += 1
            except Exception:
                pass
        self.handles.clear()
        return n

    def __len__(self) -> int:
        return len(self.handles)