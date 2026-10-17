"""Toolbox lifecycle for an already loaded OpenVLA-OFT VLA-Pruner model."""

from __future__ import annotations

from typing import Any

from methods.vla_pruner.accelerator import AcceleratorConfig, VLAPrunerAccelerator
from pruners.base import BasePruner
from pruners.registry import register


@register
class VLAPrunerAdapter(BasePruner):
    name = "vla_pruner"
    state_type = "stateful"

    def __init__(self, config=None):
        super().__init__(config)
        self._acc = VLAPrunerAccelerator(AcceleratorConfig.from_mapping(config))
        self._last_action = None
        self._last_actions = None

    def attach(self, model: Any, **resources: Any) -> None:
        if self._attached:
            raise RuntimeError("vla_pruner: already attached")
        if model is None:
            raise ValueError("attach requires an existing model; use VLAPrunerAccelerator.from_pretrained to load one")
        self._acc.attach(model, **resources)
        self._attached = True

    def _do_attach(self, model: Any) -> None:
        self._acc.attach(model)

    def detach(self, model: Any = None) -> None:
        if model is not None and self._acc.model is not None and model is not self._acc.model:
            raise ValueError("detach received a different model")
        self._acc.detach()
        self._attached = False
        self._last_action = self._last_actions = None
        self._stats = {}

    def _do_reset(self) -> None:
        self._acc.reset()
        self._last_action = self._last_actions = None
        self._stats = {}

    def enable(self) -> None:
        self._acc.enable()
        self._last_action = self._last_actions = None

    def disable(self) -> None:
        self._acc.disable()
        self._last_action = self._last_actions = None

    @property
    def enabled(self) -> bool:
        return self._acc.enabled

    def step(self, **kwargs: Any) -> dict[str, Any]:
        self._last_action = self._last_actions = None
        result = self._acc.step(**kwargs)
        self._last_action, self._last_actions = result["action"], result["actions"]
        self._stats = result["info"]
        return result

    def _do_step(self, **kwargs: Any) -> None:
        self.step(**kwargs)

    def get_action(self):
        return self._last_action

    def get_actions(self):
        return self._last_actions

    def get_stats(self) -> dict[str, Any]:
        return self._acc.get_stats()

    def end_episode(self, success: bool = False) -> dict[str, Any]:
        return self._acc.end_episode(success=success)
