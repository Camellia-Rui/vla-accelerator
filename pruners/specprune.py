"""Adapter: SpecPrune-VLA -> BasePruner."""

from __future__ import annotations

from typing import Any, Dict

from pruners.base import BasePruner
from pruners.registry import register


@register
class SpecPruneAdapter(BasePruner):
    name = "specprune"
    state_type = "stateful"

    def _do_attach(self, model: Any, **resources: Any) -> None:
        from methods.specprune.accelerator import (
            SpecPruneAccelerator, AcceleratorConfig,
        )
        cfg_kwargs = dict(self.config)
        cfg_kwargs.pop("checkpoint", None)
        acc_cfg = AcceleratorConfig(**{
            k: v for k, v in cfg_kwargs.items()
            if k in AcceleratorConfig.__dataclass_fields__
        })
        self._acc = SpecPruneAccelerator.install(
            model=model,
            processor=resources["processor"],
            action_head=resources.get("action_head"),
            proprio_projector=resources.get("proprio_projector"),
            noisy_action_projector=resources.get("noisy_action_projector"),
            config=acc_cfg,
        )
        self._last_action = None

    def _do_reset(self) -> None:
        self._acc.start_episode()
        self._last_action = None

    def _do_step(self, **kwargs: Any) -> None:
        action, info = self._acc.step(
            image=kwargs["image"],
            wrist_image=kwargs.get("wrist_image"),
            text=kwargs.get("instruction", ""),
            proprio=kwargs.get("proprio"),
        )
        self._last_action = action
        self._stats = {"action": action, **info}

    def get_action(self):
        return self._last_action

    def end_episode(self, success: bool = False) -> Dict[str, Any]:
        stats = self._acc.end_episode()
        stats["success"] = success
        return stats