"""Adapter: VLA-Pruner -> BasePruner."""

from __future__ import annotations

from typing import Any, Dict

from pruners.base import BasePruner
from pruners.registry import register


@register
class VLAPrunerAdapter(BasePruner):
    name = "vla_pruner"
    state_type = "stateful"

    def _do_attach(self, model: Any, **resources: Any) -> None:
        from methods.vla_pruner.accelerator import VLAPrunerAccelerator
        self._acc = VLAPrunerAccelerator.install(
            checkpoint=self.config["checkpoint"],
            task_suite_name=self.config.get("task_suite_name", "libero_object"),
            fastv_r=self.config.get("fastv_r", 0.9275),
            vla_pruner_layer=self.config.get("vla_pruner_layer", 15),
            vla_pruner_mode=self.config.get("vla_pruner_mode", "semantic_action"),
            vla_pruner_av_hist_w=self.config.get("vla_pruner_av_hist_w", 3),
            vla_pruner_av_decay=self.config.get("vla_pruner_av_decay", 0.8),
        )
        self._last_action = None

    def _do_reset(self) -> None:
        self._acc.start_episode()
        self._last_action = None

    def _do_step(self, **kwargs: Any) -> None:
        result = self._acc.step(
            image=kwargs["image"],
            wrist_image=kwargs.get("wrist_image"),
            instruction=kwargs.get("instruction", ""),
            proprio=kwargs.get("proprio"),
        )
        self._last_action = result["action"]
        self._stats = {
            "action": result["action"],
            "time_elapsed": result["info"].get("time_elapsed", 0.0),
        }

    def get_action(self):
        return self._last_action

    def end_episode(self, success: bool = False) -> Dict[str, Any]:
        return self._acc.end_episode(success=success)