"""LightVLA registration with lazy model dependencies."""

from typing import Any

from methods.lightvla import LightVLAAccelerator, LightVLAConfig
from pruners.base import BasePruner
from pruners.registry import register


@register
class LightVLAAdapter(BasePruner):
    name = "lightvla"
    state_type = "stateless"

    def __init__(self, config=None):
        super().__init__(config)
        self._acc = LightVLAAccelerator(LightVLAConfig.from_mapping(config))
        self._last_output = None

    def _do_attach(self, model: Any):
        if model is None:
            raise ValueError("LightVLA attach requires an existing model")
        self._acc.attach(model)

    def detach(self, model=None):
        if model is not None and self._acc.model is not None and model is not self._acc.model:
            raise ValueError("detach received a different model")
        self._acc.detach()
        self._attached = False
        self._last_output = None
        self._stats = {}

    def enable(self):
        self._acc.enable()
        self._last_output = None

    def disable(self):
        self._acc.disable()
        self._last_output = None

    def set_mode(self, mode, **options):
        result = self._acc.set_mode(mode, **options)
        self._last_output = None
        return result

    @property
    def enabled(self):
        return self._acc.enabled

    def _do_reset(self):
        self._acc.reset()
        self._last_output = None
        self._stats = {}

    def predict_action(self, *args, **kwargs):
        self._last_output = None
        self._last_output = self._acc.predict_action(*args, **kwargs)
        return self._last_output

    def step(self, **model_inputs):
        """Run prepared model inputs; preserve the native output without guessing its shape."""
        output = self.predict_action(**model_inputs)
        return {"output": output, "info": self.get_stats()}

    def _do_step(self, **kwargs):
        self.step(**kwargs)

    def get_output(self):
        return self._last_output

    def get_stats(self):
        return self._acc.get_stats()
