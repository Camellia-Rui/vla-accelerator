"""Dependency-free configuration for the LightVLA toolbox adapter."""

from dataclasses import dataclass, fields
import math
from numbers import Real
from typing import Any, Mapping


PRUNING_MODES = ("builtin", "baseline", "packaged")


@dataclass(frozen=True)
class LightVLAConfig:
    mode: str = "packaged"
    patch_start: int = 1
    patch_count: int | None = None
    score_context: str = "suffix"
    noise_scale: float | None = None
    allow_existing_pruner: bool = False

    def __post_init__(self):
        if self.mode not in PRUNING_MODES:
            raise ValueError(f"mode must be one of {PRUNING_MODES}")
        if type(self.patch_start) is not int or self.patch_start < 0:
            raise ValueError("patch_start must be a nonnegative integer")
        if self.patch_count is not None and (type(self.patch_count) is not int or self.patch_count <= 0):
            raise ValueError("patch_count must be a positive integer or None")
        if self.score_context not in ("suffix", "non_patch"):
            raise ValueError("score_context must be suffix or non_patch")
        if type(self.allow_existing_pruner) is not bool:
            raise ValueError("allow_existing_pruner must be a boolean")
        if self.noise_scale is not None and (
            isinstance(self.noise_scale, bool) or not isinstance(self.noise_scale, Real)
            or not math.isfinite(self.noise_scale) or self.noise_scale < 0
        ):
            raise ValueError("noise_scale must be a finite nonnegative number or None")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None = None):
        values = dict(config or {})
        for name in ("method", "name"):
            if values.pop(name, "lightvla") != "lightvla":
                raise ValueError(f"{name} must be lightvla")
        if values.pop("state_type", "stateless") != "stateless":
            raise ValueError("LightVLA requires state_type: stateless")
        unknown = values.keys() - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown LightVLA options: {', '.join(sorted(unknown))}")
        return cls(**values)
