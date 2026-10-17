"""Reversible configuration of the official OpenVLA-OFT VLA-Pruner backend.

Only the standard library is imported here. The algorithm is supplied by the
pinned upstream model and Transformers fork; this module manages its lifecycle.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, fields, replace
import math
from numbers import Real
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from .backend import BACKEND_COMMIT, BACKEND_REPOSITORY, check_model, load_backend


@dataclass(frozen=True)
class AcceleratorConfig:
    checkpoint: str | None = None
    task_suite_name: str = "libero_object"
    unnorm_key: str | None = None
    use_vla_pruner: bool = True
    fastv_k: int = 3
    fastv_r: float = 0.9275
    vla_pruner_layer: int = 15
    vla_pruner_mode: str = "semantic_action"
    vla_pruner_av_hist_w: int = 3
    vla_pruner_av_decay: float = 0.8
    vla_pruner_action_horizon: int = 0
    vla_pruner_semantic_weight: float = 0.5
    vla_pruner_action_weight: float = 0.5
    use_proprio: bool = True
    num_images_in_input: int = 2
    num_open_loop_steps: int = 8
    center_crop: bool = True
    use_film: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    lora_rank: int = 32

    def __post_init__(self) -> None:
        for name in ("use_vla_pruner", "use_proprio", "center_crop", "use_film",
                     "load_in_8bit", "load_in_4bit"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name, minimum in (("fastv_k", 1), ("vla_pruner_layer", 0),
                              ("vla_pruner_av_hist_w", 1), ("vla_pruner_action_horizon", 0),
                              ("num_open_loop_steps", 1), ("lora_rank", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if type(self.num_images_in_input) is not int or self.num_images_in_input not in (1, 2):
            raise ValueError("num_images_in_input must be 1 or 2")
        if self.num_open_loop_steps > 8 or self.vla_pruner_action_horizon > 8:
            raise ValueError("This backend supports action chunks of at most 8 steps")
        for name in ("fastv_r", "vla_pruner_av_decay", "vla_pruner_semantic_weight",
                     "vla_pruner_action_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0 <= self.fastv_r < 1:
            raise ValueError("fastv_r is the removed fraction and must satisfy 0 <= fastv_r < 1")
        if not 0 <= self.vla_pruner_av_decay <= 1:
            raise ValueError("vla_pruner_av_decay must be between 0 and 1")
        if min(self.vla_pruner_semantic_weight, self.vla_pruner_action_weight) < 0 or (
            self.vla_pruner_semantic_weight + self.vla_pruner_action_weight <= 0
        ):
            raise ValueError("Importance weights must be nonnegative with a positive sum")
        if self.vla_pruner_mode not in ("semantic_action", "semantic", "action", "fastv"):
            raise ValueError("Unsupported vla_pruner_mode")
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("Choose at most one quantization mode")
        if not isinstance(self.task_suite_name, str) or not self.task_suite_name:
            raise ValueError("task_suite_name must be a non-empty string")
        for name in ("checkpoint", "unnorm_key"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string or None")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> AcceleratorConfig:
        values = dict(config or {})
        if values.pop("method", "vla_pruner") != "vla_pruner":
            raise ValueError("Expected method: vla_pruner")
        if values.pop("state_type", "stateful") != "stateful":
            raise ValueError("VLA-Pruner requires state_type: stateful")
        unknown = values.keys() - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown VLA-Pruner options: {', '.join(sorted(unknown))}")
        return cls(**values)


_MISSING = object()
_OWNER = "_vla_express_pruner_owner"


class VLAPrunerAccelerator:
    """Own one model's pruning configuration without replacing its forwards.

    Concurrent inference/configuration on the same model is unsupported.
    ``step`` queries a chunk; the caller schedules actions in its environment.
    """

    def __init__(self, cfg: AcceleratorConfig | None = None):
        self.cfg = cfg or AcceleratorConfig()
        self.model = None
        self.enabled = False
        self._saved: list[tuple[Any, str, Any]] = []
        self._resources: dict[str, Any] = {}
        self._action_runner: Callable | None = None
        self._last_info: dict[str, Any] = {}
        self.last_caches = None
        self._steps = 0
        self._total_time = 0.0

    @classmethod
    def install(cls, model: Any = None, *, checkpoint: str | None = None,
                config: AcceleratorConfig | None = None, **kwargs: Any) -> VLAPrunerAccelerator:
        """Attach to a model, or explicitly load when only checkpoint is given."""
        names = {"processor", "action_head", "proprio_projector", "action_runner"}
        resources = {name: kwargs.pop(name) for name in list(kwargs) if name in names}
        if config is not None and kwargs:
            raise ValueError("Use either config or individual configuration options")
        cfg = config or AcceleratorConfig.from_mapping(kwargs)
        if checkpoint is not None:
            cfg = replace(cfg, checkpoint=checkpoint)
        if model is None:
            if resources:
                raise ValueError("Resources require an existing model")
            return cls.from_pretrained(cfg.checkpoint, config=cfg)
        acc = cls(cfg)
        acc.attach(model, **resources)
        return acc

    @classmethod
    def from_pretrained(cls, checkpoint: str | None, *, config: AcceleratorConfig | None = None,
                        **kwargs: Any) -> VLAPrunerAccelerator:
        """Load through the optional upstream backend, without checkpoint sync."""
        if not checkpoint:
            raise ValueError("from_pretrained requires a checkpoint path or model identifier")
        if config is not None and kwargs:
            raise ValueError("Use either config or individual configuration options")
        cfg = replace(config or AcceleratorConfig.from_mapping(kwargs), checkpoint=checkpoint)
        resources = load_backend().load_model(cfg)
        model = resources.pop("model")
        return cls.install(model, config=cfg, **resources)

    def attach(self, model: Any, *, processor: Any = None, action_head: Any = None,
               proprio_projector: Any = None, action_runner: Callable | None = None) -> None:
        if self.model is not None:
            raise RuntimeError("VLA-Pruner is already attached")
        if getattr(model, _OWNER, None) is not None:
            raise RuntimeError("This model already has a VLA-Pruner owner")
        check_model(model, self.cfg)
        if action_runner is not None and not callable(action_runner):
            raise TypeError("action_runner must be callable")
        self.model = model
        self._resources = dict(processor=processor, action_head=action_head,
                               proprio_projector=proprio_projector)
        self._action_runner = action_runner
        self._decoder = model.language_model.model
        self._cache_config = model.language_model.config
        self._unnorm_key = self.cfg.unnorm_key or self.cfg.task_suite_name
        stats = getattr(model, "norm_stats", {})
        if self._unnorm_key not in stats and f"{self._unnorm_key}_no_noops" in stats:
            self._unnorm_key += "_no_noops"
        try:
            self._write(model, _OWNER, self)
            # Keep the original deque untouched for exact restoration on detach.
            self._write(model, "av_hist", deque(maxlen=self.cfg.vla_pruner_av_hist_w))
            for obj, names in (
                (self._cache_config, ("reusable_patches", "proportion_attn_var")),
                (self._decoder, ("pruning_info", "_openvla_pruning_report_once_done")),
                (model.language_model, ("pruning_info",)),
                (model, ("_attention_pruning_mode_logged",)),
            ):
                for name in names:
                    self._write(obj, name, None)
            self._configure(self.cfg.use_vla_pruner)
            self.reset()
        except Exception:
            self._restore()
            self.model = None
            self.enabled = False
            self._resources.clear()
            self._action_runner = None
            self._decoder = None
            self._cache_config = None
            raise

    def _write(self, obj: Any, name: str, value: Any) -> None:
        if not any(o is obj and n == name for o, n, _ in self._saved):
            self._saved.append((obj, name, getattr(obj, name, _MISSING)))
        setattr(obj, name, value)

    def _restore(self) -> None:
        for obj, name, value in reversed(self._saved):
            if value is _MISSING:
                if hasattr(obj, name):
                    delattr(obj, name)
            else:
                setattr(obj, name, value)
        self._saved.clear()

    def _require_attached(self) -> None:
        if self.model is None:
            raise RuntimeError("Attach VLA-Pruner to a compatible model first")

    def _configure(self, enabled: bool) -> None:
        temporal = enabled and self.cfg.vla_pruner_mode in ("semantic_action", "action")
        values = {
            "use_fastv": enabled,
            "use_vla_pruner": enabled and self.cfg.vla_pruner_mode != "fastv",
            "use_temporal": temporal,
            "fastv_k": self.cfg.fastv_k, "fastv_r": self.cfg.fastv_r,
            "vla_pruner_layer": self.cfg.vla_pruner_layer,
            "vla_pruner_mode": self.cfg.vla_pruner_mode,
            "av_hist_w": self.cfg.vla_pruner_av_hist_w,
            "av_decay": self.cfg.vla_pruner_av_decay,
            "vla_pruner_semantic_weight": self.cfg.vla_pruner_semantic_weight,
            "vla_pruner_action_weight": self.cfg.vla_pruner_action_weight,
            "vla_pruner_action_horizon": self.cfg.vla_pruner_action_horizon,
            "return_attentions_for_cache": False,
            "use_text_vision_selection": False, "use_prefil_attention": True,
            "fastv_attention_source": "prefill",
        }
        for obj in (self.model, self.model.config):
            for name, value in values.items():
                self._write(obj, name, value)
        self.enabled = enabled

    def enable(self) -> None:
        self._require_attached()
        self._configure(True)
        self.reset()

    def disable(self) -> None:
        self._require_attached()
        self._configure(False)
        self.reset()

    def reset(self) -> None:
        self._require_attached()
        self.model.av_hist.clear()
        self.last_caches = None
        self._steps = 0
        self._total_time = 0.0
        self._last_info = {}
        self._cache_config.reusable_patches = None
        self._cache_config.proportion_attn_var = None
        self._decoder.pruning_info = None
        self.model.language_model.pruning_info = None
        self._decoder._openvla_pruning_report_once_done = False
        self.model._attention_pruning_mode_logged = False

    start_episode = reset

    def detach(self) -> None:
        if self.model is None:
            return
        self._restore()
        self.model = None
        self.enabled = False
        self.last_caches = None
        self._resources.clear()
        self._action_runner = None
        self._last_info = {}
        self._steps = 0
        self._total_time = 0.0
        self._decoder = None
        self._cache_config = None

    def get_stats(self) -> dict[str, Any]:
        result = {"method": "vla_pruner", "attached": self.model is not None,
                  "enabled": self.enabled, "mode": self.cfg.vla_pruner_mode,
                  "requested_prune_ratio": self.cfg.fastv_r,
                  "steps": self._steps, **self._last_info}
        info = getattr(getattr(self, "_decoder", None), "pruning_info", None)
        before = removed = kept = visual_before = visual_kept = layer = None
        if isinstance(info, dict):
            before = info.get("original_seq_length")
            indices = info.get("pruned_indices")
            if before is not None:
                # The pinned backend leaves indices None when no tokens were cut.
                removed = len(indices) if indices is not None else 0
                kept = before - removed
                backbone = self.model.vision_backbone
                visual_before = info.get("image_token_length")
                if visual_before is None:
                    visual_before = backbone.get_num_patches() * backbone.get_num_images_in_input()
                visual_kept = visual_before - removed
            layer = info.get("pruning_layer")
        # Unavailable counts stay None; never substitute the requested ratio.
        result.update(sequence_tokens_before=before, sequence_tokens_after=kept,
                      visual_tokens_before=visual_before, visual_tokens_kept=visual_kept,
                      visual_tokens_pruned=removed, pruning_layer=layer,
                      actual_prune_ratio=(removed / visual_before if visual_before else None))
        return result

    def describe(self) -> dict[str, Any]:
        return {"config": asdict(self.cfg), "backend_repository": BACKEND_REPOSITORY,
                "backend_commit": BACKEND_COMMIT, "stats": self.get_stats()}

    def step(self, image: Any, wrist_image: Any = None, instruction: str = "",
             proprio: Any = None) -> dict[str, Any]:
        """Query a full action chunk. No environment scheduling or gripper conversion."""
        self._require_attached()
        if image is None or not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("image and a non-empty instruction are required")
        if self.cfg.num_images_in_input == 2 and wrist_image is None:
            raise ValueError("wrist_image is required for two-view inference")
        if self.cfg.use_proprio and proprio is None:
            raise ValueError("proprio is required when use_proprio is enabled")
        if self._resources["processor"] is None or self._resources["action_head"] is None:
            raise ValueError("step requires processor and continuous action_head; attach them or use model.predict_action")
        if self.cfg.use_proprio and self._resources["proprio_projector"] is None:
            raise ValueError("step requires proprio_projector when use_proprio is enabled")
        if getattr(self.model, "training", False):
            raise RuntimeError("Call model.eval() before inference")
        norm_stats = getattr(self.model, "norm_stats", {})
        if self._unnorm_key not in norm_stats:
            raise ValueError(f"Missing action normalization statistics: {self._unnorm_key}")
        if self.cfg.use_proprio and "proprio" not in norm_stats[self._unnorm_key]:
            raise ValueError("Missing proprio normalization statistics")
        import torch

        if self._action_runner is None:
            self._action_runner = load_backend().get_vla_action
        values = asdict(self.cfg)
        values.update(unnorm_key=self._unnorm_key, use_vla_cache=False,
                      use_fastv=self.enabled and self.cfg.vla_pruner_mode == "fastv",
                      use_vla_pruner=self.enabled and self.cfg.vla_pruner_mode != "fastv")
        obs = {"full_image": image}
        if self.cfg.num_images_in_input == 2:
            obs["wrist_image"] = wrist_image
        if self.cfg.use_proprio:
            obs["state"] = proprio
        self._decoder.pruning_info = None
        self.model.language_model.pruning_info = None
        self._last_info = {}
        started = perf_counter()
        try:
            with torch.inference_mode():
                actions, _, _, metrics = self._action_runner(
                    SimpleNamespace(**values), self.model, self._resources["processor"], obs,
                    task_label=instruction, action_head=self._resources["action_head"],
                    proprio_projector=self._resources["proprio_projector"],
                    use_film=self.cfg.use_film, last_caches=None,
                )
            if len(actions) == 0:
                raise RuntimeError("Backend returned an empty action chunk")
            self._steps += 1
            self._last_info = {"wall_time_ms": (perf_counter() - started) * 1000,
                               "backend_time_elapsed": metrics.get("time_elapsed")}
            self._total_time += self._last_info["wall_time_ms"] / 1000
            return {"action": actions[0], "actions": actions, "info": self.get_stats()}
        except Exception:
            self.reset()  # Drop potentially incomplete temporal state on failure.
            raise

    def end_episode(self, success: bool = False) -> dict[str, Any]:
        self._require_attached()
        return {"steps": self._steps, "total_time": self._total_time,
                "avg_time_per_query": self._total_time / self._steps if self._steps else None,
                "success": success}
