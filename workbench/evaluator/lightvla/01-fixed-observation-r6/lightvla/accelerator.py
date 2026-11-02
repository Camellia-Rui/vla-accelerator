"""Reversible LightVLA integration. Model dependencies are imported on attach."""

from __future__ import annotations

from dataclasses import asdict, replace
from types import MethodType, SimpleNamespace

from .config import LightVLAConfig

_MISSING = object()
_OWNER = "_lightvla_toolbox_session"


def _load_components():
    try:
        import torch
        from transformers.models.llama.modeling_llama import LlamaModel
        from .pruner import DifferentiableTokenPruner, PrefixLlamaPruningAdapter, StaticPrefixLayoutResolver
    except ImportError as exc:
        raise ImportError(
            "LightVLA model dependencies are unavailable. Install torch and a compatible "
            "Transformers in the model environment; see methods/lightvla/README.md. "
            f"Original error: {exc}"
        ) from exc
    return SimpleNamespace(torch=torch, LlamaModel=LlamaModel,
                           Pruner=DifferentiableTokenPruner,
                           Adapter=PrefixLlamaPruningAdapter,
                           Resolver=StaticPrefixLayoutResolver)


def _patch_count(vla, configured):
    if configured is not None:
        return configured
    backbone = getattr(vla, "vision_backbone", None)
    per_image = getattr(backbone, "get_num_patches", None)
    images = getattr(backbone, "get_num_images_in_input", None)
    if callable(per_image) and callable(images):
        count = int(per_image()) * int(images())
    elif callable(getattr(vla, "get_num_patches", None)):
        count = int(vla.get_num_patches())
    else:
        raise TypeError("Cannot determine visual token count; provide verified patch_count")
    if count <= 0:
        raise ValueError(f"Model reported invalid patch count: {count}")
    return count


def _restore_attribute(obj, name, value):
    if value is _MISSING:
        if name in vars(obj):
            delattr(obj, name)
    else:
        setattr(obj, name, value)


class LightVLAAccelerator:
    """Own a model's pruning lifecycle without reloading weights.

    This adapter handles a verified contiguous visual span before a LLaMA
    decoder. It does not preprocess observations or schedule robot actions.
    """

    def __init__(self, config: LightVLAConfig | None = None):
        self.config = config or LightVLAConfig()
        self.model = None
        self._adapter = None
        self._decoder = None
        self._components = None
        self._enabled_mode = self.config.mode if self.config.mode != "baseline" else "packaged"

    @classmethod
    def install(cls, model, *, config=None, **options):
        if config is not None and options:
            raise ValueError("Use config or individual options, not both")
        if isinstance(config, dict):
            config = LightVLAConfig.from_mapping(config)
        acc = cls(config or LightVLAConfig.from_mapping(options))
        acc.attach(model)
        return acc

    @property
    def enabled(self):
        return self.model is not None and self.config.mode != "baseline"

    def _require_attached(self):
        if self.model is None:
            raise RuntimeError("Attach LightVLA to a compatible model first")

    def attach(self, model):
        if self.model is not None:
            raise RuntimeError("LightVLA is already attached")
        if getattr(model, _OWNER, None) is not None:
            raise RuntimeError("This model already has a LightVLA owner")
        if getattr(model, "_vla_express_pruner_owner", None) is not None:
            raise RuntimeError("Detach VLA-Pruner before attaching LightVLA")
        language_model = getattr(model, "language_model", None)
        decoder = getattr(language_model, "model", None)
        components = _load_components()
        if not isinstance(decoder, components.LlamaModel):
            raise TypeError("Expected an unwrapped LlamaModel at model.language_model.model")
        if hasattr(decoder, "_specprune_orig_forward"):
            raise RuntimeError("Restore the SpecPrune model before attaching LightVLA")
        self.model, self._decoder, self._components = model, decoder, components
        self._original_forward = vars(decoder).get("forward", _MISSING)
        self._effective_forward = decoder.forward
        self._original_summary = vars(model).get("_pruning_runtime_summary", _MISSING)
        try:
            self._apply(self.config)
            setattr(model, _OWNER, self)
        except Exception:
            self._restore_original()
            self.model = self._decoder = self._adapter = self._components = None
            raise

    def _prepare(self, config):
        components = self._components
        builtin = getattr(self._decoder, "pruner", None)
        has_builtin = isinstance(builtin, components.torch.nn.Module)
        if has_builtin and not config.allow_existing_pruner:
            raise ValueError("Decoder has a builtin pruner; explicitly set allow_existing_pruner=True to manage its forward")
        if config.mode == "builtin":
            if not has_builtin or getattr(self._effective_forward, "__func__", None) is components.LlamaModel.forward:
                raise ValueError("builtin mode requires a builtin pruner AND an original custom pruning forward")
            return None, None
        if config.mode == "baseline":
            return None, None
        count = _patch_count(self.model, config.patch_count)
        # Construct everything before mutating the active model.
        adapter = components.Adapter(
            decoder_model=self._decoder,
            pruner=components.Pruner(hidden_size=int(self._decoder.config.hidden_size),
                                     noise_scale=config.noise_scale),
            layout_resolver=components.Resolver(patch_count=count,
                                                patch_start=config.patch_start,
                                                score_context=config.score_context),
            allow_existing_pruner=config.allow_existing_pruner,
        )
        return adapter, count

    def _apply(self, config):
        adapter, count = self._prepare(config)
        current_decoder = self.model.language_model.model
        current_forward = vars(self._decoder).get("forward", _MISSING)
        current_summary = vars(self.model).get("_pruning_runtime_summary", _MISSING)
        summary = {"method": "lightvla", "mode": config.mode,
                   "decoder_class": type(self._decoder).__name__,
                   "pruner_class": (type(adapter.pruner).__name__ if adapter is not None else
                                    type(self._decoder.pruner).__name__ if config.mode == "builtin" else None),
                   "patch_count": count, "patch_start": config.patch_start,
                   "score_context": config.score_context}
        try:
            if config.mode == "builtin":
                _restore_attribute(self._decoder, "forward", self._original_forward)
            else:
                self._decoder.forward = MethodType(self._components.LlamaModel.forward, self._decoder)
            self.model.language_model.model = adapter if adapter is not None else self._decoder
            self.model._pruning_runtime_summary = summary
        except Exception:
            _restore_attribute(self._decoder, "forward", current_forward)
            self.model.language_model.model = current_decoder
            _restore_attribute(self.model, "_pruning_runtime_summary", current_summary)
            raise
        self._adapter, self.config = adapter, config
        if config.mode != "baseline":
            self._enabled_mode = config.mode

    def set_mode(self, mode, **options):
        self._require_attached()
        self._check_owner()
        self._apply(replace(self.config, mode=mode, **options))
        return self.get_stats()

    def _check_owner(self):
        active = self._adapter if self._adapter is not None else self._decoder
        if self.model.language_model.model is not active:
            raise RuntimeError("Model decoder changed outside LightVLA; refusing to overwrite it")

    def enable(self):
        self._require_attached()
        if not self.enabled:
            self.set_mode(self._enabled_mode)

    def disable(self):
        self._require_attached()
        if self.enabled:
            self.set_mode("baseline")

    def reset(self):
        self._require_attached()
        self._check_owner()
        if self._adapter is not None:
            self._adapter.reset_stats()

    start_episode = reset

    def _restore_original(self):
        self.model.language_model.model = self._decoder
        _restore_attribute(self._decoder, "forward", self._original_forward)
        _restore_attribute(self.model, "_pruning_runtime_summary", self._original_summary)
        if getattr(self.model, _OWNER, None) is self:
            delattr(self.model, _OWNER)

    def detach(self):
        if self.model is None:
            return
        self._check_owner()
        self._restore_original()
        self.model = self._decoder = self._adapter = self._components = None

    def get_stats(self):
        summary = {"method": "lightvla", "attached": self.model is not None,
                   "enabled": self.enabled, "mode": self.config.mode,
                   "pruned_forward_count": 0, "bypassed_forward_count": 0,
                   "last_input_sequence_length": None, "last_output_sequence_length": None,
                   "last_patch_keep_counts": None}
        if self.model is not None:
            summary.update(self.model._pruning_runtime_summary)
        if self._adapter is not None:
            for key in ("pruned_forward_count", "bypassed_forward_count", "last_input_sequence_length",
                        "last_output_sequence_length", "last_patch_keep_counts"):
                summary[key] = getattr(self._adapter, key)
        return summary

    def describe(self):
        return {"config": asdict(self.config), "stats": self.get_stats()}

    def predict_action(self, *args, **kwargs):
        self._require_attached()
        self._check_owner()
        if self.model.training:
            raise RuntimeError("Call model.eval() before inference")
        try:
            with self._components.torch.inference_mode():
                return self.model.predict_action(*args, **kwargs)
        except Exception:
            self.reset()
            raise


def configure_lightvla_pruning(vla, mode, patch_start=1, score_context="suffix", *,
                               patch_count=None, noise_scale=None, allow_existing_pruner=False):
    """Legacy function API, now with transactional mode switches and restoration."""
    options = dict(patch_start=patch_start, score_context=score_context, patch_count=patch_count,
                   noise_scale=noise_scale, allow_existing_pruner=allow_existing_pruner)
    session = getattr(vla, _OWNER, None)
    if session is None:
        session = LightVLAAccelerator.install(vla, mode=mode, **options)
    else:
        session.set_mode(mode, **options)
    return session.get_stats()


def get_pruning_runtime_summary(vla):
    session = getattr(vla, _OWNER, None)
    return session.get_stats() if session is not None else {"method": "lightvla", "attached": False}


def restore_lightvla_pruning(vla):
    session = getattr(vla, _OWNER, None)
    if session is not None:
        session.detach()
