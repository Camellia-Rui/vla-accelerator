from __future__ import annotations

from types import MethodType
from typing import Any, Dict

import torch.nn as nn


PRUNING_MODES = ("builtin", "baseline", "packaged")


def _get_decoder_model(vla: nn.Module) -> nn.Module:
    language_model = getattr(vla, "language_model", None)
    decoder_model = getattr(language_model, "model", None)
    if not isinstance(decoder_model, nn.Module):
        raise TypeError("Expected a VLA with a decoder at `vla.language_model.model`.")
    return decoder_model


def _use_vanilla_llama_forward(decoder_model: nn.Module) -> None:
    from transformers.models.llama.modeling_llama import LlamaModel

    if not isinstance(decoder_model, LlamaModel):
        raise TypeError(
            "baseline/packaged modes require `vla.language_model.model` to be a LlamaModel instance; "
            f"got {type(decoder_model).__name__}."
        )

    # LightVLA's PrunedLlamaModel has standard LLaMA weights but overrides forward to prune.
    # Binding the upstream implementation gives a no-pruning baseline without reloading weights.
    decoder_model.forward = MethodType(LlamaModel.forward, decoder_model)


def _get_patch_count(vla: nn.Module) -> int:
    get_num_patches = getattr(vla, "get_num_patches", None)
    if callable(get_num_patches):
        patch_count = int(get_num_patches())
    else:
        vision_backbone = getattr(vla, "vision_backbone", None)
        get_patches_per_image = getattr(vision_backbone, "get_num_patches", None)
        get_num_images = getattr(vision_backbone, "get_num_images_in_input", None)
        if not callable(get_patches_per_image) or not callable(get_num_images):
            raise TypeError("Could not infer the visual patch count from the VLA model.")
        patch_count = int(get_patches_per_image()) * int(get_num_images())

    if patch_count <= 0:
        raise ValueError(f"Model reported an invalid visual patch count: {patch_count}.")
    return patch_count


def configure_lightvla_pruning(
    vla: nn.Module,
    mode: str,
    patch_start: int = 1,
    score_context: str = "suffix",
) -> Dict[str, Any]:
    if mode not in PRUNING_MODES:
        raise ValueError(f"pruning mode must be one of {PRUNING_MODES}, got {mode!r}.")

    decoder_model = _get_decoder_model(vla)
    original_decoder_class = type(decoder_model).__name__

    if mode == "builtin":
        if not isinstance(getattr(decoder_model, "pruner", None), nn.Module):
            raise ValueError(
                "`builtin` mode was requested, but the loaded decoder does not expose an internal pruner. "
                "Use `baseline` or `packaged` for an unmodified LLaMA decoder."
            )
        summary = {
            "mode": mode,
            "decoder_class": original_decoder_class,
            "pruner_class": type(decoder_model.pruner).__name__,
        }
    else:
        _use_vanilla_llama_forward(decoder_model)
        if mode == "baseline":
            summary = {
                "mode": mode,
                "decoder_class": original_decoder_class,
                "pruner_class": None,
            }
        else:
            try:
                from lightvla_pruner import (
                    DifferentiableTokenPruner,
                    PrefixLlamaPruningAdapter,
                    StaticPrefixLayoutResolver,
                )
            except ImportError as exc:
                raise ImportError(
                    "The packaged pruner is not installed. Run `python -m pip install -e packaged_pruner` "
                    "from the repository root."
                ) from exc

            patch_count = _get_patch_count(vla)
            hidden_size = int(decoder_model.config.hidden_size)
            adapter = PrefixLlamaPruningAdapter(
                decoder_model=decoder_model,
                pruner=DifferentiableTokenPruner(hidden_size=hidden_size),
                layout_resolver=StaticPrefixLayoutResolver(
                    patch_count=patch_count,
                    patch_start=patch_start,
                    score_context=score_context,
                ),
                # The old module remains registered, but its pruning forward has just been disabled.
                allow_existing_pruner=True,
            )
            vla.language_model.model = adapter
            summary = {
                "mode": mode,
                "decoder_class": original_decoder_class,
                "pruner_class": type(adapter.pruner).__name__,
                "patch_count": patch_count,
                "patch_start": patch_start,
                "score_context": score_context,
            }

    vla._pruning_runtime_summary = summary
    return summary


def get_pruning_runtime_summary(vla: nn.Module) -> Dict[str, Any]:
    summary = dict(getattr(vla, "_pruning_runtime_summary", {}))
    decoder_model = _get_decoder_model(vla)

    try:
        from lightvla_pruner import PrefixLlamaPruningAdapter
    except ImportError:
        PrefixLlamaPruningAdapter = ()

    if isinstance(decoder_model, PrefixLlamaPruningAdapter):
        summary.update(
            {
                "pruned_forward_count": decoder_model.pruned_forward_count,
                "last_input_sequence_length": decoder_model.last_input_sequence_length,
                "last_output_sequence_length": decoder_model.last_output_sequence_length,
                "last_patch_keep_counts": decoder_model.last_patch_keep_counts,
            }
        )
    return summary
