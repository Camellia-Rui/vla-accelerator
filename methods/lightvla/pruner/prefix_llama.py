from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .core import DifferentiableTokenPruner, PruningResult
from .layout import (
  SequencePruningLayout,
  TokenSpan,
  build_position_ids,
  validate_score_context,
)


def _get_past_sequence_length(past_key_values) -> int:
    """Return a conservative cache length for modern and legacy HF cache formats."""
    if past_key_values is None:
        return 0

    get_seq_length = getattr(past_key_values, "get_seq_length", None)
    if callable(get_seq_length):
        try:
            return int(get_seq_length())
        except (TypeError, ValueError):
            pass

    seen_tokens = getattr(past_key_values, "seen_tokens", None)
    if seen_tokens is not None:
        try:
            return int(seen_tokens)
        except (TypeError, ValueError):
            pass

    if isinstance(past_key_values, (tuple, list)):
        if not past_key_values:
            return 0

        first_layer = past_key_values[0]
        if first_layer is None:
            return 0
        if isinstance(first_layer, (tuple, list)) and first_layer:
            key_states = first_layer[0]
            if isinstance(key_states, torch.Tensor) and key_states.ndim >= 2:
                return int(key_states.shape[-2])

        # Unknown non-empty legacy cache. Bypassing is safer than pruning decode tokens.
        return 1

    # Unknown non-None cache-like object. Treat it as populated for backwards compatibility.
    return 1


@dataclass
class StaticPrefixLayoutResolver:
    patch_count: int
    patch_start: int = 1
    score_context: str = "suffix"

    def __post_init__(self) -> None:
        if self.patch_count <= 0:
            raise ValueError(f"patch_count must be > 0, got {self.patch_count}.")
        if self.patch_start < 0:
            raise ValueError(f"patch_start must be >= 0, got {self.patch_start}.")
        validate_score_context(self.score_context)

    def set_patch_count(self, patch_count: int) -> None:
        if patch_count <= 0:
            raise ValueError(f"patch_count must be > 0, got {patch_count}.")
        self.patch_count = patch_count

    def set_score_context(self, score_context: str) -> None:
        self.score_context = validate_score_context(score_context)

    @property
    def patch_end(self) -> int:
        return self.patch_start + self.patch_count

    def resolve(self, sequence_length: int) -> SequencePruningLayout:
        if self.patch_end > sequence_length:
            raise ValueError(
                f"Expected patch span [{self.patch_start}, {self.patch_end}) inside sequence length {sequence_length}."
            )
        return SequencePruningLayout(
            patch_span=TokenSpan(start=self.patch_start, end=self.patch_end),
            score_context=self.score_context,
        )


class PrefixLlamaPruningAdapter(nn.Module):
    def __init__(
        self,
        decoder_model: nn.Module,
        pruner: DifferentiableTokenPruner,
        layout_resolver: StaticPrefixLayoutResolver,
        bypass_with_cache: bool = True,
        allow_existing_pruner: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(decoder_model, PrefixLlamaPruningAdapter):
            raise ValueError("decoder_model is already wrapped by PrefixLlamaPruningAdapter.")
        if not allow_existing_pruner and isinstance(getattr(decoder_model, "pruner", None), nn.Module):
            raise ValueError(
                "decoder_model already exposes a pruning module. Refusing to apply a second pruner; "
                "disable or remove the existing pruning path first."
            )

        self.decoder_model = decoder_model
        self.pruner = pruner
        self.layout_resolver = layout_resolver
        self.bypass_with_cache = bypass_with_cache
        self.pruned_forward_count = 0
        self.last_input_sequence_length: Optional[int] = None
        self.last_output_sequence_length: Optional[int] = None
        self.last_patch_keep_counts: Optional[tuple[int, ...]] = None
        self.train(decoder_model.training)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.decoder_model, name)

    def set_patch_count(self, patch_count: int) -> None:
        self.layout_resolver.set_patch_count(patch_count)

    def set_noise_scale(self, noise_scale: Optional[float]) -> None:
        self.pruner.set_noise_scale(noise_scale)

    def set_score_context(self, score_context: str) -> None:
        self.layout_resolver.set_score_context(score_context)

    def prune_inputs(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> PruningResult:
        layout = self.layout_resolver.resolve(inputs_embeds.shape[1])
        if position_ids is None:
            position_ids = build_position_ids(inputs_embeds, attention_mask)
        return self.pruner(
            tokens=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            layout=layout,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            return self.decoder_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=None,
                **kwargs,
            )

        past_sequence_length = _get_past_sequence_length(past_key_values)
        should_bypass = self.bypass_with_cache and past_sequence_length > 0
        if should_bypass:
            return self.decoder_model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        pruning_result = self.prune_inputs(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        self.pruned_forward_count += 1
        self.last_input_sequence_length = int(inputs_embeds.shape[1])
        self.last_output_sequence_length = int(pruning_result.tokens.shape[1])
        self.last_patch_keep_counts = (
            None
            if pruning_result.patch_keep_counts is None
            else tuple(int(value) for value in pruning_result.patch_keep_counts.detach().cpu().tolist())
        )

        if kwargs.get("cache_position") is not None:
            input_cache_position = kwargs["cache_position"]
            kwargs["cache_position"] = torch.arange(
                past_sequence_length,
                past_sequence_length + pruning_result.tokens.shape[1],
                dtype=input_cache_position.dtype,
                device=input_cache_position.device,
            )

        return self.decoder_model(
            input_ids=None,
            attention_mask=pruning_result.attention_mask,
            position_ids=pruning_result.position_ids,
            past_key_values=past_key_values,
            inputs_embeds=pruning_result.tokens,
            **kwargs,
        )
