from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

from .core import DifferentiableTokenPruner
from .layout import SequencePruningLayout, TokenSpan, validate_score_context

PruningMode = Literal["baseline", "builtin", "packaged"]
VALID_PRUNING_MODES = ("baseline", "builtin", "packaged")


@dataclass(frozen=True)
class PruningCallRecord:
    call: int
    input_length: int
    output_length: int
    original_patches: int
    keep_counts: Optional[tuple[int, ...]]


class RuntimePruner(nn.Module):
    """Drop-in three-argument pruner used to compare runtime pruning modes."""

    def __init__(
        self,
        mode: PruningMode,
        num_patches: int,
        hidden_size: int,
        builtin_pruner: Optional[nn.Module] = None,
        patch_start: int = 1,
        score_context: str = "suffix",
    ) -> None:
        super().__init__()
        if mode not in VALID_PRUNING_MODES:
            raise ValueError(f"mode must be one of {VALID_PRUNING_MODES}, got {mode!r}.")
        if num_patches <= 0:
            raise ValueError(f"num_patches must be > 0, got {num_patches}.")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {hidden_size}.")
        if patch_start < 0:
            raise ValueError(f"patch_start must be >= 0, got {patch_start}.")
        if mode == "builtin" and builtin_pruner is None:
            raise ValueError("builtin mode requires builtin_pruner.")

        self.mode = mode
        self.num_patches = num_patches
        self.hidden_size = hidden_size
        self.patch_start = patch_start
        self.score_context = validate_score_context(score_context)
        self.builtin_pruner = builtin_pruner
        self.packaged_pruner = None
        if mode == "packaged":
            self.packaged_pruner = DifferentiableTokenPruner(hidden_size=hidden_size)

        self.layout = SequencePruningLayout(
            patch_span=TokenSpan(start=patch_start, end=patch_start + num_patches),
            score_context=self.score_context,
        )
        self.call_records: list[PruningCallRecord] = []

    @property
    def calls(self) -> int:
        return len(self.call_records)

    def reset_stats(self) -> None:
        self.call_records.clear()

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "calls": self.calls,
            "num_patches": self.num_patches,
            "hidden_size": self.hidden_size,
            "patch_start": self.patch_start,
            "score_context": self.score_context,
            "records": [asdict(record) for record in self.call_records],
        }

    def forward(
        self,
        tokens: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.LongTensor], Optional[torch.Tensor]]:
        input_length = int(tokens.shape[1])
        keep_counts: Optional[tuple[int, ...]]

        if self.mode == "baseline":
            output_tokens = tokens
            output_position_ids = position_ids
            output_attention_mask = attention_mask
            keep_counts = tuple(self.num_patches for _ in range(tokens.shape[0]))
        elif self.mode == "builtin":
            output_tokens, output_position_ids, output_attention_mask = self.builtin_pruner(
                tokens, position_ids, attention_mask
            )
            non_patch_length = input_length - self.num_patches
            kept = int(output_tokens.shape[1]) - non_patch_length
            keep_counts = tuple(kept for _ in range(tokens.shape[0]))
        else:
            result = self.packaged_pruner(
                tokens=tokens,
                position_ids=position_ids,
                attention_mask=attention_mask,
                layout=self.layout,
            )
            output_tokens = result.tokens
            output_position_ids = result.position_ids
            output_attention_mask = result.attention_mask
            keep_counts = (
                None
                if result.patch_keep_counts is None
                else tuple(int(value) for value in result.patch_keep_counts.detach().cpu().tolist())
            )

        self.call_records.append(
            PruningCallRecord(
                call=self.calls + 1,
                input_length=input_length,
                output_length=int(output_tokens.shape[1]),
                original_patches=self.num_patches,
                keep_counts=keep_counts,
            )
        )
        return output_tokens, output_position_ids, output_attention_mask


def resolve_module(root: nn.Module, dotted_path: str) -> nn.Module:
    module = root
    for segment in dotted_path.split("."):
        if not segment:
            raise ValueError(f"Invalid empty segment in module path {dotted_path!r}.")
        if not hasattr(module, segment):
            raise AttributeError(f"Module path {dotted_path!r} does not exist; missing segment {segment!r}.")
        module = getattr(module, segment)
    if not isinstance(module, nn.Module):
        raise TypeError(f"Resolved object at {dotted_path!r} is not an nn.Module.")
    return module


class PruningRuntime:
    """Owns a temporary pruner replacement and can restore the original module."""

    def __init__(self, decoder: nn.Module, original_pruner: nn.Module, runtime_pruner: RuntimePruner) -> None:
        self.decoder = decoder
        self.original_pruner = original_pruner
        self.runtime_pruner = runtime_pruner
        self._restored = False

    @property
    def mode(self) -> PruningMode:
        return self.runtime_pruner.mode

    def summary(self) -> dict:
        return self.runtime_pruner.summary()

    def restore(self) -> None:
        if self._restored:
            return
        if getattr(self.decoder, "pruner", None) is not self.runtime_pruner:
            raise RuntimeError("Decoder pruner changed after runtime configuration; refusing to overwrite it.")
        self.decoder.pruner = self.original_pruner
        self._restored = True

    def __enter__(self) -> "PruningRuntime":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.restore()


def configure_existing_pruner(
    model: nn.Module,
    mode: PruningMode,
    decoder_path: str = "language_model.model",
    patch_count: Optional[int] = None,
    hidden_size: Optional[int] = None,
    patch_start: int = 1,
    score_context: str = "suffix",
) -> PruningRuntime:
    """Replace an existing decoder pruner with an instrumented runtime mode.

    This integration path is intended for models such as LightVLA whose decoder
    already calls ``self.pruner(tokens, position_ids, attention_mask)``.
    """

    decoder = resolve_module(model, decoder_path)
    original_pruner = getattr(decoder, "pruner", None)
    if not isinstance(original_pruner, nn.Module):
        raise ValueError(f"Decoder at {decoder_path!r} does not expose an nn.Module pruner.")

    inferred_patch_count = getattr(original_pruner, "num_patches", None)
    patch_count = inferred_patch_count if patch_count is None else patch_count
    if patch_count is None:
        raise ValueError("patch_count was not provided and could not be inferred from pruner.num_patches.")

    config = getattr(decoder, "config", None)
    inferred_hidden_size = getattr(config, "hidden_size", None)
    hidden_size = inferred_hidden_size if hidden_size is None else hidden_size
    if hidden_size is None:
        raise ValueError("hidden_size was not provided and could not be inferred from decoder.config.hidden_size.")

    runtime_pruner = RuntimePruner(
        mode=mode,
        num_patches=int(patch_count),
        hidden_size=int(hidden_size),
        builtin_pruner=original_pruner if mode == "builtin" else None,
        patch_start=patch_start,
        score_context=score_context,
    )
    runtime_pruner.train(decoder.training)
    decoder.pruner = runtime_pruner
    return PruningRuntime(decoder, original_pruner, runtime_pruner)
