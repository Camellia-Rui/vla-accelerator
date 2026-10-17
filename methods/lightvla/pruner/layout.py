from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

VALID_SCORE_CONTEXTS = ("suffix", "non_patch")


def validate_score_context(score_context: str) -> str:
    if score_context not in VALID_SCORE_CONTEXTS:
        valid_values = ", ".join(repr(value) for value in VALID_SCORE_CONTEXTS)
        raise ValueError(f"score_context must be one of {valid_values}, got {score_context!r}.")
    return score_context


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"TokenSpan.start must be >= 0, got {self.start}.")
        if self.end <= self.start:
            raise ValueError(f"TokenSpan.end must be > start, got start={self.start}, end={self.end}.")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class SequencePruningLayout:
    patch_span: TokenSpan
    score_context: str = "suffix"

    def __post_init__(self) -> None:
        validate_score_context(self.score_context)

    @property
    def patch_count(self) -> int:
        return self.patch_span.length

    def validate(self, sequence_length: int) -> None:
        if self.patch_span.end > sequence_length:
            raise ValueError(
                f"Patch span [{self.patch_span.start}, {self.patch_span.end}) exceeds sequence length {sequence_length}."
            )


@dataclass
class SequenceSegments:
    prefix_tokens: torch.Tensor
    patch_tokens: torch.Tensor
    suffix_tokens: torch.Tensor
    prefix_position_ids: torch.LongTensor
    patch_position_ids: torch.LongTensor
    suffix_position_ids: torch.LongTensor
    prefix_attention_mask: Optional[torch.Tensor]
    patch_attention_mask: Optional[torch.Tensor]
    suffix_attention_mask: Optional[torch.Tensor]


def build_position_ids(tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.LongTensor:
    batch_size, sequence_length = tokens.shape[:2]
    device = tokens.device

    if attention_mask is None:
        return torch.arange(sequence_length, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    if attention_mask.shape[:2] != (batch_size, sequence_length):
        raise ValueError(
            "Attention mask shape does not match tokens shape: "
            f"tokens={tuple(tokens.shape[:2])}, attention_mask={tuple(attention_mask.shape[:2])}."
        )

    position_ids = attention_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
    return position_ids.clamp_min_(0)


def split_sequence(
    tokens: torch.Tensor,
    position_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor],
    layout: SequencePruningLayout,
) -> SequenceSegments:
    layout.validate(tokens.shape[1])

    if attention_mask is not None and attention_mask.shape != tokens.shape[:2]:
        raise ValueError(
            "Attention mask shape must match [B, S]. "
            f"Got tokens={tuple(tokens.shape[:2])}, attention_mask={tuple(attention_mask.shape)}."
        )

    patch_start = layout.patch_span.start
    patch_end = layout.patch_span.end

    prefix_tokens = tokens[:, :patch_start]
    patch_tokens = tokens[:, patch_start:patch_end]
    suffix_tokens = tokens[:, patch_end:]

    prefix_position_ids = position_ids[:, :patch_start]
    patch_position_ids = position_ids[:, patch_start:patch_end]
    suffix_position_ids = position_ids[:, patch_end:]

    prefix_attention_mask = None
    patch_attention_mask = None
    suffix_attention_mask = None
    if attention_mask is not None:
        prefix_attention_mask = attention_mask[:, :patch_start]
        patch_attention_mask = attention_mask[:, patch_start:patch_end]
        suffix_attention_mask = attention_mask[:, patch_end:]

    return SequenceSegments(
        prefix_tokens=prefix_tokens,
        patch_tokens=patch_tokens,
        suffix_tokens=suffix_tokens,
        prefix_position_ids=prefix_position_ids,
        patch_position_ids=patch_position_ids,
        suffix_position_ids=suffix_position_ids,
        prefix_attention_mask=prefix_attention_mask,
        patch_attention_mask=patch_attention_mask,
        suffix_attention_mask=suffix_attention_mask,
    )


def merge_sequence(
    prefix_tokens: torch.Tensor,
    patch_tokens: torch.Tensor,
    suffix_tokens: torch.Tensor,
    prefix_position_ids: torch.LongTensor,
    patch_position_ids: torch.LongTensor,
    suffix_position_ids: torch.LongTensor,
    prefix_attention_mask: Optional[torch.Tensor],
    patch_attention_mask: Optional[torch.Tensor],
    suffix_attention_mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.LongTensor, Optional[torch.Tensor]]:
    tokens = torch.cat((prefix_tokens, patch_tokens, suffix_tokens), dim=1)
    position_ids = torch.cat((prefix_position_ids, patch_position_ids, suffix_position_ids), dim=1)

    attention_mask = None
    if prefix_attention_mask is not None:
        if patch_attention_mask is None or suffix_attention_mask is None:
            raise ValueError("Attention-mask segments must either all be present or all be None.")
        attention_mask = torch.cat((prefix_attention_mask, patch_attention_mask, suffix_attention_mask), dim=1)

    return tokens, position_ids, attention_mask
