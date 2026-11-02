from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layout import SequencePruningLayout, build_position_ids, merge_sequence, split_sequence


@dataclass
class PruningResult:
    tokens: torch.Tensor
    position_ids: torch.LongTensor
    attention_mask: Optional[torch.Tensor]
    scores: torch.Tensor
    patch_keep_mask: Optional[torch.BoolTensor]
    patch_keep_indices: Optional[torch.LongTensor]
    patch_keep_counts: Optional[torch.LongTensor]
    selection_weights: Optional[torch.Tensor]


class DifferentiableTokenPruner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        noise_scale: Optional[float] = None,
        inference_min_keep_patches: Optional[int] = None,
        inference_grid_blocks_per_side: Optional[int] = None,
        inference_merge_dropped_patches: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {hidden_size}.")
        self.hidden_size = hidden_size
        self.noise_scale = None
        self.set_noise_scale(noise_scale)
        self.inference_min_keep_patches = None
        self.set_inference_min_keep_patches(inference_min_keep_patches)
        self.inference_grid_blocks_per_side = None
        self.set_inference_grid_blocks_per_side(inference_grid_blocks_per_side)
        self.inference_merge_dropped_patches = bool(inference_merge_dropped_patches)
        self.scale_factor = 1.0 / math.sqrt(hidden_size)

    def set_noise_scale(self, noise_scale: Optional[float]) -> None:
        if noise_scale is not None and noise_scale < 0:
            raise ValueError(f"noise_scale must be >= 0 or None, got {noise_scale}.")
        self.noise_scale = noise_scale

    def set_inference_min_keep_patches(self, value: Optional[int]) -> None:
        if value is not None and value <= 0:
            raise ValueError(
                "inference_min_keep_patches must be > 0 or None, "
                f"got {value}."
            )
        self.inference_min_keep_patches = value

    def set_inference_grid_blocks_per_side(self, value: Optional[int]) -> None:
        if value is not None and value <= 0:
            raise ValueError(
                "inference_grid_blocks_per_side must be > 0 or None, "
                f"got {value}."
            )
        self.inference_grid_blocks_per_side = value

    def _grid_balanced_mask(
        self,
        score: torch.Tensor,
        patch_count: int,
        minimum: int,
    ) -> torch.BoolTensor:
        patch_side = math.isqrt(patch_count)
        if patch_side * patch_side != patch_count:
            raise ValueError(
                "Grid-balanced selection requires a square patch layout, "
                f"got patch_count={patch_count}."
            )
        blocks_per_side = self.inference_grid_blocks_per_side
        if blocks_per_side is None or patch_side % blocks_per_side != 0:
            raise ValueError(
                f"Patch grid side {patch_side} must be divisible by "
                f"inference_grid_blocks_per_side={blocks_per_side}."
            )

        block_side = patch_side // blocks_per_side
        block_capacity = block_side * block_side
        block_count = blocks_per_side * blocks_per_side
        base_quota, remainder = divmod(minimum, block_count)
        if base_quota + (1 if remainder else 0) > block_capacity:
            raise ValueError(
                f"Cannot distribute minimum={minimum} across {block_count} blocks "
                f"with capacity={block_capacity} each."
            )

        importance = score.amax(dim=1).reshape(score.shape[0], patch_side, patch_side)
        keep_mask = torch.zeros(
            score.shape[0], patch_count, dtype=torch.bool, device=score.device
        )
        block_index = 0
        for block_row in range(blocks_per_side):
            for block_column in range(blocks_per_side):
                quota = base_quota + (1 if block_index < remainder else 0)
                block_index += 1
                if quota == 0:
                    continue
                row_start = block_row * block_side
                column_start = block_column * block_side
                block = importance[
                    :,
                    row_start : row_start + block_side,
                    column_start : column_start + block_side,
                ].reshape(score.shape[0], block_capacity)
                local_indices = block.topk(k=quota, dim=-1).indices
                local_rows = torch.div(local_indices, block_side, rounding_mode="floor")
                local_columns = local_indices.remainder(block_side)
                global_indices = (
                    (local_rows + row_start) * patch_side
                    + local_columns
                    + column_start
                )
                keep_mask.scatter_(dim=1, index=global_indices, value=True)
        return keep_mask

    def _merge_dropped_patches(
        self,
        patch_tokens: torch.Tensor,
        keep_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        merged_tokens = patch_tokens.clone()
        patch_count = patch_tokens.shape[1]
        patch_side = math.isqrt(patch_count)
        blocks_per_side = self.inference_grid_blocks_per_side

        if blocks_per_side is None:
            candidate_groups = [torch.arange(patch_count, device=patch_tokens.device)]
        else:
            if patch_side * patch_side != patch_count or patch_side % blocks_per_side != 0:
                raise ValueError(
                    "Local dropped-patch merging requires a square patch grid divisible by "
                    f"inference_grid_blocks_per_side={blocks_per_side}, got patch_count={patch_count}."
                )
            block_side = patch_side // blocks_per_side
            candidate_groups = []
            for block_row in range(blocks_per_side):
                for block_column in range(blocks_per_side):
                    rows = torch.arange(
                        block_row * block_side,
                        (block_row + 1) * block_side,
                        device=patch_tokens.device,
                    )
                    columns = torch.arange(
                        block_column * block_side,
                        (block_column + 1) * block_side,
                        device=patch_tokens.device,
                    )
                    candidate_groups.append(
                        (rows[:, None] * patch_side + columns[None, :]).reshape(-1)
                    )

        for batch_index in range(patch_tokens.shape[0]):
            for candidates in candidate_groups:
                local_keep = keep_mask[batch_index, candidates]
                kept_indices = candidates[local_keep]
                dropped_indices = candidates[~local_keep]
                if dropped_indices.numel() == 0:
                    continue
                if kept_indices.numel() == 0:
                    raise ValueError("Cannot merge dropped patches into a region with no retained patch.")

                kept = patch_tokens[batch_index, kept_indices]
                dropped = patch_tokens[batch_index, dropped_indices]
                normalized_kept = self.rms_norm(kept)
                normalized_dropped = self.rms_norm(dropped)
                similarity = normalized_dropped @ normalized_kept.transpose(0, 1)
                assignments = similarity.argmax(dim=-1)

                accumulated = torch.zeros_like(kept)
                accumulated.index_add_(0, assignments, dropped)
                counts = torch.ones(
                    kept.shape[0], dtype=kept.dtype, device=kept.device
                )
                counts.index_add_(
                    0,
                    assignments,
                    torch.ones_like(assignments, dtype=kept.dtype),
                )
                merged_tokens[batch_index, kept_indices] = (
                    kept + accumulated
                ) / counts.unsqueeze(-1)
        return merged_tokens

    def rms_norm(self, hidden_states: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return hidden_states.to(input_dtype)

    def get_score(
        self,
        patch_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        context_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if context_tokens.shape[1] == 0:
            raise ValueError("Token pruning requires at least one non-patch context token.")

        patch_tokens = self.rms_norm(patch_tokens)
        context_tokens = self.rms_norm(context_tokens)

        score_attention_mask = None
        if context_attention_mask is not None:
            valid_context = context_attention_mask.to(dtype=torch.bool)
            if not torch.all(valid_context.any(dim=-1)):
                raise ValueError("Each sample requires at least one unmasked context token for pruning.")
            score_attention_mask = valid_context.unsqueeze(1).expand(-1, patch_tokens.shape[1], -1)

        queries = F.scaled_dot_product_attention(
            patch_tokens,
            context_tokens,
            context_tokens,
            attn_mask=score_attention_mask,
        )
        queries = self.rms_norm(queries)
        return queries @ patch_tokens.transpose(-2, -1) * self.scale_factor

    def score_to_mask(self, score: torch.Tensor, patch_count: int) -> torch.BoolTensor:
        batch_size = score.shape[0]
        keep_mask = torch.zeros(batch_size, patch_count, dtype=torch.bool, device=score.device)
        keep_indices = score.argmax(dim=-1)
        batch_indices = torch.arange(batch_size, device=score.device).unsqueeze(1).expand_as(keep_indices)
        keep_mask[batch_indices, keep_indices] = True
        minimum = self.inference_min_keep_patches
        if minimum is not None:
            if minimum > patch_count:
                raise ValueError(
                    f"inference_min_keep_patches={minimum} exceeds patch_count={patch_count}."
                )
            if self.inference_grid_blocks_per_side is not None:
                keep_mask = self._grid_balanced_mask(score, patch_count, minimum)
            else:
                importance = score.amax(dim=1)
                floor_indices = importance.topk(k=minimum, dim=-1).indices
                keep_mask.scatter_(dim=1, index=floor_indices, value=True)
        return keep_mask

    def score_to_selection_weights(self, score: torch.Tensor, patch_count: int) -> tuple[torch.LongTensor, torch.Tensor]:
        if self.noise_scale is not None:
            score = score + torch.rand_like(score) * self.noise_scale

        hard_weights = F.one_hot(score.argmax(dim=-1), num_classes=patch_count).to(dtype=score.dtype)
        soft_weights = torch.softmax(score, dim=-1)
        selection_weights = hard_weights + soft_weights - soft_weights.detach()
        return selection_weights.argmax(dim=-1), selection_weights

    def _pack_inference_patches(
        self,
        patch_tokens: torch.Tensor,
        patch_position_ids: torch.LongTensor,
        patch_attention_mask: Optional[torch.Tensor],
        keep_mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.LongTensor, Optional[torch.Tensor], torch.LongTensor, torch.LongTensor]:
        batch_size, _, hidden_size = patch_tokens.shape
        keep_counts = keep_mask.sum(dim=1)
        max_keep = int(keep_counts.max().item())

        packed_tokens = patch_tokens.new_zeros((batch_size, max_keep, hidden_size))
        packed_position_ids = patch_position_ids.new_zeros((batch_size, max_keep))
        packed_indices = patch_position_ids.new_full((batch_size, max_keep), fill_value=-1)

        packed_attention_mask = None
        if patch_attention_mask is not None:
            packed_attention_mask = patch_attention_mask.new_zeros((batch_size, max_keep))

        for batch_idx in range(batch_size):
            keep_indices = torch.nonzero(keep_mask[batch_idx], as_tuple=False).squeeze(-1)
            keep_count = int(keep_indices.numel())

            packed_tokens[batch_idx, :keep_count] = patch_tokens[batch_idx, keep_indices]
            packed_position_ids[batch_idx, :keep_count] = patch_position_ids[batch_idx, keep_indices]
            packed_indices[batch_idx, :keep_count] = keep_indices

            if packed_attention_mask is not None:
                packed_attention_mask[batch_idx, :keep_count] = patch_attention_mask[batch_idx, keep_indices]

        return packed_tokens, packed_position_ids, packed_attention_mask, packed_indices, keep_counts

    def _build_synthetic_attention_masks(
        self,
        tokens: torch.Tensor,
        prefix_length: int,
        packed_patch_length: int,
        suffix_length: int,
        keep_counts: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = tokens.shape[0]
        device = tokens.device

        prefix_attention_mask = torch.ones((batch_size, prefix_length), dtype=torch.long, device=device)
        packed_attention_mask = torch.zeros((batch_size, packed_patch_length), dtype=torch.long, device=device)
        suffix_attention_mask = torch.ones((batch_size, suffix_length), dtype=torch.long, device=device)

        for batch_idx in range(batch_size):
            keep_count = int(keep_counts[batch_idx].item())
            packed_attention_mask[batch_idx, :keep_count] = 1

        return prefix_attention_mask, packed_attention_mask, suffix_attention_mask

    def _select_context_tokens(
        self,
        segments,
        layout: SequencePruningLayout,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if layout.score_context == "suffix":
            context_tokens = segments.suffix_tokens
            context_attention_mask = segments.suffix_attention_mask
        elif layout.score_context == "non_patch":
            context_tokens = torch.cat((segments.prefix_tokens, segments.suffix_tokens), dim=1)
            context_attention_mask = None
            if segments.prefix_attention_mask is not None:
                context_attention_mask = torch.cat(
                    (segments.prefix_attention_mask, segments.suffix_attention_mask),
                    dim=1,
                )
        else:
            raise ValueError(f"Unsupported score_context {layout.score_context!r}.")

        if context_tokens.shape[1] == 0:
            raise ValueError(
                f"Token pruning requires at least one context token for score_context={layout.score_context!r}."
            )

        return context_tokens, context_attention_mask

    def _mask_invalid_patch_scores(
        self,
        scores: torch.Tensor,
        patch_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if patch_attention_mask is None:
            return scores

        valid_patches = patch_attention_mask.to(dtype=torch.bool)
        if not torch.all(valid_patches.any(dim=-1)):
            raise ValueError("Each sample requires at least one unmasked patch token for pruning.")
        return scores.masked_fill(~valid_patches.unsqueeze(1), torch.finfo(scores.dtype).min)

    def forward(
        self,
        tokens: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        layout: SequencePruningLayout,
    ) -> PruningResult:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens to have shape [B, S, D], got {tuple(tokens.shape)}.")
        if tokens.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected token hidden size {self.hidden_size}, got {tokens.shape[-1]}."
            )

        if position_ids is None:
            position_ids = build_position_ids(tokens, attention_mask)

        if position_ids.shape != tokens.shape[:2]:
            raise ValueError(
                f"Position ids shape must match [B, S]. Got tokens={tuple(tokens.shape[:2])}, "
                f"position_ids={tuple(position_ids.shape)}."
            )

        segments = split_sequence(tokens, position_ids, attention_mask, layout)
        context_tokens, context_attention_mask = self._select_context_tokens(segments, layout)
        scores = self.get_score(segments.patch_tokens, context_tokens, context_attention_mask)
        scores = self._mask_invalid_patch_scores(scores, segments.patch_attention_mask)

        minimum = self.inference_min_keep_patches
        if not self.training and minimum is not None and minimum > layout.patch_count:
            raise ValueError(
                f"inference_min_keep_patches={minimum} exceeds patch_count={layout.patch_count}."
            )
        if not self.training and minimum is not None and segments.patch_attention_mask is not None:
            valid_counts = segments.patch_attention_mask.to(dtype=torch.bool).sum(dim=-1)
            if torch.any(valid_counts < minimum):
                raise ValueError(
                    f"inference_min_keep_patches={minimum} exceeds the number of valid patches "
                    f"for at least one sample: valid_counts={valid_counts.detach().cpu().tolist()}."
                )

        if self.training:
            keep_indices, selection_weights = self.score_to_selection_weights(scores, layout.patch_count)
            batch_indices = torch.arange(tokens.shape[0], device=tokens.device).unsqueeze(1).expand_as(keep_indices)

            mixed_patch_tokens = selection_weights @ segments.patch_tokens
            mixed_patch_position_ids = segments.patch_position_ids[batch_indices, keep_indices]

            mixed_patch_attention_mask = None
            if segments.patch_attention_mask is not None:
                mixed_patch_attention_mask = segments.patch_attention_mask[batch_indices, keep_indices]

            new_tokens, new_position_ids, new_attention_mask = merge_sequence(
                segments.prefix_tokens,
                mixed_patch_tokens,
                segments.suffix_tokens,
                segments.prefix_position_ids,
                mixed_patch_position_ids,
                segments.suffix_position_ids,
                segments.prefix_attention_mask,
                mixed_patch_attention_mask,
                segments.suffix_attention_mask,
            )

            return PruningResult(
                tokens=new_tokens,
                position_ids=new_position_ids,
                attention_mask=new_attention_mask,
                scores=scores,
                patch_keep_mask=None,
                patch_keep_indices=keep_indices,
                patch_keep_counts=None,
                selection_weights=selection_weights,
            )

        keep_mask = self.score_to_mask(scores, layout.patch_count)
        inference_patch_tokens = segments.patch_tokens
        if self.inference_merge_dropped_patches:
            inference_patch_tokens = self._merge_dropped_patches(
                segments.patch_tokens,
                keep_mask,
            )
        packed_tokens, packed_position_ids, packed_attention_mask, packed_indices, keep_counts = (
            self._pack_inference_patches(
                inference_patch_tokens,
                segments.patch_position_ids,
                segments.patch_attention_mask,
                keep_mask,
            )
        )

        prefix_attention_mask = segments.prefix_attention_mask
        suffix_attention_mask = segments.suffix_attention_mask
        if packed_attention_mask is None and int(keep_counts.min().item()) != int(keep_counts.max().item()):
            prefix_attention_mask, packed_attention_mask, suffix_attention_mask = self._build_synthetic_attention_masks(
                tokens=tokens,
                prefix_length=segments.prefix_tokens.shape[1],
                packed_patch_length=packed_tokens.shape[1],
                suffix_length=segments.suffix_tokens.shape[1],
                keep_counts=keep_counts,
            )

        new_tokens, new_position_ids, new_attention_mask = merge_sequence(
            segments.prefix_tokens,
            packed_tokens,
            segments.suffix_tokens,
            segments.prefix_position_ids,
            packed_position_ids,
            segments.suffix_position_ids,
            prefix_attention_mask,
            packed_attention_mask,
            suffix_attention_mask,
        )

        return PruningResult(
            tokens=new_tokens,
            position_ids=new_position_ids,
            attention_mask=new_attention_mask,
            scores=scores,
            patch_keep_mask=keep_mask,
            patch_keep_indices=packed_indices,
            patch_keep_counts=keep_counts,
            selection_weights=None,
        )
