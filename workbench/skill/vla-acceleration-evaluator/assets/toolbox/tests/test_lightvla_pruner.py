from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lightvla_pruner import (
    DifferentiableTokenPruner,
    PrefixLlamaPruningAdapter,
    RuntimePruner,
    SequencePruningLayout,
    TokenSpan,
    configure_existing_pruner,
)
from lightvla_pruner.adapters import StaticPrefixLayoutResolver


def reference_rms_norm(hidden_states: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return hidden_states.to(input_dtype)


def reference_get_score(patches: torch.Tensor, task: torch.Tensor, hidden_size: int) -> torch.Tensor:
    patches = reference_rms_norm(patches)
    task = reference_rms_norm(task)
    queries = F.scaled_dot_product_attention(patches, task, task)
    queries = reference_rms_norm(queries)
    return queries @ patches.transpose(-2, -1) * (1.0 / hidden_size**0.5)


def reference_token_pruner_forward(
    tokens: torch.Tensor,
    position_ids: torch.LongTensor,
    attention_mask: torch.Tensor | None,
    num_patches: int,
    hidden_size: int,
    training: bool,
) -> tuple[torch.Tensor, torch.LongTensor, torch.Tensor | None, torch.Tensor]:
    batch_size, seq_len, hidden_dim = tokens.shape
    cls_token, patches, task = torch.split(tokens, [1, num_patches, seq_len - num_patches - 1], dim=1)
    cls_token_id, patches_id, task_id = torch.split(position_ids, [1, num_patches, seq_len - num_patches - 1], dim=1)
    if attention_mask is not None:
        cls_token_mask, patches_mask, task_mask = torch.split(
            attention_mask, [1, num_patches, seq_len - num_patches - 1], dim=1
        )

    score = reference_get_score(patches, task, hidden_size)

    if not training:
        mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=score.device)
        indices = score.argmax(dim=-1)
        batch_indices = torch.arange(batch_size, device=score.device).unsqueeze(1).expand_as(indices)
        mask[batch_indices, indices] = True

        patches = patches[mask].view(batch_size, -1, hidden_dim)
        tokens = torch.cat([cls_token, patches, task], dim=1)

        patches_id = patches_id[mask].view(batch_size, -1)
        position_ids = torch.cat([cls_token_id, patches_id, task_id], dim=1)

        if attention_mask is not None:
            patches_mask = patches_mask[mask].view(batch_size, -1)
            attention_mask = torch.cat([cls_token_mask, patches_mask, task_mask], dim=1)
    else:
        hard_score = F.one_hot(score.argmax(dim=-1), num_classes=num_patches).to(dtype=score.dtype)
        soft_score = torch.softmax(score, dim=-1)
        straight_through = hard_score + soft_score - soft_score.detach()
        indices = straight_through.argmax(dim=-1)
        patches = straight_through @ patches
        batch_indices = torch.arange(batch_size, device=patches.device).unsqueeze(1).expand_as(indices)

        tokens = torch.cat([cls_token, patches, task], dim=1)
        patches_id = patches_id[batch_indices, indices].view(batch_size, -1)
        position_ids = torch.cat([cls_token_id, patches_id, task_id], dim=1)

        if attention_mask is not None:
            patches_mask = patches_mask[batch_indices, indices].view(batch_size, -1)
            attention_mask = torch.cat([cls_token_mask, patches_mask, task_mask], dim=1)

    return tokens, position_ids, attention_mask, score


class DummyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs


class DummyCache:
    def __init__(self, sequence_length: int) -> None:
        self.sequence_length = sequence_length

    def get_seq_length(self) -> int:
        return self.sequence_length


class DecoderWithPruner(DummyDecoder):
    def __init__(self) -> None:
        super().__init__()
        self.pruner = nn.Identity()


class ExistingThreeArgumentPruner(nn.Module):
    def __init__(self, num_patches: int) -> None:
        super().__init__()
        self.num_patches = num_patches

    def forward(self, tokens, position_ids, attention_mask):
        return tokens[:, :-1], position_ids[:, :-1], attention_mask[:, :-1]


class ExistingPrunerDecoder(nn.Module):
    def __init__(self, hidden_size: int, num_patches: int) -> None:
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.pruner = ExistingThreeArgumentPruner(num_patches=num_patches)


class ExistingPrunerModel(nn.Module):
    def __init__(self, hidden_size: int, num_patches: int) -> None:
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.model = ExistingPrunerDecoder(hidden_size, num_patches)


class LightVlaPrunerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hidden_size = 8
        self.patch_count = 4
        self.layout = SequencePruningLayout(
            patch_span=TokenSpan(start=1, end=1 + self.patch_count),
            score_context="suffix",
        )
        self.position_ids = torch.arange(8).unsqueeze(0)
        self.attention_mask = torch.ones(1, 8, dtype=torch.long)
        torch.manual_seed(0)
        self.tokens = torch.randn(1, 8, self.hidden_size)

    def test_eval_matches_lightvla_suffix_only_reference(self) -> None:
        pruner = DifferentiableTokenPruner(hidden_size=self.hidden_size)
        pruner.eval()

        with torch.no_grad():
            result = pruner(self.tokens, self.position_ids, self.attention_mask, self.layout)
            ref_tokens, ref_position_ids, ref_attention_mask, ref_scores = reference_token_pruner_forward(
                self.tokens,
                self.position_ids,
                self.attention_mask,
                self.patch_count,
                self.hidden_size,
                training=False,
            )

        self.assertTrue(torch.allclose(result.tokens, ref_tokens))
        self.assertTrue(torch.equal(result.position_ids, ref_position_ids))
        self.assertTrue(torch.equal(result.attention_mask, ref_attention_mask))
        self.assertTrue(torch.allclose(result.scores, ref_scores))

    def test_train_matches_lightvla_suffix_only_reference(self) -> None:
        pruner = DifferentiableTokenPruner(hidden_size=self.hidden_size)
        pruner.train()

        result = pruner(self.tokens, self.position_ids, self.attention_mask, self.layout)
        ref_tokens, ref_position_ids, ref_attention_mask, ref_scores = reference_token_pruner_forward(
            self.tokens,
            self.position_ids,
            self.attention_mask,
            self.patch_count,
            self.hidden_size,
            training=True,
        )

        self.assertTrue(torch.allclose(result.tokens, ref_tokens))
        self.assertTrue(torch.equal(result.position_ids, ref_position_ids))
        self.assertTrue(torch.equal(result.attention_mask, ref_attention_mask))
        self.assertTrue(torch.allclose(result.scores, ref_scores))

    def test_non_patch_context_is_explicit_opt_in(self) -> None:
        suffix_layout = self.layout
        non_patch_layout = SequencePruningLayout(
            patch_span=TokenSpan(start=1, end=1 + self.patch_count),
            score_context="non_patch",
        )
        pruner = DifferentiableTokenPruner(hidden_size=self.hidden_size)
        pruner.eval()

        with torch.no_grad():
            suffix_scores = pruner(self.tokens, self.position_ids, self.attention_mask, suffix_layout).scores
            non_patch_scores = pruner(self.tokens, self.position_ids, self.attention_mask, non_patch_layout).scores

        self.assertFalse(torch.allclose(suffix_scores, non_patch_scores))

    def test_masked_context_tokens_do_not_affect_scores(self) -> None:
        attention_mask = self.attention_mask.clone()
        attention_mask[:, -1] = 0
        changed_tokens = self.tokens.clone()
        changed_tokens[:, -1] = torch.randn_like(changed_tokens[:, -1]) * 1000
        pruner = DifferentiableTokenPruner(hidden_size=self.hidden_size).eval()

        with torch.no_grad():
            original_scores = pruner(self.tokens, self.position_ids, attention_mask, self.layout).scores
            changed_scores = pruner(changed_tokens, self.position_ids, attention_mask, self.layout).scores

        self.assertTrue(torch.allclose(original_scores, changed_scores))

    def test_masked_patches_are_not_selected(self) -> None:
        attention_mask = self.attention_mask.clone()
        masked_patch_index = 2
        attention_mask[:, self.layout.patch_span.start + masked_patch_index] = 0
        pruner = DifferentiableTokenPruner(hidden_size=self.hidden_size).eval()

        with torch.no_grad():
            result = pruner(self.tokens, self.position_ids, attention_mask, self.layout)

        self.assertFalse(bool(result.patch_keep_mask[0, masked_patch_index]))

    def test_inference_minimum_keep_patch_floor_is_enforced(self) -> None:
        pruner = DifferentiableTokenPruner(
            hidden_size=self.hidden_size,
            inference_min_keep_patches=self.patch_count,
        ).eval()

        with torch.no_grad():
            result = pruner(self.tokens, self.position_ids, self.attention_mask, self.layout)

        self.assertEqual(result.patch_keep_counts.tolist(), [self.patch_count])

    def test_inference_minimum_keep_patch_floor_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "inference_min_keep_patches"):
            DifferentiableTokenPruner(
                hidden_size=self.hidden_size,
                inference_min_keep_patches=0,
            )

        pruner = DifferentiableTokenPruner(
            hidden_size=self.hidden_size,
            inference_min_keep_patches=self.patch_count + 1,
        ).eval()
        with self.assertRaisesRegex(ValueError, "exceeds patch_count"):
            pruner(self.tokens, self.position_ids, self.attention_mask, self.layout)

    def test_grid_balanced_selection_enforces_equal_local_coverage(self) -> None:
        pruner = DifferentiableTokenPruner(
            hidden_size=self.hidden_size,
            inference_min_keep_patches=8,
            inference_grid_blocks_per_side=2,
        ).eval()
        score = torch.randn(1, 3, 16)

        keep_mask = pruner.score_to_mask(score, patch_count=16).reshape(1, 4, 4)

        self.assertEqual(int(keep_mask.sum().item()), 8)
        for row_start in (0, 2):
            for column_start in (0, 2):
                block_count = keep_mask[
                    :, row_start : row_start + 2, column_start : column_start + 2
                ].sum()
                self.assertEqual(int(block_count.item()), 2)

    def test_dropped_patch_merging_changes_only_retained_tokens(self) -> None:
        pruner = DifferentiableTokenPruner(
            hidden_size=2,
            inference_merge_dropped_patches=True,
        ).eval()
        patches = torch.tensor([[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]])
        keep_mask = torch.tensor([[True, False, True, False]])

        merged = pruner._merge_dropped_patches(patches, keep_mask)

        self.assertTrue(torch.allclose(merged[0, 0], torch.tensor([0.95, 0.05])))
        self.assertTrue(torch.allclose(merged[0, 2], torch.tensor([0.05, 0.95])))
        self.assertTrue(torch.equal(merged[0, 1], patches[0, 1]))
        self.assertTrue(torch.equal(merged[0, 3], patches[0, 3]))

    def test_adapter_bypasses_pruning_with_cache(self) -> None:
        decoder = DummyDecoder()
        pruner = DifferentiableTokenPruner(hidden_size=self.hidden_size)
        adapter = PrefixLlamaPruningAdapter(
            decoder_model=decoder,
            pruner=pruner,
            layout_resolver=StaticPrefixLayoutResolver(
                patch_count=self.patch_count,
                patch_start=1,
                score_context="suffix",
            ),
        )

        adapter.eval()
        with torch.no_grad():
            prefill = adapter(
                inputs_embeds=self.tokens,
                attention_mask=self.attention_mask,
                position_ids=self.position_ids,
            )
            decode = adapter(
                inputs_embeds=torch.randn(1, 1, self.hidden_size),
                attention_mask=torch.ones(1, 1, dtype=torch.long),
                position_ids=torch.tensor([[8]]),
                past_key_values="cache",
            )

        self.assertLess(prefill["inputs_embeds"].shape[1], self.tokens.shape[1])
        self.assertEqual(decode["inputs_embeds"].shape[1], 1)
        self.assertEqual(decode["past_key_values"], "cache")

    def test_adapter_prunes_with_empty_modern_cache_and_rebuilds_cache_position(self) -> None:
        decoder = DummyDecoder()
        adapter = PrefixLlamaPruningAdapter(
            decoder_model=decoder,
            pruner=DifferentiableTokenPruner(hidden_size=self.hidden_size),
            layout_resolver=StaticPrefixLayoutResolver(patch_count=self.patch_count),
        )

        adapter.eval()
        with torch.no_grad():
            result = adapter(
                inputs_embeds=self.tokens,
                attention_mask=self.attention_mask,
                position_ids=self.position_ids,
                past_key_values=DummyCache(sequence_length=0),
                cache_position=torch.arange(self.tokens.shape[1]),
            )

        output_length = result["inputs_embeds"].shape[1]
        self.assertLess(output_length, self.tokens.shape[1])
        self.assertTrue(torch.equal(result["cache_position"], torch.arange(output_length)))
        self.assertEqual(adapter.pruned_forward_count, 1)
        self.assertEqual(adapter.last_input_sequence_length, self.tokens.shape[1])
        self.assertEqual(adapter.last_output_sequence_length, output_length)
        self.assertIsNotNone(adapter.last_patch_keep_counts)
        self.assertIsNotNone(adapter.last_patch_keep_indices)
        self.assertEqual(
            len(adapter.last_patch_keep_indices[0]),
            adapter.last_patch_keep_counts[0],
        )

    def test_adapter_bypasses_populated_modern_cache(self) -> None:
        decoder = DummyDecoder()
        adapter = PrefixLlamaPruningAdapter(
            decoder_model=decoder,
            pruner=DifferentiableTokenPruner(hidden_size=self.hidden_size),
            layout_resolver=StaticPrefixLayoutResolver(patch_count=self.patch_count),
        )

        decode_token = torch.randn(1, 1, self.hidden_size)
        with torch.no_grad():
            result = adapter(
                inputs_embeds=decode_token,
                past_key_values=DummyCache(sequence_length=8),
                cache_position=torch.tensor([8]),
            )

        self.assertIs(result["inputs_embeds"], decode_token)
        self.assertTrue(torch.equal(result["cache_position"], torch.tensor([8])))
        self.assertEqual(adapter.pruned_forward_count, 0)

    def test_adapter_rejects_existing_pruner_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "second pruner"):
            PrefixLlamaPruningAdapter(
                decoder_model=DecoderWithPruner(),
                pruner=DifferentiableTokenPruner(hidden_size=self.hidden_size),
                layout_resolver=StaticPrefixLayoutResolver(patch_count=self.patch_count),
            )

    def test_adapter_inherits_decoder_eval_mode(self) -> None:
        decoder = DummyDecoder().eval()
        adapter = PrefixLlamaPruningAdapter(
            decoder_model=decoder,
            pruner=DifferentiableTokenPruner(hidden_size=self.hidden_size),
            layout_resolver=StaticPrefixLayoutResolver(patch_count=self.patch_count),
        )

        self.assertFalse(adapter.training)
        self.assertFalse(adapter.pruner.training)
        self.assertFalse(adapter.decoder_model.training)

    def test_adapter_supports_configured_trailing_unmasked_token(self) -> None:
        decoder = DummyDecoder()
        adapter = PrefixLlamaPruningAdapter(
            decoder_model=decoder,
            pruner=DifferentiableTokenPruner(hidden_size=self.hidden_size),
            layout_resolver=StaticPrefixLayoutResolver(patch_count=self.patch_count),
            trailing_unmasked_tokens=1,
        ).eval()

        short_attention_mask = self.attention_mask[:, :-1]
        with torch.no_grad():
            result = adapter(
                inputs_embeds=self.tokens,
                attention_mask=short_attention_mask,
            )

        self.assertLess(result["inputs_embeds"].shape[1], self.tokens.shape[1])
        self.assertEqual(
            result["attention_mask"].shape[1],
            result["inputs_embeds"].shape[1] - 1,
        )

    def test_adapter_rejects_unconfigured_attention_mask_shortfall(self) -> None:
        adapter = PrefixLlamaPruningAdapter(
            decoder_model=DummyDecoder(),
            pruner=DifferentiableTokenPruner(hidden_size=self.hidden_size),
            layout_resolver=StaticPrefixLayoutResolver(patch_count=self.patch_count),
        ).eval()

        with self.assertRaisesRegex(ValueError, "trailing_unmasked_tokens"):
            adapter(
                inputs_embeds=self.tokens,
                attention_mask=self.attention_mask[:, :-1],
            )

    def test_input_shapes_and_noise_scale_are_validated(self) -> None:
        pruner = DifferentiableTokenPruner(hidden_size=self.hidden_size)
        with self.assertRaisesRegex(ValueError, "hidden size"):
            pruner(
                torch.randn(1, 8, self.hidden_size + 1),
                self.position_ids,
                self.attention_mask,
                self.layout,
            )
        with self.assertRaisesRegex(ValueError, "Attention mask shape"):
            pruner(self.tokens, self.position_ids, torch.ones(1, 7), self.layout)
        with self.assertRaisesRegex(ValueError, "noise_scale"):
            pruner.set_noise_scale(-0.1)

    def test_eval_builds_attention_mask_when_batch_padding_is_needed(self) -> None:
        pruner = DifferentiableTokenPruner(hidden_size=self.hidden_size)
        pruner.eval()

        tokens = None
        result = None
        for seed in range(256):
            torch.manual_seed(seed)
            candidate_tokens = torch.randn(2, 8, self.hidden_size)
            with torch.no_grad():
                candidate_result = pruner(candidate_tokens, self.position_ids.expand(2, -1), None, self.layout)
            if int(candidate_result.patch_keep_counts.min().item()) != int(candidate_result.patch_keep_counts.max().item()):
                tokens = candidate_tokens
                result = candidate_result
                break

        self.assertIsNotNone(tokens)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.attention_mask)

        max_keep = int(result.patch_keep_counts.max().item())
        min_keep = int(result.patch_keep_counts.min().item())
        self.assertGreater(max_keep, min_keep)

        shorter_batch = int(torch.argmin(result.patch_keep_counts).item())
        prefix_length = 1
        patch_mask = result.attention_mask[shorter_batch, prefix_length : prefix_length + max_keep]
        self.assertEqual(int(patch_mask[min_keep].item()), 0)

    def test_runtime_baseline_is_noop_and_records_calls(self) -> None:
        runtime = RuntimePruner("baseline", self.patch_count, self.hidden_size).eval()
        output = runtime(self.tokens, self.position_ids, self.attention_mask)

        self.assertIs(output[0], self.tokens)
        self.assertIs(output[1], self.position_ids)
        self.assertIs(output[2], self.attention_mask)
        self.assertEqual(runtime.calls, 1)
        self.assertEqual(runtime.summary()["records"][0]["keep_counts"], (self.patch_count,))

    def test_runtime_packaged_prunes_and_records_counts(self) -> None:
        runtime = RuntimePruner("packaged", self.patch_count, self.hidden_size).eval()
        with torch.no_grad():
            output_tokens, _, _ = runtime(self.tokens, self.position_ids, self.attention_mask)

        self.assertLess(output_tokens.shape[1], self.tokens.shape[1])
        record = runtime.call_records[0]
        self.assertEqual(record.output_length, output_tokens.shape[1])
        self.assertIsNotNone(record.keep_counts)

    def test_configure_existing_pruner_infers_shape_and_restores(self) -> None:
        model = ExistingPrunerModel(self.hidden_size, self.patch_count).eval()
        decoder = model.language_model.model
        original_pruner = decoder.pruner

        runtime = configure_existing_pruner(model, mode="baseline")
        self.assertIs(decoder.pruner, runtime.runtime_pruner)
        self.assertEqual(runtime.runtime_pruner.hidden_size, self.hidden_size)
        self.assertEqual(runtime.runtime_pruner.num_patches, self.patch_count)

        runtime.restore()
        self.assertIs(decoder.pruner, original_pruner)

    def test_configure_builtin_preserves_original_behavior(self) -> None:
        model = ExistingPrunerModel(self.hidden_size, self.patch_count).eval()
        runtime = configure_existing_pruner(model, mode="builtin")

        output_tokens, _, _ = runtime.runtime_pruner(self.tokens, self.position_ids, self.attention_mask)
        self.assertEqual(output_tokens.shape[1], self.tokens.shape[1] - 1)
        self.assertEqual(runtime.runtime_pruner.call_records[0].output_length, self.tokens.shape[1] - 1)

    def test_runtime_restore_refuses_to_overwrite_external_change(self) -> None:
        model = ExistingPrunerModel(self.hidden_size, self.patch_count).eval()
        runtime = configure_existing_pruner(model, mode="baseline")
        model.language_model.model.pruner = nn.Identity()

        with self.assertRaisesRegex(RuntimeError, "changed after runtime configuration"):
            runtime.restore()


if __name__ == "__main__":
    unittest.main()
