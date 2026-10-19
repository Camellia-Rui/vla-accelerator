"""Lightweight configuration checks and real CPU tensor/LLaMA contract checks."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import subprocess
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

from methods.lightvla import (
    LightVLAAccelerator, LightVLAConfig, configure_lightvla_pruning,
    get_pruning_runtime_summary, restore_lightvla_pruning,
)
from pruners import available, build

HAS_MODEL_DEPS = bool(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"))
if HAS_MODEL_DEPS:
    import torch
    from transformers import LlamaModel
    from transformers.cache_utils import DynamicCache
    from methods.lightvla.cli import _smoke_bundle, main
    from methods.lightvla.pruner import DifferentiableTokenPruner, SequencePruningLayout, TokenSpan


class ConfigTests(unittest.TestCase):
    def test_registered_without_loading_model_dependencies(self):
        self.assertIn("lightvla", available())
        self.assertFalse(build("lightvla").get_stats()["attached"])
        code = "from pruners import build; import sys; build('lightvla'); assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_config_accepts_metadata_and_rejects_invalid_values(self):
        self.assertEqual(LightVLAConfig.from_mapping({"method": "lightvla", "name": "lightvla"}).mode, "packaged")
        for config in ({"mode": "unknown"}, {"patch_start": -1}, {"patch_start": True},
                       {"patch_count": 0}, {"patch_count": 4.5}, {"score_context": "wrong"},
                       {"noise_scale": float("nan")}, {"noise_scale": -1},
                       {"allow_existing_pruner": "true"}, {"state_type": "stateful"},
                       {"name": "different"}, {"misspelled": 1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                LightVLAConfig.from_mapping(config)

    def test_operations_require_attach(self):
        pruner = build("lightvla")
        for method in (pruner.enable, pruner.disable, pruner.reset):
            with self.assertRaisesRegex(RuntimeError, "Attach"):
                method()
        with self.assertRaisesRegex(ValueError, "existing model"):
            pruner.attach(None)
        pruner.detach()


@unittest.skipUnless(HAS_MODEL_DEPS, "Install LightVLA model dependencies for real CPU tensor tests")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        bundle = _smoke_bundle()
        self.model, self.inputs = bundle["model"], bundle["inputs"]
        self.decoder = self.model.language_model.model
        self.pruner = build("lightvla")
        self.addCleanup(self.pruner.detach)

    def builtin_model(self):
        class BuiltinDecoder(LlamaModel):
            def __init__(self, config):
                super().__init__(config)
                self.pruner = torch.nn.Identity()
                self.native_calls = 0

            def forward(self, *args, **kwargs):
                self.native_calls += 1
                return super().forward(*args, **kwargs)
        self.model.language_model.model = BuiltinDecoder(self.decoder.config).eval()
        self.decoder = self.model.language_model.model

    def test_real_prefill_prunes_and_preserves_native_api(self):
        native_predict = self.model.predict_action.__func__
        self.pruner.attach(self.model)
        output = self.pruner.predict_action(**self.inputs)
        self.assertEqual(tuple(output.shape), (1, 5, 16))
        self.assertFalse(output.requires_grad)
        self.assertIs(self.model.predict_action.__func__, native_predict)
        stats = self.pruner.get_stats()
        self.assertEqual(stats["last_patch_keep_counts"], (1,))
        self.assertEqual(stats["last_input_sequence_length"], 8)
        self.assertEqual(stats["last_output_sequence_length"], 5)

    def test_enable_disable_and_detach_restore_numerical_baseline_and_weights(self):
        with torch.inference_mode():
            baseline = self.model.predict_action(**self.inputs)
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        original_forward = self.decoder.forward.__func__
        self.pruner.attach(self.model)
        self.pruner.disable()
        self.assertTrue(torch.equal(self.pruner.predict_action(**self.inputs), baseline))
        self.pruner.enable()
        self.assertEqual(self.pruner.predict_action(**self.inputs).shape[1], 5)
        self.pruner.detach()
        self.assertIs(self.model.language_model.model, self.decoder)
        self.assertNotIn("forward", vars(self.decoder))
        self.assertIs(self.decoder.forward.__func__, original_forward)
        self.assertEqual(set(before), set(self.model.state_dict()))
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        self.pruner.detach()

    def test_all_three_modes_restore_original_builtin_forward(self):
        self.builtin_model()
        pruner = build("lightvla", {"allow_existing_pruner": True})
        self.addCleanup(pruner.detach)
        pruner.attach(self.model)
        pruner.predict_action(**self.inputs)
        self.assertEqual(self.decoder.native_calls, 0)
        for mode in ("baseline", "builtin", "packaged", "builtin"):
            pruner.set_mode(mode)
            output = pruner.predict_action(**self.inputs)
            self.assertEqual(output.shape[1], 5 if mode == "packaged" else 8)
        self.assertEqual(self.decoder.native_calls, 2)
        pruner.detach()
        self.assertIs(self.decoder.forward.__func__, type(self.decoder).forward)

    def test_function_api_switches_and_restores(self):
        self.model._pruning_runtime_summary = {"old": True}
        previous = self.model._pruning_runtime_summary
        configure_lightvla_pruning(self.model, "packaged")
        try:
            configure_lightvla_pruning(self.model, "baseline")
            self.assertIs(self.model.language_model.model, self.decoder)
            configure_lightvla_pruning(self.model, "packaged")
            self.assertEqual(get_pruning_runtime_summary(self.model)["mode"], "packaged")
        finally:
            restore_lightvla_pruning(self.model)
        self.assertIs(self.model._pruning_runtime_summary, previous)
        restore_lightvla_pruning(self.model)

    def test_duplicate_owner_wrong_model_and_external_wrapping(self):
        self.pruner.attach(self.model)
        with self.assertRaisesRegex(RuntimeError, "owner"):
            LightVLAAccelerator.install(self.model)
        with self.assertRaisesRegex(ValueError, "different model"):
            self.pruner.detach(object())
        active = self.model.language_model.model
        self.model.language_model.model = torch.nn.Identity()
        with self.assertRaisesRegex(RuntimeError, "outside LightVLA"):
            self.pruner.detach()
        self.model.language_model.model = active

    def test_rejects_other_pruner_owner(self):
        self.model._vla_express_pruner_owner = object()
        with self.assertRaisesRegex(RuntimeError, "Detach VLA-Pruner"):
            self.pruner.attach(self.model)
        self.assertIs(self.model.language_model.model, self.decoder)

    def test_builtin_requires_explicit_permission_and_real_custom_forward(self):
        self.builtin_model()
        with self.assertRaisesRegex(ValueError, "allow_existing_pruner"):
            self.pruner.attach(self.model)
        self.assertNotIn("forward", vars(self.decoder))
        ordinary = _smoke_bundle()["model"]
        ordinary.language_model.model.pruner = torch.nn.Identity()
        with self.assertRaisesRegex(ValueError, "custom pruning forward"):
            LightVLAAccelerator.install(ordinary, mode="builtin", allow_existing_pruner=True)

    def test_failed_mode_switch_keeps_previous_active_adapter(self):
        self.pruner.attach(self.model)
        active = self.model.language_model.model
        with self.assertRaisesRegex(ValueError, "builtin"):
            self.pruner.set_mode("builtin")
        self.assertIs(self.model.language_model.model, active)
        self.assertEqual(self.pruner.get_stats()["mode"], "packaged")
        self.assertEqual(self.pruner.predict_action(**self.inputs).shape[1], 5)

    def test_failed_model_assignment_rolls_back_forward_and_wrapper(self):
        class Holder(torch.nn.Module):
            fail = False

            def __setattr__(self, name, value):
                if name == "model" and self.fail:
                    self.fail = False
                    raise RuntimeError("assignment failed")
                super().__setattr__(name, value)
        holder = Holder()
        holder.model = self.decoder
        self.model.language_model = holder
        self.pruner.attach(self.model)
        active = holder.model
        forward = vars(self.decoder)["forward"]
        holder.fail = True
        with self.assertRaisesRegex(RuntimeError, "assignment failed"):
            self.pruner.set_mode("baseline")
        self.assertIs(holder.model, active)
        self.assertIs(vars(self.decoder)["forward"], forward)
        self.assertEqual(self.pruner.get_stats()["mode"], "packaged")

    def test_per_instance_forward_and_training_flags_restored(self):
        def custom(this, *args, **kwargs):
            return LlamaModel.forward(this, *args, **kwargs)
        custom_forward = MethodType(custom, self.decoder)
        self.decoder.forward = custom_forward
        self.decoder.layers[0].training = True
        before_flags = [module.training for module in self.decoder.modules()]
        self.pruner.attach(self.model)
        self.pruner.detach()
        self.assertIs(self.decoder.forward, custom_forward)
        self.assertEqual([module.training for module in self.decoder.modules()], before_flags)

    def test_cached_decode_bypasses_pruning_after_empty_cache_prefill(self):
        self.pruner.attach(self.model)
        cache = DynamicCache()
        with torch.inference_mode():
            output = self.model.language_model.model(inputs_embeds=self.inputs["inputs_embeds"],
                                                     past_key_values=cache, use_cache=True)
            # 4.40.x may convert the input Cache and return a legacy tuple.
            # Decode must consume the cache actually returned by the backend.
            cache = output.past_key_values
            length = cache[0][0].shape[-2]
            decoded = self.model.language_model.model(
                inputs_embeds=torch.randn(1, 1, 16), past_key_values=cache, use_cache=True,
                position_ids=torch.tensor([[8]]), attention_mask=torch.ones(1, length + 1),
            )
        self.assertEqual(output.last_hidden_state.shape[1], 5)
        self.assertEqual(length, 5)
        self.assertEqual(decoded.last_hidden_state.shape[1], 1)
        self.assertEqual(self.pruner.get_stats()["pruned_forward_count"], 1)
        self.assertEqual(self.pruner.get_stats()["bypassed_forward_count"], 1)
        self.assertEqual(self.pruner.get_stats()["last_output_sequence_length"], 1)

    def test_batch_broadcast_positions_and_preservation_of_nonvisual_tokens(self):
        self.pruner.attach(self.model)
        adapter = self.model.language_model.model
        tokens = self.inputs["inputs_embeds"].expand(2, -1, -1).clone()
        positions = torch.arange(8).unsqueeze(0)
        result = adapter.prune_inputs(tokens, position_ids=positions)
        self.assertEqual(result.tokens.shape, (2, 5, 16))
        self.assertTrue(torch.equal(result.tokens[:, :1], tokens[:, :1]))
        self.assertTrue(torch.equal(result.tokens[:, -3:], tokens[:, -3:]))
        self.assertTrue(torch.equal(result.position_ids[0], torch.tensor([0, 1, 5, 6, 7])))

    def test_variable_batch_keep_counts_get_padding_mask(self):
        pruner = DifferentiableTokenPruner(16).eval()
        tokens = torch.randn(2, 8, 16)
        scores = torch.zeros(2, 4, 4)
        scores[0, :, 0] = 1
        scores[1] = torch.eye(4)
        with patch.object(pruner, "get_score", return_value=scores):
            result = pruner(tokens, None, None, SequencePruningLayout(TokenSpan(1, 5)))
        self.assertEqual(result.patch_keep_counts.tolist(), [1, 4])
        self.assertEqual(result.attention_mask[0].tolist(), [1, 1, 0, 0, 0, 1, 1, 1])
        self.assertEqual(result.attention_mask[1].tolist(), [1] * 8)

    def test_unknown_layout_and_mask_errors_do_not_leave_stale_metrics(self):
        self.pruner.attach(self.model)
        self.pruner.predict_action(**self.inputs)
        with self.assertRaisesRegex(ValueError, "exceeds|Expected patch span"):
            self.pruner.predict_action(inputs_embeds=torch.randn(1, 3, 16), use_cache=False)
        self.assertEqual(self.pruner.get_stats()["pruned_forward_count"], 0)
        self.assertIsNone(self.pruner.get_stats()["last_output_sequence_length"])
        with self.assertRaisesRegex(ValueError, "shape"):
            self.pruner.predict_action(**self.inputs, attention_mask=torch.ones(1, 1, 8, 8))

    def test_prepared_step_preserves_entire_native_output(self):
        expected = ([[1.0] * 7, [2.0] * 7], {"hidden": "native-result"})
        self.model.predict_action = lambda **kwargs: expected
        self.pruner.attach(self.model)
        result = self.pruner.step(**self.inputs)
        self.assertIs(result["output"], expected)
        self.assertIs(self.pruner.get_output(), expected)
        self.pruner.reset()
        self.assertIsNone(self.pruner.get_output())

    def test_builtin_enable_remembers_mode_and_invalid_types_reject(self):
        self.builtin_model()
        acc = LightVLAAccelerator.install(self.model, mode="builtin", allow_existing_pruner=True)
        try:
            acc.disable()
            acc.enable()
            self.assertEqual(acc.config.mode, "builtin")
        finally:
            acc.detach()
        with self.assertRaisesRegex(TypeError, "LlamaModel"):
            LightVLAAccelerator.install(SimpleNamespace(language_model=SimpleNamespace(model=torch.nn.Identity())))

    def test_noise_option_is_forwarded_to_low_level_pruner(self):
        acc = LightVLAAccelerator.install(self.model, noise_scale=0.1)
        try:
            self.assertEqual(self.model.language_model.model.pruner.noise_scale, 0.1)
        finally:
            acc.detach()

    def test_cli_smoke_reports_counts_and_detach(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["--smoke"])
        self.assertEqual(code, 0)
        report = json.loads(output.getvalue())
        self.assertTrue(report["detached"])
        self.assertEqual(report["stats"]["last_output_sequence_length"], 5)

    def test_cli_factory_restores_model_on_exception(self):
        module = SimpleNamespace(create=lambda: {"model": self.model, "run": lambda: 1 / 0})
        with patch("methods.lightvla.cli.importlib.import_module", return_value=module), redirect_stderr(io.StringIO()):
            code = main(["--factory", "user_project:create"])
        self.assertEqual(code, 2)
        self.assertIs(self.model.language_model.model, self.decoder)
        self.assertNotIn("forward", vars(self.decoder))
        self.assertFalse(hasattr(self.model, "_lightvla_toolbox_session"))


if __name__ == "__main__":
    unittest.main()
