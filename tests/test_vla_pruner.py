"""Toolbox contract checks, using a fake backend (no torch/model/data required)."""

from collections import deque
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from methods.vla_pruner import AcceleratorConfig, VLAPrunerAccelerator
from methods.vla_pruner.backend import BACKEND_COMMIT, OpenVLAOFTBackend, load_backend
from pruners import build


class FakeDecoder:
    def __init__(self):
        self.layers = [object() for _ in range(32)]

    def fastv_forward(self):
        pass

    def _fastv_pruning_indices(self):
        pass


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(use_fastv=False, fastv_r=0.1)
        self.language_model = SimpleNamespace(
            model=FakeDecoder(), config=SimpleNamespace(reusable_patches="old-cache"),
            fastv_forward=lambda: None,
        )
        self.vision_backbone = SimpleNamespace(
            get_num_patches=lambda: 256, get_num_images_in_input=lambda: 2,
        )
        self.av_hist = deque(["original-history"], maxlen=5)
        self.training = False
        self.norm_stats = {"libero_object_no_noops": {"proprio": {}}}

    def _build_fastv_config(self):
        pass

    def reset_av_history(self):
        self.av_hist.clear()

    def predict_action(self):
        # Simulate only the backend output contract, not the pruning algorithm.
        removed = list(range(100)) if self.use_fastv and self.fastv_r else []
        self.language_model.model.pruning_info = {
            "original_seq_length": 600, "pruned_indices": removed,
            "pruning_layer": self.fastv_k if removed else None,
        }
        if self.use_temporal:
            self.av_hist.append("new-attention")
        return [[1.0] * 7, [2.0] * 7]


def fake_runner(cfg, model, processor, obs, **kwargs):
    if cfg.use_vla_cache or kwargs["last_caches"] is not None:
        raise AssertionError("Cross-observation KV reuse must be disabled")
    if cfg.unnorm_key != "libero_object_no_noops":
        raise AssertionError("Normalization suffix was not resolved")
    return model.predict_action(), {"unused": True}, [], {"time_elapsed": 0.01}


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.pruner = build("vla_pruner")
        self.addCleanup(self.pruner.detach)

    def attach_with_resources(self, **kwargs):
        self.pruner.attach(self.model, processor=object(), action_head=object(),
                           proprio_projector=object(), action_runner=fake_runner, **kwargs)

    def query(self):
        with patch.dict("sys.modules", {"torch": SimpleNamespace(inference_mode=nullcontext)}):
            return self.pruner.step(image=object(), wrist_image=object(),
                                    instruction="pick up", proprio=[0] * 8)

    def test_attach_does_not_load_or_replace_forward(self):
        forward = self.model.predict_action.__func__
        with patch("methods.vla_pruner.accelerator.load_backend", side_effect=AssertionError("unexpected load")):
            self.pruner.attach(self.model)
        self.assertIs(self.model.predict_action.__func__, forward)
        self.assertTrue(self.model.use_vla_pruner)
        self.assertTrue(self.model.config.use_vla_pruner)
        self.assertEqual(self.model.fastv_k, 3)

    def test_detach_restores_all_managed_attributes_and_history_identity(self):
        objects = (self.model, self.model.config, self.model.language_model,
                   self.model.language_model.config, self.model.language_model.model)
        before = [vars(obj).copy() for obj in objects]
        self.pruner.attach(self.model)
        self.model.predict_action()
        self.pruner.disable()
        self.pruner.enable()
        self.pruner.reset()
        self.pruner.detach()
        for obj, snapshot in zip(objects, before):
            self.assertEqual(vars(obj), snapshot)
        self.assertIs(self.model.av_hist, before[0]["av_hist"])
        self.assertEqual(list(self.model.av_hist), ["original-history"])
        self.pruner.detach()  # idempotent

    def test_duplicate_owner_and_wrong_model_are_rejected(self):
        self.pruner.attach(self.model)
        other = build("vla_pruner")
        with self.assertRaisesRegex(RuntimeError, "owner"):
            other.attach(self.model)
        with self.assertRaisesRegex(RuntimeError, "already attached"):
            self.pruner.attach(FakeModel())
        with self.assertRaisesRegex(ValueError, "different model"):
            self.pruner.detach(FakeModel())
        self.assertTrue(self.pruner.enabled)

    def test_reset_and_mode_switch_clear_owned_state(self):
        self.pruner.attach(self.model)
        self.model.predict_action()
        self.pruner.disable()
        self.assertFalse(self.model.use_fastv)
        self.assertFalse(self.model.use_vla_pruner)
        self.assertFalse(self.model.use_temporal)
        self.assertEqual(len(self.model.av_hist), 0)
        self.assertIsNone(self.model.language_model.config.reusable_patches)
        self.assertIsNone(self.pruner.get_stats()["actual_prune_ratio"])
        self.pruner.enable()
        self.assertTrue(self.model.use_temporal)

    def test_detach_then_attach_to_another_model(self):
        self.pruner.attach(self.model)
        self.pruner.detach()
        other = FakeModel()
        self.pruner.attach(other)
        self.assertTrue(other.use_vla_pruner)
        self.assertFalse(hasattr(self.model, "use_vla_pruner"))

    def test_native_model_statistics_use_actual_counts(self):
        self.pruner.attach(self.model)
        self.assertIsNone(self.pruner.get_stats()["visual_tokens_kept"])
        self.model.predict_action()
        stats = self.pruner.get_stats()
        self.assertEqual(stats["visual_tokens_pruned"], 100)
        self.assertEqual(stats["visual_tokens_kept"], 412)
        self.assertEqual(stats["sequence_tokens_after"], 500)
        self.assertAlmostEqual(stats["actual_prune_ratio"], 100 / 512)
        self.assertEqual(stats["steps"], 0)  # Native calls bypass the query counter.

    def test_full_chunk_and_query_statistics_are_preserved(self):
        self.attach_with_resources()
        result = self.query()
        self.assertEqual(result["actions"], [[1.0] * 7, [2.0] * 7])
        self.assertIs(result["action"], result["actions"][0])
        self.assertIs(self.pruner.get_actions(), result["actions"])
        self.assertEqual(result["info"]["steps"], 1)
        self.assertEqual(result["info"]["backend_time_elapsed"], 0.01)
        self.assertEqual(self.pruner.end_episode(True)["steps"], 1)

    def test_disabled_step_passes_matching_backend_flags(self):
        def runner(cfg, model, processor, obs, **kwargs):
            self.assertFalse(cfg.use_vla_pruner)
            self.assertFalse(cfg.use_fastv)
            self.assertFalse(model.use_fastv)
            return fake_runner(cfg, model, processor, obs, **kwargs)
        self.pruner.attach(self.model, processor=object(), action_head=object(),
                           proprio_projector=object(), action_runner=runner)
        self.pruner.disable()
        self.assertEqual(self.query()["info"]["visual_tokens_pruned"], 0)

    def test_failed_query_clears_history_and_stale_actions(self):
        self.attach_with_resources()
        self.query()
        def fail(*args, **kwargs):
            self.model.av_hist.append("partial")
            raise RuntimeError("inference failed")
        self.pruner._acc._action_runner = fail
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.query()
        self.assertEqual(len(self.model.av_hist), 0)
        self.assertIsNone(self.pruner.get_action())
        self.assertIsNone(self.pruner.get_actions())
        self.assertIsNone(self.pruner.get_stats()["visual_tokens_pruned"])

    def test_input_and_resource_errors_are_actionable(self):
        self.pruner.attach(self.model)
        with self.assertRaisesRegex(ValueError, "wrist_image"):
            self.pruner.step(image=object(), instruction="pick")
        with self.assertRaisesRegex(ValueError, "proprio is required"):
            self.pruner.step(image=object(), wrist_image=object(), instruction="pick")
        with self.assertRaisesRegex(ValueError, "processor"):
            self.query()

    def test_training_mode_rejected_without_changing_it(self):
        self.attach_with_resources()
        self.model.training = True
        with self.assertRaisesRegex(RuntimeError, "eval"):
            self.query()
        self.assertTrue(self.model.training)

    def test_calls_before_attach_and_after_detach_fail(self):
        for method in (self.pruner.reset, self.pruner.enable, self.pruner.disable):
            with self.assertRaisesRegex(RuntimeError, "Attach"):
                method()
        with self.assertRaisesRegex(ValueError, "existing model"):
            self.pruner.attach(None)


class CompatibilityTests(unittest.TestCase):
    def test_unpatched_models_and_wrong_layers_rejected_before_mutation(self):
        for model in (object(), SimpleNamespace(predict_action=lambda: None)):
            with self.assertRaises(TypeError):
                VLAPrunerAccelerator.install(model)
        model = FakeModel()
        model.language_model.fastv_forward = None
        with self.assertRaisesRegex(TypeError, "LLaMA"):
            VLAPrunerAccelerator.install(model)
        self.assertFalse(hasattr(model, "_vla_express_pruner_owner"))
        with self.assertRaisesRegex(ValueError, "depth"):
            VLAPrunerAccelerator.install(FakeModel(), fastv_k=32)
        with self.assertRaisesRegex(ValueError, "view count"):
            VLAPrunerAccelerator.install(FakeModel(), num_images_in_input=1)

    def test_attach_failure_rolls_back_partial_configuration(self):
        class RejectingConfig(SimpleNamespace):
            def __setattr__(self, name, value):
                if name == "use_vla_pruner" and value:
                    raise RuntimeError("immutable config")
                super().__setattr__(name, value)
        model = FakeModel()
        model.config = RejectingConfig()
        before = vars(model).copy()
        acc = VLAPrunerAccelerator()
        with self.assertRaisesRegex(RuntimeError, "immutable config"):
            acc.attach(model)
        self.assertEqual(vars(model), before)
        self.assertEqual(vars(model.config), {})
        self.assertIsNone(acc.model)

    def test_strict_config_validation(self):
        for values in ({"fastv_r": 1}, {"fastv_r": float("nan")}, {"fastv_k": True},
                       {"fastv_k": 0}, {"vla_pruner_av_hist_w": 0}, {"typo": 1},
                       {"vla_pruner_av_decay": 2}, {"use_vla_pruner": "false"},
                       {"vla_pruner_mode": "unknown"}, {"state_type": "stateless"},
                       {"num_open_loop_steps": 9}, {"num_images_in_input": 3},
                       {"vla_pruner_semantic_weight": 0, "vla_pruner_action_weight": 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                AcceleratorConfig.from_mapping(values)
        self.assertEqual(AcceleratorConfig.from_mapping({"method": "vla_pruner"}).fastv_k, 3)

    def test_zero_ratio_and_fastv_mode(self):
        for values in ({"fastv_r": 0}, {"vla_pruner_mode": "fastv"}):
            model = FakeModel()
            acc = VLAPrunerAccelerator.install(model, **values)
            try:
                model.predict_action()
                if "fastv_r" in values:
                    self.assertEqual(acc.get_stats()["visual_tokens_pruned"], 0)
                else:
                    self.assertFalse(model.use_vla_pruner)
                    self.assertFalse(model.use_temporal)
                    self.assertTrue(model.use_fastv)
            finally:
                acc.detach()

    def test_warmup_without_pruned_indices_reports_zero(self):
        acc = VLAPrunerAccelerator.install(FakeModel())
        try:
            acc.model.language_model.model.pruning_info = {
                "original_seq_length": 600, "pruned_indices": None,
                "kept_indices": None, "pruning_layer": None,
            }
            self.assertEqual(acc.get_stats()["visual_tokens_pruned"], 0)
            self.assertEqual(acc.get_stats()["visual_tokens_kept"], 512)
        finally:
            acc.detach()

    def test_backend_missing_import_preserves_cause(self):
        with patch("methods.vla_pruner.backend.importlib.import_module", side_effect=ModuleNotFoundError("torch")):
            with self.assertRaisesRegex(ImportError, "pinned backend") as caught:
                load_backend()
            self.assertIsInstance(caught.exception.__cause__, ModuleNotFoundError)

    def test_checkpoint_sync_is_never_implicitly_enabled(self):
        backend = OpenVLAOFTBackend(SimpleNamespace(get_vla_action=fake_runner))
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENVLA_SKIP_CHECKPOINT_SYNC"):
                backend.load_model(AcceleratorConfig(checkpoint="test"))

    def test_checkpoint_loading_is_separate_and_attach_uses_returned_model(self):
        model = FakeModel()
        resources = {"model": model, "processor": object(), "action_head": object()}
        with patch("methods.vla_pruner.accelerator.load_backend") as backend:
            backend.return_value.load_model.return_value = resources.copy()
            acc = VLAPrunerAccelerator.from_pretrained("checkpoint")
            try:
                self.assertIs(acc.model, model)
                self.assertEqual(backend.return_value.load_model.call_args.args[0].checkpoint, "checkpoint")
            finally:
                acc.detach()

    def test_lock_matches_backend_reference(self):
        path = Path(__file__).resolve().parents[1] / "methods/vla_pruner/backend.lock.json"
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["commit"], BACKEND_COMMIT)


if __name__ == "__main__":
    unittest.main()
