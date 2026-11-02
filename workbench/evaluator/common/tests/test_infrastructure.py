"""No checkpoint, dataset or GPU jobs. Tensor tests use CPU only when installed."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paired_stats import paired_summary, join_pairs
from result_io import RecordStore, atomic_json
from preflight import check
from mechanism import LayerInputTrace


class StatisticsTests(unittest.TestCase):
    def records(self):
        base = dict(pair_id="one", task=0, trial=0, seed=7, initial_state_hash="s",
                    input_hash="i", checkpoint_id="weights", protocol_hash="p", device_id="cpu", error=None)
        return [dict(base, mode="accelerated", wall_ms=8), dict(base, mode="baseline", wall_ms=10)]

    def test_pairs_use_identity_not_record_order(self):
        self.assertEqual(join_pairs(self.records()), [(10, 8)])

    def test_mismatched_inputs_are_not_paired(self):
        for field in ("initial_state_hash", "input_hash", "device_id", "checkpoint_id", "protocol_hash"):
            records = self.records()
            records[0][field] = "other"
            with self.subTest(field=field), self.assertRaises(ValueError):
                join_pairs(records)

    def test_duplicates_and_errors_are_rejected(self):
        with self.assertRaises(ValueError):
            join_pairs(self.records() + self.records()[:1])
        records = self.records()
        records[0]["error"] = "failed"
        with self.assertRaises(ValueError):
            join_pairs(records)

    def test_bootstrap_is_reproducible_and_paired(self):
        pairs = [(10, 8), (20, 18), (30, 28)]
        a = paired_summary(pairs, bootstrap=100)
        self.assertEqual(a, paired_summary(pairs, bootstrap=100))
        self.assertEqual(a["paired_mean_saved_ms_ci95"], [2, 2])
        self.assertEqual(a["raw_paired_saved_ms"], [2, 2, 2])

    def test_invalid_samples_are_not_silently_filtered(self):
        for pairs in ([], [(1, 1)], [(1, 0), (1, 1)], [(1, float("nan")), (1, 1)], [(True, 1), (1, 1)]):
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                paired_summary(pairs, bootstrap=100)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="evaluator-test-", dir=Path(__file__).resolve().parents[4])
        self.addCleanup(self.directory.cleanup)
        self.store = RecordStore(self.directory.name, {"method": "test", "config_hash": "a"})

    def test_completed_record_cannot_be_overwritten(self):
        key = ["baseline", 0, 0, 7]
        self.store.save(key, {"status": "complete", "error": None, "wall_ms": 10})
        self.assertTrue(self.store.completed(key))
        with self.assertRaises(FileExistsError):
            self.store.save(key, {"status": "complete", "error": None})

    def test_errors_are_retryable(self):
        key = ["accelerated", 0, 0, 7]
        self.store.save(key, {"status": "error", "error": "temporary failure"})
        self.assertFalse(self.store.completed(key))
        self.store.save(key, {"status": "complete", "error": None})
        self.assertTrue(self.store.completed(key))

    def test_tuple_keys_survive_json_round_trip(self):
        key = ("baseline", 1, 2, 7)
        self.store.save(key, {"status": "complete", "error": None})
        self.assertTrue(self.store.completed(key))

    def test_nonfinite_json_does_not_destroy_previous_evidence(self):
        path = Path(self.directory.name) / "result.json"
        atomic_json(path, {"valid": True})
        with self.assertRaises(ValueError):
            atomic_json(path, {"wall_ms": float("nan")})
        self.assertEqual(json.loads(path.read_text()), {"valid": True})

    def test_configuration_identity_separates_records(self):
        self.store.save([1], {"status": "complete", "error": None})
        other = RecordStore(self.directory.name, {"method": "test", "config_hash": "b"})
        self.assertFalse(other.completed([1]))

    def test_corrupt_records_fail_instead_of_being_overwritten(self):
        path = self.store.path_for([1])
        path.write_text("invalid json")
        with self.assertRaises(ValueError):
            self.store.completed([1])

    def test_corrupt_status_cannot_be_replaced(self):
        self.store.path_for([1]).write_text(json.dumps({"status": "unknown"}))
        with self.assertRaises(ValueError):
            self.store.save([1], {"status": "complete", "error": None})


class PreflightTests(unittest.TestCase):
    def config(self):
        return dict(method="specprune", task_suite="libero_object", device="cuda:0", python="/missing/python",
                    views=2, action_dim=7, chunk_size=8,
                    paths={k: "/missing/" + k for k in ("vla_repository", "method_repository", "checkpoint", "libero_assets", "toolbox_src")})

    def test_remote_missing_paths_are_reported_without_importing_models(self):
        result = check(self.config(), "specprune")
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["gpu_experiment_started"])
        self.assertIsNone(result["runtime_compatible"])
        self.assertIn("checkpoint", result["missing"])

    def test_unsupported_layout_is_rejected(self):
        cfg = self.config()
        cfg["action_dim"] = 14
        with self.assertRaises(ValueError):
            check(cfg, "specprune")

    def test_method_confusion_is_rejected(self):
        with self.assertRaises(ValueError):
            check(self.config(), "vla-pruner")

    def test_checkpoint_index_cannot_escape_checkpoint_directory(self):
        with tempfile.TemporaryDirectory(prefix="checkpoint-fixture-", dir=Path(__file__).resolve().parents[4]) as directory:
            config = self.config()
            config["paths"]["checkpoint"] = directory
            index = Path(directory) / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"weight": "../escape.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "escapes"):
                check(config, "specprune")

    def test_missing_checkpoint_shard_is_reported(self):
        with tempfile.TemporaryDirectory(prefix="checkpoint-fixture-", dir=Path(__file__).resolve().parents[4]) as directory:
            config = self.config()
            config["paths"]["checkpoint"] = directory
            index = Path(directory) / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"weight": "missing.safetensors"}}))
            result = check(config, "specprune")
            self.assertIn("checkpoint shard/missing.safetensors", result["missing"])
            self.assertFalse(result["model_loaded"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch not installed")
class CpuAuditTests(unittest.TestCase):
    def test_actual_precompute_shape_hook_is_cleaned_up(self):
        import torch
        layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        trace = LayerInputTrace(layers)
        with trace:
            layers[0](torch.zeros(1, 10, 4))
            layers[1](torch.zeros(1, 6, 4))
            self.assertTrue(trace.summary()["shortening_observed"])
        self.assertTrue(all(len(layer._forward_pre_hooks) == 0 for layer in layers))

    def test_parameters_and_hooks_audit_real_cpu_modules(self):
        import torch
        from lifecycle import ParameterAudit, DetachAudit
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
        values = ParameterAudit(model)
        structure = DetachAudit(model)
        self.assertTrue(values.active(model)["sampled_original_values_unchanged"])
        with LayerInputTrace([model[0], model[1]]):
            self.assertFalse(structure.check(model)["hook_counts_restored"])
        self.assertTrue(structure.check(model)["hook_counts_restored"])
        with torch.no_grad():
            model[0].weight[0, 0] += 1
        self.assertFalse(values.active(model)["sampled_original_values_unchanged"])


if __name__ == "__main__":
    unittest.main()
