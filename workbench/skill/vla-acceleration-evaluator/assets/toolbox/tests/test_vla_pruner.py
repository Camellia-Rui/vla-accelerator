from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vla_pruner import AcceleratorConfig, VLAPrunerAccelerator


class DummyVisionBackbone:
    def get_num_patches(self) -> int:
        return 256

    def get_num_images_in_input(self) -> int:
        return 2


class DummyDecoder:
    def __init__(self) -> None:
        self.layers = [object() for _ in range(32)]
        self.pruning_info = None

    def fastv_forward(self):
        return None

    def _fastv_pruning_indices(self):
        return None


class DummyLanguageModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace()
        self.model = DummyDecoder()
        self.pruning_info = None

    def fastv_forward(self):
        return None


class DummyModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace()
        self.language_model = DummyLanguageModel()
        self.vision_backbone = DummyVisionBackbone()
        self.av_hist = deque(["original"], maxlen=7)
        self.norm_stats = {"libero_object": {"proprio": {}}}
        self.training = False

    def predict_action(self):
        return None

    def _build_fastv_config(self):
        return None

    def reset_av_history(self):
        return None


class VLAPrunerContractTests(unittest.TestCase):
    def test_config_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "fastv_r"):
            AcceleratorConfig(fastv_r=1.0)
        with self.assertRaisesRegex(ValueError, "num_images_in_input"):
            AcceleratorConfig(num_images_in_input=3)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            AcceleratorConfig(vla_pruner_mode="unknown")

    def test_attach_stats_disable_enable_reset_and_detach(self) -> None:
        model = DummyModel()
        original_history = model.av_hist
        pruner = VLAPrunerAccelerator(AcceleratorConfig())

        pruner.attach(model)
        self.assertTrue(pruner.enabled)
        self.assertTrue(model.use_fastv)
        self.assertTrue(model.use_vla_pruner)
        self.assertIsNot(model.av_hist, original_history)
        self.assertEqual(len(model.av_hist), 0)

        model.av_hist.append("frame")
        model.language_model.model.pruning_info = {
            "original_seq_length": 598,
            "pruned_indices": list(range(474)),
            "image_token_length": 512,
            "pruning_layer": 3,
        }
        stats = pruner.get_stats()
        self.assertEqual(stats["sequence_tokens_before"], 598)
        self.assertEqual(stats["sequence_tokens_after"], 124)
        self.assertEqual(stats["visual_tokens_kept"], 38)
        self.assertEqual(stats["pruning_layer"], 3)

        pruner.disable()
        self.assertFalse(pruner.enabled)
        self.assertFalse(model.use_fastv)
        self.assertFalse(model.use_vla_pruner)
        self.assertEqual(len(model.av_hist), 0)

        pruner.enable()
        self.assertTrue(pruner.enabled)
        self.assertEqual(len(model.av_hist), 0)

        pruner.detach()
        self.assertIs(model.av_hist, original_history)
        self.assertFalse(hasattr(model, "use_fastv"))
        self.assertFalse(hasattr(model, "_vla_express_pruner_owner"))
        pruner.detach()

    def test_one_owner_per_model(self) -> None:
        model = DummyModel()
        first = VLAPrunerAccelerator()
        second = VLAPrunerAccelerator()
        first.attach(model)
        try:
            with self.assertRaisesRegex(RuntimeError, "already has"):
                second.attach(model)
        finally:
            first.detach()

    def test_incompatible_model_is_rejected_without_mutation(self) -> None:
        model = SimpleNamespace()
        pruner = VLAPrunerAccelerator()
        with self.assertRaisesRegex(TypeError, "Expected"):
            pruner.attach(model)
        self.assertIsNone(pruner.model)
        self.assertFalse(hasattr(model, "_vla_express_pruner_owner"))


if __name__ == "__main__":
    unittest.main()
