import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import server_pruner_pair as pair


class PairProtocolTests(unittest.TestCase):
    def test_mechanism_requires_actual_shorter_matching_layer(self):
        self.assertTrue(pair.shortened({"0": [[1, 12, 8]]}, {"0": [[1, 8, 8]]}))
        self.assertFalse(pair.shortened({"0": [[1, 12, 8]]}, {"0": [[1, 12, 8]]}))
        self.assertFalse(pair.shortened({"0": [[1, 12, 8]]}, {"1": [[1, 8, 8]]}))

    def test_action_comparison_separates_gripper_and_arm(self):
        import numpy as np
        a = np.ones((8, 7))
        b = a.copy()
        b[:, 6] = -1
        result = pair.compare(a, b)
        self.assertEqual(result["arm_mae"], 0)
        self.assertEqual(result["raw_gripper_sign_agreement"], 0)
        self.assertFalse(result["close"])

    def test_diagnostic_hooks_removed_after_failed_query(self):
        import torch
        from types import SimpleNamespace
        layer = torch.nn.Linear(8, 8)
        model = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=[layer])))
        def fail():
            layer(torch.ones(1, 12, 8))
            raise RuntimeError("query failed")
        with self.assertRaisesRegex(RuntimeError, "query failed"):
            pair.trace_query(model, fail)
        self.assertEqual(len(layer._forward_pre_hooks), 0)


if __name__ == "__main__":
    unittest.main()
