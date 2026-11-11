import json
from pathlib import Path
import unittest

from core.select_method import decide, metrics, paired_metrics


class SelectionTests(unittest.TestCase):
    report = (Path(__file__).resolve().parents[1] / "docs/server-evidence/2026-11-04"
              / "oft-choice-closedloop-001/report.json")

    def test_fastest_failed_candidate_excluded(self):
        scores = metrics(self.report)
        self.assertLess(scores["lightvla_style"][2], scores["vla_pruner"][2])
        self.assertEqual(decide(scores, 90, 6, 0)["selected"], "vla_pruner")
        self.assertIsNone(decide(scores, 90, 7, 0)["selected"])

    def test_unpaired_state_rejected(self):
        report = json.loads(self.report.read_bytes())
        report["records"][0]["initial_state_hash"] = "other"
        with self.assertRaises(ValueError):
            paired_metrics(report)

    def test_no_benefit_retains_native(self):
        scores = {"baseline": (2, 2, 10), "alternative": (2, 2, 11)}
        self.assertEqual(decide(scores, 100, 2, 0)["selected"], "baseline")

    def test_scoring_run_distinguishes_backend_modes_and_sample_size(self):
        path = (Path(__file__).resolve().parents[1] / "docs/server-evidence/2026-11-10"
                / "oft-scoring-object/pilot/report.json")
        scores = metrics(path)
        self.assertEqual(set(scores), {"baseline", "vla_pruner", "fastv_scoring"})
        self.assertTrue(all((wins, n) == (6, 6) for wins, n, _ in scores.values()))
        self.assertIsNone(decide(scores, 90, 50, 0)["selected"])
