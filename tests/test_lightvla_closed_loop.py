import json
from pathlib import Path
import tempfile
import unittest
from scripts.server_lightvla_closed_loop import atomic, summarize
from scripts.summarize_lightvla_closed_loop import summarize as audit


class ClosedLoopReportingTests(unittest.TestCase):
    def test_audit_rejects_duplicate_missing_or_mismatched_pairs(self):
        base=dict(task_id=0,trial=0,mode="baseline",seed=7,initial_state_hash="same",complete=True,
                  error=None,success=True,episode_seconds=1.0,queries=[{"ms":10.0}])
        other=dict(base,mode="packaged")
        report=dict(protocol=dict(tasks=[0],trials=[0]),records=[base,other])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path=Path(folder)/"report.json"
            atomic(path,report)
            self.assertEqual(audit(path)["paired_episodes"],1)
            self.assertFalse(audit(path)["formal_speedup_accepted"])
            for rows in ([base], [base,other,other], [base,dict(other,initial_state_hash="different")]):
                report["records"]=rows
                atomic(path,report)
                with self.assertRaises(ValueError):
                    audit(path)

    def test_episode_failures_are_not_execution_errors_or_quality_pass(self):
        rows=[dict(mode="baseline",complete=True,error=None,success=True),
              dict(mode="packaged",complete=True,error=None,success=False)]
        result=summarize(rows,2)
        self.assertTrue(result["execution_complete"])
        self.assertEqual(result["successes"],dict(baseline=1,packaged=0))
        self.assertEqual(result["quality_decision"],"INCONCLUSIVE")

    def test_missing_or_error_episode_never_counts_as_complete(self):
        rows=[dict(mode="baseline",complete=True,error=None,success=True),
              dict(mode="packaged",complete=False,error="CUDA error",success=False)]
        result=summarize(rows,2)
        self.assertFalse(result["execution_complete"])
        self.assertEqual(result["error_records"],1)
        self.assertFalse(summarize(rows[:1],2)["execution_complete"])

    def test_atomic_report_replaces_previous_and_rejects_nan(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path=Path(folder)/"report.json"
            atomic(path,{"value":1})
            atomic(path,{"value":2})
            self.assertEqual(json.loads(path.read_text()),{"value":2})
            with self.assertRaises(ValueError):
                atomic(path,{"value":float("nan")})
            self.assertEqual(json.loads(path.read_text()),{"value":2})
