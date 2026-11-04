import copy
import unittest
from scripts.audit_oft_choices import audit


class AuditTests(unittest.TestCase):
    def report(self):
        rows=[dict(task_id=0,trial=0,mode=m,seed=7,initial_state_hash="same",complete=True,error=None,
            success=m!="lightvla",queries=[dict(ms=1.0,actions=[[0.0]*7])]) for m in ("baseline","lightvla","vla_pruner")]
        return dict(protocol=dict(tasks=[0],trials=[0]),records=rows,restored_vs_native=dict(matches=True),
            parameters_preserved=True,modules_restored=True)

    def test_behavior_loss_is_visible_without_fabricating_acceptance(self):
        result=audit(self.report())
        self.assertEqual(result["baseline_only_successes"]["lightvla"],1)
        self.assertEqual(result["successes"]["vla_pruner"],1)
        self.assertEqual(result["quality_decision"],"INCONCLUSIVE")

    def test_missing_duplicate_mismatched_or_nonfinite_records_rejected(self):
        base=self.report()
        for kind in ("missing","duplicate","mismatch","nan"):
            report=copy.deepcopy(base)
            if kind=="missing":report["records"].pop()
            elif kind=="duplicate":report["records"].append(report["records"][0])
            elif kind=="mismatch":report["records"][0]["initial_state_hash"]="other"
            else:report["records"][0]["queries"][0]["actions"][0][0]=float("nan")
            with self.assertRaises(ValueError):audit(report)
