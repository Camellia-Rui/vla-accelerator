import json
from pathlib import Path
import tempfile
import unittest
from scripts.audit_lightvla_history import audit


class HistoryTests(unittest.TestCase):
    def test_pairing_and_device_mismatch_are_not_acceptance(self):
        row=dict(task_suite="spatial",task_id=0,episode=1,success=True,error=None,
                 initial_state_hash="a",seed=0,checkpoint="ckpt",task_description="pick",device="cuda:0")
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            a,b=Path(directory)/"a.jsonl",Path(directory)/"b.jsonl"
            a.write_text(json.dumps(row)+"\n")
            b.write_text(json.dumps(dict(row,device="cuda:1"))+"\n")
            result=audit(a,b)
            self.assertEqual(result["device_mismatch_pairs"],1)
            self.assertFalse(result["current_adapter_acceptance"])
            self.assertFalse(result["formal_speedup_accepted"])
            b.write_text(json.dumps(dict(row,initial_state_hash="other"))+"\n")
            with self.assertRaises(ValueError):
                audit(a,b)
            b.write_text(a.read_text()*2)
            with self.assertRaises(ValueError):
                audit(a,b)
