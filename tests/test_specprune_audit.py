import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_specprune import audit


class SpecPruneAuditTests(unittest.TestCase):
    def report(self):
        episode = dict(task=0, trial=5, seed=7, initial_state_hash='abc', success=True,
                       environment_steps=2, records=[
                           dict(action=[0]*7, execute_action=True, ms=100, queried=True, infer_step=1),
                           dict(action=[0]*7, execute_action=True, ms=1, queried=False, infer_step=1)])
        return dict(passed=True, control={'matches':True}, restored={'matches':True},
                    mechanism_observed=True, modules_restored=True, parameters_preserved=True,
                    timings=[dict(pair=i, mode=m, ms=100, actions=[[0]*7 for _ in range(8)])
                             for i in range(2) for m in ('baseline','accelerated')],
                    closed_loop_tested=True, post_loop_restored={'matches':True},
                    episodes=[dict(copy.deepcopy(episode), mode=m) for m in ('baseline','accelerated')])

    def check(self, report):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            for e in report['episodes']:
                (root/(e['mode']+'-episode.json')).write_text(json.dumps(e))
            path=root/'report.json'
            path.write_text(json.dumps(report))
            return audit(path)

    def test_queue_hits_are_not_counted_as_model_calls(self):
        e=self.check(self.report())['episodes'][0]
        self.assertEqual((e['policy_calls'],e['model_calls'],e['policy_ms'],e['query_policy_ms']), (2,1,101,100))

    def test_invalid_counters_and_times_are_rejected(self):
        for field,value in [('infer_step',2),('queried',None),('ms',float('nan')),('ms',-1)]:
            r=self.report()
            r['episodes'][0]['records'][1][field]=value
            with self.assertRaises(ValueError): self.check(r)

    def test_historical_records_remain_readable_without_invented_query_counts(self):
        r=self.report()
        for e in r['episodes']:
            for x in e['records']:
                del x['queried'];del x['infer_step']
        self.assertNotIn('model_calls',self.check(r)['episodes'][0])
