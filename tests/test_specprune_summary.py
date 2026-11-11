import json
from pathlib import Path
import tempfile
import unittest
from scripts.audit_specprune import audit
from scripts.batch_specprune import aggregate
from scripts.summarize_specprune import summarize
from tests import test_specprune_audit


class SpecPruneSummaryTests(unittest.TestCase):
    def batch(self, root, task, checkpoint='same-model'):
        root.mkdir()
        pair=root/f'{task}-5';pair.mkdir()
        r=test_specprune_audit.SpecPruneAuditTests().report()
        for e in r['episodes']:
            e['task']=task
            (pair/(e['mode']+'-episode.json')).write_text(json.dumps(e))
        (pair/'report.json').write_text(json.dumps(r))
        checked=audit(pair/'report.json')
        rows=[dict(task=task,trial=5,directory=f'/server/{task}-5',audit=checked)]
        (root/'report.json').write_text(json.dumps(aggregate(rows,1)))
        (root/'batch-config.json').write_text(json.dumps(dict(tasks=[task],trials=[5],checkpoint=checkpoint,gpu=str(task))))
        return root/'report.json'

    def test_pooling_checks_raw_evidence_and_protocol(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root=Path(tmp)
            a=self.batch(root/'a',0)
            b=self.batch(root/'b',1)
            c=self.batch(root/'c',2,'different-model')
            r=summarize([a,b])
            self.assertEqual(r['successes'],{'baseline':2,'accelerated':2})
            self.assertEqual(r['episode_policy_speedup'],1)
            self.assertEqual(r['paired_successful_episodes'],2)
            self.assertEqual(r['paired_successful_episode_policy_speedup'],1)
            with self.assertRaises(ValueError):summarize([a,a])
            with self.assertRaises(ValueError):summarize([a,c])
            p=root/'a/0-5/baseline-episode.json'
            p.write_text('{}')
            with self.assertRaises(ValueError):summarize([a,b])
