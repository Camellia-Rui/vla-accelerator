import copy
import json
from pathlib import Path
import tempfile
import unittest
import sys
import subprocess
from core.live_evaluation import assess_progress
from core.select_method import main

class LiveEvaluationTests(unittest.TestCase):
    path = Path(__file__).resolve().parents[1]/'docs/server-evidence/2026-11-10/oft-scoring-object/pilot/report.json'

    def test_project_preserves_virtualenv_interpreter_path(self):
        from core.project import read_project
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root=Path(tmp);interpreter=root/'env/bin/python'
            project=root/'project.json'
            project.write_text(json.dumps(dict(schema_version=1,profile='oft_object',
                workspace='source',checkpoint='weights',python='env/bin/python',
                output_root='results',gpu=0)))
            original=Path.resolve
            def resolve(p,*a,**kw):
                if p == interpreter:raise AssertionError('Interpreter symlink must not be dereferenced')
                return original(p,*a,**kw)
            with patch.object(Path,'resolve',resolve):
                self.assertEqual(read_project(project)['python'],str(interpreter))

    def test_fastv_pair_journal_rejects_mismatched_state(self):
        from methods.fastv.live import publish_modes
        r=json.loads(self.path.read_bytes())
        row=copy.deepcopy(r['records'][0])
        protocol=dict(tasks=[row['task_id']],trials=[row['trial']],seed=row['seed'])
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root=Path(tmp)
            for mode in ('baseline','fastv'):
                (root/mode).mkdir()
                row['mode']=mode
                report=dict(protocol=dict(protocol,modes=[mode]),records=[copy.deepcopy(row)])
                combined=publish_modes(root/mode,report)
            self.assertEqual(assess_progress(combined)['complete_pairs'],1)
            report['records'][0]['initial_state_hash']='corrupt'
            with self.assertRaises(ValueError):publish_modes(root/'fastv',report)
            self.assertEqual(json.loads((root/'assessment.json').read_text())['status'],'FAILED')

    def test_specprune_queue_calls_are_not_model_queries(self):
        from core.live_evaluation import specprune_progress
        calls=[dict(queried=True,infer_step=1,ms=10.,action=[0.]*7,execute_action=True),
               dict(queried=False,infer_step=1,ms=1.,action=[0.]*7,execute_action=True)]
        e=dict(task=0,trial=0,mode='baseline',seed=7,initial_state_hash='state',success=True,
               records=calls,episode_seconds=.1,environment_steps=2)
        checked=assess_progress(specprune_progress([e],0,0,7))
        self.assertEqual(checked['metrics']['baseline']['queries'],1)
        self.assertEqual(checked['metrics']['baseline']['policy_ms'],11.)
        calls[1]['infer_step']=2
        with self.assertRaises(ValueError):specprune_progress([e],0,0,7)

    def test_supervisor_propagates_process_failure(self):
        from core.assessment import run_monitored
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            out=Path(tmp);(out/'measurement').mkdir()
            (out/'measurement/assessment.json').write_text('{"status":"RUNNING"}')
            run_monitored([sys.executable,'-c','pass'],Path.cwd(),out,dict(status='RUNNING'))
            self.assertEqual(json.loads((out/'assessment.json').read_text())['live']['status'],'RUNNING')
            with self.assertRaises(subprocess.CalledProcessError):
                run_monitored([sys.executable,'-c','raise SystemExit(3)'],Path.cwd(),out,{})

    def test_final_audit_requires_all_episodes(self):
        from core.live_evaluation import publish
        r=json.loads(self.path.read_bytes());r['records'].pop()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            with self.assertRaises(ValueError):publish(Path(tmp),r,final=True)
            self.assertEqual(json.loads((Path(tmp)/'assessment.json').read_text())['status'],'FAILED')

    def test_explicit_mechanism_failure_cannot_be_selected(self):
        from core.select_method import paired_metrics
        r=json.loads(self.path.read_bytes());r['mechanism_observed']=False
        with self.assertRaises(ValueError):paired_metrics(r)

    def test_failed_run_also_writes_human_report(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root=Path(tmp);out=root/'failed'
            self.assertEqual(main(['--report',str(root/'missing.json'),'--output',str(out)]),2)
            self.assertIn('FAILED',(out/'assessment.md').read_text(encoding='utf-8'))
            self.assertEqual(json.loads((out/'assessment.json').read_text())['quality_decision'],'INCONCLUSIVE')

    def test_real_records_checked_incrementally(self):
        r=json.loads(self.path.read_bytes())
        rows=r['records']; r['records']=[]
        for row in rows:
            r['records'].append(row)
            checked=assess_progress(r)
            self.assertEqual(checked['checked_episodes'],len(r['records']))
            self.assertEqual(checked['quality_decision'],'INCONCLUSIVE')
        self.assertEqual(checked['complete_pairs'],6)

    def test_corruption_rejected_but_task_failure_retained(self):
        r=json.loads(self.path.read_bytes())
        r['records'][0]['success']=False
        assess_progress(r)
        for field,value in [('seed',999),('policy_ms',-1),('episode_seconds',1e-9),('initial_state_hash','wrong'),('error','crash')]:
            broken=copy.deepcopy(r);broken['records'][0][field]=value
            with self.assertRaises(ValueError):assess_progress(broken)

    def test_one_command_writes_decision_without_second_review(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            out=Path(tmp)/'result'
            args=['--report',str(self.path),'--output',str(out),'--min-success-pct','90','--max-drop-pp','0','--min-pairs','50']
            self.assertEqual(main(args),0)
            r=json.loads((out/'assessment.json').read_text())
            self.assertEqual(r['sample_policy_status'],'NOT_SATISFIED')
            self.assertEqual(r['quality_decision'],'INCONCLUSIVE')
            self.assertTrue((out/'assessment.md').exists())
            before=(out/'assessment.json').read_bytes()
            self.assertEqual(main(args),2)
            self.assertEqual(before,(out/'assessment.json').read_bytes())
            other=Path(tmp)/'no-policy'
            self.assertEqual(main(['--report',str(self.path),'--output',str(other)]),0)
            r=json.loads((other/'assessment.json').read_text())
            self.assertEqual(r['sample_policy_status'],'NOT_SPECIFIED')
            self.assertIsNone(r['selected'])
