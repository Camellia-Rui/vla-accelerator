import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from core.project import main, read_project, plan

class ProjectTests(unittest.TestCase):
    def test_unknown_missing_resources_never_execute(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as d:
            p=Path(d)/'project.json'
            self.assertEqual(main(['init','--project',str(p)]),0)
            with patch('core.project.subprocess.run') as run:
                self.assertEqual(main(['execute','--project',str(p)]),2)
                run.assert_not_called()
            self.assertEqual(main(['init','--project',str(p)]),2)

    def test_paths_are_relative_to_project_and_no_traversal(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as d:
            p=Path(d)/'project.json'
            main(['init','--project',str(p)])
            data=read_project(p)
            self.assertEqual(data['checkpoint'],str(Path(d)/'weights'))
            for name in ('../escape','a/b','a\\b','..'):
                with self.assertRaises(ValueError): plan(data,name)

    def test_matching_metadata_is_not_compatibility_claim(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as d:
            root=Path(d)
            (root/'experiments/robot').mkdir(parents=True)
            (root/'experiments/robot/openvla_utils.py').touch()
            (root/'config.json').write_text(json.dumps({'norm_stats':{'libero_spatial':{}}}))
            (root/'tokenizer_config.json').write_text('{}')
            (root/'model.safetensors').write_bytes(b'fixture')
            data=dict(profile='lightvla_spatial',workspace=d,checkpoint=d,python=__file__,gpu='0',observation_dirs=[],output_root=d)
            result=plan(data,'new')
            self.assertEqual(result['status'],'READY_FOR_RUNTIME_CHECK')
            self.assertFalse(result['compatibility_verified'])
            self.assertEqual(result['quality_decision'],'INCONCLUSIVE')
            self.assertEqual(result['candidate_method'],'lightvla')



class ResourceTests(unittest.TestCase):
    def test_missing_shard_and_escape_are_blocked(self):
        from core.project_checks import weights
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as d:
            p=Path(d)
            (p/'model.safetensors').write_bytes(b'fixture')
            (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'missing.safetensors','b':'../outside'}}))
            issues=weights(p)
            self.assertTrue(any('Missing or empty weight shard' in x for x in issues))
            self.assertTrue(any('escapes' in x for x in issues))

    def test_environment_failure_is_explicit(self):
        from core.project_checks import environment
        import subprocess
        with patch('core.project_checks.subprocess.run',side_effect=subprocess.TimeoutExpired('python',20)):
            self.assertIn('error',environment('python'))
