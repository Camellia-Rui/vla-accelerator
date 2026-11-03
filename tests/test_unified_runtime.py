from contextlib import nullcontext, redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from unittest.mock import patch

from core.runtime_env import runtime_environment
from methods.vla_pruner.cli import parser, read_config, worker, main, validate_bundle
from methods.vla_pruner.evaluation import actions_array, difference, screen
from test_vla_pruner import FakeModel, fake_runner


class UnifiedRuntimeTests(unittest.TestCase):
    def temp(self):
        return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)

    def bundle(self, runner=fake_runner):
        self.model = FakeModel()
        return dict(model=self.model, processor=object(), action_head=object(), proprio_projector=object(),
                    action_runner=runner, observations=[dict(full_image=[1],wrist_image=[2],state=[0]*8,instruction="pick up")])

    def execute(self, directory, bundle):
        args = parser().parse_args(["--workspace", ".", "--factory", "user:build", "--output", str(directory)])
        with patch("methods.vla_pruner.runtime.load_bundle", return_value=bundle), \
             patch.dict("sys.modules", {"torch":SimpleNamespace(inference_mode=nullcontext)}), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return worker(args, read_config(args))

    def test_complete_actions_saved_and_original_model_restored(self):
        with self.temp() as value:
            bundle=self.bundle()
            original_history=self.model.av_hist
            self.assertEqual(self.execute(Path(value),bundle),0)
            report=json.loads((Path(value)/"report.json").read_text())
            self.assertEqual(report["records"][0]["actions"],[[1.0]*7,[2.0]*7])
            self.assertTrue(report["detached"])
            self.assertIs(self.model.av_hist,original_history)
            self.assertFalse(hasattr(self.model,"use_vla_pruner"))

    def test_failed_query_still_restores_and_reports_failure(self):
        def fail(*args,**kwargs):
            raise RuntimeError("native query failed")
        with self.temp() as value:
            bundle=self.bundle(fail)
            original_history=self.model.av_hist
            self.assertEqual(self.execute(Path(value),bundle),2)
            report=json.loads((Path(value)/"report.json").read_text())
            self.assertFalse(report["passed"])
            self.assertTrue(report["detached"])
            self.assertIs(self.model.av_hist,original_history)

    def test_existing_output_prevents_subprocess_or_factory(self):
        with self.temp() as value, patch("subprocess.Popen",side_effect=AssertionError("must not launch")), redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--workspace",".","--factory","x:y","--output",value]),2)

    def test_all_supported_cache_paths_stay_in_requested_directory(self):
        with self.temp() as value:
            env=runtime_environment(value,"0")
            for key in ("HF_HOME","HF_MODULES_CACHE","MPLCONFIGDIR","CUDA_CACHE_PATH","TRITON_CACHE_DIR","TMPDIR"):
                self.assertTrue(Path(env[key]).is_relative_to(Path(value).resolve()))
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"],"0")

    def test_nonfinite_actions_and_shape_changes_rejected(self):
        with self.assertRaises(ValueError):
            actions_array([[float('nan')]*7])
        with self.assertRaises(ValueError):
            difference([[0]*7],[[0]*7]*2)

    def test_missing_thresholds_never_accept_quality(self):
        d=[dict(mae=0.0,max_abs=0.0,gripper_sign_agreement=1.0)]
        limits=dict(max_action_mae=None,max_action_error=None,min_gripper_agreement=None,min_speedup=None)
        self.assertEqual(screen(d,1.2,limits)["decision"],"INCONCLUSIVE")
        limits.update(max_action_mae=.01,max_action_error=.1,min_gripper_agreement=1.0,min_speedup=1.0)
        result=screen(d,1.2,limits)
        self.assertEqual(result["decision"],"INPUT_SCREEN_PASS")
        self.assertFalse(result["closed_loop_quality_accepted"])
        self.assertEqual(screen(d,.9,limits)["decision"],"INPUT_SCREEN_FAIL")

    def test_duplicate_observations_are_rejected(self):
        bundle=self.bundle()
        bundle["observations"][0]["id"]="duplicate"
        bundle["observations"]*=2
        with self.assertRaisesRegex(ValueError,"unique"):
            validate_bundle(bundle)


if __name__=="__main__":
    unittest.main()
