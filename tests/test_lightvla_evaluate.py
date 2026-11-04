import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.run import main
from methods.lightvla.evaluation import main as pair_main


class LightVLAEvaluationTests(unittest.TestCase):
    def test_unified_routes_both_protocols_without_loading_model(self):
        for flag, target in (([], "evaluation"), (["--closed-loop"], "closed_loop")):
            with patch(f"methods.lightvla.{target}.main", return_value=0) as run:
                self.assertEqual(main(["--method", "lightvla", *flag, "--help"], evaluate=True), 0)
                run.assert_called_once_with(["--help"])

    def test_child_uses_absolute_paths_and_isolated_cache(self):
        with tempfile.TemporaryDirectory(dir=".") as root:
            base = Path(root)
            output = base / "result"
            with patch("methods.lightvla.evaluation.subprocess.run", return_value=SimpleNamespace(returncode=0)) as run:
                self.assertEqual(pair_main(["--workspace", str(base), "--checkpoint", str(base),
                                           "--output", str(output), "--gpu", "2"]), 0)
                command = run.call_args.args[0]
                self.assertIn(str(output.resolve()), command)
                env = run.call_args.kwargs["env"]
                self.assertTrue(Path(env["HF_HOME"]).is_relative_to(output.resolve()))
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
            with patch("subprocess.run") as run:
                with self.assertRaises(FileExistsError):
                    pair_main(["--workspace", str(base), "--checkpoint", str(base),
                               "--output", str(output), "--gpu", "2"])
                run.assert_not_called()
