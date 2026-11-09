import json
from pathlib import Path
import tempfile
import unittest

from methods.fastv.evaluation import summarize


class FastVReportTests(unittest.TestCase):
    def test_different_initial_states_cannot_be_reported_as_paired(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            output = Path(temp)
            for mode in ("baseline", "fastv"):
                directory = output / mode
                directory.mkdir()
                (directory / "summary.json").write_text(json.dumps({"success_rate": 1}))
                (directory / "episodes.csv").write_text(
                    f"task_id,episode_id,initial_state_sha256\n0,0,{mode}\n")
            with self.assertRaisesRegex(ValueError, "initial-state pairs"):
                summarize(output, "oft", "workspace", "checkpoint", [0], 1, 7)
            self.assertFalse((output / "report.json").exists())

