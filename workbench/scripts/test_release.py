"""End-user tamper rejection tests; operate only on disposable copies."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class ReleaseTests(unittest.TestCase):
    def check_failure(self, kind):
        source = Path(__file__).resolve().parents[1]
        # Keep disposable writes within this authorized workspace and outside the release.
        with tempfile.TemporaryDirectory(prefix="release-check-", dir=source.parent) as directory:
            root = Path(directory) / "bundle"
            shutil.copytree(source, root)
            if kind == "modified":
                with (root / "PROJECT_STATUS.md").open("a", encoding="utf-8") as stream:
                    stream.write("\nUnexpected modification\n")
            else:
                (root / "unexpected.txt").write_text("not manifested", encoding="utf-8")
            result = subprocess.run([sys.executable, "-B", str(root / "scripts/verify_bundle.py")],
                                    capture_output=True, text=True, errors="replace")
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            expected = "hash mismatch" if kind == "modified" else "unmanifested file"
            self.assertIn(expected, result.stdout)

    def test_modified_payload_is_rejected(self):
        self.check_failure("modified")

    def test_unmanifested_payload_is_rejected(self):
        self.check_failure("extra")


if __name__ == "__main__":
    unittest.main()
