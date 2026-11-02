import json
from pathlib import Path
import sys
import tempfile
import unittest

from core.toolbox import inspect_workspace, write_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_product import verify
import hashlib


class ProductTests(unittest.TestCase):
    def temp(self):
        # Local scratch avoids platform-specific locked global temp directories.
        return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)

    def test_discovery_never_imports_and_excludes_assets(self):
        with self.temp() as value:
            root = Path(value)
            (root / "policy.py").write_text("raise RuntimeError('MUST NOT IMPORT')\n# openvla predict_action", encoding="utf-8")
            (root / "models").mkdir()
            (root / "models/secret.py").write_text("specprune", encoding="utf-8")
            result = inspect_workspace(root)
            self.assertEqual(result["candidates_for_review"], ["lightvla", "vla_pruner"])
            self.assertFalse(result["compatibility_verified"])
            self.assertNotIn("specprune", result["source_signals"])
            self.assertEqual(result["methods"]["vla_cache"]["status"], "quarantined")

    def test_checkpoint_paths_resolve_against_user_workspace(self):
        with self.temp() as value:
            root = Path(value)
            (root / "weights").mkdir()
            (root / "toolbox-project.json").write_text('{"checkpoint":"weights"}', encoding="utf-8")
            self.assertTrue(inspect_workspace(root)["checkpoint"]["exists"])

    def test_reports_escape_source_and_refuse_overwrite(self):
        with self.temp() as value:
            root = Path(value)
            report = inspect_workspace(root)
            report["project_config"] = {"value": "<script>alert(1)</script>"}
            write_report(report, root / "output")
            self.assertNotIn("<script>", (root / "output/report.html").read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                write_report(report, root / "output")

    def test_release_rejects_tampering_unknown_files_and_escape(self):
        with self.temp() as value:
            root = Path(value)
            payload = root / "README.md"
            payload.write_text("original", encoding="utf-8")
            manifest = {"files": [{"path": "README.md", "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}]}
            (root / "product-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(verify(root), [])
            payload.write_text("modified", encoding="utf-8")
            self.assertTrue(any("changed" in error for error in verify(root)))
            (root / "extra.txt").touch()
            self.assertTrue(any("unexpected" in error for error in verify(root)))
            manifest["files"][0]["path"] = "../outside.txt"
            (root / "product-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(any("invalid" in error for error in verify(root)))


if __name__ == "__main__":
    unittest.main()
