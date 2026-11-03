import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "scripts/server_native_smoke.py"
spec = importlib.util.spec_from_file_location("server_native_smoke", path)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class GPUSelectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(index=0, free_mib=70000, utilization=99),
                     dict(index=1, free_mib=20000, utilization=0),
                     dict(index=2, free_mib=50000, utilization=44)]

    def test_auto_skips_insufficient_memory_and_prefers_less_busy_card(self):
        self.assertEqual(smoke.select_gpu(self.rows, "auto", 32000)["index"], 2)

    def test_explicit_selection_is_not_silently_redirected(self):
        with self.assertRaises(RuntimeError):
            smoke.select_gpu(self.rows, "1", 32000)
        self.assertEqual(smoke.select_gpu(self.rows, "0", 32000)["index"], 0)

    def test_no_gpu_or_insufficient_memory_stops(self):
        for rows in ([], self.rows):
            with self.assertRaises(RuntimeError):
                smoke.select_gpu(rows, "auto", 100000)


if __name__ == "__main__":
    unittest.main()
