"""Assess an existing evidence manifest; this does not execute model benchmarks."""
from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    entry = Path(__file__).resolve().parents[1] / "workbench/skill/vla-acceleration-evaluator/scripts/admission_decision.py"
    sys.path.insert(0, str(entry.parent))
    runpy.run_path(str(entry), run_name="__main__")
