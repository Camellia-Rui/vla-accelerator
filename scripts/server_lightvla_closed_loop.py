"""Backward-compatible entry point; implementation ships with the Python package."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods.lightvla.closed_loop import main, worker, atomic, summarize

if __name__ == "__main__":
    raise SystemExit(main())
