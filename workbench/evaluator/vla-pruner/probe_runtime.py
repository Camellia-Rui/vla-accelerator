"""Metadata preflight only; native runtime probe awaits the target server stack."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from preflight import main

if __name__ == "__main__":
    raise SystemExit(main("vla-pruner"))

