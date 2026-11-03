"""Compatibility wrapper for the unified LightVLA/VLA-Pruner inference entry."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from core.run import main as run
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
