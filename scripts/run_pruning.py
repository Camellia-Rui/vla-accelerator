"""Explicit factory entry for LightVLA; other methods are not silently routed here."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["lightvla"])
    args, rest = parser.parse_known_args()
    from methods.lightvla.cli import main as lightvla_main
    return lightvla_main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
