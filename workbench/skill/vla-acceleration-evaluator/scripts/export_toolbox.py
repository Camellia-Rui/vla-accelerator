#!/usr/bin/env python3
"""Copy the bundled accelerator toolbox to a new experiment directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    source = Path(__file__).resolve().parents[1] / "assets" / "toolbox"
    destination = args.destination.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Bundled toolbox is missing: {source}")
    if destination.exists():
        raise FileExistsError(
            f"Destination already exists: {destination}. Choose a new directory so existing work is not overwritten."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    print(f"VLA_ACCELERATION_TOOLBOX_EXPORTED: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
