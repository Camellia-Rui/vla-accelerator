#!/usr/bin/env python3
import argparse
from pathlib import Path
import shlex


def first_valid(candidates, marker):
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / marker).is_file():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cloud = Path("/cloud/cloud-ssd1")
    project_candidates = [
        cloud / "lightvla_workspace/repos/vla-cache",
        cloud / "lightvla_workspace/repos/openvla",
        cloud / "vla_cache_acceptance_2026-10-23/sources/vla-cache",
    ]
    for pattern in ("*/sources/*", "*/source/*", "lightvla_workspace/repos/*"):
        project_candidates.extend(cloud.glob(pattern))
    project = first_valid(project_candidates, "experiments/robot/libero/libero_utils.py")

    checkpoint_candidates = [
        cloud / "lightvla_workspace/models/LightVLA-libero-spatial",
        cloud / "lightvla_workspace/checkpoints/LightVLA-libero-spatial",
    ]
    checkpoint_candidates.extend((cloud / "lightvla_workspace/models").glob("*LightVLA*spatial*"))
    checkpoint_candidates.extend((cloud / "lightvla_workspace/checkpoints").glob("*LightVLA*spatial*"))
    checkpoint = first_valid(checkpoint_candidates, "config.json")

    python = cloud / "lightvla_workspace/envs/vla-cache/bin/python"
    missing = []
    if project is None:
        missing.append("OpenVLA project root containing experiments/robot/libero/libero_utils.py")
    if checkpoint is None:
        missing.append("LightVLA-libero-spatial checkpoint containing config.json")
    if not python.is_file():
        missing.append(str(python))
    if missing:
        raise SystemExit("Missing required paths: " + "; ".join(missing))

    values = {"PY": python, "PROJECT": project, "CHECKPOINT": checkpoint}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(f"{key}={shlex.quote(str(value))}\n" for key, value in values.items()),
        encoding="utf-8",
    )
    for key, value in values.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
