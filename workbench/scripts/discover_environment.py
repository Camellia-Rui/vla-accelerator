"""Read-only discovery: no model imports, downloads, SSH, installation or GPU computation."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=15, errors="replace")
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", default=[])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--task-assets", type=Path)
    parser.add_argument("--gpu-info", action="store_true", help="nvidia-smi metadata only")
    args = parser.parse_args(argv)
    report = {"scope": "read-only metadata, not runtime compatibility proof",
              "platform": platform.platform(), "python": sys.executable,
              "python_version": platform.python_version(), "cwd": str(Path.cwd()),
              "packages": {}, "repositories": [], "assets": {}}
    for name in ("torch", "transformers", "tokenizers", "numpy", "timm", "libero", "mujoco"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    for value in args.repo:
        path = Path(value).resolve()
        item = {"path": str(path), "exists": path.is_dir()}
        if path.is_dir() and shutil.which("git"):
            item["commit"] = command(["git", "-C", str(path), "rev-parse", "HEAD"])
            item["dirty"] = command(["git", "-C", str(path), "status", "--short"])
        report["repositories"].append(item)
    for name in ("checkpoint", "task_assets"):
        path = getattr(args, name)
        report["assets"][name] = {"path": str(path) if path else None,
            "exists": path.is_dir() if path else False, "validated": False}
        if path and path.is_dir():
            report["assets"][name]["top_level_files"] = sorted(p.name for p in path.iterdir())[:100]
    if args.gpu_info:
        report["gpu_metadata"] = command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    else:
        report["gpu_metadata"] = "NOT_REQUESTED"
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
