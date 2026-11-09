"""Launch the team FastV Spatial backends in their model-specific environment."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
from statistics import mean
import subprocess
import sys

from core.runtime_env import runtime_environment


def openvla_summary(directory, expected):
    log, = directory.glob("EVAL*.txt")
    actions, = directory.glob("*-actions.csv")
    outcomes = re.findall(r"^Success: (True|False)$", log.read_text(), re.M)
    if len(outcomes) != expected:
        raise ValueError("Incomplete OpenVLA episode log")
    with actions.open() as stream:
        rows = list(csv.DictReader(stream))
    episodes = {(int(row["task_id"]), int(row["episode_idx"])) for row in rows}
    if episodes != {(task, trial) for task in range(10) for trial in range(expected // 10)}:
        raise ValueError("Incomplete OpenVLA action records")
    times = [1000 * float(row["latency_s"]) for row in rows]
    if len(times) <= 5 or not all(math.isfinite(t) and t > 0 for t in times):
        raise ValueError("Invalid OpenVLA latency records")
    return {"episodes": expected, "successes": outcomes.count("True"),
            "success_rate": outcomes.count("True") / expected,
            "latency_excluding_first_5_calls": {"count": len(times) - 5, "mean_ms": mean(times[5:])}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["openvla", "oft"])
    for name in ("workspace", "checkpoint", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--trials", type=int, default=1, help="initial states per task (1..50)")
    parser.add_argument("--tasks", type=int, nargs="+", help="OFT task IDs, default all ten")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cublas-dir", type=Path, help="validated cuBLAS 12.4 library directory")
    args = parser.parse_args(argv)
    workspace, checkpoint, output = (p.resolve() for p in (args.workspace, args.checkpoint, args.output))
    oft = args.backend == "oft"
    if not 1 <= args.trials <= 50 or args.gpu < 0:
        parser.error("invalid trial count or GPU")
    if args.tasks and (not oft or len(set(args.tasks)) != len(args.tasks) or not set(args.tasks) <= set(range(10))):
        parser.error("--tasks requires distinct OFT task IDs in 0..9")
    marker = "fastv_oft.py" if oft else "fastv_openvla.py"
    if not (workspace / "experiments/robot" / marker).is_file() or not (checkpoint / "config.json").is_file():
        parser.error("missing FastV backend or checkpoint config.json")
    if any(output.is_relative_to(p) for p in (workspace, checkpoint)):
        parser.error("--output must be outside model and backend")
    output.mkdir(parents=True, exist_ok=False)
    env = runtime_environment(output / "cache", args.gpu)
    env.update(FASTV_WORKSPACE=str(workspace), FASTV_CHECKPOINT=str(checkpoint),
               PYTHONPATH=os.pathsep.join([str(workspace), env.get("PYTHONPATH", "")]),
               MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID=str(args.gpu))
    if (workspace / ".libero-config/config.yaml").is_file():
        env["LIBERO_CONFIG_PATH"] = str(workspace / ".libero-config")
    if args.cublas_dir:
        libs = args.cublas_dir.resolve()
        preload = [libs / n for n in ("libcublasLt.so.12", "libcublas.so.12")]
        if not all(p.is_file() for p in preload):
            parser.error("missing cuBLAS libraries")
        env["LD_PRELOAD"] = os.pathsep.join(map(str, preload))
        site = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages/nvidia"
        env["LD_LIBRARY_PATH"] = os.pathsep.join(map(str, [libs, site/"cudnn/lib", site/"cuda_runtime/lib",
                                                           Path(sys.prefix)/"lib", env.get("LD_LIBRARY_PATH", "")]))
    if oft:
        # OFT rewrites config and synchronizes two Python files. Isolate those writes.
        view = output / "checkpoint"
        view.mkdir()
        for path in checkpoint.iterdir():
            if path.name.startswith(".") or ".back." in path.name:
                continue
            if path.name == "config.json" or path.suffix == ".py":
                shutil.copy2(path, view / path.name)
            else:
                (view / path.name).symlink_to(path, target_is_directory=path.is_dir())
        env["FASTV_CHECKPOINT"] = str(view)
    runner = Path(__file__).with_name(args.backend + "_runner.py").resolve()
    for mode in ("baseline", "fastv"):
        directory = output / mode
        directory.mkdir()
        command = [sys.executable, "-B", str(runner), "--mode", mode, "--trials", str(args.trials),
                   "--seed", str(args.seed)]
        if oft:
            command += ["--gpu", str(args.gpu)]
            if args.tasks:
                command += ["--tasks", *map(str, args.tasks)]
        subprocess.run(command, cwd=workspace if oft else directory, check=True,
                       env=dict(env, FASTV_OUTPUT=str(directory)))
    summarize(output, args.backend, workspace, checkpoint, args.tasks, args.trials, args.seed)
    print(output / "report.json")
    return 0



def summarize(output, backend, workspace, checkpoint, tasks, trials, seed):
    oft = backend == "oft"
    results = {}
    for mode in ("baseline", "fastv"):
        directory = output / mode
        results[mode] = (json.loads((directory / "summary.json").read_text()) if oft
                         else openvla_summary(directory, 10 * trials))
    if oft:
        paired = []
        for mode in results:
            with (output / mode / "episodes.csv").open() as stream:
                paired.append({(row["task_id"], row["episode_id"], row["initial_state_sha256"])
                               for row in csv.DictReader(stream)})
        if paired[0] != paired[1] or len(paired[0]) != trials * len(tasks or range(10)):
            raise ValueError("OFT initial-state pairs differ or are incomplete")
    report = dict(backend=backend, checkpoint=str(checkpoint), workspace=str(workspace),
                  integration_passed=True, closed_loop_tested=True,
                  seed=seed, trials_per_task=trials, tasks=tasks or list(range(10)),
                  k=2, prune_ratio=0.5, quality_decision="INCONCLUSIVE", results=results)
    base = results["baseline"]
    fast = results["fastv"]
    report["success_rate_change_pp"] = 100 * (fast["success_rate"] - base["success_rate"])
    timings = [result["latency_excluding_first_5_calls"] for result in (base, fast)]
    report["mean_inference_speedup"] = (
        timings[0]["mean_ms"] / timings[1]["mean_ms"] if all(timings) else None)
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(output / "report.json")
