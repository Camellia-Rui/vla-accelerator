"""Run/evaluate the pinned OFT backend or an explicit user factory, in an isolated process."""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import os
import subprocess
import sys
import traceback

from .accelerator import AcceleratorConfig, VLAPrunerAccelerator


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True, help="model source import root")
    p.add_argument("--checkpoint", type=Path, help="local weights; relative to workspace")
    p.add_argument("--observation-dir", type=Path, action="append", default=[], help="repeat for multiple captured inputs")
    p.add_argument("--factory", help="module:function returning model/resources/observations")
    p.add_argument("--factory-kwargs", default="{}")
    p.add_argument("--config", type=Path, help="JSON AcceleratorConfig")
    p.add_argument("--unnorm-key", help="checkpoint normalization key")
    p.add_argument("--prune-ratio", type=float, help="fraction removed, in [0,1)")
    p.add_argument("--mode", choices=("baseline", "accelerated"), default="accelerated")
    p.add_argument("--output", type=Path, required=True, help="new run directory")
    p.add_argument("--cache-dir", type=Path, help="all supported caches/temp dirs; default output/cache")
    p.add_argument("--gpu", help="single physical GPU index; otherwise preserve caller visibility")
    p.add_argument("--prefer-env-cuda", action="store_true", help="Linux: prioritize current environment cuDNN/cuBLAS")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--pairs", type=int, default=6)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--max-action-mae", type=float)
    p.add_argument("--max-action-error", type=float)
    p.add_argument("--min-gripper-agreement", type=float)
    p.add_argument("--min-speedup", type=float)
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return p


def read_config(args):
    values = json.loads(args.config.read_text(encoding="utf-8-sig")) if args.config else {}
    cfg = AcceleratorConfig.from_mapping(values)
    updates = {"use_vla_pruner": args.mode == "accelerated"}
    if args.checkpoint is not None:
        path = args.checkpoint if args.checkpoint.is_absolute() else args.workspace / args.checkpoint
        updates["checkpoint"] = str(path.resolve())
    elif cfg.checkpoint is not None:
        path = Path(cfg.checkpoint)
        updates["checkpoint"] = str((path if path.is_absolute() else args.workspace / path).resolve())
    if args.unnorm_key is not None:
        updates["unnorm_key"] = args.unnorm_key
    if args.prune_ratio is not None:
        updates["fastv_r"] = args.prune_ratio
    cfg = replace(cfg, **updates)
    if not args.factory and not cfg.unnorm_key:
        raise ValueError("Built-in loader requires explicit --unnorm-key (or config.unnorm_key)")
    return cfg


def validate_bundle(bundle):
    for key in ("model", "processor", "action_head", "observations"):
        if key not in bundle or bundle[key] is None:
            raise ValueError(f"factory bundle missing {key}")
    if getattr(bundle["model"], "training", True):
        raise ValueError("factory must return model.eval()")
    observations = bundle["observations"]
    if not isinstance(observations, list) or not observations:
        raise ValueError("observations must be a non-empty list")
    ids = []
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict) or "full_image" not in observation:
            raise ValueError("Each observation requires full_image")
        if not isinstance(observation.get("instruction"), str) or not observation["instruction"].strip():
            raise ValueError("Each observation requires a nonempty instruction")
        ids.append(str(observation.get("id", index)))
    if len(set(ids)) != len(ids):
        raise ValueError("Observation identifiers must be unique")


def worker(args, cfg):
    from .runtime import load_bundle
    from .evaluation import query, evaluate
    report = {"method": "vla_pruner", "operation": "evaluate" if args.evaluate else "run",
              "passed": False, "config": asdict(cfg), "python": sys.executable,
              "workspace": str(args.workspace.resolve()), "records": []}
    exit_code = 2
    try:
        bundle = load_bundle(args, cfg)
        validate_bundle(bundle)
        import importlib.metadata
        report["environment"] = {"gpu_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cache_root": os.environ.get("HF_HOME"),
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")}}
        report["provenance"] = bundle.get("provenance", {"source": "user_factory", "factory": args.factory})
        if args.evaluate:
            thresholds = {key: getattr(args, key) for key in
                          ("max_action_mae", "max_action_error", "min_gripper_agreement", "min_speedup")}
            evaluation = evaluate(bundle, cfg, args.output, args.pairs, args.warmup, thresholds)
            report.update(integration_passed=evaluation["integration_passed"], input_screen=evaluation["input_screen"],
                          evaluation_file="evaluation.json", closed_loop_tested=False)
            exit_code = 3 if evaluation["input_screen"]["decision"] == "INPUT_SCREEN_FAIL" else 0
        else:
            session = None
            try:
                session = VLAPrunerAccelerator.install(bundle["model"], config=cfg,
                    **{key: bundle[key] for key in ("processor", "action_head", "proprio_projector", "action_runner") if key in bundle})
                for i, observation in enumerate(bundle["observations"]):
                    session.reset()
                    actions = query(session, observation)
                    row = {"id": str(observation.get("id", i)), "actions": actions.tolist(),
                           "action_shape": list(actions.shape), "stats_during_query": session.get_stats()}
                    report["records"].append(row)
                    with (args.output / "actions.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
            finally:
                if session is not None:
                    session.detach()
                    report["detached"] = True
            exit_code = 0
        report["passed"] = exit_code == 0
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
        print(report["traceback"], file=sys.stderr)
    finally:
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output / "report.json"), "exit_code": exit_code}, ensure_ascii=False), flush=True)
    return exit_code


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    try:
        cfg = read_config(args)
        if args.evaluate and cfg.fastv_r == 0:
            raise ValueError("Evaluation of pruning mechanism requires a positive prune ratio")
        if args.pairs < 2 or args.warmup < 1:
            raise ValueError("pairs >= 2 and warmup >= 1 required")
        for key in ("max_action_mae", "max_action_error", "min_gripper_agreement", "min_speedup"):
            value = getattr(args, key)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{key} must be finite and nonnegative")
        if args.min_gripper_agreement is not None and args.min_gripper_agreement > 1:
            raise ValueError("min-gripper-agreement must be <= 1")
        args.output = args.output.resolve()
        if args._worker:
            return worker(args, cfg)
        if not args.workspace.is_dir():
            raise ValueError("workspace does not exist")
        # Reserve output before importing user code or loading the model.
        args.output.mkdir(parents=True, exist_ok=False)
        from core.runtime_env import runtime_environment
        env = runtime_environment(args.cache_dir or args.output / "cache", args.gpu, args.prefer_env_cuda)
        package_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = os.pathsep.join([package_root, env.get("PYTHONPATH", "")])
        command = [sys.executable, "-u", "-B", "-m", "methods.vla_pruner", *argv, "--_worker"]
        with (args.output / "process.log").open("x", encoding="utf-8") as log:
            child = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace")
            try:
                for line in child.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = child.wait()
            except KeyboardInterrupt:
                child.terminate()
                try:
                    code = child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    code = child.wait()
            finally:
                child.stdout.close()
        (args.output / "process.json").write_text(json.dumps({"exit_code": code, "command": command,
            "cache_dir": str(Path(args.cache_dir or args.output / "cache").resolve())}, indent=2), encoding="utf-8")
        return code if code >= 0 else 2
    except Exception as exc:
        print(json.dumps({"error": str(exc), "type": type(exc).__name__}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
