"""Resource checks and reproducible execution plans; never infer compatibility from names."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

PROFILES = {"oft_object": "vla_pruner", "lightvla_spatial": "lightvla"}


def read_project(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Expected project schema_version=1")
    allowed = {"schema_version", "profile", "workspace", "checkpoint", "python", "gpu", "observation_dirs", "output_root"}
    if data.keys() - allowed:
        raise ValueError("Unknown project fields: " + str(sorted(data.keys() - allowed)))
    for key in ("workspace", "checkpoint", "python", "output_root"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing project field: {key}")
        p = Path(value)
        data[key] = str((p if p.is_absolute() else path.parent / p).resolve())
    observations = data.get("observation_dirs", [])
    if not isinstance(observations, list) or any(not isinstance(p, str) for p in observations):
        raise ValueError("observation_dirs must be a list of paths")
    data["observation_dirs"] = [str((Path(p) if Path(p).is_absolute() else path.parent / p).resolve()) for p in observations]
    if not str(data.get("gpu", "")).isdigit():
        raise ValueError("gpu must be a physical GPU index")
    return data


def plan(data, name):
    if not name or Path(name).name != name or name in (".", "..") or any(c in name for c in '/\\:'):
        raise ValueError("run name must be one directory name")
    issues = []
    profile = data.get("profile")
    method = PROFILES.get(profile)
    if method is None:
        issues.append("Unsupported profile: original OpenVLA and unknown variants require adaptation")
    for key in ("workspace", "checkpoint"):
        if not Path(data[key]).is_dir():
            issues.append(f"Missing directory: {key}={data[key]}")
    if not Path(data["python"]).is_file():
        issues.append("Model Python executable is missing")
    cp = Path(data["checkpoint"])
    from core.project_checks import weights
    issues.extend(weights(cp))
    stats = {}
    try:
        config = json.loads((cp / "config.json").read_text())
        stats = config.get("norm_stats", {})
        if (cp / "dataset_statistics.json").is_file():
            stats = json.loads((cp / "dataset_statistics.json").read_text())
        if not isinstance(stats, dict):
            raise ValueError("Normalization statistics must be an object")
    except (OSError, ValueError, AttributeError) as exc:
        issues.append(f"Cannot inspect checkpoint metadata: {exc}")
    key = "libero_object_no_noops" if profile == "oft_object" else "libero_spatial_no_noops"
    if profile == "lightvla_spatial" and key not in stats:
        key = "libero_spatial"
    if key not in stats:
        issues.append(f"Missing normalization key: {key}")
    if not (cp / "tokenizer_config.json").is_file():
        issues.append("Missing tokenizer_config.json")
    if not any(cp.glob("*.safetensors")) and not any(cp.glob("pytorch_model*.bin")):
        issues.append("No local weight files found")
    if not (Path(data["workspace"]) / "experiments/robot/openvla_utils.py").is_file():
        issues.append("Missing native model utilities")
    if profile == "oft_object":
        if not data["observation_dirs"]:
            issues.append("OFT evaluation requires captured observations")
        for p in data["observation_dirs"]:
            if not all((Path(p) / n).is_file() for n in ("report.json", "observation.npz")):
                issues.append(f"Incomplete captured input: {p}")
    output = Path(data["output_root"]) / name
    if output.exists():
        issues.append("Output already exists; choose a new run name")
    command = [data["python"], "-B", "-m", "core", "evaluate", "--method", method or "unsupported",
               "--workspace", data["workspace"], "--checkpoint", data["checkpoint"],
               "--gpu", str(data["gpu"]), "--output", str(output)]
    if profile == "oft_object":
        command += ["--unnorm-key", key, "--prune-ratio", "0.5", "--prefer-env-cuda"]
        for p in data["observation_dirs"]:
            command += ["--observation-dir", p]
    return dict(schema_version=1, profile=profile, candidate_method=method,
                status="BLOCKED" if issues else "READY_FOR_RUNTIME_CHECK", blockers=issues,
                compatibility_verified=False, quality_decision="INCONCLUSIVE",
                reason="Explicit profile plus resource checks; runtime backend, structure and restoration checks remain mandatory",
                alternatives={"lightvla_on_ordinary_oft": "NOT_RECOMMENDED: observed Object quality failure"},
                command=command, output=str(output))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("init", "check", "execute"))
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", default="evaluation-001")
    parser.add_argument("--probe-env", action="store_true", help="inspect selected Python package metadata without model/GPU imports")
    args = parser.parse_args(argv)
    try:
        if args.operation == "init":
            template = dict(schema_version=1, profile="unknown", workspace="model-project", checkpoint="weights",
                            python="model-env/bin/python", gpu="0", observation_dirs=[], output_root="results")
            with args.project.open("x", encoding="utf-8") as f:
                json.dump(template, f, indent=2)
            print("Created template; replace paths and confirm profile before execution")
            return 0
        data = read_project(args.project)
        result = plan(data, args.name)
        if args.probe_env or args.operation == "execute":
            from core.project_checks import environment
            if Path(data["python"]).is_file():
                result["environment"] = environment(data["python"])
                if result["environment"].get("error") or result["environment"].get("missing"):
                    result["blockers"].append("Model environment probe failed or required packages are missing")
                    result["status"] = "BLOCKED"
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if result["blockers"]:
            return 2
        if args.operation == "execute":
            output = Path(result["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            # Existing evaluators reserve output and isolate model caches in their child.
            execution = subprocess.run(result["command"], cwd=Path(__file__).resolve().parents[1])
            if output.is_dir():
                result["exit_code"] = execution.returncode
                result["status"] = "EVALUATION_FINISHED" if execution.returncode == 0 else "EVALUATION_FAILED"
                report_path = output / "report.json"
                if report_path.is_file():
                    report = json.loads(report_path.read_text())
                    result["integration_passed"] = report.get("integration_passed", report.get("passed", False))
                    result["input_screen"] = report.get("input_screen")
                    result["closed_loop_tested"] = report.get("closed_loop_tested", False)
                elif execution.returncode == 0:
                    raise ValueError("Evaluator exited zero without report.json")
                with (output / "project-plan.json").open("x", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            return execution.returncode
        return 0
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
