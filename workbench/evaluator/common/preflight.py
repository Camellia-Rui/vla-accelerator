"""No-model configuration and source readiness inspection for both target methods."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import subprocess
import sys

PINNED_VLA_PRUNER = "84d4b7192c77abf1585610e2f12393319b7ebff9"
STAGES = {
    "01": "paired real fixed-observation timing, action fidelity and actual mechanism",
    "02": "fresh-environment paired closed-loop smoke, no errors, reset audit",
    "03": "multitask paired confirmation with overall and per-task confidence intervals",
    "04": "conditional adjudication with predeclared trigger and unseen initial states",
    "05": "active-wrapper parameter value audit and post-detach topology/output restoration",
}


def check(config, expected_method):
    if not isinstance(config, dict) or config.get("method") != expected_method:
        raise ValueError("Wrong method configuration")
    if expected_method not in ("specprune", "vla-pruner"):
        raise ValueError("Unsupported evaluator route")
    for name in ("task_suite", "device", "python"):
        if not isinstance(config.get(name), str) or not config[name]:
            raise ValueError(f"{name} must be explicit")
    if config.get("views") != 2 or config.get("action_dim") != 7 or config.get("chunk_size") != 8:
        raise ValueError("These target wrappers require the recorded two-view, 7-D, 8-action OFT contract")
    missing = []
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("paths must be a mapping")
    resolved = {}
    for name in ("vla_repository", "method_repository", "checkpoint", "libero_assets", "toolbox_src"):
        value = paths.get(name)
        if not isinstance(value, str) or not value:
            missing.append(name)
            continue
        path = Path(value)
        if not (path.is_absolute() or PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()):
            raise ValueError(f"{name} must be an absolute target-machine path")
        exists = path.is_dir() if path.is_absolute() else False
        resolved[name] = {"path": value, "exists": exists, "native_absolute_path": path.is_absolute()}
        if not exists:
            missing.append(name)
    backend = None
    source_identity = {}
    for name in ("vla_repository", "method_repository"):
        item = resolved.get(name, {})
        if item.get("exists"):
            try:
                head = subprocess.run(["git", "-C", item["path"], "rev-parse", "HEAD"],
                                      capture_output=True, text=True, timeout=10)
                dirty = subprocess.run(["git", "-C", item["path"], "status", "--short"],
                                       capture_output=True, text=True, timeout=10)
                source_identity[name] = {"head": head.stdout.strip() if head.returncode == 0 else None,
                                         "dirty": dirty.stdout.strip() if dirty.returncode == 0 else None}
            except OSError as exc:
                source_identity[name] = {"error": str(exc)}
    method_path = resolved.get("method_repository", {})
    if expected_method == "vla-pruner" and method_path.get("exists"):
        p = subprocess.run(["git", "-C", method_path["path"], "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        backend = p.stdout.strip() if p.returncode == 0 else None
        if backend != PINNED_VLA_PRUNER:
            missing.append("pinned_official_backend_revision")
        if source_identity.get("method_repository", {}).get("dirty"):
            missing.append("pinned_backend_has_unverified_local_changes")
    checkpoint = resolved.get("checkpoint", {})
    metadata = {}
    if checkpoint.get("exists"):
        for name in ("config.json", "dataset_statistics.json", "model.safetensors.index.json"):
            path = Path(checkpoint["path"]) / name
            if path.is_file():
                metadata[name] = {"size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            else:
                missing.append(f"checkpoint/{name}")
        index_path = Path(checkpoint["path"]) / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map or not all(isinstance(v, str) for v in weight_map.values()):
                raise ValueError("Invalid checkpoint shard index")
            metadata["shards"] = []
            for value in sorted(set(weight_map.values())):
                path = (Path(checkpoint["path"]) / value).resolve()
                if not path.is_relative_to(Path(checkpoint["path"]).resolve()):
                    raise ValueError("Checkpoint index escapes checkpoint directory")
                exists = path.is_file()
                metadata["shards"].append({"name": value, "exists": exists,
                                            "size": path.stat().st_size if exists else None})
                if not exists:
                    missing.append(f"checkpoint shard/{value}")
    packages = {}
    for name in ("torch", "transformers", "timm", "libero"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    python_matches = Path(config["python"]).resolve() == Path(sys.executable).resolve()
    if not python_matches:
        missing.append("selected_python_not_current_interpreter")
    return {"method": expected_method, "development_status": "INCOMPLETE_REAL_RUNTIME_VALIDATION",
            "config_parsed": True, "missing": missing, "paths": resolved, "backend_commit": backend,
            "checkpoint_metadata": metadata, "source_identity": source_identity,
            "packages_visible_to_current_python": packages,
            "actual_python": sys.executable, "selected_python_matches": python_matches,
            "runtime_compatible": None, "model_loaded": False, "gpu_experiment_started": False,
            "stages_pending": STAGES,
            "next_gate": "native real-observation probe after source/environment confirmation and authorization"}


def main(method):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, default="01")
    parser.add_argument("--dry-run", action="store_true", help="metadata only; this entry never runs models")
    args = parser.parse_args()
    try:
        report = check(json.loads(args.config.read_text(encoding="utf-8")), method)
        report["requested_stage"] = args.stage
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 2 if report["missing"] else 0
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"error": str(exc), "model_loaded": False}, ensure_ascii=True))
        return 2
