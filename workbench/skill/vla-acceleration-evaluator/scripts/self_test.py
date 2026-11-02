#!/usr/bin/env python3
"""Dependency-free checks for decisions and bundled source integrity."""

from __future__ import annotations

from copy import deepcopy
import ast
import hashlib
import json
from pathlib import Path

from admission_decision import decide
from validate_manifest import validate


def valid_manifest() -> dict:
    return {
        "schema_version": 3,
        "module": {
            "name": "temporary-pruner",
            "state_type": "stateless",
            "repository": "local",
            "commit": "0123456789abcdef",
        },
        "target": {
            "vla": "OpenVLA",
            "checkpoint": "/models/openvla",
            "task_suite": "libero_spatial",
            "hardware": "RTX 4090",
            "software": {"torch": "2.2.0", "transformers": "4.47.0"},
        },
        "adapter": {
            "attachment_point": "language_model.model prefill",
            "parameters": {"target_token_num": 224},
            "control_mode_tested": True,
            "reset_hook": None,
        },
        "evidence": {
            "structure_compatible": True,
            "control_matches_baseline": True,
            "mechanism_proven": True,
            "state_reset_valid": True,
            "weights_unchanged": True,
            "parameter_names_restored_after_detach": True,
            "episode_errors": 0,
            "paired_samples": 50,
            "wall_speedup": 1.08,
            "paired_mean_saved_ms_ci_low": 1.2,
            "paired_median_saved_ms_ci_low": 0.8,
            "action_mae": 0.01,
            "action_max_abs_difference": 0.15,
            "gripper_agreement": 1.0,
            "closed_loop_baseline_success_rate": 0.8,
            "closed_loop_accelerated_success_rate": 0.8,
            "closed_loop_per_task_max_success_drop": 0.0,
            "raw_evidence": [str(Path(__file__).resolve().parent / "fixtures" / "synthetic-only.json")],
        },
        "thresholds": {
            "minimum_paired_samples": 20,
            "minimum_wall_speedup": 1.03,
            "maximum_action_mae": 0.05,
            "maximum_action_max_abs_difference": 0.5,
            "minimum_gripper_agreement": 0.95,
            "require_positive_paired_ci": True,
            "maximum_closed_loop_success_drop": 0.05,
            "maximum_closed_loop_per_task_success_drop": 0.05,
            "require_closed_loop": True,
        },
        "tuning_remaining": False,
        "notes": [],
    }


def validate_bundled_specprune() -> None:
    skill_root = Path(__file__).resolve().parents[1]
    source = skill_root / "assets" / "toolbox" / "src" / "specprune" / "accelerator.py"
    readme = source.with_name("README.md")
    manifest_path = skill_root / "assets" / "toolbox" / "SOURCE_MANIFEST.json"
    assert source.is_file() and source.stat().st_size > 50_000
    assert readme.is_file() and readme.stat().st_size > 10_000

    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    definitions = {
        node.name for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required = {
        "AcceleratorConfig",
        "SpecPruneAccelerator",
        "get_final_precise_topk",
        "get_similarity_indices",
    }
    assert required <= definitions

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {item["package"]: item for item in manifest["implementations"]}
    assert entries["specprune"]["included"] is True
    for name, expected in entries["specprune"]["files"].items():
        actual = hashlib.sha256(source.with_name(name).read_bytes()).hexdigest().upper()
        assert expected.removeprefix("sha256:") == actual


def validate_bundled_vla_pruner() -> None:
    skill_root = Path(__file__).resolve().parents[1]
    package = skill_root / "assets" / "toolbox" / "src" / "vla_pruner"
    manifest_path = skill_root / "assets" / "toolbox" / "SOURCE_MANIFEST.json"
    accelerator = package / "accelerator.py"
    backend = package / "backend.py"
    lock_path = package / "backend.lock.json"

    assert accelerator.is_file() and accelerator.stat().st_size > 10_000
    assert backend.is_file() and backend.stat().st_size > 2_000
    tree = ast.parse(accelerator.read_text(encoding="utf-8"), filename=str(accelerator))
    definitions = {
        node.name for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"AcceleratorConfig", "VLAPrunerAccelerator"} <= definitions

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["repository"] == "https://github.com/MINT-SJTU/VLA-Pruner"
    assert len(lock["commit"]) == 40

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {item["package"]: item for item in manifest["implementations"]}
    entry = entries["vla_pruner"]
    assert entry["included"] is True
    assert entry["algorithm_source_included"] is False
    assert entry["provenance"]["backend_commit"] == lock["commit"]
    for name, expected in entry["files"].items():
        actual = hashlib.sha256((package / name).read_bytes()).hexdigest().upper()
        assert expected.removeprefix("sha256:") == actual


def validate_vla_cache_quarantine() -> None:
    skill_root = Path(__file__).resolve().parents[1]
    package = skill_root / "assets" / "vla-cache-openvla"
    reference = skill_root / "references" / "vla-cache.md"
    assert not package.exists()
    assert reference.is_file()
    text = reference.read_text(encoding="utf-8")
    assert "REJECTED for closed-loop deployment" in text
    assert "Do not repeat" in text


def validate_autoprune_retirement() -> None:
    skill_root = Path(__file__).resolve().parents[1]
    toolbox = skill_root / "assets" / "toolbox"
    assert not (toolbox / "src" / "vla_accelerators").exists()
    assert not (toolbox / "integrations" / "openvla").exists()
    for source in toolbox.rglob("*.py"):
        assert "autoprune" not in source.read_text(encoding="utf-8", errors="ignore").lower()

    manifest = json.loads((toolbox / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    entries = {item["package"]: item for item in manifest["implementations"]}
    assert entries["autoprune"]["included"] is False
    assert entries["autoprune"]["status"] == "permanently-retired"


def main() -> None:
    validate_bundled_specprune()
    validate_bundled_vla_pruner()
    validate_vla_cache_quarantine()
    validate_autoprune_retirement()
    passing = valid_manifest()
    assert validate(passing) == []
    assert decide(passing)["decision"] == "PASS"
    for field in ("wall_speedup", "action_mae", "paired_mean_saved_ms_ci_low"):
        for value in (float("nan"), float("inf"), True, "1"):
            invalid_number = deepcopy(passing)
            invalid_number["evidence"][field] = value
            assert decide(invalid_number)["decision"] == "INCONCLUSIVE"
    missing_raw = deepcopy(passing)
    missing_raw["evidence"]["raw_evidence"] = [str(Path(__file__).parent / "missing-evidence.json")]
    assert decide(missing_raw)["decision"] == "INCONCLUSIVE"
    no_control = deepcopy(passing)
    no_control["adapter"]["control_mode_tested"] = False
    assert decide(no_control)["decision"] == "REJECT"

    tuning = deepcopy(passing)
    tuning["evidence"]["wall_speedup"] = 1.01
    tuning["tuning_remaining"] = True
    assert decide(tuning)["decision"] == "TUNE"

    rejected = deepcopy(passing)
    rejected["evidence"]["control_matches_baseline"] = False
    assert decide(rejected)["decision"] == "REJECT"

    changed_weights = deepcopy(passing)
    changed_weights["evidence"]["weights_unchanged"] = False
    assert decide(changed_weights)["decision"] == "REJECT"

    broken_detach = deepcopy(passing)
    broken_detach["evidence"]["parameter_names_restored_after_detach"] = False
    assert decide(broken_detach)["decision"] == "REJECT"

    episode_error = deepcopy(passing)
    episode_error["evidence"]["episode_errors"] = 1
    assert decide(episode_error)["decision"] == "REJECT"

    noisy_speed = deepcopy(passing)
    noisy_speed["evidence"]["paired_mean_saved_ms_ci_low"] = -0.1
    noisy_speed["tuning_remaining"] = True
    assert decide(noisy_speed)["decision"] == "TUNE"

    invalid = deepcopy(passing)
    del invalid["target"]["checkpoint"]
    assert decide(invalid)["decision"] == "INCONCLUSIVE"

    stateful_without_reset = deepcopy(passing)
    stateful_without_reset["module"]["state_type"] = "stateful"
    assert "stateful module requires adapter.reset_hook" in validate(stateful_without_reset)

    print("VLA_ACCELERATION_EVALUATOR_SELF_TEST: PASS")


if __name__ == "__main__":
    main()
