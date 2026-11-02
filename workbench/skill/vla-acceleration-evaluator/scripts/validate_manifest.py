#!/usr/bin/env python3
"""Validate a VLA acceleration evidence manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_PATHS = (
    "schema_version",
    "module.name",
    "module.state_type",
    "module.commit",
    "target.vla",
    "target.checkpoint",
    "target.task_suite",
    "target.hardware",
    "adapter.attachment_point",
    "adapter.control_mode_tested",
    "evidence.structure_compatible",
    "evidence.control_matches_baseline",
    "evidence.mechanism_proven",
    "evidence.state_reset_valid",
    "evidence.weights_unchanged",
    "evidence.parameter_names_restored_after_detach",
    "evidence.episode_errors",
    "evidence.paired_samples",
    "evidence.wall_speedup",
    "evidence.paired_mean_saved_ms_ci_low",
    "evidence.paired_median_saved_ms_ci_low",
    "evidence.action_mae",
    "evidence.action_max_abs_difference",
    "evidence.gripper_agreement",
    "evidence.raw_evidence",
    "thresholds.minimum_paired_samples",
    "thresholds.minimum_wall_speedup",
    "thresholds.maximum_action_mae",
    "thresholds.maximum_action_max_abs_difference",
    "thresholds.minimum_gripper_agreement",
    "thresholds.require_positive_paired_ci",
    "thresholds.maximum_closed_loop_success_drop",
    "evidence.closed_loop_per_task_max_success_drop",
    "thresholds.maximum_closed_loop_per_task_success_drop",
    "thresholds.require_closed_loop",
    "tuning_remaining",
)


def get_path(data: dict[str, Any], dotted: str) -> Any:
    value: Any = data
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted)
        value = value[part]
    return value


def preflight(data: Any, evidence_root: Path | None) -> list[str]:
    if not isinstance(data, dict):
        return ["manifest must be an object"]
    errors = []
    for section in ("module", "target", "adapter", "evidence", "thresholds"):
        if not isinstance(data.get(section), dict):
            errors.append(f"{section} must be an object")
    if errors:
        return errors
    e, t, a = data["evidence"], data["thresholds"], data["adapter"]
    boolean_fields = [(a, "control_mode_tested"), (data, "tuning_remaining")]
    boolean_fields += [(e, k) for k in ("structure_compatible", "control_matches_baseline", "mechanism_proven",
        "state_reset_valid", "weights_unchanged", "parameter_names_restored_after_detach")]
    boolean_fields += [(t, k) for k in ("require_positive_paired_ci", "require_closed_loop")]
    for obj, k in boolean_fields:
        if k in obj and type(obj[k]) is not bool:
            errors.append(f"{k} must be boolean")
    for obj, k in [(e, "paired_samples"), (e, "episode_errors"), (t, "minimum_paired_samples")]:
        if k in obj and (type(obj[k]) is not int or obj[k] < 0):
            errors.append(f"{k} must be a nonnegative integer")
    for obj, names in ((e, ("wall_speedup", "paired_mean_saved_ms_ci_low", "paired_median_saved_ms_ci_low",
        "action_mae", "action_max_abs_difference", "gripper_agreement", "closed_loop_baseline_success_rate",
        "closed_loop_accelerated_success_rate", "closed_loop_per_task_max_success_drop")),
        (t, ("minimum_wall_speedup", "maximum_action_mae", "maximum_action_max_abs_difference",
        "minimum_gripper_agreement", "maximum_closed_loop_success_drop", "maximum_closed_loop_per_task_success_drop"))):
        for k in names:
            if k in obj and (type(obj[k]) not in (int, float) or not math.isfinite(obj[k])):
                errors.append(f"{k} must be a finite number")
    if errors:
        return errors
    for k in ("action_mae", "action_max_abs_difference"):
        if k in e and e[k] < 0:
            errors.append(f"{k} must be nonnegative")
    if "wall_speedup" in e and e["wall_speedup"] <= 0:
        errors.append("wall_speedup must be positive")
    for k in ("maximum_action_mae", "maximum_action_max_abs_difference"):
        if k in t and t[k] < 0:
            errors.append(f"{k} must be nonnegative")
    for k in ("minimum_gripper_agreement", "maximum_closed_loop_success_drop", "maximum_closed_loop_per_task_success_drop"):
        if k in t and not 0 <= t[k] <= 1:
            errors.append(f"{k} must be between 0 and 1")
    if "minimum_wall_speedup" in t and t["minimum_wall_speedup"] <= 0:
        errors.append("minimum_wall_speedup must be positive")
    paths = e.get("raw_evidence")
    if isinstance(paths, list):
        for value in paths:
            if isinstance(value, str) and value:
                p = Path(value)
                if not p.is_absolute():
                    p = (evidence_root or Path.cwd()) / p
                if not p.is_file():
                    errors.append(f"raw evidence file missing: {value}")
    return errors


def validate(data: dict[str, Any], evidence_root: Path | None = None) -> list[str]:
    errors = preflight(data, evidence_root)
    if errors:
        return errors
    errors: list[str] = []
    for dotted in REQUIRED_PATHS:
        try:
            value = get_path(data, dotted)
        except KeyError:
            errors.append(f"missing field: {dotted}")
            continue
        if value is None:
            errors.append(f"null mandatory field: {dotted}")

    if data.get("schema_version") != 3:
        errors.append("schema_version must be 3")

    state_type = data.get("module", {}).get("state_type")
    if state_type not in {"stateless", "stateful"}:
        errors.append("module.state_type must be stateless or stateful")

    evidence = data.get("evidence", {})
    thresholds = data.get("thresholds", {})
    for field in ("paired_samples", "episode_errors"):
        value = evidence.get(field)
        if value is not None and (not isinstance(value, int) or value < 0):
            errors.append(f"evidence.{field} must be a non-negative integer")
    for field in (
        "wall_speedup",
        "paired_mean_saved_ms_ci_low",
        "paired_median_saved_ms_ci_low",
        "action_mae",
        "action_max_abs_difference",
        "gripper_agreement",
        "closed_loop_per_task_max_success_drop",
    ):
        value = evidence.get(field)
        if value is not None and not isinstance(value, (int, float)):
            errors.append(f"evidence.{field} must be numeric")

    raw_evidence = evidence.get("raw_evidence")
    if raw_evidence is not None and (
        not isinstance(raw_evidence, list)
        or not raw_evidence
        or not all(isinstance(path, str) and path for path in raw_evidence)
    ):
        errors.append("evidence.raw_evidence must be a non-empty list of paths")

    if thresholds.get("require_closed_loop"):
        for field in ("closed_loop_baseline_success_rate", "closed_loop_accelerated_success_rate"):
            if evidence.get(field) is None:
                errors.append(f"closed-loop evidence required: evidence.{field}")

    for field in ("gripper_agreement", "closed_loop_baseline_success_rate", "closed_loop_accelerated_success_rate"):
        value = evidence.get(field)
        if value is not None and not 0 <= value <= 1:
            errors.append(f"evidence.{field} must be between 0 and 1")

    per_task_drop = evidence.get("closed_loop_per_task_max_success_drop")
    if per_task_drop is not None and not -1 <= per_task_drop <= 1:
        errors.append("evidence.closed_loop_per_task_max_success_drop must be between -1 and 1")

    if state_type == "stateful" and not data.get("adapter", {}).get("reset_hook"):
        errors.append("stateful module requires adapter.reset_hook")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = validate(data, args.manifest.resolve().parent)
    print(json.dumps({"valid": not errors, "errors": errors}, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
