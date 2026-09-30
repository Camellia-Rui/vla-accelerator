#!/usr/bin/env python3
"""Validate a VLA acceleration evidence manifest."""

from __future__ import annotations

import argparse
import json
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
    "evidence.state_reset_valid",
    "evidence.weights_unchanged",
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


def validate(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for dotted in REQUIRED_PATHS:
        try:
            value = get_path(data, dotted)
        except KeyError:
            errors.append(f"missing field: {dotted}")
            continue
        if value is None:
            errors.append(f"null mandatory field: {dotted}")

    if data.get("schema_version") != 2:
        errors.append("schema_version must be 2")

    state_type = data.get("module", {}).get("state_type")
    if state_type not in {"stateless", "stateful"}:
        errors.append("module.state_type must be stateless or stateful")

    evidence = data.get("evidence", {})
    thresholds = data.get("thresholds", {})
    for field in ("paired_samples",):
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

    if state_type == "stateful" and not data.get("adapter", {}).get("reset_hook"):
        errors.append("stateful module requires adapter.reset_hook")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = validate(data)
    print(json.dumps({"valid": not errors, "errors": errors}, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
