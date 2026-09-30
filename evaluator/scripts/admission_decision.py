#!/usr/bin/env python3
"""Make a deterministic admission decision from a validated manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from validate_manifest import validate


def decide(data: dict[str, Any]) -> dict[str, Any]:
    errors = validate(data)
    if errors:
        return {"decision": "INCONCLUSIVE", "failed_gates": [], "missing_or_invalid": errors}

    e = data["evidence"]
    t = data["thresholds"]
    correctness_failures: list[str] = []
    quality_failures: list[str] = []
    performance_failures: list[str] = []

    for field in (
        "structure_compatible",
        "control_matches_baseline",
        "state_reset_valid",
        "weights_unchanged",
    ):
        if e[field] is not True:
            correctness_failures.append(field)

    if e["paired_samples"] < t["minimum_paired_samples"]:
        quality_failures.append("paired_samples")
    if e["action_mae"] > t["maximum_action_mae"]:
        quality_failures.append("action_mae")
    if e["action_max_abs_difference"] > t["maximum_action_max_abs_difference"]:
        quality_failures.append("action_max_abs_difference")
    if e["gripper_agreement"] < t["minimum_gripper_agreement"]:
        quality_failures.append("gripper_agreement")

    closed_loop_drop = None
    if e.get("closed_loop_baseline_success_rate") is not None and e.get("closed_loop_accelerated_success_rate") is not None:
        closed_loop_drop = e["closed_loop_baseline_success_rate"] - e["closed_loop_accelerated_success_rate"]
        if closed_loop_drop > t["maximum_closed_loop_success_drop"]:
            correctness_failures.append("closed_loop_success_drop")

    if e["wall_speedup"] < t["minimum_wall_speedup"]:
        performance_failures.append("wall_speedup")
    if t["require_positive_paired_ci"] and (
        e["paired_mean_saved_ms_ci_low"] <= 0.0
        or e["paired_median_saved_ms_ci_low"] <= 0.0
    ):
        performance_failures.append("paired_speed_confidence")

    failed = correctness_failures + quality_failures + performance_failures
    if not failed:
        decision = "PASS"
    elif correctness_failures:
        decision = "REJECT"
    elif data["tuning_remaining"]:
        decision = "TUNE"
    else:
        decision = "REJECT"

    return {
        "decision": decision,
        "failed_gates": failed,
        "missing_or_invalid": [],
        "observations": {
            "wall_speedup": e["wall_speedup"],
            "closed_loop_success_drop": closed_loop_drop,
            "state_type": data["module"]["state_type"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = decide(data)
    print(json.dumps(result, indent=2 if args.pretty else None, ensure_ascii=False))
    return 0 if result["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
