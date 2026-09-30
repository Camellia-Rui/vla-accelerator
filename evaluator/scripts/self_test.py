#!/usr/bin/env python3
"""Dependency-free behavioral checks for manifest validation and admission."""

from __future__ import annotations

from copy import deepcopy

from admission_decision import decide
from validate_manifest import validate


def valid_manifest() -> dict:
    return {
        "schema_version": 2,
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
            "state_reset_valid": True,
            "weights_unchanged": True,
            "paired_samples": 50,
            "wall_speedup": 1.08,
            "paired_mean_saved_ms_ci_low": 1.2,
            "paired_median_saved_ms_ci_low": 0.8,
            "action_mae": 0.01,
            "action_max_abs_difference": 0.15,
            "gripper_agreement": 1.0,
            "closed_loop_baseline_success_rate": 0.8,
            "closed_loop_accelerated_success_rate": 0.8,
            "raw_evidence": ["/results/raw.json"],
        },
        "thresholds": {
            "minimum_paired_samples": 20,
            "minimum_wall_speedup": 1.03,
            "maximum_action_mae": 0.05,
            "maximum_action_max_abs_difference": 0.5,
            "minimum_gripper_agreement": 0.95,
            "require_positive_paired_ci": True,
            "maximum_closed_loop_success_drop": 0.05,
            "require_closed_loop": True,
        },
        "tuning_remaining": False,
        "notes": [],
    }


def main() -> None:
    passing = valid_manifest()
    assert validate(passing) == []
    assert decide(passing)["decision"] == "PASS"

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
