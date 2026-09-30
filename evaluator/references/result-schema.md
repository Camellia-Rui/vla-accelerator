# Result manifest schema

Use JSON. Paths may point to external raw evidence files.

```json
{
  "schema_version": 2,
  "module": {
    "name": "example",
    "state_type": "stateless",
    "repository": "https://example/repo",
    "commit": "full-commit-id"
  },
  "target": {
    "vla": "OpenVLA",
    "checkpoint": "/path/to/checkpoint",
    "task_suite": "libero_spatial",
    "hardware": "RTX 4090",
    "software": {"torch": "2.2.0", "transformers": "4.47.0"}
  },
  "adapter": {
    "attachment_point": "language_model.model prefill",
    "parameters": {},
    "control_mode_tested": true,
    "reset_hook": null
  },
  "evidence": {
    "structure_compatible": true,
    "control_matches_baseline": true,
    "state_reset_valid": true,
    "weights_unchanged": true,
    "paired_samples": 50,
    "wall_speedup": 1.08,
    "paired_mean_saved_ms_ci_low": 1.2,
    "paired_median_saved_ms_ci_low": 0.8,
    "action_mae": 0.01,
    "action_max_abs_difference": 0.15,
    "gripper_agreement": 0.98,
    "closed_loop_baseline_success_rate": 0.8,
    "closed_loop_accelerated_success_rate": 0.8,
    "raw_evidence": ["/path/result.json"]
  },
  "thresholds": {
    "minimum_paired_samples": 20,
    "minimum_wall_speedup": 1.03,
    "maximum_action_mae": 0.05,
    "maximum_action_max_abs_difference": 0.5,
    "minimum_gripper_agreement": 0.95,
    "require_positive_paired_ci": true,
    "maximum_closed_loop_success_drop": 0.05,
    "require_closed_loop": true
  },
  "tuning_remaining": false,
  "notes": []
}
```

`state_reset_valid` means "not applicable and verified stateless" for stateless modules. For stateful modules it must represent an explicit reset test. Null measurement values are missing evidence, not zero.

`weights_unchanged` is mandatory evidence for a training-free claim. Compute a reproducible fingerprint over representative parameters before and after the experiment; a false value is a correctness failure.

`paired_mean_saved_ms_ci_low` and `paired_median_saved_ms_ci_low` are the lower bounds of paired confidence intervals for `baseline wall time - accelerated wall time`. When `require_positive_paired_ci` is true, both must be greater than zero. This prevents a noisy median speedup from passing admission.

`raw_evidence` must be a non-empty list of paths to raw JSON evidence. Keep thresholds in the manifest so the decision is auditable.

The decision script returns a decision, failed gates, missing fields, and observations.
