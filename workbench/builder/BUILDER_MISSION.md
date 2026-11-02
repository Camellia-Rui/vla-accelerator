# Active mission: two production-grade five-stage evaluators

> r5: development backlog, activated only by a current user request to develop.
> Initialization alone does not activate this mission or authorize GPU experiments.

## Objective

Build separate, directly runnable five-stage real evaluators for:

1. SpecPrune on its compatible OpenVLA-OFT runtime.
2. VLA-Pruner on its pinned official backend and compatible VLA runtime.

The final deliverable must reduce later user interaction: for a supported server stack, a
user supplies paths/configuration once and receives one experiment ZIP plus one pasteable
launch command. The evaluators may share infrastructure but must have separate adapters,
mechanism counters, compatibility probes, configuration, provenance, results, and verdicts.

## Existing assets to reuse

- `evaluator/lightvla/`: working structural reference for the five stages and remote launch.
- `skill/vla-acceleration-evaluator/`: authoritative evaluation workflow, schema-v3
  validators, admission decision logic, and remote experiment rules.
- `skill/.../assets/toolbox/src/specprune/`: bundled SpecPrune engineering wrapper.
- `skill/.../assets/toolbox/src/vla_pruner/`: bundled VLA-Pruner adapter plus pinned-backend
  metadata. The external backend remains the algorithmic authority.
- `templates/remote-experiment/`: background, PID, resume, status, and summary templates.

Do not replace these with AutoPrune, VLA-Cache, GRIDs, or OpenPI code.

## Required development sequence

### Phase A — provenance and runtime discovery

Record repository URLs/paths, commit hashes, dirty state, model/checkpoint identities,
environment versions, GPU/driver, task suite, official inference entry point, attention
backend, dtype, action head, proprio path, camera/token layout, and episode reset behavior.
Run one unaccelerated real observation before choosing an attachment point.

### Phase B — method-specific reversible adapters

Implement `attach/enable/reset/disable/detach/status` or equivalent lifecycle operations.
All monkey patches and hooks must be restored in `finally`. Baseline, retain-all/no-effect
control, and accelerated modes must be selectable in separate processes when in-process
restoration is not proven. Never mutate model weights or checkpoint files.

### Phase C — five stages per method

Create under `evaluator/specprune/` and `evaluator/vla-pruner/`:

1. `01-fixed-observation-r1`: paired real-input timing/action/mechanism screen.
2. `02-closed-loop-smoke-r1`: small paired smoke with fresh state and zero episode errors.
3. `03-multitask-paired-r1`: sufficient paired tasks/episodes with overall and per-task CI.
4. `04-adjudication-r1`: conditional targeted replication of ambiguous/worst cases.
5. `05-weight-audit-r1`: active-wrapper tensor equality plus post-detach topology restoration.

Add a method-level `README-ZH.md`, `launch-stage.sh`, `status-stage.sh`, provenance file,
configuration schema, and dependency-free tests. Follow the expected tree in the delivery
contract; version names may advance only when behavior changes are recorded.

### Phase D — validation

For each method, validate three distinct claims:

- **attachment correctness:** the intended real model code path is affected;
- **mechanism truth:** expensive work is skipped before computation, with observed counters;
- **outcome acceptability:** paired closed-loop quality and end-to-end latency meet recorded
  thresholds without errors, weight changes, or state leakage.

Run the bundled schema validator and deterministic decision tool. PASS, TUNE, REJECT, and
INCONCLUSIVE are all legitimate outcomes. Do not tune after a correctness failure without
first identifying and retesting the cause.

### Phase E — packaging

Each remote experiment must contain all temporary Python/shell code needed to run, a Chinese
README, source manifest and hashes, expected output paths, background launch/status scripts,
record-level resume, retry of errored records, atomic JSON writes, and one pasteable command
block. Re-extract and verify every ZIP before handing it to the user.

## Definition of done

An evaluator is complete only when:

- its supported source revisions and runtime stack are explicit;
- the compatibility probe rejects unsupported stacks before model mutation;
- the real algorithm source is invoked, not a mock or post-compute filter;
- all five stages exist and dependency-free tests pass;
- at least one real supported stack has executed the required stages;
- raw results and schema-v3 manifest are preserved;
- mechanism, reset, error, paired CI, per-task quality, end-to-end timing, active weight, and
  detach restoration gates are evaluated;
- the final verdict scope is explicit and historical claims are not substituted for evidence.

If real GPU execution is unavailable, deliver a validated **builder/scaffold** and mark the
evaluator `INCOMPLETE_REAL_RUNTIME_VALIDATION`; never label it directly runnable/validated.
