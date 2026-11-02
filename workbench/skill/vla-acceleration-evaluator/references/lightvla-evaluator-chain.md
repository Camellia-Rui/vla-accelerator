# Latest packaged LightVLA evaluator chain

The release bundle contains a five-stage chain under `evaluator/lightvla/`.

1. `01-fixed-observation-r6`: real LIBERO-Spatial observation, official two-image/proprio/L1-head/eight-action configuration, normalization-statistics gate, alternating paired timing, action fidelity, structural pruning proof, and explicit clearing of inherited VLA-Cache fields.
2. `02-closed-loop-smoke-r1`: tasks 0 and 1, three paired initial states per task, fresh environment per episode.
3. `03-multitask-paired-r1`: tasks 0-3, five paired initial states per task; requires the prior smoke PASS.
4. `04-adjudication-r1`: only for the documented single per-task failure pattern; adds unseen trials 5-19 on tasks 0 and 3. Do not run it after an arbitrary failure.
5. `05-weight-audit-r1`: compares sampled tensor values through original parameter objects while the wrapper is active, then verifies name/topology restoration after detach. It supplements rather than rewrites earlier evidence.

The root helper `evaluator/lightvla/launch-stage.sh` launches one stage with a PID guard. The evaluator is environment-pinned and should first be audited against the actual repository/checkpoint paths discovered on the target machine.

The chain contains executable code but not the remote checkpoint, LIBERO assets, or previously generated raw LightVLA result JSON. Existing valid evidence should be reused; do not rerun a completed gate without a concrete reason.

