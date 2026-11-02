# VLA-Pruner integration guide

Read this reference when the target is the bundled `assets/toolbox/src/vla_pruner`
package or the official MINT-SJTU VLA-Pruner OpenVLA-OFT backend.

## What is bundled

The toolbox contains a dependency-light lifecycle adapter, configuration validation,
backend compatibility checks, pruning statistics, and a lock file. It does **not** vendor
the modified model or Transformers pruning kernels. Those remain an external dependency
pinned to repository `https://github.com/MINT-SJTU/VLA-Pruner` commit
`84d4b7192c77abf1585610e2f12393319b7ebff9`.

Use:

```python
from vla_pruner import AcceleratorConfig, VLAPrunerAccelerator
```

Do not use the historical `pruners.registry` or `methods.vla_pruner` imports from the
source directory's old integration context.

## Classification and compatibility

Classify VLA-Pruner as stateful temporal token pruning when its semantic/action mode is
enabled: action-attention history persists across model queries. The bundled adapter only
accepts the pinned VLA-Pruner OpenVLA-OFT model with its modified `openvla_utils`, model
configuration, LLaMA decoder, and vision-backbone methods. Ordinary OpenVLA,
unmodified OpenVLA-OFT, PyPI Transformers, SpecPrune models, and arbitrary decoder models
are incompatible.

Before model loading, verify the external checkout commit and expose its
`src/openvla-oft` on `PYTHONPATH`. Installing the toolbox alone does not install the
backend. `from_pretrained()` also requires `OPENVLA_SKIP_CHECKPOINT_SYNC=1`; this prevents
the upstream loader from rewriting local checkpoint code. Never modify the user's
original checkpoint to satisfy compatibility.

## Lifecycle and isolation

Call `attach(model)` only after `model.eval()`. Call `reset()`/`start_episode()` after
every environment reset, instruction/task switch, and inference exception. `enable()` and
`disable()` both clear temporal history. `detach()` restores every attribute managed by
the adapter and is idempotent, but concurrent inference or stacking another pruner on the
same model is unsupported.

Use separate processes or separately loaded models for baseline and accelerated timing.
The disabled path is useful as a configuration control, but it still uses the patched
upstream backend and is not proof of an untouched baseline.

## Evidence gates

First run the dependency-free contract test:

```bash
python tests/test_vla_pruner.py -v
```

Then validate the real target in this order:

1. Record upstream/checkpoint commits, dependency versions, model structure, view count,
   token layout, and a parameter fingerprint.
2. Run an untouched baseline and a zero-prune control (`fastv_r=0`) on identical saved
   observations; compare full action chunks.
3. Confirm `get_stats()` reports actual sequence/visual counts from backend
   `pruning_info`; requested ratios alone are not evidence of pruning.
4. Check first query, warmed history, repeated frames, episode reset, task switch, and
   exception recovery. The first temporal query may legitimately prune differently while
   history is empty.
5. Alternate separate baseline/accelerated processes, warm up, synchronize CUDA, save raw
   samples, and report median/p95 plus paired confidence intervals.
6. Run closed-loop paired episodes and compare success rate, action/gripper fidelity, and
   episode length. Verify the parameter fingerprint is unchanged.

The adapter's `wall_time_ms` does not add CUDA synchronization and is not a strict
benchmark. Historical performance numbers in the supplied README are provenance only;
never present them as validation of a new environment.

## Deployment contract

After PASS, persist the pinned backend commit, selected pruning parameters, number of
views, reset hook, compatibility probe, evidence manifest path, and detach procedure.
Keep the upstream backend and checkpoint identity explicit because the bundled adapter
does not contain the pruning kernel itself.
