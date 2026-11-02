---
name: vla-acceleration-evaluator
description: Audit, integrate, benchmark, tune, and adjudicate training-free VLA inference accelerators using reversible adapters, paired real-observation tests, closed-loop task evidence, and reproducible experiment packages. Use for OpenVLA/LightVLA/OpenVLA-OFT acceleration work, temporary scripts, latency attribution, weight-integrity checks, or PASS/TUNE/REJECT decisions. AutoPrune is permanently retired and VLA-Cache is quarantined; this release must not restore, integrate, or deploy either method.
---

# VLA Acceleration Evaluator

Treat every accelerator as an experimental intervention. Token reduction, FLOPs, decoder CUDA time, or a fixed-observation speedup is not enough to claim closed-loop acceleration.

## Start here

For initialization follow the release START_HERE.md: verify then audit read-only.
Read PROJECT_STATUS.md, EVIDENCE_INDEX.json and COLLABORATION.md in the release root.
Builder files are backlog, not automatic execution authority; preserve current user scope.

LightVLA's original method includes fine-tuning. A fixed-weight inference intervention
does not establish that the original method is training-free.

1. Read [project-state.md](references/project-state.md) and [retired-methods.md](references/retired-methods.md) to avoid reviving retired work, repeating completed tests, or mixing unrelated projects.
2. Read [evaluation-protocol.md](references/evaluation-protocol.md) and classify the requested module before changing code.
3. For temporary probes, adapters, benchmarks, or ZIPs, read [temporary-script-playbook.md](references/temporary-script-playbook.md) and [cloud-operations.md](references/cloud-operations.md).
4. When using bundled implementations, read [toolbox.md](references/toolbox.md), then export `assets/toolbox/` with `scripts/export_toolbox.py`. Never edit the skill asset in place.
5. For the packaged LightVLA acceptance chain, read [lightvla-evaluator-chain.md](references/lightvla-evaluator-chain.md).

## Scope and boundaries

Require or discover the VLA source, checkpoint, acceleration source, task suite, official inference entry point, hardware/software environment, and output root. Keep repositories, weights, datasets, generated scripts, and evidence outside the skill directory.

Do not overwrite a checkpoint or upstream repository. If remote-code files must differ, create a separate checkpoint view and link or copy immutable weight shards. Preserve the user's unrelated changes.

This release does not authorize AutoPrune or VLA-Cache integration or deployment. AutoPrune is permanently retired by user decision, regardless of historical source or results. VLA-Cache evidence establishes internal compute reduction and fixed-observation acceleration, but rejects the tested cross-frame reuse implementation for closed-loop use. Read [retired-methods.md](references/retired-methods.md) for the binding boundary and [vla-cache.md](references/vla-cache.md) only when auditing its archived evidence.

Do not mix GRIDs/OpenPI results into OpenVLA, LightVLA, VLA-Pruner, SpecPrune, or VLA-Cache conclusions.

## Route the module

- **LightVLA / stateless prefix pruning:** read [lightvla-pruner.md](references/lightvla-pruner.md).
- **SpecPrune / hybrid OpenVLA-OFT controller:** read [specprune.md](references/specprune.md); use separate baseline and accelerated processes unless every replaced module can be restored.
- **VLA-Pruner / stateful OpenVLA-OFT pruning:** read [vla-pruner.md](references/vla-pruner.md); validate temporal reset and use the pinned official backend.
- **VLA-Cache:** evidence review only in this release; do not export or attach an implementation.
- **AutoPrune:** permanently retired; stop and explain the archive boundary. Do not recreate it under another name or from historical files.
- **Unknown module:** inspect source and run an unaccelerated real forward before selecting an attachment point.

## Required gates

Execute gates in order and stop when a hard gate fails:

1. **Audit:** record source revisions, dirty state, environment, checkpoint identity, official entry point, defaults, and modifications.
2. **Runtime probe:** observe actual classes, tensor shapes, prefill/decode calls, cache format, action output, device, dtype, and attention backend on a real input.
3. **Reversible attachment:** implement the smallest adapter and a no-op/retain-all control. Restore hooks, modules, and configuration in `finally`.
4. **Paired fixed-observation screen:** identical real observations, warm-up of every mode, alternating order, CUDA synchronization, raw samples, full actions, memory, and real kept/reused counts.
5. **Mechanism proof:** demonstrate actual skipped computation, shortened decoder input, or documented reuse. A post-compute filter does not count.
6. **Closed-loop smoke:** new environment per episode, identical task/initial state/seed per pair, alternating order, explicit state reset, atomic result writes, and retry of error records.
7. **Multitask confirmation:** expand only after smoke passes. Report overall and per-task success, query and episode-level time, paired confidence intervals, and errors.
8. **Weight audit:** while wrappers are active, compare original parameter objects and sampled tensor values; after detach, verify parameter names/topology are restored. Do not use a name-sensitive hash alone as the active-wrapper integrity test.
9. **Decision:** write a manifest using [result-schema.md](references/result-schema.md), validate it, and run the admission script.

Use:

```bash
python scripts/validate_manifest.py result.json
python scripts/admission_decision.py result.json --pretty
```

## Decision policy

- **PASS:** structure, control, mechanism, reset, weight, fidelity, paired timing, and required closed-loop gates all pass for the recorded target and environment.
- **TUNE:** correctness gates pass and profiling shows credible tunable headroom.
- **REJECT:** incompatible attachment, invalid control, state leak, weight change, failed action/closed-loop quality, or no credible end-to-end gain remains.
- **INCONCLUSIVE:** mandatory evidence is absent or invalid.

Never promote an adapter after a fixed-observation-only result. Never convert a rejected experimental wrapper into deployment code.

## Experiment-package contract

When the GPU/model is remote, hand off one ZIP containing the scripts, Chinese README, exact output paths, source manifest, and status tools. Long jobs must run via `nohup setsid`, save a PID, refuse duplicate live PIDs, write progress/results atomically, resume completed records, retry error records, and leave the Jupyter terminal open. Provide one pasteable command block whenever practical.

## Deployment output

Generate a persistent adapter only after PASS. It must expose enable/disable, selected parameters, compatibility checks, state reset when applicable, evidence path, and clean detach/uninstall. Preserve the original model path whenever hooks or wrappers suffice.

