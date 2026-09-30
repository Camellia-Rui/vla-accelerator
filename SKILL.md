---
name: vla-acceleration-evaluator
description: Evaluate, tune, and integrate training-free acceleration modules for vision-language-action (VLA) inference. Use when inspecting a VLA or acceleration repository, writing temporary probe/adapter/benchmark scripts, locating an attachment point, distinguishing stateless token pruning from stateful cross-frame caching, building paired baseline/accelerated experiments, checking action fidelity and cache reset behavior, profiling latency by stage, tuning module hyperparameters, deciding PASS/TUNE/REJECT, or generating a deployment adapter only after validation.
---

# VLA Acceleration Evaluator

Evaluate an acceleration module as an experimental intervention. Do not assume that fewer tokens, patches, or cache entries imply end-to-end acceleration or acceptable control behavior.

## Inputs and boundaries

Require paths or identifiers for the VLA implementation, checkpoint, acceleration module, evaluation task, and output directory. Keep repositories, checkpoints, datasets, and generated evidence outside this skill directory.

Do not edit a user's original checkpoint. Use a separate checkpoint directory and link or copy immutable weight shards when remote-code files must be changed.

## Temporary experiment scripts

Temporary, reversible scripts and wrappers are expected during evaluation; they are not deployment adapters. When asked to implement or test a new integration, read [references/temporary-script-playbook.md](references/temporary-script-playbook.md) before writing code.

Generate scripts in the user-selected experiment/output directory, never inside this skill. Inspect the target repository first and use its real loading and inference entry points. A usable experiment must include a runtime probe or recorded discovery evidence, baseline and accelerated paths, a zero-effect control when possible, raw paired samples, action outputs, actual retained/reused counts, environment metadata, and cleanup of hooks or monkeypatches in `finally`.

For training-free claims, put the model in evaluation mode, run under inference/no-grad context, record a parameter fingerprint before and after, and fail if sampled weights change. Treat a script that only shortens tensors or reports a local kernel time as incomplete.

## Route the module

Classify the module before writing an adapter:

- **Stateless:** transforms the current forward input, such as visual-token pruning before the language decoder. Read [references/lightvla-pruner.md](references/lightvla-pruner.md).
- **Stateful:** reuses information across observations, such as KV-cache or patch reuse. It requires explicit episode reset validation. Read [references/vla-cache.md](references/vla-cache.md).
- **Unknown:** inspect source and run one unaccelerated forward before deciding. Do not guess tensor layout from class names alone.

## Evaluation workflow

Follow [references/evaluation-protocol.md](references/evaluation-protocol.md) and execute these gates in order:

1. **Audit:** record repository commits, environment versions, model/checkpoint identity, official entry points, defaults, and all local modifications.
2. **Probe:** load the model without acceleration and observe runtime classes, tensor shapes, prefill/decode calls, cache format, device, and memory.
3. **Attach and control:** implement the smallest reversible adapter. Verify no-op or zero-acceleration mode reproduces baseline exactly or within a justified tolerance.
4. **Pair measurements:** alternate baseline and accelerated runs on identical inputs. Include warm-up, CUDA synchronization, repeated samples, per-stage timing, action outputs, and resource counts.
5. **Validate semantics:** test action fidelity, gripper agreement, repeated-frame behavior, state reset, order effects, and closed-loop task success.
6. **Diagnose and tune:** attribute gains and overhead to stages. Tune only documented parameters and preserve an untouched default regression.
7. **Decide:** write a manifest following [references/result-schema.md](references/result-schema.md), validate it, then run the deterministic admission script.

Use:

```bash
python scripts/validate_manifest.py result.json
python scripts/admission_decision.py result.json --pretty
```

## Decision policy

- **PASS:** structure/control gates pass, required reset and closed-loop evidence pass, fidelity is within thresholds, and measured end-to-end speedup meets the target.
- **TUNE:** structural and correctness gates pass, but performance or non-critical fidelity still has credible tunable headroom.
- **REJECT:** attachment is incompatible, the control is invalid, state leaks across episodes, action/closed-loop quality fails, or no credible speedup remains after profiling.
- **INCONCLUSIVE:** mandatory evidence is missing or invalid.

Never describe a module as accelerated based only on theoretical FLOPs, token retention, prefill improvement, or a synthetic input. Never ship an adapter for a rejected configuration.

## Deployment output

Generate a persistent, task-specific deployment adapter only after PASS. Evaluation-time scripts may use temporary hooks or wrappers before PASS, but must remain reversible and outside the original checkpoint. A deployment adapter must expose enable/disable, selected hyperparameters, reset behavior for stateful methods, compatibility checks, evidence path, and a clean uninstall/restore path. Preserve the original model forward implementation when hooks or wrappers suffice.

## Bundled resources

- `scripts/validate_manifest.py`: checks evidence completeness and schema invariants.
- `scripts/admission_decision.py`: emits deterministic PASS/TUNE/REJECT/INCONCLUSIVE results.
- `scripts/profile_vla_cache_stages.py`: reference CUDA stage profiler for VLA-Cache/OpenVLA; adapt model-loading imports rather than assuming universal compatibility.
- `scripts/self_test.py`: exercises manifest validation and PASS/TUNE/REJECT/INCONCLUSIVE decisions without model dependencies.
- `references/temporary-script-playbook.md`: contract and implementation patterns for disposable probe, adapter, benchmark, and closed-loop scripts.
- `references/result-schema.md`: canonical manifest fields and meanings.
- `references/lightvla-pruner.md`: known stateless-pruning evidence and hazards.
- `references/vla-cache.md`: known stateful-cache evidence, reset requirements, and suitability.
