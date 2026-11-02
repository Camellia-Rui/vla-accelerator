# Evaluator delivery contract

## Required output tree for each method

```text
evaluator/<method>/
├── README-ZH.md
├── SOURCE_MANIFEST.json
├── config.example.json
├── probe_runtime.py
├── adapter.py
├── common/
│   ├── paired_stats.py
│   ├── result_io.py
│   ├── lifecycle.py
│   └── mechanism.py
├── 01-fixed-observation-r1/
│   ├── benchmark_real_observation.py
│   ├── run.sh
│   └── README-ZH.md
├── 02-closed-loop-smoke-r1/
│   ├── closed_loop_paired.py
│   ├── run.sh
│   └── README-ZH.md
├── 03-multitask-paired-r1/
│   ├── closed_loop_paired.py
│   ├── run.sh
│   └── README-ZH.md
├── 04-adjudication-r1/
│   ├── closed_loop_paired.py
│   ├── run.sh
│   └── README-ZH.md
├── 05-weight-audit-r1/
│   ├── weight_audit.py
│   ├── run.sh
│   └── README-ZH.md
├── launch-stage.sh
├── status-stage.sh
├── summarize.py
└── tests/
```

Equivalent organization is allowed only if every capability remains obvious and individually
runnable. Do not copy LightVLA method code blindly; extract only method-neutral infrastructure.

## Command-line contract

Each stage must support explicit paths and at least:

- `--mode baseline|control|accelerated` where applicable;
- task/suite, seed/initial-state, output root, resume, and device selection;
- a dry-run/config-validation mode that does not load the checkpoint;
- deterministic output names and nonzero exit on structural failure.

Long stages must be launchable through `nohup setsid`, write a PID, reject a duplicate live
PID, preserve terminal availability, and expose status without attaching to the process.

## Evidence contract

Preserve raw per-query and per-episode records. Summary-only output is insufficient. Every
record must identify method/mode/config/source/checkpoint/task/seed/initial state/order and
contain timing, actions/outcome, mechanism counters, and error state. CUDA timing must be
synchronized and warm-up excluded. Paired comparisons must use the same task and initial
state, alternate order, and compute paired confidence intervals.

The final schema-v3 manifest must be accepted by:

```bash
python skill/vla-acceleration-evaluator/scripts/validate_manifest.py RESULT.json
python skill/vla-acceleration-evaluator/scripts/admission_decision.py RESULT.json
```

Do not weaken the validators to make a method pass. Extend schemas only additively and add
tests when method-specific evidence is needed.

## Safety and source-integrity contract

- Original repositories and checkpoints remain read-only.
- Temporary patches live in the evaluator/experiment directory.
- Cleanup/restoration runs in `finally` and is verified, not assumed.
- Separate processes are the default when global monkey patches or caches cannot be proven
  reversible.
- No training, optimizer, backward pass, parameter assignment, or checkpoint rewrite.
- A copied upstream component must retain license/provenance; absent licensing blocks public
  redistribution but not private inspection.

## Handoff contract

Before asking the user to run an experiment, Codex must have already generated and checked
the full ZIP. The user receives:

1. the ZIP;
2. one pasteable command block;
3. expected result/log paths;
4. `status.sh` usage;
5. a precise statement of what the experiment proves and what it does not prove.
