# Temporary experiment script playbook

Use this reference when the user asks Codex to write scripts that attach, benchmark, tune, or validate a training-free VLA accelerator. The scripts are disposable experimental tooling; raw evidence and selected configuration are durable outputs.

## Choose the smallest useful script set

Do not force a fixed file layout when one script is enough. For a new or uncertain integration, the following separation is usually useful:

1. `probe_runtime.py`: load the unmodified model and record runtime structure and tensor shapes.
2. `run_paired_benchmark.py`: run baseline, zero-effect control, and accelerated modes in one process and save raw samples.
3. `run_closed_loop.py`: evaluate selected candidates on real episodes after structural and fixed-observation gates pass.

A parameter sweep may be a fourth script or a mode in the paired benchmark. Do not duplicate model-loading code when a small shared helper is sufficient.

## Inspect before coding

Read the target repository's official model-loading and action-inference entry points. Search for the real decoder, vision tower, projector, cache type, action unnormalization, and reset path. Run one baseline forward and record:

- runtime classes and relevant named-module paths;
- image/vision output and projected embedding shapes;
- decoder prefill versus iterative decode calls;
- attention mask, position IDs, cache position, and cache representation;
- action return type and any auxiliary state;
- batch size, dtype, attention backend, device, and peak memory.

Do not infer a 256- or 512-patch layout from a checkpoint name. Do not copy imports from a different fork without verifying them.

## Script contract

Every generated experiment script should satisfy the applicable items below.

### Inputs and outputs

- Accept checkpoint, repository/module paths, output path, warm-up count, measured iterations, and method parameters through CLI arguments or a clearly documented configuration block.
- Keep checkpoint and repository inputs read-only. Write JSON evidence and logs only to the requested output directory.
- Record repository revisions, checkpoint identity, GPU, driver/CUDA when available, Python, PyTorch, Transformers, dtype, attention backend, task, episode, seed, and every non-default parameter.
- Save raw per-run timings and actions, not only aggregates.

### Training-free invariant

- Call `model.eval()` and use `torch.inference_mode()` or `torch.no_grad()` for measured inference.
- Do not create an optimizer, call backward, or load learned adapter weights unless the user explicitly changes the scope.
- Fingerprint representative parameters before and after the experiment. Report `weights_unchanged`; fail or mark the result invalid if it is false.

### Reversible attachment

- Prefer a wrapper, pre/post hook, or replaceable module reference over editing upstream model source.
- Keep the original callable/module and restore it in `finally`.
- Store every hook handle and remove it in `finally`, including failure paths.
- Expose baseline and accelerated modes in the same process. Add a zero-effect control that exercises the adapter path while retaining all information whenever the method permits it.
- Preserve the model's training/eval state and documented return type.

### Measurement

- Use identical observations, prompts, task state, and decode settings for paired modes.
- Warm up every mode. Rotate or alternate execution order to reduce thermal and order bias.
- Synchronize CUDA immediately before starting and after ending measured regions.
- Record end-to-end wall time and CUDA time. Add stage timing for vision, projector, accelerator overhead, LLM prefill, and decode when diagnosing performance.
- Record median, p95, raw paired differences, memory, and actual patch/token/cache counts. Use repeated samples; a single run is a smoke test.

### Behavior

- Save full action vectors and compute mean/max absolute error plus per-dimension error. Report gripper sign agreement separately.
- Check deterministic repeats and execution-order effects.
- Treat fixed observations as screening only. Run known-success closed-loop episodes for final admission.

## Stateless prefix-pruning pattern

For a LLaMA-style VLA with a continuous visual patch span:

1. Discover the patch span from a real prefill and keep special/prefix/suffix tokens explicit.
2. Attach only to prefill. Bypass one-token cached decode unless the method explicitly supports decode-time changes.
3. When selecting patches, update hidden states, attention mask, position IDs, cache position, and any layout metadata consistently.
4. Preserve token order unless the method specifies another positional convention.
5. Validate batch-size assumptions explicitly; fail clearly rather than silently broadcasting.
6. If the algorithm requires a vision attention prior, capture it from the current vision forward, normalize its documented shape, inject it once into the matching prefill, and prove one capture/one injection per inference. Do not substitute random or stale attention in the real gate.
7. Implement a retain-all control. It should reproduce baseline exactly or within a tolerance justified by an attention-backend change.

Do not assume fewer patches guarantees speed. Full attention materialization, Python selection, host-device scalar transfers, sequence rebuilding, and unchanged decode may erase the gain.

## Stateful cross-frame pattern

Keep state in an explicit context owned by the experiment, not in untracked globals. Define and test:

- first frame with empty state;
- cached consecutive frame;
- explicit episode reset;
- task or instruction change;
- missing previous image/cache;
- exception recovery and model reload.

Clear frame-dependent configuration before every inference and repopulate it only when all required previous-state components are valid. A reset call must reproduce first-frame behavior.

## Minimal architecture

Repository-specific names vary, but a temporary paired benchmark should have this shape:

```python
def load_target(args): ...
def capture_baseline_input(...): ...
def attach_accelerator(model, config): ...  # returns adapter + cleanup
def parameter_fingerprint(model): ...
def infer(mode, observation, measured): ...

model, processor = load_target(args)
model.eval()
fingerprint_before = parameter_fingerprint(model)
adapter, cleanup = attach_accelerator(model, config)
try:
    warm_up_all_modes()
    run_rotating_paired_measurements()
    validate_control_and_behavior()
finally:
    cleanup()
fingerprint_after = parameter_fingerprint(model)
write_raw_and_summary_json(weights_unchanged=(fingerprint_before == fingerprint_after))
```

This is a responsibility map, not code to paste blindly. Use actual signatures and return types discovered in the target repository.

## Stop conditions

Stop and report `INCONCLUSIVE` instead of improvising when the checkpoint, real input, task state, required prior, or official inference entry point is unavailable. Mark a configuration `REJECT` when the zero-effect control is invalid, state leaks across episodes, weights change, or behavior/closed-loop gates fail. Do not convert a rejected experimental wrapper into deployment code.

Before handing off a temporary script, compile it, run its dependency-free checks, and provide the exact command plus expected output path. If the GPU/model is remote, package the script and validation helpers, then analyze the returned raw JSON rather than asking the user to transcribe aggregate numbers.
