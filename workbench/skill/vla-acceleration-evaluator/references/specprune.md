# SpecPrune-VLA integration guide

Read this reference when the target module is the bundled `assets/toolbox/src/specprune`
or when a user asks to evaluate SpecPrune on a VLA. Treat the bundled source README as
provenance and detailed reproduction notes; this reference defines how Codex should
route and safely evaluate it.

## What is bundled

`src/specprune/accelerator.py` is the user-supplied engineering wrapper, not merely an
API stub. It implements two-view frame-similarity selection, attention-guided pruning,
deep action-aware dynamic pruning, an eight-action queue, draft/precise mode control,
and runtime injection into OpenVLA-OFT. The original `README.md` is preserved beside it.
After exporting/installing the toolbox, import the packaged copy with
`from specprune.accelerator import AcceleratorConfig, SpecPruneAccelerator,
get_final_precise_topk`. The original README's `spec_prune_accelerator` import assumes
the source file was copied flat into an experiment root.

The supplied directory did not include a license file. Preserve its paper/upstream
attribution and verify upstream licensing before redistributing it outside the user's
toolbox.

## Compatibility gates

Do not attach SpecPrune until a runtime probe confirms all of the following:

- the target is OpenVLA-OFT, not ordinary OpenVLA or the restored VLA-Cache fork;
- `model.language_model.model` is the expected LLaMA decoder;
- the model exposes the helper methods consumed by the patched `predict_action`;
- a continuous L1 action head is available, plus the proprio projector when enabled;
- input uses one primary and one wrist view, each mapped to 256 visual tokens after BOS;
- action chunks contain eight 7-D actions and proprio has the expected eight values;
- eager attention weights and absolute token positions are available;
- the target Transformers/OpenVLA-OFT fork is compatible with the bundled forward.

The recorded validated stack used Ubuntu 22.04, Python 3.10, H20, PyTorch 2.6/CUDA
12.4, and the OpenVLA-OFT Transformers 4.40.1 fork. That evidence does not transfer to
RTX 4090, ordinary Transformers, another checkpoint, another camera layout, or another
task suite without revalidation. Keep SpecPrune in an isolated environment because its
dependencies differ from the existing LightVLA/VLA-Cache environments.

## Lifecycle and state

Classify this method as hybrid/stateful. `SpecPruneAccelerator.step()` retains previous
images, previous attention indices, dynamic importance state, an action queue, and
draft/precise controller state. Call `start_episode()` after every environment reset and
before the first policy step. Also reset on task/instruction changes and after exceptions.
Only advance the environment when `info["execute_action"]` is true; the action adapter
intentionally consumes initial queued actions without advancing the environment.

Validated precise-mode budgets are Object `15/12` and Goal `40/31`. Obtain them with
`get_final_precise_topk(suite)`. The fallback `15/12` for other suites is not a validated
recommendation. Keep `num_open_loop_steps=8` and `layer_skip=False` unless a separate
ablation is explicitly requested.

## Attachment and baseline isolation

`SpecPruneAccelerator.install()` patches the decoder forward and `predict_action`, and
can replace every self-attention module with eager `LlamaAttention`. Although it stores
the original forward callables, it has no public uninstall operation and does not retain
the replaced attention modules. Therefore:

1. Run the authoritative baseline in a separately loaded model or separate process.
2. Treat `token_prune=False`, `dynamic_prune=False`, and `action_adapter=False` as an
   adapter-path control only, not as the untouched baseline.
3. If a same-process paired test is required, write a temporary attachment manager that
   snapshots the decoder forward, `predict_action`, every attention module, model config,
   and eval state before installation and restores all of them in `finally`.
4. Fingerprint weights before and after both paths; runtime module replacement must not
   be confused with learned-weight modification.

## Evaluation requirements

Start with source compilation and a runtime structure probe. Then run:

- one untouched baseline episode and one adapter-path retain-all/control episode;
- first-query, queued-action, precise-mode transition, and episode-reset checks;
- paired end-to-end query timing with identical observations and action-head weights;
- full action, gripper, query-count, environment-step, and success comparisons;
- Object and Goal separately because their checkpoint heads and precise budgets differ.

The supplied README records H20 results for 500 episodes per suite and independent
latency trials. Preserve the distinction between closed-loop query latency and the
separate timing experiment. Never reuse those numbers as evidence for a new server or
integration; save fresh raw per-episode and per-query evidence and apply the evaluator's
manifest gates.
