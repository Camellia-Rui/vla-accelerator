# School-server discovery checklist

Perform read-only discovery first. Do not install or change environments during this phase.

## Access topology

- Confirm Codex is operating in the same VS Code Remote-SSH workspace as the terminal.
- Record `pwd`, repository roots, writable experiment/output root, and available disk space.
- Confirm that paths are remote Linux paths, not the local Windows upload machine.

## Hardware and software

Record OS/kernel, GPU model/count/memory, driver, CUDA runtime visible to PyTorch, Python,
PyTorch, Transformers/fork identity, tokenizers, timm, attention backend, and relevant
LIBERO/MuJoCo/rendering packages. Use the environment already associated with each method;
SpecPrune and VLA-Pruner may require separate environments.

## Source identity

For every repository: absolute path, remote URL, HEAD commit, branch/detached state, and
`git status --short`. Locate the official inference/evaluation entry point. Do not infer
compatibility from directory names.

Required real sources:

- target OpenVLA/OpenVLA-OFT repository used by the checkpoint;
- SpecPrune upstream/fork or a confirmed compatible copy;
- VLA-Pruner official backend pinned by `backend.lock.json`;
- LIBERO/task evaluator source and assets.

## Model and task assets

Record checkpoint path, hashes/manifest if available, processor/tokenizer/config files,
action-head and proprio-projector identities, dataset statistics, task definitions, BDDL,
assets, and initial states. Never hash multi-gigabyte shards without need; filenames, sizes,
index/config hash, and repository/revision may be sufficient for discovery.

## One native observation probe

Before attaching either accelerator, run the official native path on one real observation and
capture runtime class names, tensor shapes, image/token layout, cache/prefill/decode behavior,
action shape/range, query latency, and peak memory. This is diagnostic evidence, not a speed
result.

## Consolidated missing-input request

Only after discovery, ask once for unresolved essentials. Prefer asking the user to provide
paths, not upload checkpoints. The request should cover both methods at once and distinguish:

- missing entirely;
- present but incompatible/unverified;
- present and ready;
- optional for later real closed-loop execution.
