# Method-specific mechanism and control requirements

## Shared modes

- **baseline:** untouched official inference path.
- **control:** adapter/lifecycle overhead with retain-all or no-effect settings, when the method
  can represent a semantically valid no-effect configuration.
- **accelerated:** intended pruning/speculation configuration.

If a valid retain-all control is impossible, document why and use separate native versus
accelerated processes. Never create a fake control that silently changes model semantics.

## SpecPrune

The probe must validate OpenVLA-OFT-specific model structure, continuous action head, proprio
projector if used, two-view layout, action chunk/queue semantics, eager attention availability,
and exact source compatibility described in `references/specprune.md`.

At minimum record:

- primary/wrist tokens before and actually computed after pruning by stage/layer;
- draft versus precise selections and transitions;
- candidate/draft production and acceptance/use counts where speculative behavior exists;
- fallback/full-compute counts and reasons;
- action-queue length, `execute_action`, resets, and episode-boundary state;
- real attention/kernel path reached and proof that removed work was not computed first;
- adapter overhead, accelerated kernel time, and full query/environment end-to-end time.

Post-compute Top-K, token estimates, or queue reuse without avoided model work do not prove
acceleration. Control timing must include the same wrapper/measurement overhead.

## VLA-Pruner

The probe must verify the exact official backend revision pinned in
`assets/toolbox/src/vla_pruner/backend.lock.json`, target model structure, supported attention
implementation, pruning layer/location, visual token boundaries, semantic/action score path,
EMA history reset, and action-generation compatibility.

At minimum record:

- total and visual sequence lengths before/after at the real pruning site;
- per-layer kept/dropped counts and configured versus observed retention;
- semantic score and action-history/EMA availability and update/reset counts;
- fallback/unsupported-shape counts;
- number of downstream blocks/kernels that received the shorter sequence;
- measured adapter overhead and synchronized full query/end-to-end latency;
- episode start/end reset proof and no cross-episode history leakage.

`pruning-report`, estimated FLOPs, or a shorter tensor observed only after expensive blocks are
not sufficient by themselves. The official backend must be actually invoked.

## Weight and detach audit for both

While the wrapper is active, compare original parameter tensor values (not only names or a
checkpoint file hash). After detach, verify module classes, parameter names, hook counts,
patched callables, relevant buffers/caches, and representative native output are restored.
Any unexplained mutation yields REJECT or INCONCLUSIVE, never PASS.
