# VLA-Cache

## Mechanism and suitability

VLA-Cache is stateful across consecutive observations. It is most plausible when the camera viewpoint and much of the scene remain stable, task-relevant regions are localized, observations arrive sequentially within an episode, and the decoder/cache implementation supports selective reuse. It is less suitable for rapid camera motion, broad scene changes, frequent task switches, single-frame inference, or short horizons where setup overhead dominates.

The tested implementation finds visually stable patches between frames, filters them using prior attention, computes a layer-wise reuse schedule, and reuses cache content in a modified Transformers/Llama path.

## Lifecycle invariant

Before every inference call, clear frame-dependent `reusable_patches` and `proportion_attn_var`. Populate them only when valid previous-image, previous-attention, and previous-cache state exists. On episode reset, exception, task switch, or missing state, clear all cache state. A reset test must reproduce first-frame baseline behavior.

## Verified evidence on OpenVLA 7B, LIBERO-Spatial, RTX 4090

The corrected state machine passed first/cached/reset checks. Real-frame profiling showed approximately 82 reusable patches on average. Default stage medians were roughly:

- baseline wall 300.089 ms; cache wall 301.236 ms; speedup 0.996x;
- cache overhead: static-patch 1.561 ms, task selection 2.236 ms, layer schedule 4.753 ms;
- baseline/cache prefill 46.212/42.236 ms;
- baseline/cache decode 208.462/208.568 ms.

The module improved prefill but not end-to-end latency because selection/scheduling overhead and unchanged decode dominated. Tested growth factors 0, 0.25, and 0.55 and patch top-k 100, 130, and 160 did not produce end-to-end speedup. Action differences remained nonzero, with some large-difference frames.

## Prior conclusion

REJECT for the current OpenVLA 7B/LIBERO-Spatial/RTX 4090 implementation and parameters. Future tuning should first reduce CPU/GPU selection and schedule overhead or move reuse to a stage that reduces the dominant decoder cost. Do not claim acceleration from prefill gains alone.
