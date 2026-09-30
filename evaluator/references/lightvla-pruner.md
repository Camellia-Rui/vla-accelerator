# LightVLA-style prefix pruning

## Mechanism

The tested module scores visual patch tokens, retains a subset, rebuilds position IDs and attention masks, and passes a shorter prefix to the language decoder. It is stateless across observations and training-free, but it can materially change actions.

## Verified integration facts

- A packaged adapter reproduced the built-in LightVLA pruner on the LightVLA checkpoint.
- On an original OpenVLA checkpoint, runtime discovery found a 256-patch visual prefix and a standard Llama decoder without a built-in pruner.
- A zero-pruning control retaining 256 patches produced exactly equal actions, proving the adapter path itself was valid.
- Retaining fewer patches reduced LLM prefill time on fixed observations, but observed end-to-end gains were modest and a known-success LIBERO episode failed under tested pruning configurations.

## Required checks

Discover patch start/count from a real prefill instead of hard-coding 512 or 256. Keep non-patch context explicit. Preserve attention masks, positions, cache behavior, and padding. Compare selection strategies for spatial collapse. Require closed-loop validation; action similarity alone did not predict episode success in the tested setup.

## Prior conclusion

The tested original-OpenVLA/LIBERO-Spatial configuration is REJECT for deployment. This does not reject the method universally; a new model, task, selector, or retention policy must repeat the full protocol.
