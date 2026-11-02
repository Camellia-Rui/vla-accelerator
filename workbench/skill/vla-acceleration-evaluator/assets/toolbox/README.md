# VLA Acceleration Toolbox

This asset contains the active reusable implementations bundled with the
`vla-acceleration-evaluator` skill. Export it to a separate experiment workspace before
editing. Do not write results or checkpoint data into the Skill asset.

## Active implementations

### `lightvla_pruner`

Learned-weight-free LightVLA-style visual-prefix pruning with explicit patch layout,
spatially balanced selection, optional dropped-patch merging, populated-cache decode
bypass, reversible replacement, and baseline/builtin/packaged modes.

### `specprune`

The supplied experimental SpecPrune-VLA wrapper for OpenVLA-OFT. It requires the
continuous L1 action head, two camera views, the expected token/action layout, eager
attention, and explicit lifecycle handling. Read `references/specprune.md` and
`src/specprune/README.md` before adapting it. Use an untouched baseline process unless
every mutated module can be restored.

### `vla_pruner`

The reversible lifecycle adapter for the pinned official VLA-Pruner OpenVLA-OFT
backend. The external modified model and Transformers pruning kernels are not vendored.
Read `references/vla-pruner.md` and `src/vla_pruner/INTEGRATION.md`; reset at every
episode/task boundary and after errors.

## Permanently unavailable here

AutoPrune is permanently retired. This toolbox intentionally contains no
`vla_accelerators` package, AutoPrune adapter, attention-prior bridge, example, or test.
Do not reconstruct it from older workspaces or archives. VLA-Cache is also absent and
remains quarantined after failed closed-loop acceptance.

## Install in an experiment environment

```bash
python -m pip install -e /path/to/exported/toolbox
```

Or add the exported `src` directory to `PYTHONPATH`.

## Dependency-light checks

```bash
python -m unittest tests/test_lightvla_pruner.py
python tests/test_vla_pruner.py -v
python -c "from pathlib import Path; compile(Path('src/specprune/accelerator.py').read_text(encoding='utf-8'), 'accelerator.py', 'exec'); print('SPECPRUNE_SOURCE: OK')"
```

Real checkpoint, paired latency, reset, action, and closed-loop validation remain target-
repository tasks. No configuration is universally approved.

