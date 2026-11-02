# Bundled acceleration toolbox

The active reusable implementations are stored in `assets/toolbox/`. Export them to the
experiment workspace before modification:

```bash
python scripts/export_toolbox.py /path/to/new/vla-acceleration-toolbox
```

The exporter refuses to overwrite an existing destination. Keep the Skill asset pristine.

## Route by target

- Use `lightvla_pruner` when the target exposes a compatible three-argument pruner or
  needs learned-weight-free patch scoring through a prefix wrapper.
- Use `specprune` only for a compatible OpenVLA-OFT model with the continuous L1 action
  head, two camera views, eight-action chunks, and required OFT helpers. It is hybrid and
  stateful; read `references/specprune.md` first.
- Use `vla_pruner` only with the pinned official VLA-Pruner OpenVLA-OFT backend. Its
  external patched model and Transformers implementation are not bundled. Read
  `references/vla-pruner.md` first.

## Excluded methods

- AutoPrune is permanently retired. No source, bridge, example, or test is present. Do
  not restore it from older folders or recreate it under a generic name.
- VLA-Cache is quarantined after failed closed-loop acceptance. No overlay is present.

SpecPrune does not expose a proven clean reversible toggle for every mutation. Use an
untouched baseline process and a separate accelerated process, or snapshot and restore
every replaced module in a temporary adapter.

VLA-Pruner is stateful when action history is enabled. Call `reset()` at every episode
and task boundary and after inference errors. Prefer separate baseline and accelerated
processes for timing so backend selection, allocator state, and warm-up do not contaminate
the control.

Read the toolbox `README.md` and `SOURCE_MANIFEST.json` after export. Then follow
`temporary-script-playbook.md` for runtime discovery, reversible attachment, pairing,
behavior tests, and evidence output.

