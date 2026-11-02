# Project state as of 2026-10-31

## Active evaluation routes

- **LightVLA:** the latest local acceptance chain is fixed-observation r6, closed-loop smoke r1, multitask paired r1, targeted adjudication r1, and numeric weight audit r1. Use the packaged chain; do not restart from r1-r5.
- **SpecPrune:** bundled as an experimental OpenVLA-OFT implementation with lifecycle caveats; it needs an untouched baseline process and real closed-loop validation.
- **VLA-Pruner:** bundled lifecycle adapter requires the pinned official external backend and explicit temporal reset validation.

## Retired and quarantined routes

- AutoPrune is permanently retired by user decision. It is not an active candidate and is absent from the toolbox.
- VLA-Cache is not an active integration candidate. Its fixed-observation gain did not survive closed-loop behavior. See `vla-cache.md`.

Neither route may be reconstructed from older local folders, archived ZIPs, logs, or prose. See `retired-methods.md`.

## Evidence discipline

The project distinguishes internal compute, single-query wall time, and closed-loop behavior. Claims are scoped to the exact source, checkpoint, tasks, hardware, software, adapter, and parameters recorded in the evidence.

Do not repeat completed historical tests merely to reproduce a summary. Start a new experiment only when it closes a missing gate, tests a materially different algorithm, or diagnoses a specific observed failure.

Do not include GRIDs/OpenPI results in this project.

