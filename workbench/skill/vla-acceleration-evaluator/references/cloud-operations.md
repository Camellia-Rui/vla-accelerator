# Remote/Jupyter experiment operations

Use these requirements for long GPU experiments.

## Package contents

Every handoff ZIP should include the actual benchmark, `run.sh`, `launch.sh`, `status.sh`, a summarizer when useful, a Chinese README, expected result/log paths, and a source/version manifest. Compile Python and run `bash -n` before packaging.

## Launch and PID guard

`launch.sh` must create the result/log/PID directories, reject a still-live PID, remove only a stale PID, then start `nohup setsid bash run.sh ... </dev/null &` and save `$!`. It must return promptly and print `TERMINAL_REMAINS_OPEN`. Never call `exit` in the user's pasteable wrapper in a way that closes the Jupyter terminal.

## Resume and atomic evidence

Use a stable record key such as `(mode, task, trial, seed, configuration)`. After every episode, write a temporary JSON in the same filesystem and atomically replace the result. Completed valid records are retained; records with errors are retried. A completed summary must state expected records, valid records, errors, and decision.

## Status

`status.sh` should report PID state, record progress, result status/decision when JSON is parseable, and a short log tail. A missing live PID plus incomplete result means failed/stopped, not completed.

## One-block handoff

Prefer one pasteable block that defines `ZIP`, `PKG`, and `ROOT`, extracts without overwriting source/checkpoints, makes scripts executable, launches in the background, waits briefly, prints status, and ends with `TERMINAL_REMAINS_OPEN`.

