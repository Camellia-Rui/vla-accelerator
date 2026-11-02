# Remote experiment handoff dialogue

When evidence shows a new cloud experiment is necessary, the Codex should do the build
work first. It should not ask the user to create files manually.

## Required Codex actions before replying

1. Create a new uniquely versioned experiment directory.
2. Include the actual benchmark, `run.sh`, `launch.sh`, `status.sh`, summarizer, source
   manifest, and Chinese README.
3. Implement live-PID duplicate protection, record-level resume, retry of error records,
   atomic writes, exact output paths, and `TERMINAL_REMAINS_OPEN`.
4. Run Python syntax/self-tests, `bash -n`, create checksums, ZIP it, re-extract it, and
   verify the extracted package.
5. Preserve old repositories, checkpoints, results, and unrelated user files.

## Reply format

Return:

1. Why the experiment is needed and which missing gate it closes.
2. One link to the complete ZIP and its SHA256.
3. One pasteable command block that extracts, launches in the background, waits briefly,
   runs status, and leaves the Jupyter terminal open.
4. Expected JSON/log paths and the result files the user should upload.

Do not provide multiple alternative command blocks unless the current environment truly
requires a user choice.

