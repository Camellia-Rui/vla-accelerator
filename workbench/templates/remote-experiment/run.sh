#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:?Set ROOT to the experiment output directory}"
PY="${PY:?Set PY to the target environment Python}"
LOG="${LOG:-$ROOT/logs/experiment.log}"
OUTPUT="${OUTPUT:-$ROOT/results/result.json}"

mkdir -p "$ROOT/logs" "$ROOT/results"
exec >>"$LOG" 2>&1
echo "===== EXPERIMENT START $(date -Is) ====="
"$PY" "$SCRIPT_DIR/benchmark.py" --output "$OUTPUT"
"$PY" "$SCRIPT_DIR/summarize.py" "$OUTPUT"
echo "RESULT_JSON=$OUTPUT"
echo "===== EXPERIMENT FINISH $(date -Is) ====="

