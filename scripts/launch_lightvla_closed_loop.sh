#!/usr/bin/env bash
set -euo pipefail
# Usage: bash launch_lightvla_closed_loop.sh PYTHON WORKSPACE CHECKPOINT OUTPUT GPU
PYTHON=$1
WORKSPACE=$2
CHECKPOINT=$3
OUTPUT=$4
GPU=$5
shift 5
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PID_FILE="${OUTPUT}.launcher.pid"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "A launcher is still running; refusing duplicate execution" >&2
  exit 2
fi
if [[ -e "$OUTPUT" ]]; then
  echo "Output already exists; inspect results before choosing a fresh directory" >&2
  exit 2
fi
nohup setsid "$PYTHON" -u -B "$SCRIPT_DIR/server_lightvla_closed_loop.py" \
  --workspace "$WORKSPACE" --checkpoint "$CHECKPOINT" --output "$OUTPUT" --gpu "$GPU" "$@" \
  >"${OUTPUT}.launcher.log" 2>&1 </dev/null &
echo $! > "$PID_FILE"
echo "PID=$(cat "$PID_FILE") OUTPUT=$OUTPUT"
echo TERMINAL_REMAINS_OPEN
