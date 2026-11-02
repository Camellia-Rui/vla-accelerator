#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:?Set ROOT to the experiment output directory}"
PID_FILE="${PID_FILE:-$ROOT/run.pid}"
LAUNCH_LOG="${LAUNCH_LOG:-$ROOT/logs/launcher.log}"
mkdir -p "$ROOT/logs" "$ROOT/results"

if [[ -s "$PID_FILE" ]]; then
  old_pid="$(tr -dc '0-9' <"$PID_FILE")"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "ALREADY_RUNNING pid=$old_pid"
    echo TERMINAL_REMAINS_OPEN
    exit 0
  fi
  rm -f -- "$PID_FILE"
fi

nohup setsid env ROOT="$ROOT" PY="${PY:?Set PY to the target environment Python}" \
  bash "$SCRIPT_DIR/run.sh" >"$LAUNCH_LOG" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
echo "STARTED pid=$pid"
echo TERMINAL_REMAINS_OPEN

