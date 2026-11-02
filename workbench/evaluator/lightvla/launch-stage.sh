#!/usr/bin/env bash
set -Eeuo pipefail

BASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/cloud/cloud-ssd1/lightvla_updated_acceptance_2026-10-26}"
STAGE="${1:?usage: launch-stage.sh 01|02|03|04|05}"

case "$STAGE" in
  01) DIR="$BASE_DIR/01-fixed-observation-r6"; RESULT="$ROOT/results/01_real_fixed_observation_r6.json" ;;
  02) DIR="$BASE_DIR/02-closed-loop-smoke-r1"; RESULT="$ROOT/results/02_closed_loop_smoke_r1.json" ;;
  03) DIR="$BASE_DIR/03-multitask-paired-r1"; RESULT="$ROOT/results/03_multitask_paired_r1.json" ;;
  04) DIR="$BASE_DIR/04-adjudication-r1"; RESULT="$ROOT/results/04_targeted_adjudication_r1.json" ;;
  05) DIR="$BASE_DIR/05-weight-audit-r1"; RESULT="$ROOT/results/05_weight_audit_r1.json" ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac

mkdir -p "$ROOT/logs" "$ROOT/results" "$ROOT/pids"
PID_FILE="$ROOT/pids/lightvla-stage-$STAGE.pid"
LAUNCH_LOG="$ROOT/logs/lightvla-stage-$STAGE-launcher.log"

if [[ -s "$PID_FILE" ]]; then
  old_pid="$(tr -dc '0-9' <"$PID_FILE")"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "ALREADY_RUNNING stage=$STAGE pid=$old_pid"
    echo TERMINAL_REMAINS_OPEN
    exit 0
  fi
  rm -f -- "$PID_FILE"
fi

if [[ -s "$RESULT" && "${FORCE:-0}" != "1" ]]; then
  echo "RESULT_ALREADY_EXISTS stage=$STAGE path=$RESULT"
  echo "Refusing to rerun. Inspect/reuse it, or archive it and set FORCE=1 with an explicit reason."
  echo TERMINAL_REMAINS_OPEN
  exit 0
fi

nohup setsid env ROOT="$ROOT" PACKAGE="$DIR" DISCOVERY_PY="${DISCOVERY_PY:-/cloud/cloud-ssd1/lightvla_workspace/envs/vla-cache/bin/python}" \
  bash "$DIR/run.sh" >"$LAUNCH_LOG" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
echo "STARTED stage=$STAGE pid=$pid result=$RESULT"
echo TERMINAL_REMAINS_OPEN

