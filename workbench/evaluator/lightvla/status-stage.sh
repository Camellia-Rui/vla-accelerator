#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/cloud/cloud-ssd1/lightvla_updated_acceptance_2026-10-26}"
STAGE="${1:?usage: status-stage.sh 01|02|03|04|05}"

case "$STAGE" in
  01) RESULT="$ROOT/results/01_real_fixed_observation_r6.json"; LOG="$ROOT/logs/01_preflight_real_benchmark_r6.log" ;;
  02) RESULT="$ROOT/results/02_closed_loop_smoke_r1.json"; LOG="$ROOT/logs/02_closed_loop_smoke_r1.log" ;;
  03) RESULT="$ROOT/results/03_multitask_paired_r1.json"; LOG="$ROOT/logs/03_multitask_paired_r1.log" ;;
  04) RESULT="$ROOT/results/04_targeted_adjudication_r1.json"; LOG="$ROOT/logs/04_targeted_adjudication_r1.log" ;;
  05) RESULT="$ROOT/results/05_weight_audit_r1.json"; LOG="$ROOT/logs/05_weight_audit_r1.log" ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac

PID_FILE="$ROOT/pids/lightvla-stage-$STAGE.pid"
if [[ -s "$PID_FILE" ]]; then
  pid="$(tr -dc '0-9' <"$PID_FILE")"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "STATE=running STAGE=$STAGE PID=$pid"
  else
    echo "STATE=not_running STAGE=$STAGE STALE_PID=${pid:-invalid}"
  fi
else
  echo "STATE=no_pid STAGE=$STAGE"
fi

if [[ -f "$RESULT" ]]; then
  python - "$RESULT" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    d = json.loads(p.read_text(encoding="utf-8"))
except Exception as exc:
    print(f"RESULT=unreadable {type(exc).__name__}: {exc}")
else:
    print(f"RESULT={p}")
    print("STATUS=", d.get("status", "complete-file-no-status-field"), sep="")
    print("DECISION=", d.get("decision"), sep="")
    print("RECORDS=", len(d.get("records", [])), sep="")
PY
else
  echo "RESULT=missing PATH=$RESULT"
fi

[[ -f "$LOG" ]] && tail -n 30 "$LOG" || true
echo TERMINAL_REMAINS_OPEN

