#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:?Set ROOT to the experiment output directory}"
PID_FILE="${PID_FILE:-$ROOT/run.pid}"
LOG="${LOG:-$ROOT/logs/experiment.log}"
OUTPUT="${OUTPUT:-$ROOT/results/result.json}"

if [[ -s "$PID_FILE" ]]; then
  pid="$(tr -dc '0-9' <"$PID_FILE")"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "STATE=running PID=$pid"
  else
    echo "STATE=not_running STALE_PID=${pid:-invalid}"
  fi
else
  echo STATE=no_pid
fi

if [[ -f "$OUTPUT" ]]; then
  python - "$OUTPUT" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    d = json.loads(p.read_text(encoding="utf-8"))
except Exception as exc:
    print(f"RESULT=unreadable {type(exc).__name__}: {exc}")
else:
    print("RESULT_STATUS=", d.get("status"), sep="")
    print("RECORDS=", len(d.get("records", [])), sep="")
    print("DECISION=", d.get("decision"), sep="")
PY
else
  echo RESULT=missing
fi

[[ -f "$LOG" ]] && tail -n 30 "$LOG" || true
echo TERMINAL_REMAINS_OPEN

