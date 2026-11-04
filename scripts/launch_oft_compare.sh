#!/usr/bin/env bash
set -euo pipefail
# Usage: bash launch_oft_compare.sh PYTHON OUTPUT [server_oft_compare.py arguments]
PYTHON=$1
OUTPUT=$2
shift 2
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -e "$OUTPUT" ]]; then
  echo "Output exists; inspect results before restarting" >&2
  exit 2
fi
if [[ -f "${OUTPUT}.launcher.pid" ]] && kill -0 "$(cat "${OUTPUT}.launcher.pid")" 2>/dev/null; then
  echo "Existing process is live" >&2
  exit 2
fi
nohup setsid "$PYTHON" -u -B "$SCRIPT_DIR/server_oft_compare.py" --output "$OUTPUT" "$@" \
  >"${OUTPUT}.launcher.log" 2>&1 </dev/null &
echo $! > "${OUTPUT}.launcher.pid"
echo "PID=$(cat "${OUTPUT}.launcher.pid") OUTPUT=$OUTPUT"
echo TERMINAL_REMAINS_OPEN
