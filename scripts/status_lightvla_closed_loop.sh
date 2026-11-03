#!/usr/bin/env bash
set -euo pipefail
OUTPUT=$1
if [[ -f "${OUTPUT}.launcher.pid" ]] && kill -0 "$(cat "${OUTPUT}.launcher.pid")" 2>/dev/null; then
  echo RUNNING
else
  echo "NOT_RUNNING: only process.json and the complete report establish completion"
fi
for name in process.json progress.json; do
  if [[ -f "$OUTPUT/$name" ]]; then cat "$OUTPUT/$name"; fi
done
if [[ -f "$OUTPUT/process.log" ]]; then tail -8 "$OUTPUT/process.log"; fi
