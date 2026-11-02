#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/cloud/cloud-ssd1/lightvla_updated_acceptance_2026-10-26}"
PACKAGE="${PACKAGE:-$SCRIPT_DIR}"
LOG="$ROOT/logs/04_targeted_adjudication_r1.log"
PATHS="$ROOT/paths.env"
DEPS="$ROOT/python_packages"
PREFLIGHT="$ROOT/results/01_real_fixed_observation_r6.json"
PRIOR="$ROOT/results/03_multitask_paired_r1.json"
OUTPUT="$ROOT/results/04_targeted_adjudication_r1.json"

mkdir -p "$ROOT/logs" "$ROOT/results"
exec >>"$LOG" 2>&1

echo "===== LIGHTVLA TARGETED ADJUDICATION START $(date -Is) ====="
DISCOVERY_PY="${DISCOVERY_PY:-/cloud/cloud-ssd1/lightvla_workspace/envs/vla-cache/bin/python}"
"$DISCOVERY_PY" "$PACKAGE/discover_paths.py" --output "$PATHS"
set -a
source "$PATHS"
set +a

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="$PACKAGE:$PROJECT:$DEPS${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ROOT/huggingface-cache"

test -f "$PREFLIGHT"
test -f "$PRIOR"

"$PY" "$PACKAGE/closed_loop_paired.py" \
    --checkpoint "$CHECKPOINT" \
    --preflight "$PREFLIGHT" \
    --prior-result "$PRIOR" \
    --output "$OUTPUT" \
    --task-ids 0,3 \
    --trials-per-task 15 \
    --trial-start 5 \
    --seed 7

echo "RESULT_JSON=$OUTPUT"
echo "LIGHTVLA_UPDATED_TARGETED_ADJUDICATION: PASS"
echo "===== LIGHTVLA TARGETED ADJUDICATION FINISH $(date -Is) ====="
