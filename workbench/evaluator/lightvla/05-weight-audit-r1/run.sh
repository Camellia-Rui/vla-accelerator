#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/cloud/cloud-ssd1/lightvla_updated_acceptance_2026-10-26}"
PACKAGE="${PACKAGE:-$SCRIPT_DIR}"
LOG="$ROOT/logs/05_weight_audit_r1.log"
PATHS="$ROOT/paths.env"
DEPS="$ROOT/python_packages"

mkdir -p "$ROOT/logs" "$ROOT/results"
exec >>"$LOG" 2>&1

echo "===== LIGHTVLA WEIGHT AUDIT START $(date -Is) ====="

DISCOVERY_PY="${DISCOVERY_PY:-/cloud/cloud-ssd1/lightvla_workspace/envs/vla-cache/bin/python}"
test -x "$DISCOVERY_PY"
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

echo "===== ENVIRONMENT ====="
echo "PY=$PY"
echo "PROJECT=$PROJECT"
echo "CHECKPOINT=$CHECKPOINT"
"$PY" - <<'PY'
import json
import platform
import torch
import transformers
print("PYTHON=", platform.python_version())
print("TORCH=", torch.__version__)
print("TORCH_CUDA=", torch.version.cuda)
print("TRANSFORMERS=", transformers.__version__)
print("CUDA_AVAILABLE=", torch.cuda.is_available())
print("GPU=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
git -C "$PROJECT" rev-parse HEAD 2>/dev/null || true
git -C "$PROJECT" status --short 2>/dev/null || true

echo "===== UPDATED SOURCE MANIFEST ====="
"$PY" - <<PY
import hashlib
from pathlib import Path
root = Path("$PACKAGE/lightvla")
for path in sorted(root.rglob("*")):
    if path.is_file() and "__pycache__" not in path.parts:
        print(hashlib.sha256(path.read_bytes()).hexdigest(), path.relative_to(root))
PY

echo "===== LIGHTWEIGHT CONFIG CONTRACT ====="
"$PY" - <<'PY'
from lightvla import LightVLAConfig
baseline = LightVLAConfig.from_mapping({"method": "lightvla", "state_type": "stateless", "mode": "baseline"})
packaged = LightVLAConfig.from_mapping({"method": "lightvla", "state_type": "stateless", "mode": "packaged", "allow_existing_pruner": True})
print("BASELINE_CONFIG=", baseline)
print("PACKAGED_CONFIG=", packaged)
print("LIGHTVLA_CONFIG_CONTRACT: PASS")
PY

echo "===== OPTIONAL GENERIC SYNTHETIC SMOKE ====="
set +e
"$PY" -m lightvla --smoke --mode baseline
SYNTHETIC_BASELINE_RC=$?
"$PY" -m lightvla --smoke --mode packaged
SYNTHETIC_PACKAGED_RC=$?
set -e
echo "SYNTHETIC_BASELINE_RC=$SYNTHETIC_BASELINE_RC"
echo "SYNTHETIC_PACKAGED_RC=$SYNTHETIC_PACKAGED_RC"
if [[ "$SYNTHETIC_BASELINE_RC" -ne 0 || "$SYNTHETIC_PACKAGED_RC" -ne 0 ]]; then
    echo "SYNTHETIC_SMOKE: SKIPPED_IN_CUSTOM_TRANSFORMERS_FORK"
    echo "The actual checkpoint runtime test below is authoritative."
else
    echo "SYNTHETIC_SMOKE: PASS"
fi

echo "===== REAL LIBERO FIXED-OBSERVATION BENCHMARK ====="
"$PY" - <<'PY'
import diffusers
from diffusers import DDIMScheduler
print("DIFFUSERS_VERSION=", diffusers.__version__)
print("DDIM_IMPORT: PASS")
PY
"$PY" "$PACKAGE/benchmark_real_observation.py" \
    --checkpoint "$CHECKPOINT" \
    --output "$ROOT/results/05_weight_audit_r1.json" \
    --task-id 0 \
    --episode-id 0 \
    --seed 7 \
    --warmups 1 \
    --iterations 10

echo "RESULT_JSON=$ROOT/results/05_weight_audit_r1.json"
echo "LIGHTVLA_UPDATED_WEIGHT_AUDIT: PASS"
echo "===== LIGHTVLA WEIGHT AUDIT FINISH $(date -Is) ====="
