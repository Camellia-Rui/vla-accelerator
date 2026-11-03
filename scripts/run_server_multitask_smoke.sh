#!/usr/bin/env bash
# Bounded extension of the existing task-0 integration check. No policy rollout.
set -euo pipefail
PYTHON=/home/wxy/miniconda3/envs/openvla-oft/bin/python
PACKAGE=/home/wxy/Rui_work/vla-pruner-server-check-r1
OUTPUT=/home/wxy/Rui_work/toolbox-native-check
unset LD_PRELOAD
export LD_LIBRARY_PATH=/home/wxy/miniconda3/envs/openvla-oft/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/wxy/miniconda3/envs/openvla-oft/lib/python3.10/site-packages/nvidia/cublas/lib:/home/wxy/miniconda3/envs/openvla-oft/lib
cd "$PACKAGE"
for task_id in 1 2; do
    capture="$OUTPUT/capture-task${task_id}-trial0"
    pair="$OUTPUT/pruner-task${task_id}-trial0"
    if [[ -e "$capture" || -e "$pair" ]]; then
        printf 'Existing output for task %s; inspect before retrying.\n' "$task_id" >&2
        exit 2
    fi
    "$PYTHON" -u -B scripts/server_native_smoke.py capture --task "$task_id" --trial 0 \
        --gpu auto --output "$capture"
    "$PYTHON" -u -B scripts/launch_pruner_check.py --gpu auto \
        --observation-dir "$capture" --output "$pair"
done
