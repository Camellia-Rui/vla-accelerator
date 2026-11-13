"""Run one method of the paired OFT / OFT+FastV LIBERO evaluation."""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from methods.fastv.live import publish_modes
from core.live_evaluation import publish

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["baseline", "fastv"], required=True)
parser.add_argument("--trials", type=int, default=50)
parser.add_argument("--tasks", type=int, nargs="+", default=None)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--gpu", type=int, default=3)
args = parser.parse_args()

if args.trials < 1:
    raise SystemExit("--trials must be positive")

# Require the validated launcher settings; do not silently choose a GPU.
if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
    raise SystemExit("CUDA_VISIBLE_DEVICES must match --gpu")
if os.environ.get("MUJOCO_EGL_DEVICE_ID") != str(args.gpu):
    raise SystemExit("MUJOCO_EGL_DEVICE_ID must match --gpu")

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

import inspect
import numpy as np
import torch

import experiments.robot.libero.run_libero_eval as evaluation
from experiments.robot.fastv_oft import enable_fastv_oft

repo = Path(os.environ["FASTV_WORKSPACE"])
output = Path(os.environ["FASTV_OUTPUT"])
cfg = evaluation.GenerateConfig(
    pretrained_checkpoint=os.environ["FASTV_CHECKPOINT"],
    use_l1_regression=True,
    use_diffusion=False,
    use_film=False,
    num_images_in_input=2,
    use_proprio=True,
    center_crop=True,
    num_open_loop_steps=8,
    load_in_8bit=False,
    load_in_4bit=False,
    task_suite_name="libero_spatial",
    num_trials_per_task=args.trials,
    use_wandb=False,
    seed=args.seed,
)

evaluation.validate_config(cfg)
evaluation.set_seed_everywhere(cfg.seed)

suite = evaluation.benchmark.get_benchmark_dict()[cfg.task_suite_name]()
task_ids = args.tasks if args.tasks is not None else list(range(suite.n_tasks))
live_report=dict(protocol=dict(tasks=task_ids,trials=list(range(args.trials)),seed=args.seed,
                              modes=[args.mode],action_chunk=8),records=[])

if len(set(task_ids)) != len(task_ids):
    raise SystemExit("Duplicate task IDs are not allowed")
if any(task_id < 0 or task_id >= suite.n_tasks for task_id in task_ids):
    raise SystemExit("Invalid task ID")

# Verify that both methods can use the same requested initial states.
for task_id in task_ids:
    states = suite.get_task_init_states(task_id)
    if len(states) < args.trials:
        raise SystemExit(f"Task {task_id} has too few initial states")

source_files = [
    "experiments/robot/openvla_utils.py",
    "experiments/robot/libero/run_libero_eval.py",
    "experiments/robot/fastv_oft.py",
    "prismatic/extern/hf/modeling_prismatic.py",
]
source_hashes = {
    name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
    for name in source_files
}

source_hashes["toolbox_runner"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

metadata = {
    "status": "initializing",
    "method": args.mode,
    "checkpoint": cfg.pretrained_checkpoint,
    "task_suite": cfg.task_suite_name,
    "task_ids": task_ids,
    "trials_per_task": args.trials,
    "seed": args.seed,
    "attention": "eager",
    "precision": "bfloat16",
    "use_cache": False,
    "num_images": 2,
    "use_proprio": True,
    "actions_per_chunk": 8,
    "fastv_k": 2 if args.mode == "fastv" else None,
    "prune_ratio": 0.5 if args.mode == "fastv" else None,
    "physical_gpu": args.gpu,
    "torch_version": torch.__version__,
    "cublas_preload": os.environ.get("LD_PRELOAD"),
    "timing_scope": (
        "evaluation.get_action, including policy preprocessing; "
        "excluding prepare_observation, env.step and video"
    ),
    "excluded_initial_calls": 5,
    "save_videos": False,
    "source_sha256": source_hashes,
    "LD_PRELOAD": os.environ.get("LD_PRELOAD"),
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
}


def save_metadata():
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


save_metadata()

# Keep a snapshot of GPU sharing conditions.
snapshot = subprocess.run(
    ["nvidia-smi"],
    capture_output=True,
    text=True,
    check=False,
)
(output / "nvidia-smi-start.txt").write_text(
    snapshot.stdout + snapshot.stderr,
    encoding="utf-8",
)

print("Output directory:", output, flush=True)
print("Loading model...", flush=True)

model, action_head, proprio_projector, noisy_action_projector, processor = (
    evaluation.initialize_model(cfg)
)
decoder = model.language_model.model

if decoder.config._attn_implementation != "eager":
    raise RuntimeError("Loaded decoder is not using eager attention")
if "removed_visual_tokens" not in inspect.getsource(
    type(model)._regression_or_discrete_prediction
):
    raise RuntimeError("Loaded model is missing the action-position correction")

model.language_model.config.use_cache = False
decoder.config.use_cache = False

if args.mode == "fastv":
    enable_fastv_oft(model, k=2, prune_ratio=0.5)

metadata["gpu_name"] = torch.cuda.get_device_name(0)
metadata["status"] = "running"
save_metadata()

resize_size = evaluation.get_image_resize_size(cfg)
os.chdir(output)

original_get_action = evaluation.get_action
original_run_episode = evaluation.run_episode
original_log_message = evaluation.log_message
episode_signature = inspect.signature(original_run_episode)

context = {"task_id": None, "episode_id": 0}
calls = 0
episode_calls = []
all_latencies = []
measured_latencies = []

action_file = (output / "actions.csv").open("w", newline="", buffering=1)
episode_file = (output / "episodes.csv").open("w", newline="", buffering=1)
log_file = (output / "eval.log").open("w", buffering=1)

action_writer = csv.DictWriter(
    action_file,
    fieldnames=[
        "task_id", "episode_id", "call_id",
        "latency_ms", "excluded_as_warmup",
    ],
)
episode_writer = csv.DictWriter(
    episode_file,
    fieldnames=[
        "task_id", "episode_id", "initial_state_sha256",
        "success", "inference_calls", "mean_chunk_ms",
    ],
)
action_writer.writeheader()
episode_writer.writeheader()


def timed_get_action(*positional, **keywords):
    global calls

    torch.cuda.synchronize()
    start = time.perf_counter()
    actions = original_get_action(*positional, **keywords)
    torch.cuda.synchronize()
    latency = (time.perf_counter() - start) * 1000

    values = np.asarray(actions)
    if values.shape != (8, 7) or not np.isfinite(values).all():
        raise RuntimeError("Invalid action chunk")

    calls += 1
    excluded = calls <= 5
    episode_calls.append({
        "task_id": context["task_id"],
        "episode_id": context["episode_id"],
        "call_id": calls,
        "latency_ms": latency,
        "excluded_as_warmup": excluded,
        "actions":values.tolist(),
    })
    all_latencies.append(latency)
    if not excluded:
        measured_latencies.append(latency)

    return actions


def strict_log_message(message, *positional, **keywords):
    original_log_message(message, *positional, **keywords)
    # The original evaluator otherwise counts exceptions as task failures.
    if str(message).startswith("Episode error:"):
        raise RuntimeError(str(message))


def recorded_run_episode(*positional, **keywords):
    episode_calls.clear()
    bound = episode_signature.bind(*positional, **keywords)
    bound.apply_defaults()

    state = np.ascontiguousarray(bound.arguments["initial_state"])
    state_hash = hashlib.sha256(
        str(state.shape).encode()
        + str(state.dtype).encode()
        + state.tobytes()
    ).hexdigest()

    started=time.perf_counter()
    try:
        success, images = original_run_episode(*positional, **keywords)
    finally:
        # Write outside the per-call timer, retaining calls on exceptions too.
        action_writer.writerows({k:v for k,v in r.items() if k!='actions'} for r in episode_calls)
        action_file.flush()

    episode_writer.writerow({
        "task_id": context["task_id"],
        "episode_id": context["episode_id"],
        "initial_state_sha256": state_hash,
        "success": bool(success),
        "inference_calls": len(episode_calls),
        "mean_chunk_ms": (
            float(np.mean([row["latency_ms"] for row in episode_calls]))
            if episode_calls else None
        ),
    })
    episode_file.flush()
    live_report['records'].append(dict(task_id=context['task_id'],trial=context['episode_id'],mode=args.mode,
        seed=args.seed,initial_state_hash=state_hash,complete=True,error=None,success=bool(success),
        queries=[dict(ms=r['latency_ms'],actions=r['actions']) for r in episode_calls],
        policy_ms=sum(r['latency_ms'] for r in episode_calls),episode_seconds=time.perf_counter()-started))
    publish_modes(output,live_report)
    context["episode_id"] += 1
    return success, images


evaluation.get_action = timed_get_action
evaluation.run_episode = recorded_run_episode
evaluation.log_message = strict_log_message

# Disable video encoding and disk growth for the formal comparison.
evaluation.save_rollout_video = lambda *positional, **keywords: None

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

total_episodes = 0
total_successes = 0

try:
    for task_id in task_ids:
        context["task_id"] = task_id
        context["episode_id"] = 0

        total_episodes, total_successes = evaluation.run_task(
            cfg,
            suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
        )

    expected = len(task_ids) * args.trials
    if total_episodes != expected:
        raise RuntimeError(f"Expected {expected} episodes, got {total_episodes}")
    publish(output,live_report,final=True)

    def latency_stats(values):
        if not values:
            return None
        return {
            "count": len(values),
            "mean_ms": float(np.mean(values)),
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
        }

    summary = {
        "method": args.mode,
        "episodes": total_episodes,
        "successes": total_successes,
        "success_rate": total_successes / total_episodes,
        "raw_latency": latency_stats(all_latencies),
        "latency_excluding_first_5_calls": latency_stats(measured_latencies),
        "peak_torch_allocated_gib": (
            torch.cuda.max_memory_allocated() / 1024**3
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metadata["status"] = "completed"
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

except BaseException as error:
    metadata["status"] = "aborted"
    metadata["error"] = repr(error)
    raise

finally:
    save_metadata()
    action_file.close()
    episode_file.close()
    log_file.close()
    print("Results:", output, flush=True)
