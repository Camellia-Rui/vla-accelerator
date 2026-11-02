#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import inspect
import json
from pathlib import Path
import platform
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch
from libero.libero import benchmark
from transformers import AutoModelForVision2Seq

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla_action,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from lightvla import LightVLAAccelerator


def aggregate(values):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p95": None, "raw": []}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
        "raw": [float(value) for value in array],
    }


def parameter_fingerprint(model):
    digest = hashlib.sha256()
    parameters = list(model.named_parameters())
    indices = np.linspace(0, len(parameters) - 1, num=min(16, len(parameters)), dtype=int)
    for index in indices:
        name, parameter = parameters[int(index)]
        sample = parameter.detach().reshape(-1)[:4096].to(dtype=torch.float32, device="cpu").numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(sample.tobytes())
    return digest.hexdigest()


def capture_observation(obs, cfg):
    resize_size = get_image_resize_size(cfg)
    signature = inspect.signature(get_libero_image)
    try:
        signature.bind(obs, resize_size)
    except TypeError:
        full = get_libero_image(obs)
    else:
        full = get_libero_image(obs, resize_size)
    wrist = get_libero_wrist_image(obs)
    return {
        "full_image": resize_image_for_policy(full, resize_size),
        "wrist_image": resize_image_for_policy(wrist, resize_size),
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-ids", default="0,1")
    parser.add_argument("--trials-per-task", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    task_ids = [int(value) for value in args.task_ids.split(",") if value.strip()]
    if not task_ids or args.trials_per_task < 1:
        raise ValueError("At least one task and one trial are required")

    preflight = json.loads(Path(args.preflight).read_text(encoding="utf-8"))
    comparison = preflight["comparison"]
    preflight_gate = {
        "structure": bool(preflight["checks"]["structure_compatible"]),
        "weights": bool(preflight["checks"]["weights_unchanged"]),
        "speedup": comparison["wall_speedup"] >= 1.03,
        "action_mae": comparison["action_mae"] <= 0.05,
        "action_max": comparison["action_max_abs_difference"] <= 0.5,
        "gripper": comparison["gripper_sign_agreement"] >= 0.95,
    }
    print("PREFLIGHT_GATE=", json.dumps(preflight_gate, sort_keys=True), flush=True)
    if not all(preflight_gate.values()):
        raise RuntimeError("Fixed-observation preflight did not satisfy closed-loop admission gates")

    cfg = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=args.checkpoint,
        load_in_8bit=False,
        load_in_4bit=False,
        use_l1_regression=True,
        use_diffusion=False,
        num_diffusion_steps_train=50,
        num_diffusion_steps_inference=50,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        num_open_loop_steps=8,
        lora_rank=32,
        center_crop=True,
        task_suite_name="libero_spatial",
        num_steps_wait=10,
        unnorm_key="libero_spatial_no_noops",
        seed=args.seed,
    )

    set_seed_everywhere(cfg.seed)
    print("===== LOAD OFFICIAL LIGHTVLA COMPONENTS =====", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    model.set_num_images_in_input(cfg.num_images_in_input)
    model.language_model.config.reusable_patches = None
    model.language_model.config.proportion_attn_var = None

    stats_path = Path(cfg.pretrained_checkpoint) / "dataset_statistics.json"
    checkpoint_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if cfg.unnorm_key not in checkpoint_stats:
        raise KeyError(f"Missing {cfg.unnorm_key!r} in {stats_path}")
    model.norm_stats = checkpoint_stats

    processor = get_processor(cfg)
    action_head = get_action_head(cfg, model.llm_dim).to("cuda").eval()
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8).to("cuda").eval()
    fingerprint_before = parameter_fingerprint(model)

    accelerator = LightVLAAccelerator.install(
        model,
        mode="baseline",
        patch_start=1,
        patch_count=None,
        score_context="suffix",
        allow_existing_pruner=True,
    )
    suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []

    def run_episode(mode, task_id, trial, pair_seed):
        set_seed_everywhere(pair_seed)
        accelerator.set_mode(mode)
        accelerator.start_episode()
        model.language_model.config.reusable_patches = None
        model.language_model.config.proportion_attn_var = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        if trial >= len(initial_states):
            raise IndexError(f"trial {trial} unavailable for task {task_id}")
        env = None
        query_ms = []
        steps = 0
        success = False
        error = None
        task_description = None
        queue = deque(maxlen=cfg.num_open_loop_steps)
        try:
            env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
            env.reset()
            obs = env.set_init_state(initial_states[trial])
            for _ in range(cfg.num_steps_wait):
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

            for _ in range(220):
                if not queue:
                    observation = capture_observation(obs, cfg)
                    torch.cuda.synchronize()
                    started = time.perf_counter_ns()
                    with torch.inference_mode():
                        actions = get_vla_action(
                            cfg,
                            model,
                            processor,
                            observation,
                            task_description,
                            action_head,
                            proprio_projector,
                            None,
                            cfg.use_film,
                        )
                    torch.cuda.synchronize()
                    query_ms.append((time.perf_counter_ns() - started) / 1e6)
                    actions = np.asarray(actions, dtype=np.float64)
                    if actions.shape != (8, 7) or not np.isfinite(actions).all():
                        raise RuntimeError(f"Unexpected action chunk: shape={actions.shape}")
                    queue.extend(actions)

                action = np.asarray(queue.popleft(), dtype=np.float64).copy()
                action = normalize_gripper_action(action, binarize=True)
                action = invert_gripper_action(action)
                obs, _, done, _ = env.step(action.tolist())
                steps += 1
                if done:
                    success = True
                    break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            if env is not None:
                close = getattr(env, "close", None)
                if callable(close):
                    close()

        record = {
            "mode": mode,
            "task_id": task_id,
            "trial": trial,
            "pair_seed": pair_seed,
            "task_description": task_description,
            "success": success,
            "steps": steps,
            "queries": len(query_ms),
            "query_latency_ms": aggregate(query_ms),
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "runtime_stats": accelerator.get_stats(),
            "error": error,
        }
        print(
            f"EPISODE_END mode={mode} task={task_id} trial={trial} "
            f"success={success} steps={steps} queries={len(query_ms)} error={error}",
            flush=True,
        )
        return record

    try:
        for task_id in task_ids:
            for trial in range(args.trials_per_task):
                pair_seed = args.seed + task_id * 1000 + trial
                order = ("baseline", "packaged") if (task_id + trial) % 2 == 0 else ("packaged", "baseline")
                for mode in order:
                    records.append(run_episode(mode, task_id, trial, pair_seed))
                    output.write_text(
                        json.dumps({"status": "running", "records": records}, indent=2), encoding="utf-8"
                    )

        fingerprint_after = parameter_fingerprint(model)
        by_mode = {mode: [item for item in records if item["mode"] == mode] for mode in ("baseline", "packaged")}
        latencies = {
            mode: [value for item in by_mode[mode] for value in item["query_latency_ms"]["raw"]]
            for mode in by_mode
        }
        baseline_latency = aggregate(latencies["baseline"])
        packaged_latency = aggregate(latencies["packaged"])
        baseline_successes = sum(item["success"] for item in by_mode["baseline"])
        packaged_successes = sum(item["success"] for item in by_mode["packaged"])
        errors = [item for item in records if item["error"]]
        pruning_observed = any(
            item["runtime_stats"].get("pruned_forward_count", 0) > 0
            and item["runtime_stats"].get("last_output_sequence_length", 10**9)
            < item["runtime_stats"].get("last_input_sequence_length", 0)
            for item in by_mode["packaged"]
        )
        speedup = baseline_latency["median"] / packaged_latency["median"]
        checks = {
            "all_pairs_complete": len(records) == len(task_ids) * args.trials_per_task * 2,
            "no_episode_errors": not errors,
            "weights_unchanged": fingerprint_before == fingerprint_after,
            "packaged_pruning_observed": pruning_observed,
            "inference_speedup_at_least_1_03x": speedup >= 1.03,
            "packaged_success_not_below_baseline": packaged_successes >= baseline_successes,
        }
        result = {
            "schema_version": 1,
            "status": "complete",
            "module": "updated-lightvla",
            "protocol": {
                "suite": cfg.task_suite_name,
                "task_ids": task_ids,
                "trials_per_task": args.trials_per_task,
                "paired_initial_states": True,
                "fresh_environment_per_episode": True,
                "alternating_mode_order": True,
                "official_inference": "two images + proprio + L1 head + 8-action chunk",
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "gpu": torch.cuda.get_device_name(0),
            },
            "baseline": {
                "episodes": len(by_mode["baseline"]),
                "successes": baseline_successes,
                "success_rate": baseline_successes / len(by_mode["baseline"]),
                "query_latency_ms": baseline_latency,
            },
            "packaged": {
                "episodes": len(by_mode["packaged"]),
                "successes": packaged_successes,
                "success_rate": packaged_successes / len(by_mode["packaged"]),
                "query_latency_ms": packaged_latency,
            },
            "comparison": {
                "query_latency_speedup": speedup,
                "success_delta": packaged_successes - baseline_successes,
            },
            "checks": checks,
            "decision": "PASS" if all(checks.values()) else "REVIEW_OR_FAIL",
            "fingerprint_before": fingerprint_before,
            "fingerprint_after": fingerprint_after,
            "records": records,
        }
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({key: result[key] for key in ("baseline", "packaged", "comparison", "checks", "decision")}, indent=2), flush=True)
        if not all(checks.values()):
            raise SystemExit(2)
        print("LIGHTVLA_UPDATED_CLOSED_LOOP_SMOKE: PASS", flush=True)
    finally:
        accelerator.detach()


if __name__ == "__main__":
    main()
