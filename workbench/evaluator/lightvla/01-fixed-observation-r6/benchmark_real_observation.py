#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform
import time
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
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere
from lightvla import LightVLAAccelerator


def aggregate(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
        "raw": [float(value) for value in values],
    }


def parameter_fingerprint(model) -> str:
    digest = hashlib.sha256()
    parameters = list(model.named_parameters())
    if not parameters:
        raise RuntimeError("Model has no parameters")
    indices = np.linspace(0, len(parameters) - 1, num=min(16, len(parameters)), dtype=int)
    for index in indices:
        name, parameter = parameters[int(index)]
        sample = parameter.detach().reshape(-1)[:4096].to(dtype=torch.float32, device="cpu").numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(sample.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    if args.warmups < 1 or args.iterations < 10:
        raise ValueError("Use at least 1 warmup and 10 measured iterations")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

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
        seed=args.seed,
        unnorm_key="libero_spatial_no_noops",
    )
    set_seed_everywhere(cfg.seed)

    print("===== LOAD MODEL =====", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    model.set_num_images_in_input(cfg.num_images_in_input)

    # This server reuses the VLA-Cache Transformers fork. Its LlamaModel.forward
    # reads these fields unconditionally, even when VLA-Cache is disabled.
    # Setting both to None selects the fork's ordinary, non-cache execution path
    # and prevents cache state from contaminating the LightVLA comparison.
    cache_config = model.language_model.config
    cache_config.reusable_patches = None
    cache_config.proportion_attn_var = None
    if cache_config.reusable_patches is not None or cache_config.proportion_attn_var is not None:
        raise RuntimeError("Failed to disable inherited VLA-Cache runtime state")
    print("VLA_CACHE_FIELDS_RESET: PASS", flush=True)

    # The official LightVLA LIBERO checkpoint stores its fine-tuning statistics
    # in dataset_statistics.json. Its config.json may still carry base OpenVLA
    # pretraining statistics, which use an incompatible action scale.
    stats_source = "model.config"
    statistics_path = Path(cfg.pretrained_checkpoint) / "dataset_statistics.json"
    if cfg.unnorm_key not in model.norm_stats and statistics_path.is_file():
        checkpoint_stats = json.loads(statistics_path.read_text(encoding="utf-8"))
        if cfg.unnorm_key in checkpoint_stats:
            model.norm_stats = checkpoint_stats
            stats_source = str(statistics_path.resolve())

    if cfg.unnorm_key not in model.norm_stats and "libero_spatial" in model.norm_stats:
        cfg.unnorm_key = "libero_spatial"
    if cfg.unnorm_key not in model.norm_stats:
        raise KeyError(
            f"Missing LIBERO-Spatial normalization key {cfg.unnorm_key!r}; "
            f"model keys={list(model.norm_stats)}; "
            f"dataset_statistics_exists={statistics_path.is_file()}"
        )

    action_stats = model.norm_stats[cfg.unnorm_key].get("action", {})
    required_action_stats = {"q01", "q99", "mask"}
    missing_action_stats = sorted(required_action_stats.difference(action_stats))
    if missing_action_stats:
        raise KeyError(f"Incomplete action statistics: missing {missing_action_stats}")
    if len(action_stats["q01"]) != 7 or len(action_stats["q99"]) != 7:
        raise ValueError("LIBERO action statistics must contain seven action dimensions")
    print(f"NORM_STATS_SOURCE={stats_source}", flush=True)
    print(f"UNNORM_KEY={cfg.unnorm_key}", flush=True)
    print("NORM_STATS_GATE: PASS", flush=True)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, model.llm_dim).to("cuda").eval()
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8).to("cuda").eval()
    print("OFFICIAL_COMPONENTS: PASS", flush=True)
    print(
        f"OFFICIAL_CONFIG images={cfg.num_images_in_input} proprio={cfg.use_proprio} "
        f"l1={cfg.use_l1_regression} chunk={cfg.num_open_loop_steps}",
        flush=True,
    )
    fingerprint_before = parameter_fingerprint(model)

    print("===== REAL LIBERO OBSERVATION =====", flush=True)
    suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.episode_id < len(initial_states):
        raise IndexError(f"episode-id {args.episode_id} outside [0, {len(initial_states)})")
    env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
    env.reset()
    obs = env.set_init_state(initial_states[args.episode_id])
    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
    resize_size = get_image_resize_size(cfg)
    image_signature = inspect.signature(get_libero_image)
    try:
        image_signature.bind(obs, resize_size)
    except TypeError:
        image = get_libero_image(obs)
        image_api = "get_libero_image(obs)"
    else:
        image = get_libero_image(obs, resize_size)
        image_api = "get_libero_image(obs, resize_size)"
    wrist_image = get_libero_wrist_image(obs)
    image = resize_image_for_policy(image, resize_size)
    wrist_image = resize_image_for_policy(wrist_image, resize_size)
    print(f"LIBERO_IMAGE_API={image_api}", flush=True)
    fixed_observation = {
        "full_image": image,
        "wrist_image": wrist_image,
        "state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
    }

    accelerator = None
    try:
        accelerator = LightVLAAccelerator.install(
            model,
            mode="baseline",
            patch_start=1,
            patch_count=None,
            score_context="suffix",
            allow_existing_pruner=True,
        )

        def infer(mode: str, measured: bool):
            accelerator.set_mode(mode)
            accelerator.reset()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            with torch.inference_mode():
                # get_vla_action normalizes obs["state"] in place, so each run
                # receives a fresh copy of the identical real observation.
                observation = {
                    "full_image": fixed_observation["full_image"],
                    "wrist_image": fixed_observation["wrist_image"],
                    "state": fixed_observation["state"].copy(),
                }
                action = get_vla_action(
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
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            action = np.asarray(action, dtype=np.float64)
            if not np.isfinite(action).all():
                raise RuntimeError(f"Non-finite action in mode {mode}")
            stats = accelerator.get_stats()
            record = {
                "wall_ms": elapsed_ms,
                "action": action.tolist(),
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "runtime_stats": stats,
            }
            if measured:
                print(
                    f"MEASURED mode={mode} wall_ms={elapsed_ms:.6f} "
                    f"input_len={stats.get('last_input_sequence_length')} "
                    f"output_len={stats.get('last_output_sequence_length')} "
                    f"kept={stats.get('last_patch_keep_counts')}",
                    flush=True,
                )
            return record

        print("===== WARMUP =====", flush=True)
        for _ in range(args.warmups):
            infer("baseline", measured=False)
            infer("packaged", measured=False)

        records = {"baseline": [], "packaged": []}
        print("===== ALTERNATING PAIRED MEASUREMENTS =====", flush=True)
        for iteration in range(args.iterations):
            order = ("baseline", "packaged") if iteration % 2 == 0 else ("packaged", "baseline")
            for mode in order:
                records[mode].append(infer(mode, measured=True))

        baseline_actions = np.asarray([item["action"] for item in records["baseline"]], dtype=np.float64)
        packaged_actions = np.asarray([item["action"] for item in records["packaged"]], dtype=np.float64)
        differences = np.abs(baseline_actions - packaged_actions)
        baseline_ms = [item["wall_ms"] for item in records["baseline"]]
        packaged_ms = [item["wall_ms"] for item in records["packaged"]]
        paired_saved_ms = [left - right for left, right in zip(baseline_ms, packaged_ms)]
        packaged_stats = records["packaged"][-1]["runtime_stats"]
        structural_gate = (
            packaged_stats.get("mode") == "packaged"
            and packaged_stats.get("pruned_forward_count", 0) > 0
            and packaged_stats.get("last_output_sequence_length") is not None
            and packaged_stats.get("last_input_sequence_length") is not None
            and packaged_stats["last_output_sequence_length"] < packaged_stats["last_input_sequence_length"]
        )
        fingerprint_after = parameter_fingerprint(model)
        weights_unchanged = fingerprint_before == fingerprint_after
        baseline_summary = aggregate(baseline_ms)
        packaged_summary = aggregate(packaged_ms)
        result = {
            "schema_version": 1,
            "module": "updated-lightvla",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "normalization_statistics_source": stats_source,
            "unnorm_key": cfg.unnorm_key,
            "task_suite": cfg.task_suite_name,
            "task_id": args.task_id,
            "episode_id": args.episode_id,
            "task_description": str(task_description),
            "seed": args.seed,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
            "protocol": {
                "input": "one fixed real LIBERO-Spatial observation",
                "official_lightvla_path": "two images + proprio + L1 action head + 8-action chunk",
                "warmups_per_mode": args.warmups,
                "paired_iterations": args.iterations,
                "alternating_order": True,
                "same_model_instance_and_weights": True,
            },
            "baseline": {
                "latency_ms": baseline_summary,
                "peak_memory_bytes": max(item["peak_memory_bytes"] for item in records["baseline"]),
                "first_action": records["baseline"][0]["action"],
            },
            "packaged": {
                "latency_ms": packaged_summary,
                "peak_memory_bytes": max(item["peak_memory_bytes"] for item in records["packaged"]),
                "first_action": records["packaged"][0]["action"],
                "runtime_stats": packaged_stats,
            },
            "comparison": {
                "wall_speedup": baseline_summary["median"] / packaged_summary["median"],
                "wall_latency_change_percent": (packaged_summary["median"] / baseline_summary["median"] - 1.0) * 100.0,
                "paired_saved_ms": aggregate(paired_saved_ms),
                "action_mae": float(differences.mean()),
                "action_max_abs_difference": float(differences.max()),
                "gripper_sign_agreement": float(
                    np.mean(np.sign(baseline_actions[..., -1]) == np.sign(packaged_actions[..., -1]))
                ),
            },
            "checks": {
                "structure_compatible": bool(structural_gate),
                "weights_unchanged": bool(weights_unchanged),
                "actions_finite": bool(np.isfinite(baseline_actions).all() and np.isfinite(packaged_actions).all()),
                "fingerprint_before": fingerprint_before,
                "fingerprint_after": fingerprint_after,
            },
            "raw_records": records,
        }
    finally:
        if accelerator is not None and accelerator.model is not None:
            accelerator.detach()
        close = getattr(env, "close", None)
        if callable(close):
            close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"comparison": result["comparison"], "checks": result["checks"], "packaged_stats": result["packaged"]["runtime_stats"]}, indent=2), flush=True)
    if not all(result["checks"].values()):
        raise SystemExit(2)
    print("LIGHTVLA_UPDATED_REAL_BENCHMARK: PASS", flush=True)


if __name__ == "__main__":
    main()
