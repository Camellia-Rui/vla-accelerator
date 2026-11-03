"""Two-stage LIBERO-Object capture/native OFT smoke check; no acceleration benchmark."""
import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

DEFAULT_BACKEND = "/home/wxy/Rui_work/VLA-Pruner/VLA-Pruner/src/openvla-oft"
DEFAULT_CHECKPOINT = "/home/wxy/Rui_work/openvla-7b-oft-libero-object"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_inventory():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    rows = []
    for line in output.strip().splitlines():
        index, uuid, free, utilization = [x.strip() for x in line.split(",")]
        rows.append(dict(index=int(index), uuid=uuid, free_mib=int(free), utilization=int(utilization)))
    return rows


def select_gpu(rows, requested, minimum):
    candidates = [r for r in rows if r["free_mib"] >= minimum
                  and (requested == "auto" or str(r["index"]) == requested)]
    if not candidates:
        raise RuntimeError(f"No selected GPU has at least {minimum} MiB free; no process was stopped")
    return min(candidates, key=lambda r: (r["utilization"], -r["free_mib"], r["index"]))


def config(checkpoint):
    return SimpleNamespace(
        pretrained_checkpoint=str(checkpoint), model_family="openvla", task_suite_name="libero_object",
        unnorm_key="libero_object_no_noops", use_l1_regression=True, use_diffusion=False,
        use_film=False, num_images_in_input=2, use_proprio=True, load_in_8bit=False,
        load_in_4bit=False, center_crop=True, num_open_loop_steps=8, lora_rank=32,
        use_vla_cache=False, use_fastv=False, use_vla_pruner=False,
        fastv_k=3, fastv_r=0.0, vla_pruner_layer=15, vla_pruner_mode="semantic_action",
        vla_pruner_av_hist_w=3, vla_pruner_av_decay=0.8, vla_pruner_action_horizon=0,
        vla_pruner_semantic_weight=0.5, vla_pruner_action_weight=0.5,
        merge_local_lora_adapter=False)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checkpoint_snapshot(checkpoint):
    # Small metadata/code content hashes plus weight file sizes; not a weight integrity proof.
    return {p.name: {"bytes": p.stat().st_size,
                     "sha256": digest(p) if p.suffix in (".py", ".json") else None}
            for p in sorted(checkpoint.iterdir()) if p.is_file()}


def capture(args, report):
    import numpy as np
    import yaml
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", str(Path.home() / ".libero")))
    asset_config = config_dir / "config.yaml"
    if not asset_config.is_file():
        raise FileNotFoundError(f"LIBERO config missing: {asset_config}; refusing interactive initialization")
    asset_paths = yaml.safe_load(asset_config.read_text())
    for key in ("assets", "bddl_files", "init_states"):
        if not Path(asset_paths[key]).is_dir():
            raise FileNotFoundError(f"LIBERO {key} missing: {asset_paths[key]}")
    report["libero_assets"] = asset_paths
    from libero.libero import benchmark
    from experiments.robot.libero import libero_utils as lu
    from experiments.robot import openvla_utils as vu
    from experiments.robot.robot_utils import get_image_resize_size
    report["libero_module"] = benchmark.__file__
    cfg = config(args.checkpoint)
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(args.task)
    states = suite.get_task_init_states(args.task)
    initial = np.asarray(states[args.trial]).copy()
    env = None
    try:
        env, description = lu.get_libero_env(task, cfg.model_family, resolution=256)
        env.seed(args.seed)
        env.reset()
        obs = env.set_init_state(initial)
        for _ in range(10):
            obs, _, done, _ = env.step(lu.get_libero_dummy_action(cfg.model_family))
            if done:
                raise RuntimeError("Episode terminated during settling; capture rejected")
        size = get_image_resize_size(cfg)
        try:
            inspect.signature(lu.get_libero_image).bind(obs, size)
        except TypeError:
            primary = lu.get_libero_image(obs)
        else:
            primary = lu.get_libero_image(obs, size)
        wrist = lu.get_libero_wrist_image(obs)
        primary = np.asarray(vu.resize_image_for_policy(primary, size))
        wrist = np.asarray(vu.resize_image_for_policy(wrist, size))
        state = np.concatenate((obs["robot0_eef_pos"], lu.quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"]))
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Expected finite 8D proprioception")
        for image in (primary, wrist):
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError("Expected uint8 RGB images after official preprocessing")
        np.savez_compressed(args.output / "observation.npz", full_image=primary, wrist_image=wrist,
                            state=state, initial_state=initial,
                            raw_primary=obs["agentview_image"], raw_wrist=obs["robot0_eye_in_hand_image"])
        report.update(task_suite="libero_object", task_id=args.task, trial=args.trial,
                      task_description=description, seed=args.seed, settle_steps=10,
                      observation_sha256=digest(args.output / "observation.npz"),
                      initial_state_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                      full_image_shape=list(primary.shape), wrist_image_shape=list(wrist.shape),
                      state_shape=list(state.shape), policy_actions_executed=0)
    finally:
        if env is not None:
            env.close()


def baseline(args, report):
    import numpy as np
    import torch
    from experiments.robot import openvla_utils as vu
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM
    if (NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM) != (8, 7, 8):
        raise ValueError("Unexpected backend action/proprio constants")
    source = args.observation_dir.resolve()
    capture_report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    observation_path = source / "observation.npz"
    if (capture_report.get("stage") != "capture" or not capture_report.get("passed")
            or digest(observation_path) != capture_report.get("observation_sha256")
            or capture_report.get("task_suite") != "libero_object"):
        raise ValueError("Observation provenance/hash validation failed")
    with np.load(observation_path, allow_pickle=False) as raw:
        obs = {key: raw[key].copy() for key in ("full_image", "wrist_image", "state")}
    cfg = config(args.checkpoint)
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
    for filename in set(index["weight_map"].values()):
        shard = (args.checkpoint / filename).resolve()
        if not shard.is_relative_to(args.checkpoint.resolve()) or not shard.is_file():
            raise FileNotFoundError(f"Invalid/missing weight shard: {filename}")
    before = checkpoint_snapshot(args.checkpoint)
    report.update(checkpoint=str(args.checkpoint), checkpoint_before=before, config=vars(cfg),
                  observation_sha256=capture_report["observation_sha256"],
                  task_description=capture_report["task_description"],
                  device=torch.cuda.get_device_name(0), torch_version=torch.__version__)
    try:
        print("Loading native OFT model; all acceleration flags disabled", flush=True)
        model = vu.get_vla(cfg)
        report["loaded_model_class_file"] = inspect.getfile(type(model))
        processor = vu.get_processor(cfg)
        head = vu.get_action_head(cfg, model.llm_dim)
        projector = vu.get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
        if cfg.unnorm_key not in model.norm_stats:
            raise ValueError("LIBERO-Object normalization key missing")
        with torch.inference_mode():
            torch.cuda.synchronize()
            start = time.perf_counter()
            actions, _, _, _ = vu.get_vla_action(
                cfg, model, processor, obs, capture_report["task_description"],
                action_head=head, proprio_projector=projector, use_film=False, last_caches=None)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
        actions = np.asarray(actions)
        if actions.shape != (8, 7) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid native actions: shape={actions.shape}")
        np.save(args.output / "actions.npy", actions, allow_pickle=False)
        report.update(actions=actions.tolist(), action_shape=list(actions.shape),
                      first_query_ms=elapsed, benchmark_valid=False, success_rate_measured=False,
                      note="One cold native query only; no speedup or task-success conclusion")
    finally:
        after = checkpoint_snapshot(args.checkpoint)
        report["checkpoint_metadata_and_sizes_unchanged"] = before == after
        report["weight_content_hash_verified"] = False
        if before != after:
            raise RuntimeError("Checkpoint metadata or file sizes changed during smoke check")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("capture", "baseline"))
    parser.add_argument("--backend", type=Path, default=Path(DEFAULT_BACKEND))
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--gpu", default="auto", help="physical nvidia-smi index, or auto")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--observation-dir", type=Path)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if min(args.task, args.trial, args.seed) < 0:
        parser.error("task/trial/seed must be nonnegative")
    if args.stage == "baseline" and args.observation_dir is None:
        parser.error("baseline requires --observation-dir")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"stage": args.stage, "passed": False, "python": sys.executable,
              "script_sha256": digest(__file__), "started_at_unix": time.time()}
    try:
        rows = gpu_inventory()
        chosen = select_gpu(rows, args.gpu, 32000 if args.stage == "baseline" else 4000)
        report.update(gpu_inventory=rows, selected_gpu=chosen)
        print("Selected GPU:", chosen, flush=True)
        os.environ.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(chosen["index"]),
                          MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                          MUJOCO_EGL_DEVICE_ID=str(chosen["index"]), TF_FORCE_GPU_ALLOW_GROWTH="true",
                          OPENVLA_SKIP_CHECKPOINT_SYNC="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(args.backend.resolve()))
        repo = args.backend.resolve().parents[1]
        report["backend_head"] = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        # Check actual working files against pinned core, while permitting separate evaluation script changes.
        paths = ["src/openvla-oft/prismatic", "src/openvla-oft/transformers",
                 "src/openvla-oft/experiments/robot/openvla_utils.py"]
        subprocess.run(["git", "-C", str(repo), "diff", "--quiet",
                        "84d4b7192c77abf1585610e2f12393319b7ebff9", "--", *paths], check=True)
        report["tracked_core_matches_pinned"] = True
        if args.stage == "capture":
            capture(args, report)
        else:
            baseline(args, report)
        report["passed"] = True
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
        print(report["traceback"], file=sys.stderr)
    finally:
        write_json(args.output / "report.json", report)
    print("REPORT:", args.output / "report.json", flush=True)
    print("PASS" if report["passed"] else "FAIL", flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
