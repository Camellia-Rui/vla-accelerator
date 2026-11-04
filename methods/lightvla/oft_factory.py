"""Read-only loader for an audited LightVLA OFT workspace; no checkpoint synchronization."""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def build(workspace, checkpoint, task_id=0, trial=0, seed=7):
    import numpy as np
    import torch
    from transformers import AutoModelForVision2Seq
    root, checkpoint = Path(workspace).resolve(), Path(checkpoint).resolve()
    if not root.is_dir() or not checkpoint.is_dir():
        raise ValueError("Existing local workspace and checkpoint required")
    # Fail before allocating GPU memory for an incompatible normalization target.
    stats = json.loads((checkpoint / "dataset_statistics.json").read_text())
    unnorm_key = next((key for key in ("libero_spatial_no_noops", "libero_spatial") if key in stats), None)
    if unnorm_key is None:
        raise ValueError("This loader requires dedicated LightVLA LIBERO-Spatial weights and statistics")
    if type(task_id) is not int or not 0 <= task_id < 10 or type(trial) is not int or trial < 0:
        raise ValueError("Spatial task_id must be 0..9 and trial must be nonnegative")
    sys.path[:0] = [str(root), str(root / "LIBERO")]
    from experiments.robot import openvla_utils as vu
    from experiments.robot.libero import libero_utils as lu
    from experiments.robot.robot_utils import set_seed_everywhere, get_image_resize_size
    from libero.libero import benchmark
    if Path(vu.__file__).resolve() != root / "experiments/robot/openvla_utils.py":
        raise ValueError("Model utilities imported from another workspace")
    cfg = SimpleNamespace(pretrained_checkpoint=str(checkpoint), model_family="openvla",
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True, num_open_loop_steps=8, lora_rank=32,
        center_crop=True, task_suite_name="libero_spatial", load_in_8bit=False,
        load_in_4bit=False, num_diffusion_steps_train=50, num_diffusion_steps_inference=50,
        unnorm_key=unnorm_key)
    set_seed_everywhere(seed)
    # Do not call get_vla: that upstream helper can rewrite the original checkpoint.
    model = AutoModelForVision2Seq.from_pretrained(str(checkpoint), trust_remote_code=True,
        local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda:0").eval()
    model.set_num_images_in_input(2)
    cache_config = model.language_model.config
    cache_config.reusable_patches = None
    cache_config.proportion_attn_var = None
    model.norm_stats = stats
    processor = vu.get_processor(cfg)
    head = vu.get_action_head(cfg, model.llm_dim).to("cuda:0").eval()
    proprio = vu.get_proprio_projector(cfg, model.llm_dim, 8).to("cuda:0").eval()
    suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    task = suite.get_task(task_id)
    states = suite.get_task_init_states(task_id)
    env, instruction = lu.get_libero_env(task, cfg.model_family, resolution=256)
    try:
        env.reset()
        obs = env.set_init_state(states[trial])
        for _ in range(10):
            obs, _, _, _ = env.step(lu.get_libero_dummy_action(cfg.model_family))
        size = get_image_resize_size(cfg)
        observation = dict(full_image=vu.resize_image_for_policy(lu.get_libero_image(obs), size),
            wrist_image=vu.resize_image_for_policy(lu.get_libero_wrist_image(obs), size),
            state=np.concatenate((obs["robot0_eef_pos"], lu.quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])))
    finally:
        env.close()
    def query(observation, instruction):
        with torch.inference_mode():
            return np.asarray(vu.get_vla_action(cfg, model, processor, copy.deepcopy(observation),
                instruction, head, proprio, None, False))
    def run():
        return query(observation, instruction)
    def prepare(obs):
        size = get_image_resize_size(cfg)
        return dict(full_image=vu.resize_image_for_policy(lu.get_libero_image(obs),size),
            wrist_image=vu.resize_image_for_policy(lu.get_libero_wrist_image(obs),size),
            state=np.concatenate((obs["robot0_eef_pos"],lu.quat2axisangle(obs["robot0_eef_quat"]),obs["robot0_gripper_qpos"])))
    paths = [Path(vu.__file__), Path(inspect.getfile(type(model))), checkpoint / "config.json",
             checkpoint / "dataset_statistics.json"]
    return dict(model=model, run=run, query=query, prepare=prepare, observation=observation, instruction=instruction,
        provenance={"workspace":str(root), "checkpoint":str(checkpoint), "task":task_id,
            "trial":trial, "seed":seed, "unnorm_key":cfg.unnorm_key,
            "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})
