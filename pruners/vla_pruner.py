"""
VLA-Pruner 加速插件封装

自包含、免训练、可被 Agent 直接调用的推理加速插件。
封装 OpenVLA-OFT + VLA-Pruner 的模型加载和推理流程。

用法:
    from vla_pruner_accelerator import VLAPrunerAccelerator, AcceleratorConfig

    acc = VLAPrunerAccelerator.install(
        checkpoint="/path/to/model",
        task_suite_name="libero_object",
        fastv_r=0.9275,
    )
    acc.start_episode()
    action = acc.step(image, wrist_image, instruction, proprio)
    stats = acc.end_episode()
"""
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class AcceleratorConfig:
    """加速器配置"""
    checkpoint: str = ""
    task_suite_name: str = "libero_object"
    unnorm_key: str = "libero_object"

    # VLA-Pruner 剪枝参数
    use_vla_pruner: bool = True
    use_fastv: bool = False
    use_vla_cache: bool = False
    fastv_k: int = 3
    fastv_r: float = 0.9275
    vla_pruner_layer: int = 15
    vla_pruner_mode: str = "semantic_action"
    vla_pruner_av_hist_w: int = 3
    vla_pruner_av_decay: float = 0.8
    vla_pruner_action_horizon: int = 0

    # 模型参数
    model_family: str = "openvla"
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    use_proprio: bool = True
    num_images_in_input: int = 2
    num_open_loop_steps: int = 8
    num_steps_wait: int = 10

    # 推理参数
    center_crop: bool = True
    device: str = "cuda:0"


class VLAPrunerAccelerator:
    """VLA-Pruner 加速器，供 Agent 调用"""

    def __init__(self, cfg: AcceleratorConfig):
        self.cfg = cfg
        self.model = None
        self.action_head = None
        self.proprio_projector = None
        self.noisy_action_projector = None
        self.processor = None
        self.last_caches = None
        self._episode_stats = {"steps": 0, "total_time": 0.0, "success": False}

    @classmethod
    def install(cls, checkpoint: str = None, task_suite_name: str = "libero_object", **kwargs):
        """一行安装加速器

        Args:
            checkpoint: 模型权重路径
            task_suite_name: LIBERO 任务套件名
            **kwargs: 其他 AcceleratorConfig 参数
        """
        cfg = AcceleratorConfig(
            checkpoint=checkpoint or "",
            task_suite_name=task_suite_name,
            unnorm_key=task_suite_name,
            **kwargs,
        )
        acc = cls(cfg)
        acc._load_model()
        return acc

    def _load_model(self):
        """加载模型和 VLA-Pruner"""
        import importlib
        import experiments.robot.libero.run_libero_eval as eval_module
        import experiments.robot.openvla_utils as openvla_utils

        # 构造 cfg 对象（复用现有 GenerateConfig）
        generate_cfg = eval_module.GenerateConfig(
            pretrained_checkpoint=self.cfg.checkpoint,
            task_suite_name=self.cfg.task_suite_name,
            model_family=self.cfg.model_family,
            use_l1_regression=self.cfg.use_l1_regression,
            use_diffusion=self.cfg.use_diffusion,
            use_film=self.cfg.use_film,
            use_proprio=self.cfg.use_proprio,
            num_images_in_input=self.cfg.num_images_in_input,
            num_open_loop_steps=self.cfg.num_open_loop_steps,
            num_steps_wait=self.cfg.num_steps_wait,
            center_crop=self.cfg.center_crop,
            use_vla_pruner=self.cfg.use_vla_pruner,
            use_fastv=self.cfg.use_fastv,
            use_vla_cache=self.cfg.use_vla_cache,
            fastv_k=self.cfg.fastv_k,
            fastv_r=self.cfg.fastv_r,
            vla_pruner_layer=self.cfg.vla_pruner_layer,
            vla_pruner_mode=self.cfg.vla_pruner_mode,
            vla_pruner_av_hist_w=self.cfg.vla_pruner_av_hist_w,
            vla_pruner_av_decay=self.cfg.vla_pruner_av_decay,
            vla_pruner_action_horizon=self.cfg.vla_pruner_action_horizon,
        )
        self.generate_cfg = generate_cfg

        # 加载模型
        (self.model, self.action_head, self.proprio_projector,
         self.noisy_action_projector, self.processor) = eval_module.initialize_model(generate_cfg)

    def start_episode(self):
        """开始新 episode，重置跨帧状态"""
        self.last_caches = None
        self._episode_stats = {"steps": 0, "total_time": 0.0, "success": False}
        if hasattr(self.model, "av_hist"):
            self.model.av_hist.clear()

    def step(self, image, wrist_image=None, instruction="", proprio=None) -> Dict[str, Any]:
        """闭环单步推理

        Args:
            image: 主视角图像 (np.ndarray 或 PIL Image)
            wrist_image: 腕部相机图像
            instruction: 语言指令
            proprio: 本体感知状态 (np.ndarray)
        Returns:
            {"action": 动作, "info": {"time_elapsed": ..., "success": ...}}
        """
        from experiments.robot.openvla_utils import get_vla_action

        # 组装 observation
        obs = {"full_image": image}
        if self.cfg.num_images_in_input > 1 and wrist_image is not None:
            obs["wrist_image"] = wrist_image
        if self.cfg.use_proprio and proprio is not None:
            obs["state"] = proprio

        # 调用推理
        actions, last_caches, result_image, metrics = get_vla_action(
            self.generate_cfg,
            self.model,
            self.processor,
            obs,
            task_label=instruction,
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            noisy_action_projector=self.noisy_action_projector,
            use_film=self.cfg.use_film,
            last_caches=self.last_caches,
        )
        self.last_caches = last_caches
        self._episode_stats["steps"] += 1
        self._episode_stats["total_time"] += metrics.get("time_elapsed", 0.0)

        action = actions[0] if isinstance(actions, list) else actions
        return {
            "action": action,
            "info": {
                "time_elapsed": metrics.get("time_elapsed", 0.0),
                "num_static_tokens_primary": metrics.get("num_static_tokens_primary", 0),
                "num_static_tokens_wrist": metrics.get("num_static_tokens_wrist", 0),
            },
        }

    def end_episode(self, success: bool = False) -> Dict[str, Any]:
        """结束 episode，返回统计"""
        self._episode_stats["success"] = success
        steps = self._episode_stats["steps"]
        total_time = self._episode_stats["total_time"]
        return {
            "steps": steps,
            "total_time": total_time,
            "avg_time_per_step": total_time / max(steps, 1),
            "control_freq_hz": steps / max(total_time, 1e-8),
            "success": success,
        }
