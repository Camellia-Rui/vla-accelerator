"""External backend contract. No model imports or downloads at import time."""

from __future__ import annotations

import importlib
import os
from types import SimpleNamespace
from typing import Any

BACKEND_REPOSITORY = "https://github.com/MINT-SJTU/VLA-Pruner"
BACKEND_COMMIT = "84d4b7192c77abf1585610e2f12393319b7ebff9"


def check_model(model: Any, cfg: Any) -> None:
    """Reject unpatched models before changing attributes."""
    required = ("predict_action", "_build_fastv_config", "reset_av_history")
    if model is None or any(not callable(getattr(model, name, None)) for name in required):
        raise TypeError("Expected the VLA-Pruner OpenVLA-OFT model; see methods/vla_pruner/INTEGRATION.md")
    lm = getattr(model, "language_model", None)
    decoder = getattr(lm, "model", None)
    if (getattr(model, "config", None) is None or getattr(lm, "config", None) is None
            or not callable(getattr(lm, "fastv_forward", None))
            or not callable(getattr(decoder, "fastv_forward", None))
            or not callable(getattr(decoder, "_fastv_pruning_indices", None))):
        raise TypeError("Missing VLA-Pruner LLaMA backend; install the pinned upstream Transformers fork")
    depth = len(decoder.layers)
    if cfg.fastv_k >= depth or cfg.vla_pruner_layer >= depth:
        raise ValueError(f"fastv_k and vla_pruner_layer must be below decoder depth {depth}")
    backbone = getattr(model, "vision_backbone", None)
    for name in ("get_num_patches", "get_num_images_in_input"):
        if not callable(getattr(backbone, name, None)):
            raise TypeError(f"Missing vision_backbone.{name}")
    if backbone.get_num_images_in_input() != cfg.num_images_in_input:
        raise ValueError("Model view count differs from num_images_in_input; configure the model before attach")
    if not hasattr(model, "av_hist"):
        raise TypeError("Missing model.av_hist temporal state")


class OpenVLAOFTBackend:
    def __init__(self, utils: Any):
        self.utils = utils
        self.get_vla_action = utils.get_vla_action

    def load_model(self, cfg: Any) -> dict[str, Any]:
        # Upstream otherwise rewrites local checkpoint remote-code/config files.
        if os.environ.get("OPENVLA_SKIP_CHECKPOINT_SYNC") != "1":
            raise RuntimeError("Set OPENVLA_SKIP_CHECKPOINT_SYNC=1 before from_pretrained; checkpoint files must already match the backend")
        values = dict(vars(cfg))
        values.update(pretrained_checkpoint=cfg.checkpoint, model_family="openvla",
                      use_l1_regression=True, use_diffusion=False, use_vla_cache=False,
                      use_vla_pruner=False, use_fastv=False, merge_local_lora_adapter=False)
        load_cfg = SimpleNamespace(**values)
        model = self.utils.get_vla(load_cfg)
        return {"model": model, "processor": self.utils.get_processor(load_cfg),
                "action_head": self.utils.get_action_head(load_cfg, model.llm_dim),
                "proprio_projector": self.utils.get_proprio_projector(load_cfg, model.llm_dim, proprio_dim=8)
                if cfg.use_proprio else None}


def load_backend() -> OpenVLAOFTBackend:
    try:
        utils = importlib.import_module("experiments.robot.openvla_utils")
    except ImportError as exc:
        raise ImportError("VLA-Pruner inference dependencies are unavailable. Install the pinned backend and expose src/openvla-oft on PYTHONPATH; see methods/vla_pruner/INTEGRATION.md") from exc
    if not callable(getattr(utils, "_configure_attention_pruning", None)):
        raise ImportError("Imported an incompatible openvla_utils; check PYTHONPATH and the pinned backend revision")
    return OpenVLAOFTBackend(utils)
