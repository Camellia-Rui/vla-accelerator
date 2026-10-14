"""SpecPrune-VLA inference accelerator for OpenVLA-OFT.

封装推理期视觉 token 剪枝、动态剪枝和动作感知控制。
通过 install() 注入已加载的 OpenVLA-OFT 模型。
"""

import logging
import os
import time
import types
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast

logger = logging.getLogger("spec_prune_accelerator")


# 视觉 token 布局：BOS | 主视图 256 | 腕视图 256 | prompt | 动作 token | stop
NUM_VISION_TOKENS_PER_VIEW = 256
PRIMARY_VISUAL_TOKEN_START = 1
WRIST_VISUAL_TOKEN_START = PRIMARY_VISUAL_TOKEN_START + NUM_VISION_TOKENS_PER_VIEW   # 257
VISUAL_TOKEN_END = WRIST_VISUAL_TOKEN_START + NUM_VISION_TOKENS_PER_VIEW            # 513

DYNAMIC_PRUNE_RATIO = 0.9        # 深层动态剪枝保留比例
STATIC_PRUNE_RATIO = 0.8         # 静态 top-K 缩放系数

# 文本引导 top-K：15/12 是 original/base precise 默认值。
ATTN_TOPK = int(30 * STATIC_PRUNE_RATIO)            # 24
ATTN_TOPK_WRIST = int(24 * STATIC_PRUNE_RATIO)      # 19
ATTN_TOPK_PRECISE = int(ATTN_TOPK * STATIC_PRUNE_RATIO**2)               # 15
ATTN_TOPK_PRECISE_WRIST = int(ATTN_TOPK_WRIST * STATIC_PRUNE_RATIO**2)   # 12

# 按 suite 冻结的 precise 配置：Object 15/12，Goal 40/31。
FINAL_PRECISE_TOPK = {
    "libero_object": (15, 12),
    "libero_goal": (40, 31),
}


def get_final_precise_topk(suite: str) -> Tuple[int, int]:
    """未冻结 suite 回退 base 15/12，不代表最终参数。"""
    return FINAL_PRECISE_TOPK.get(
        suite, (ATTN_TOPK_PRECISE, ATTN_TOPK_PRECISE_WRIST)
    )

# 帧差相似度预算：每视图可剪除的低变化 patch 上限。
PRIMARY_TOPK = 236
PRIMARY_TOPK_PRECISE = 240
WRIST_TOPK = 230
WRIST_TOPK_PRECISE = 236
PRIMARY_SIM_THRESHOLD = 0.986
WRIST_SIM_THRESHOLD = 0.98

# 动态重要性在更新层估计，在剪枝层实际移除 token。
DYNAMIC_IMPORTANCE_UPDATE_LAYERS = (14, 19, 24)
DYNAMIC_PRUNE_LAYERS = (10, 15, 20, 25)
DYNAMIC_IMPORTANCE_EMA_BETA = 0.2
MIN_VISUAL_RETAIN_TOKENS = 60
DYNAMIC_PRUNE_SAFETY_MARGIN = 58

# 动作感知阈值：低速进入 precise，Z/旋转较快且队列耗尽时退出。
VELOCITY_THRES = 0.35
ROT_THRES = 0.2
XY_THRES = 0.1
Z_THRES = 0.1
PRECISE_EXIT_Z_THRES = 0.2
PRECISE_EXIT_ROT_THRES = 0.3
SKIP_LAYER = 3

ACTION_DIM = 7
NUM_ACTIONS_CHUNK = 8              # 单次模型查询预测 8 步动作
IGNORE_INDEX = -100
EMPTY_TOKEN_ID = 29871

class NormalizationType(str, Enum):
    BOUNDS = "bounds"
    BOUNDS_Q99 = "bounds_q99"
ACTION_PROPRIO_NORMALIZATION_TYPE = NormalizationType.BOUNDS_Q99

OPENVLA_IMAGE_SIZE = 224
DEFAULT_NUM_PROMPT_TOKENS = 34


def resize_image_for_policy(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """按 RLDS 预处理：JPEG 往返、lanczos3 resize、取整。"""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    return img.numpy()


def crop_and_resize(image: tf.Tensor, crop_scale: float, batch_size: int) -> tf.Tensor:
    """推理时以中心裁剪对应训练阶段的随机裁剪。"""
    assert image.shape.ndims in (3, 4), "Image must be 3D or 4D tensor"
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [height_offsets, width_offsets, height_offsets + new_heights, width_offsets + new_widths], axis=1
    )
    image = tf.image.crop_and_resize(
        image, bounding_boxes, tf.range(batch_size), (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE)
    )
    if expanded_dims:
        image = image[0]
    return image


def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """中心裁剪 90% 面积。"""
    batch_size = 1
    crop_scale = 0.9
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))
    orig_dtype = image.dtype
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = crop_and_resize(image, crop_scale, batch_size)
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
    return Image.fromarray(image.numpy()).convert("RGB")


def check_image_format(image: Any) -> None:
    is_numpy_array = isinstance(image, np.ndarray)
    has_correct_shape = len(image.shape) == 3 and image.shape[-1] == 3
    has_correct_dtype = image.dtype == np.uint8
    assert is_numpy_array and has_correct_shape and has_correct_dtype, (
        "Incorrect image format detected! input must be np.uint8 (H, W, 3)."
    )


def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    """按 checkpoint 的 BOUNDS_Q99 统计归一化本体状态。"""
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")
    normalized_proprio = np.clip(
        np.where(mask, 2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1, proprio),
        a_min=-1.0, a_max=1.0,
    )
    return normalized_proprio


def prepare_images_for_vla(images: List[np.ndarray], cfg: Any) -> List[Image.Image]:
    processed_images = []
    for image in images:
        check_image_format(image)
        if image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            image = resize_image_for_policy(image, OPENVLA_IMAGE_SIZE)
        pil_image = Image.fromarray(image).convert("RGB")
        if cfg.center_crop:
            pil_image = center_crop_image(pil_image)
        processed_images.append(pil_image)
    return processed_images


def multi_image_processing(cfg: Any, obs):
    all_images = [obs["full_image"]]
    if cfg.num_images_in_input > 1:
        all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])
    all_images = prepare_images_for_vla(all_images, cfg)
    result_image = all_images.copy()
    prev_images = prepare_images_for_vla(obs["prev_images"], cfg)
    return all_images, result_image, prev_images


def _patchify(image, patch_size=14):
    image = np.array(image)
    assert image.shape[0] % patch_size == 0 and image.shape[1] % patch_size == 0, \
        "Image dimensions must be divisible by patch size."
    h, w = image.shape[:2]
    if image.ndim == 3:
        c = image.shape[2]
        patches = image.reshape(h // patch_size, patch_size, w // patch_size, patch_size, c)
        patches = patches.transpose(0, 2, 1, 3, 4).reshape(-1, patch_size, patch_size, c)
    else:
        patches = image.reshape(h // patch_size, patch_size, w // patch_size, patch_size)
        patches = patches.transpose(0, 2, 1, 3).reshape(-1, patch_size, patch_size)
    return patches


def _calculate_patch_similarity(patch1, patch2):
    """逐 patch 计算余弦相似度。"""
    flat1 = patch1.reshape(len(patch1), -1).astype(np.float32)
    flat2 = patch2.reshape(len(patch2), -1).astype(np.float32)
    norm1 = np.linalg.norm(flat1, axis=1)
    norm2 = np.linalg.norm(flat2, axis=1)
    dot = np.sum(flat1 * flat2, axis=1)
    return dot / (norm1 * norm2 + 1e-8)


def _preprocess_image_for_patchify(image, patch_size=14):
    image = np.array(image)
    h, w = image.shape[:2]
    target_h = (h // patch_size) * patch_size
    target_w = (w // patch_size) * patch_size
    if target_h == 224 and target_w == 224:
        return image
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    return image[start_h:start_h + target_h, start_w:start_w + target_w]


def get_similarity_indices(img_0, img_1, top_k, patch_size=14, sim_threshold=0.996, view="primary"):
    """找出低变化 patch；返回值使用含 BOS 偏移的绝对 token id。"""
    img_0 = _preprocess_image_for_patchify(img_0)
    img_1 = _preprocess_image_for_patchify(img_1)
    patches1 = _patchify(img_0, patch_size)
    patches2 = _patchify(img_1, patch_size)
    similarity = _calculate_patch_similarity(patches1, patches2)
    grid_h = img_0.shape[0] // patch_size
    grid_w = img_0.shape[1] // patch_size
    flat_indices = np.where(similarity >= sim_threshold)[0]
    if len(flat_indices) == 0:
        return np.array([], dtype=int)
    k = min(top_k, len(flat_indices))
    top_k_flat = flat_indices[np.argpartition(similarity[flat_indices], -k)[-k:]]
    row_ids, col_ids = np.unravel_index(top_k_flat, (grid_h, grid_w))
    patch_ids = row_ids * grid_w + col_ids + 1  # 跳过 BOS
    if view == "wrist":
        patch_ids += 256
    return patch_ids


def vlm_layer_attn(multihead_attention, num_tokens=34, layer_indices=None, primary=True):
    """汇总各层文本到视觉的注意力。"""
    num_layers = len(multihead_attention)
    layer_indices = range(layer_indices) if layer_indices is not None else range(num_layers)
    v_token_start = 1 if primary else 257
    v_token_end = v_token_start + 256
    t_token_start = 513
    t_token_end = t_token_start + num_tokens
    attn_dict = {}
    token_position = {}
    for layer_idx in layer_indices:
        attn_map = multihead_attention[layer_idx][0].to(torch.float32).squeeze(0).mean(dim=0)
        token_idx = multihead_attention[layer_idx][1]
        text_mask = (token_idx >= t_token_start + 9) & (token_idx < t_token_end)
        vision_mask = (token_idx >= v_token_start) & (token_idx < v_token_end)
        relation = attn_map[text_mask][:, vision_mask]
        token_position[layer_idx] = token_idx
        attn_dict[layer_idx] = relation.mean(dim=0)
    return attn_dict, token_position


def get_layer_attn_indices(multihead_attention, topk, primary=True, goal_layers=None):
    """提取供下一帧复用的高注意力视觉 token id。"""
    if goal_layers is None:
        goal_layers = []
    attn_dict, position = vlm_layer_attn(multihead_attention, layer_indices=None, primary=primary)
    offset = 1 if primary else 257
    total_patches = 256 + offset
    global_top = []
    for layer_idx in sorted(attn_dict.keys()):
        if layer_idx not in goal_layers:
            continue
        position_id = position[layer_idx]
        attn_scores = attn_dict[layer_idx]
        filtered_ids = (position_id[(position_id >= offset) & (position_id < total_patches)] - offset).cpu().numpy()
        full_attn = torch.zeros(256, device=attn_scores.device)
        full_attn[filtered_ids] = attn_scores
        attn = full_attn.cpu().numpy().astype(np.float32).reshape(16, 16)
        flat = [(i * 16 + j, attn[i, j]) for i in range(16) for j in range(16)]
        flat.sort(key=lambda x: x[1], reverse=True)
        global_top.extend(idx for idx, _ in flat[:topk])
    unique_global_top = sorted(set(global_top))
    return torch.tensor(unique_global_top) + offset


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """将 gripper 从 [0,1] 映射到 [-1,1]，可选二值化。"""
    normalized = action.copy()
    normalized[..., -1] = 2 * (normalized[..., -1] - 0.0) / (1.0 - 0.0) - 1
    if binarize:
        normalized[..., -1] = np.sign(normalized[..., -1])
    return normalized


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """翻转与环境相反的 OpenVLA gripper 符号。"""
    inverted = action.copy()
    inverted[..., -1] *= -1.0
    return inverted


def _as_token_tensor(indices, device):
    if indices is None:
        return torch.empty(0, dtype=torch.long, device=device)
    if isinstance(indices, torch.Tensor):
        return indices.to(device=device, dtype=torch.long).flatten()
    return torch.as_tensor(indices, device=device, dtype=torch.long).flatten()


def _task_relevant_set(cur_layer_attention, indices=None, num_prompt=DEFAULT_NUM_PROMPT_TOKENS,
                       topk=40, primary=True):
    """按文本到视觉的注意力选取前两层候选 token。"""
    offset = 1 if primary else 257
    v_token_start = 1 if primary else 257
    v_token_end = v_token_start + 256
    t_token_start = 513
    t_token_end = t_token_start + num_prompt
    attn_map = cur_layer_attention.to(torch.float32).squeeze(0).mean(dim=0)
    topk = max(int(topk), 0)

    if indices is None:  # 层 0：从完整视觉序列中选取
        token_ids = torch.arange(attn_map.size(0), device=attn_map.device)
        text_mask = (token_ids >= t_token_start + 10) & (token_ids < t_token_end)
        v_attn = attn_map[text_mask, v_token_start:v_token_end]
        if topk == 0 or v_attn.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=attn_map.device)
        attn_scores = v_attn.mean(dim=0)
        flat = attn_scores.reshape(-1)
        k = min(topk, flat.numel())
        _, selected = flat.topk(k)
        return selected + offset
    else:  # 层 1：只从当前保留的视觉 token 中选取
        indices = _as_token_tensor(indices, attn_map.device).sort().values
        vision_mask = (indices >= v_token_start) & (indices < v_token_end)
        visible_visual_tokens = indices[vision_mask]
        if topk == 0 or visible_visual_tokens.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=attn_map.device)
        v_attn = attn_map[-1 - ACTION_DIM * NUM_ACTIONS_CHUNK - num_prompt + 10: -1 - ACTION_DIM * NUM_ACTIONS_CHUNK,
                          vision_mask]
        if v_attn.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=attn_map.device)
        attn_scores = v_attn.mean(dim=0)
        flat = attn_scores.reshape(-1)
        k = min(topk, flat.numel())
        _, selected = flat.topk(k)
        return visible_visual_tokens[selected]


def _update_causal_mask(model, attention_mask, input_tensor, cache_position, past_seen_tokens):
    """构造兼容 SDPA 的 causal mask。"""
    if model.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None
    if model.config._attn_implementation == "sdpa":
        if AttentionMaskConverter._ignore_causal_mask_sdpa(
            attention_mask, inputs_embeds=input_tensor, past_key_values_length=past_seen_tokens
        ):
            return None

    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    if hasattr(getattr(model.layers[0], "self_attn", {}), "past_key_value"):
        target_length = model.config.max_position_embeddings
    else:
        target_length = (
            attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

    causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)

    if attention_mask is not None:
        causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
        if attention_mask.dim() == 2:
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
            causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
        elif attention_mask.dim() == 4:
            offset = cache_position[0] if attention_mask.shape[-2] < cache_position[0] + sequence_length else 0
            mask_shape = attention_mask.shape
            mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
            causal_mask[
                : mask_shape[0], : mask_shape[1], offset: mask_shape[2] + offset, : mask_shape[3]
            ] = mask_slice

    if (model.config._attn_implementation == "sdpa"
            and attention_mask is not None and attention_mask.device.type == "cuda"):
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
    return causal_mask


def _pruned_llama_forward(self, cfg, input_ids=None, attention_mask=None, position_ids=None,
                          past_key_values=None, inputs_embeds=None, use_cache=None,
                          output_attentions=None, output_hidden_states=None, return_dict=None,
                          cache_position=None, **kwargs):
    """注入 LlamaModel 的剪枝前向，保留原版 token 位置与注意力语义。"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time, "
                         "and must specify either one")
    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once = getattr(logger, "warning_once", logger.warning)
        logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
        use_cache = False
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    past_seen_tokens = 0
    if use_cache:
        if not isinstance(past_key_values, StaticCache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
    if cache_position is None:
        if isinstance(past_key_values, StaticCache):
            raise ValueError("cache_position is a required argument when using StaticCache.")
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    causal_mask = _update_causal_mask(self, attention_mask, inputs_embeds, cache_position, past_seen_tokens)

    # 输入剪枝状态
    precise_mode = kwargs["precise_mode"]
    NUM_PROMPT_TOKENS = kwargs.get("num_prompt", DEFAULT_NUM_PROMPT_TOKENS)
    skip_layer_list = kwargs.get("skip_layer_list", None)
    high_similarity_indices = kwargs["high_similarity_indices"]
    high_similarity_indices_wrist = kwargs["high_similarity_indices_wrist"]

    if not isinstance(high_similarity_indices, torch.Tensor):
        static_indices = torch.tensor(high_similarity_indices, device=cache_position.device)
        static_indices_wrist = torch.tensor(high_similarity_indices_wrist, device=cache_position.device)
    else:
        static_indices = high_similarity_indices
        static_indices_wrist = high_similarity_indices_wrist
    cur_local_indices = torch.cat([static_indices, static_indices_wrist]).long()

    hidden_states = inputs_embeds
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None
    dtype, device = inputs_embeds.dtype, inputs_embeds.device
    mask_size = VISUAL_TOKEN_END
    retain_mask = torch.ones(mask_size, dtype=torch.bool, device=cache_position.device)
    all_retain_mask = torch.ones(len(cache_position), dtype=torch.bool, device=cache_position.device)

    prev_attn_indices = kwargs["prev_attn_indices"]
    prev_attn_indices = _as_token_tensor(prev_attn_indices, cache_position.device)
    action_token_count = kwargs.get("action_token_count", None)
    if action_token_count is None:
        action_token_count = ACTION_DIM * NUM_ACTIONS_CHUNK   # 56
    action_token_start = hidden_states.shape[1] - action_token_count - 1
    action_token_end = hidden_states.shape[1] - 1

    dynamic_prune = cfg.dynamic_prune
    infer_step = kwargs.get("infer_step", None)
    episode_key = kwargs.get("episode_id", kwargs.get("total_episodes", None))
    if episode_key is None:
        episode_key = "active_episode"
    reset_dynamic_confidence = (
        (infer_step == 1 and self.dynamic_importance_last_infer_step != 1)
        or self.dynamic_importance_episode_key != episode_key
    )
    # 层置信度按 episode 缓存，跨 episode 或首次查询时清空。
    if reset_dynamic_confidence:
        self.dynamic_importance_confidence = {}
        self.dynamic_importance_episode_key = episode_key
    self.dynamic_importance_last_infer_step = infer_step

    self.token_importance_scores = torch.zeros(hidden_states.shape[1], device=hidden_states.device)
    last_cache_position = cache_position
    high_attn_indices = torch.empty(0, dtype=torch.long, device=cache_position.device)
    high_attn_indices_wrist = torch.empty(0, dtype=torch.long, device=cache_position.device)

    def set_mask(mask, indices, value):
        indices = _as_token_tensor(indices, mask.device)
        if indices.numel() == 0:
            return
        valid = (indices >= 0) & (indices < mask.shape[0])
        if valid.any():
            mask[indices[valid]] = value

    def apply_sequence_mask(sequence_mask, prune_columns_only=False):
        """同步裁剪 hidden states、绝对位置与因果掩码。"""
        nonlocal cache_position, position_ids, causal_mask, hidden_states
        cache_position = cache_position[sequence_mask].sort().values
        position_ids = cache_position.unsqueeze(0)
        if causal_mask is not None:
            causal_mask = causal_mask[..., sequence_mask, :]
            if not prune_columns_only:
                causal_mask = causal_mask[..., sequence_mask]
        hidden_states = hidden_states[..., sequence_mask, :]
        self.token_importance_scores = self.token_importance_scores[sequence_mask]

    def visual_token_mask(token_ids):
        return (token_ids >= PRIMARY_VISUAL_TOKEN_START) & (token_ids < VISUAL_TOKEN_END)

    for decoder_layer in self.layers:
        if skip_layer_list is not None:
            if decoder_layer.layer_idx in skip_layer_list:
                continue

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__, hidden_states, causal_mask, position_ids,
                past_key_values, output_attentions, use_cache, cache_position,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            layer_attention = layer_outputs[1]

            attn_topk = cfg.attn_topk_precise if precise_mode else ATTN_TOPK
            attn_topk_wrist = cfg.attn_topk_precise_wrist if precise_mode else ATTN_TOPK_WRIST

            if cfg.token_prune:
                # 前两层结合文本候选、帧差与跨帧动作中心剪枝。
                if decoder_layer.layer_idx == 0:
                    cur_attn_indices = _task_relevant_set(
                        layer_attention, indices=None, num_prompt=NUM_PROMPT_TOKENS, topk=2 * attn_topk, primary=True)
                    cur_attn_indices_wrist = _task_relevant_set(
                        layer_attention, indices=None, num_prompt=NUM_PROMPT_TOKENS, topk=2 * attn_topk_wrist, primary=False)
                    high_attn_indices = cur_attn_indices[0: attn_topk]
                    high_attn_indices_wrist = cur_attn_indices_wrist[0: attn_topk_wrist]
                    set_mask(retain_mask, cur_local_indices, False)
                    set_mask(retain_mask, cur_attn_indices, True)
                    set_mask(retain_mask, cur_attn_indices_wrist, True)
                    set_mask(retain_mask, prev_attn_indices, True)
                    all_retain_mask = torch.ones(len(cache_position), dtype=torch.bool, device=cache_position.device)
                    replace_length = min(len(retain_mask), len(cache_position))
                    all_retain_mask[0: replace_length] = retain_mask[:replace_length]
                    apply_sequence_mask(all_retain_mask, prune_columns_only=True)

                if decoder_layer.layer_idx == 1:
                    cur_attn_indices = _task_relevant_set(
                        layer_attention, indices=cache_position, num_prompt=NUM_PROMPT_TOKENS, topk=attn_topk, primary=True)
                    cur_attn_indices_wrist = _task_relevant_set(
                        layer_attention, indices=cache_position, num_prompt=NUM_PROMPT_TOKENS, topk=attn_topk_wrist, primary=False)
                    set_mask(all_retain_mask, cur_local_indices, False)
                    set_mask(all_retain_mask, cur_attn_indices, True)
                    set_mask(all_retain_mask, cur_attn_indices_wrist, True)
                    set_mask(all_retain_mask, high_attn_indices, True)
                    set_mask(all_retain_mask, high_attn_indices_wrist, True)
                    set_mask(all_retain_mask, prev_attn_indices, True)
                    current_retain_mask = all_retain_mask[cache_position]
                    last_cache_position = cache_position
                    apply_sequence_mask(current_retain_mask, prune_columns_only=True)

                # 深层动态剪枝：重要性更新层与实际剪枝层分离。
                if dynamic_prune:
                    if decoder_layer.layer_idx in DYNAMIC_IMPORTANCE_UPDATE_LAYERS:
                        current_positions = last_cache_position
                        vision_mask = visual_token_mask(current_positions)
                        action_mask = (current_positions >= action_token_start) & (current_positions < action_token_end)
                        if vision_mask.any() and action_mask.any():
                            attn_map = layer_attention.to(torch.float32).squeeze(0).mean(dim=0)
                            action_to_vision = attn_map[action_mask][:, vision_mask]
                            mean_attn = action_to_vision.mean(dim=0)
                            # 动作相关性排名与注意力熵共同决定重要性权重。
                            _, sorted_ranks = torch.sort(mean_attn, descending=True)
                            ranks = torch.empty_like(sorted_ranks)
                            ranks[sorted_ranks] = torch.arange(len(sorted_ranks), device=mean_attn.device)
                            sigma_vals = torch.sigmoid(-ranks.float())
                            omega_rank = sigma_vals / (sigma_vals.sum() + 1e-8)
                            eps = 1e-8
                            omega_conf = self.dynamic_importance_confidence.get(decoder_layer.layer_idx)
                            if omega_conf is None:
                                prob = action_to_vision + eps
                                prob = prob / prob.sum(dim=-1, keepdim=True)
                                entropy = -(prob * torch.log(prob)).sum(dim=-1).mean()
                                entropy = entropy / np.log(max(int(vision_mask.sum().item()), 2))
                                omega_conf = (1.0 / (entropy + eps)).detach()
                                self.dynamic_importance_confidence[decoder_layer.layer_idx] = omega_conf
                            else:
                                omega_conf = omega_conf.to(device=omega_rank.device, dtype=omega_rank.dtype)
                            visual_seq_indices = vision_mask.nonzero(as_tuple=True)[0]
                            s_l = omega_rank * omega_conf
                            old_scores = self.token_importance_scores[visual_seq_indices]
                            self.token_importance_scores[visual_seq_indices] = (
                                (1 - DYNAMIC_IMPORTANCE_EMA_BETA) * old_scores
                                + DYNAMIC_IMPORTANCE_EMA_BETA * s_l
                            )

                    if decoder_layer.layer_idx in DYNAMIC_PRUNE_LAYERS:
                        if self.token_importance_scores is not None:
                            current_seq_len = hidden_states.shape[1]
                            current_positions = cache_position
                            non_visual_mask = ~visual_token_mask(current_positions)
                            self.token_importance_scores[non_visual_mask] = float("inf")  # 非视觉 token 不剪
                            _, sorted_indices = torch.sort(
                                self.token_importance_scores[:current_seq_len], descending=True)
                            retain_ratio = DYNAMIC_PRUNE_RATIO
                            min_retain = MIN_VISUAL_RETAIN_TOKENS + NUM_PROMPT_TOKENS + DYNAMIC_PRUNE_SAFETY_MARGIN
                            retain_num = max(int(retain_ratio * current_seq_len), min_retain)
                            retain_num = min(retain_num, current_seq_len)
                            if retain_num >= current_seq_len:
                                dynamic_prune = False
                            retain_indices = sorted_indices[:retain_num]
                            new_retain_mask = torch.zeros(current_seq_len, dtype=torch.bool,
                                                          device=hidden_states.device)
                            new_retain_mask[retain_indices] = True
                            last_cache_position = cache_position
                            apply_sequence_mask(new_retain_mask, prune_columns_only=False)

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]
        if output_attentions:
            all_self_attns += ((layer_outputs[1], last_cache_position),)
            last_cache_position = cache_position

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
        )
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def _patched_predict_action(self, cfg, input_ids=None, unnorm_key=None, proprio=None,
                            proprio_projector=None, action_head=None,
                            noisy_action_projector=None, use_film=False, **kwargs):
    """向模型透传剪枝状态，返回动作、隐藏状态、注意力及跳层列表。

    precise 模式不跳层；输出注意力供动态剪枝和跨帧复用。
    """
    # 训练输入要求提示词末尾有空 token；fake labels 仅用于 action 掩码。
    if not torch.all(input_ids[:, -1] == EMPTY_TOKEN_ID):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.Tensor([EMPTY_TOKEN_ID]).long(), dim=0).to(input_ids.device)),
            dim=1,
        )

    pixel_values = kwargs["pixel_values"]
    attention_mask = kwargs.pop("attention_mask", None)

    labels = input_ids.clone()
    labels[:] = IGNORE_INDEX
    NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1

    input_ids, attention_mask = self._prepare_input_for_action_prediction(input_ids, attention_mask)
    labels = self._prepare_labels_for_action_prediction(labels, input_ids)

    input_embeddings = self.get_input_embeddings()(input_ids)
    all_actions_mask = self._process_action_masks(labels)
    language_embeddings = input_embeddings[~all_actions_mask].reshape(
        input_embeddings.shape[0], -1, input_embeddings.shape[2]
    )

    projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)

    use_proprio = proprio_projector is not None and proprio is not None
    if use_proprio:
        proprio = torch.Tensor(proprio).to(projected_patch_embeddings.device,
                                           dtype=projected_patch_embeddings.dtype)
        projected_patch_embeddings = self._process_proprio_features(
            projected_patch_embeddings, proprio, proprio_projector
        )

    all_actions_mask_u = all_actions_mask.unsqueeze(-1)
    input_embeddings = input_embeddings * ~all_actions_mask_u
    multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
        input_embeddings, projected_patch_embeddings, attention_mask
    )

    skip_layer_list = kwargs.pop("skip_layer_list", None)
    precise_mode = kwargs.get("precise_mode", False)
    language_model_output = self.language_model(
        cfg=cfg,
        input_ids=None,
        attention_mask=multimodal_attention_mask,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=multimodal_embeddings,
        labels=None,
        use_cache=None,
        output_attentions=True,      # 剪枝依赖 attention weights/positions
        output_hidden_states=True,
        return_dict=True,
        num_prompt=NUM_PROMPT_TOKENS,
        skip_layer_list=[] if precise_mode else skip_layer_list,
        **kwargs,
    )

    if action_head is None:
        raise ValueError("predict_action 需要 action_head（L1 回归动作头）。")
    last_hidden_states = language_model_output.hidden_states[-1]
    actions_hidden_states = last_hidden_states[:, -ACTION_DIM * NUM_ACTIONS_CHUNK - 1: -1, :]
    last_attn = language_model_output.attentions
    normalized_actions = action_head.predict_action(actions_hidden_states)
    normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
    normalized_actions = normalized_actions.float().cpu().detach().numpy()

    actions = self._unnormalize_actions(normalized_actions, unnorm_key)
    return actions, actions_hidden_states, last_attn, skip_layer_list


def _inject_pruning(model) -> None:
    """向已加载模型注入剪枝前向。"""
    llama = getattr(getattr(model, "language_model", None), "model", None)
    if llama is None:
        raise ValueError(
            "未找到 model.language_model.model（LlamaModel）。请传入 OpenVLA-OFT 模型。"
        )
    # 保留原始路径，供 parity 临时恢复。
    if not hasattr(llama, "_specprune_orig_forward"):
        llama._specprune_orig_forward = llama.forward
    llama.forward = types.MethodType(_pruned_llama_forward, llama)
    for attr, init in [
        ("dynamic_importance_confidence", {}),
        ("dynamic_importance_episode_key", None),
        ("dynamic_importance_last_infer_step", None),
        ("token_importance_scores", None),
    ]:
        if not hasattr(llama, attr):
            setattr(llama, attr, init)
    # 剪枝需要 eager attention 的权重及绝对位置语义。
    try:
        from transformers.models.llama.modeling_llama import LlamaAttention
        replaced = 0
        for layer in llama.layers:
            if type(layer.self_attn).__name__ != "LlamaAttention":
                new_attn = LlamaAttention(layer.self_attn.config, layer.self_attn.layer_idx)
                new_attn.load_state_dict(layer.self_attn.state_dict())
                new_attn = new_attn.to(layer.self_attn.q_proj.weight.device, layer.self_attn.q_proj.weight.dtype)
                layer.self_attn = new_attn
                replaced += 1
        if replaced:
            print(f"[spec_prune] 已把 {replaced} 层 self_attn 替换为 eager LlamaAttention（剪枝需要）。")
    except Exception as e:  # pragma: no cover
        print(f"[spec_prune] [WARN] 强制 eager 失败（{e}），若模型非 eager 可能影响正确性。")

    required = ["_prepare_input_for_action_prediction", "_prepare_labels_for_action_prediction",
                "_process_vision_features", "_process_proprio_features",
                "_build_multimodal_attention", "_process_action_masks", "_unnormalize_actions"]
    missing = [m for m in required if not hasattr(model, m)]
    if missing:
        raise ValueError(
            f"模型缺少 patch predict_action 所需的辅助方法 {missing}；请确认传入的是 OpenVLA-OFT 模型。"
        )
    if not hasattr(model, "predict_action"):
        raise ValueError("未找到 model.predict_action，请传入 OpenVLA-OFT 模型。")
    if not hasattr(model, "_orig_predict_action"):
        model._orig_predict_action = getattr(model, "predict_action", None)
    model.predict_action = types.MethodType(_patched_predict_action, model)
    print("[spec_prune] 已 patch predict_action（透传剪枝 kwargs + 返回 4 元组）。")


@dataclass
class AcceleratorConfig:
    token_prune: bool = True
    dynamic_prune: bool = True
    layer_skip: bool = False          # 原版关闭跳层
    action_adapter: bool = True
    log_mode: bool = False

    # 类默认仍是 original/base 15/12；suite 最终值由调用方选择。
    attn_topk_precise: int = ATTN_TOPK_PRECISE
    attn_topk_precise_wrist: int = ATTN_TOPK_PRECISE_WRIST

    model_family: str = "openvla"
    use_film: bool = False
    num_images_in_input: int = 2      # 1 主视图 + 1 腕部视图
    center_crop: bool = True          # 训练使用随机裁剪，推理取中心 90%
    use_proprio: bool = True
    unnorm_key: str = ""              # 空则自动检测（norm_stats 仅 1 个 key 时）
    num_open_loop_steps: int = 8      # 动作 chunk 长（须等于 NUM_ACTIONS_CHUNK）
    image_resize: int = 224
    device: Optional[str] = None      # 设备（默认取模型参数所在设备）

    pretrained_checkpoint: str = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    lora_rank: int = 32
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50


class SpecPruneAccelerator:
    """已加载 OpenVLA-OFT 模型的闭环推理加速器。"""

    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any,
        action_head: Optional[torch.nn.Module] = None,
        proprio_projector: Optional[torch.nn.Module] = None,
        noisy_action_projector: Optional[torch.nn.Module] = None,
        config: Optional[AcceleratorConfig] = None,
    ):
        self.model = model
        self.processor = processor
        self.action_head = action_head
        self.proprio_projector = proprio_projector
        self.noisy_action_projector = noisy_action_projector
        self.cfg = config if config is not None else AcceleratorConfig()

        if self.cfg.device is not None:
            self.device = torch.device(self.cfg.device)
        else:
            self.device = next(model.parameters()).device

        if self.action_head is None:
            raise ValueError(
                "SpecPrune 需要连续动作头（L1 回归动作头）才能拿到 last_attn；请传入 action_head。"
            )
        if self.cfg.num_open_loop_steps != 8:
            print(
                f"WARNING: cfg.num_open_loop_steps ({self.cfg.num_open_loop_steps}) != "
                f"NUM_ACTIONS_CHUNK (8)。建议用 8（整段 action chunk）以保性能。"
            )
        self._resolve_unnorm_key()
        self._reset_episode_state()

    @classmethod
    def install(
        cls,
        model: torch.nn.Module,
        processor: Any,
        action_head: Optional[torch.nn.Module] = None,
        proprio_projector: Optional[torch.nn.Module] = None,
        noisy_action_projector: Optional[torch.nn.Module] = None,
        config: Optional[AcceleratorConfig] = None,
    ) -> "SpecPruneAccelerator":
        """将剪枝路径注入已加载模型。"""
        _inject_pruning(model)
        return cls(model, processor, action_head=action_head,
                   proprio_projector=proprio_projector,
                   noisy_action_projector=noisy_action_projector, config=config)

    def start_episode(self) -> None:
        """重置跨帧注意力状态和动态重要性缓存。"""
        self._reset_episode_state()
        try:
            llm_model = getattr(getattr(self.model, "language_model", None), "model", None)
            if llm_model is not None:
                if hasattr(llm_model, "dynamic_importance_confidence"):
                    llm_model.dynamic_importance_confidence = {}
                if hasattr(llm_model, "dynamic_importance_episode_key"):
                    llm_model.dynamic_importance_episode_key = None
                if hasattr(llm_model, "dynamic_importance_last_infer_step"):
                    llm_model.dynamic_importance_last_infer_step = None
        except Exception:  # pragma: no cover
            pass

    def end_episode(self) -> Dict[str, Any]:
        return {
            "avg_infer_time": round(self.all_infer_time / max(self.infer_step, 1), 4),
            "infer_steps": self.infer_step,
            "acting_steps": self.acting_steps,
        }

    def describe(self) -> Dict[str, Any]:
        info = {
            "unnorm_key": self.cfg.unnorm_key,
            "device": str(self.device),
            "switches": {
                "token_prune": self.cfg.token_prune,
                "dynamic_prune": self.cfg.dynamic_prune,
                "layer_skip": self.cfg.layer_skip,
                "action_adapter": self.cfg.action_adapter,
            },
            "文本引导 topk（粗/细）": {
                "ATTN_TOPK": ATTN_TOPK,
                "ATTN_TOPK_PRECISE": self.cfg.attn_topk_precise,
                "ATTN_TOPK_WRIST": ATTN_TOPK_WRIST,
                "ATTN_TOPK_PRECISE_WRIST": self.cfg.attn_topk_precise_wrist,
            },
            "帧差 topk（粗/细）": {
                "PRIMARY_TOPK": PRIMARY_TOPK,
                "PRIMARY_TOPK_PRECISE": PRIMARY_TOPK_PRECISE,
                "WRIST_TOPK": WRIST_TOPK,
                "WRIST_TOPK_PRECISE": WRIST_TOPK_PRECISE,
            },
            "动态剪枝": {"DYNAMIC_PRUNE_LAYERS": "10/15/20/25", "DYNAMIC_PRUNE_RATIO": DYNAMIC_PRUNE_RATIO},
        }
        print("\n=== SpecPruneAccelerator 生效配置 ===")
        for k, v in info.items():
            print(f"  {k}: {v}")
        return info

    def step(
        self,
        image: np.ndarray,
        wrist_image: np.ndarray,
        text: str,
        proprio: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """返回单步动作和状态；调用方仅在 execute_action 为真时推进环境。

        动作队列按 8 步 chunk 消费；查询时复用历史帧与 prev_attn_indices。
        """
        self.acting_steps += 1
        self.frame_idx += 1
        self.replay_images.append(image)
        self.replay_images_wrist.append(wrist_image)

        queried = False
        infer_time = 0.0

        if len(self.action_queue) == 0:
            # 参考帧间隔由动作速度决定，限制在 2～6 帧。
            move_v = float(np.linalg.norm(self.history_action[:3]))
            prev_T = int((-16 / 3 * move_v / 6 + 22 / 3)) + 3   # prev_T ∈ (2, 6]
            prev_T = min(prev_T, 6)
            if self.prev_img is None or len(self.replay_images) < prev_T:
                prev_img = self.replay_images[-1]
                prev_img_wrist = self.replay_images_wrist[-1]
            else:
                prev_img = self.replay_images[-prev_T]
                prev_img_wrist = self.replay_images_wrist[-prev_T]
            # 写回参考帧，避免后续帧差始终与当前帧比较。
            self.prev_img = prev_img
            self.prev_img_wrist = prev_img_wrist

            full_resized = resize_image_for_policy(image, self.cfg.image_resize)
            wrist_resized = resize_image_for_policy(wrist_image, self.cfg.image_resize)
            obs = {
                "full_image": full_resized,
                "wrist_image": wrist_resized,
                "state": np.asarray(proprio, dtype=np.float64) if proprio is not None else None,
                "prev_images": [prev_img, prev_img_wrist],
            }

            self.infer_step += 1
            action_chunk, self.prev_attn_indices, infer_time = self._query_model(obs, text)
            self.action_queue.extend(action_chunk)
            self.history_action = np.zeros(7, dtype=np.float64)
            self.if_replan = False
            self.all_infer_time += infer_time
            queried = True

        action = self.action_queue.popleft()
        action = normalize_gripper_action(action, binarize=True)
        if self.cfg.model_family == "openvla":
            action = invert_gripper_action(action)

        # action_adapter 前 9 步消耗队列动作，但不推进环境。
        execute_action = not (self.cfg.action_adapter and self.acting_steps < 10)

        self._update_controller(action)

        info = {
            "infer_time": infer_time,
            "precise_mode": self.precise_mode,
            "queried": queried,
            "infer_step": self.infer_step,
            "acting_steps": self.acting_steps,
            "execute_action": execute_action,
        }
        return action, info

    def _reset_episode_state(self) -> None:
        self.replay_images: List[np.ndarray] = []
        self.replay_images_wrist: List[np.ndarray] = []
        self.prev_img: Optional[np.ndarray] = None
        self.prev_img_wrist: Optional[np.ndarray] = None
        self.prev_attn_indices: Optional[torch.Tensor] = None
        self.skip_layer_list = None
        self.action_queue: deque = deque(maxlen=self.cfg.num_open_loop_steps)
        self.history_action: np.ndarray = np.zeros(7, dtype=np.float64)
        self.precise_mode: bool = False
        self.cur_mode: bool = True               # 初始为 draft 模式
        self.if_replan: bool = False
        self.acting_steps: int = 0
        self.infer_step: int = 0
        self.frame_idx: int = 0
        self.all_infer_time: float = 0.0
        self._cur_velocity: Optional[float] = None
        self._xy_velocity: Optional[float] = None
        self._rot_velocity: Optional[float] = None
        self._z_velocity: Optional[float] = None
        self._start_event = None
        self._end_event = None
        self._start_time = 0.0

    def _resolve_unnorm_key(self) -> None:
        """匹配 checkpoint 的动作反归一化统计 key。"""
        norm_stats = self.model.norm_stats
        key = self.cfg.unnorm_key
        if key:
            if key not in norm_stats and f"{key}_no_noops" in norm_stats:
                key = f"{key}_no_noops"
        else:
            keys = list(norm_stats.keys())
            if len(keys) == 1:
                key = keys[0]
            else:
                raise ValueError(
                    f"模型 norm_stats 有多个 key: {keys}，请通过 "
                    f"AcceleratorConfig(unnorm_key=...) 显式指定（如 'libero_spatial'）。"
                )
        assert key in norm_stats, f"Action un-norm key '{key}' not found in `norm_stats`!"
        self.cfg.unnorm_key = key

    def _start_timer(self) -> None:
        if self.device.type == "cuda":
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            self._start_event.record()
        else:
            self._start_time = time.perf_counter()

    def _stop_timer(self) -> float:
        if self.device.type == "cuda":
            self._end_event.record()
            torch.cuda.synchronize()
            return round(self._start_event.elapsed_time(self._end_event) / 1000, 4)
        return round(time.perf_counter() - self._start_time, 4)

    def _prepare_inputs(self, obs, task_label, time_it=False) -> Dict[str, Any]:
        """复用原版预处理；parity 对照也使用同一输入构造。"""
        cfg = self.cfg
        precise_mode = self.precise_mode

        all_images, result_image, prev_images = multi_image_processing(cfg, obs)

        if cfg.token_prune:
            primary_topk = PRIMARY_TOPK_PRECISE if precise_mode else PRIMARY_TOPK
            wrist_topk = WRIST_TOPK_PRECISE if precise_mode else WRIST_TOPK
            attn_topk_p = cfg.attn_topk_precise if precise_mode else ATTN_TOPK
            attn_topk_w = cfg.attn_topk_precise_wrist if precise_mode else ATTN_TOPK_WRIST
            high_similarity_indices = get_similarity_indices(
                result_image[0], prev_images[0],
                top_k=primary_topk, patch_size=14,
                sim_threshold=PRIMARY_SIM_THRESHOLD, view="primary",
            )
            high_similarity_indices_wrist = get_similarity_indices(
                result_image[1], prev_images[1],
                top_k=wrist_topk, patch_size=14,
                sim_threshold=WRIST_SIM_THRESHOLD, view="wrist",
            )
        else:
            high_similarity_indices = np.array([], dtype=int)
            high_similarity_indices_wrist = np.array([], dtype=int)
            attn_topk_p = cfg.attn_topk_precise if precise_mode else ATTN_TOPK
            attn_topk_w = cfg.attn_topk_precise_wrist if precise_mode else ATTN_TOPK_WRIST

        primary_image = all_images.pop(0)
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

        if time_it:
            self._start_timer()   # 计时从 processor 开始
        inputs = self.processor(prompt, primary_image).to(self.device, dtype=torch.bfloat16)

        if all_images:
            all_wrist_inputs = [
                self.processor(prompt, img_w).to(self.device, dtype=torch.bfloat16)
                for img_w in all_images
            ]
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [w["pixel_values"] for w in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        proprio = None
        if cfg.use_proprio:
            proprio = obs["state"]
            proprio_norm_stats = self.model.norm_stats[cfg.unnorm_key]["proprio"]
            obs["state"] = normalize_proprio(proprio, proprio_norm_stats)
            proprio = obs["state"]

        return {
            "inputs": inputs,
            "proprio": proprio,
            "high_similarity_indices": high_similarity_indices,
            "high_similarity_indices_wrist": high_similarity_indices_wrist,
            "attn_topk_p": attn_topk_p,
            "attn_topk_w": attn_topk_w,
        }

    def _query_model(
        self,
        obs: Dict[str, Any],
        task_label: str,
    ) -> Tuple[List[np.ndarray], torch.Tensor, float]:
        cfg = self.cfg
        precise_mode = self.precise_mode
        prep = self._prepare_inputs(obs, task_label, time_it=True)

        with torch.inference_mode():
            action, _, last_attn, skip_layer_list = self.model.predict_action(
                **prep["inputs"],
                cfg=cfg,
                unnorm_key=cfg.unnorm_key,
                do_sample=False,
                proprio=prep["proprio"],
                proprio_projector=self.proprio_projector,
                noisy_action_projector=self.noisy_action_projector,
                action_head=self.action_head,
                use_film=cfg.use_film,
                high_similarity_indices=prep["high_similarity_indices"],
                high_similarity_indices_wrist=prep["high_similarity_indices_wrist"],
                precise_mode=precise_mode,
                prev_attn_indices=self.prev_attn_indices,
                acting_steps=self.acting_steps,
                infer_step=self.infer_step,
                skip_layer_list=self.skip_layer_list,
            )
            infer_time = self._stop_timer()
            self.skip_layer_list = skip_layer_list

            # 计时止于 predict_action；层 14/30 注意力供下一帧静态复用。
            goal_layers = [14, 30]
            prev_attn_indices = get_layer_attn_indices(
                last_attn, int(prep["attn_topk_p"] / 2), goal_layers=goal_layers, primary=True
            )
            prev_attn_indices_wrist = get_layer_attn_indices(
                last_attn, int(prep["attn_topk_w"] / 2), goal_layers=goal_layers, primary=False
            )

            if cfg.log_mode:
                import json
                log_path = os.path.join(os.getcwd(), "log", "high_attn_index.jsonl")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                primary_list = (
                    prev_attn_indices.cpu().numpy().tolist()
                    if hasattr(prev_attn_indices, "cpu") else prev_attn_indices.tolist()
                )
                wrist_list = (
                    prev_attn_indices_wrist.cpu().numpy().tolist()
                    if hasattr(prev_attn_indices_wrist, "cpu") else prev_attn_indices_wrist.tolist()
                )
                log_data = {
                    "Primary view high attention indices": primary_list,
                    "Wrist view high attention indices": wrist_list,
                    "step": self.infer_step,
                    "skip_layers": (skip_layer_list.tolist() if hasattr(skip_layer_list, "tolist")
                                    else skip_layer_list),
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_data) + "\n")

            prev_attn_indices = torch.cat([prev_attn_indices, prev_attn_indices_wrist])

        return [action[i] for i in range(len(action))], prev_attn_indices, infer_time

    def _update_controller(self, action: np.ndarray) -> None:
        """按动作速度切换 draft/precise；切入 precise 时清队重规划。"""
        if not self.cfg.action_adapter:
            return
        if self.acting_steps < 10:
            return

        if not self.if_replan:
            self.history_action += action
            if action is not None:
                self._cur_velocity = float(np.linalg.norm(action[0:3]))
                self._xy_velocity = float(np.linalg.norm(action[0:2]))
                self._rot_velocity = float(np.linalg.norm(action[3:6]))
                self._z_velocity = float(action[2])

        if self._cur_velocity is not None and self.acting_steps > 10:
            if self.precise_mode:
                if (
                    (self._z_velocity > PRECISE_EXIT_Z_THRES or self._rot_velocity > PRECISE_EXIT_ROT_THRES)
                    and len(self.action_queue) == 0
                ):
                    self.precise_mode = False
            else:
                self.precise_mode = bool(
                    self._cur_velocity < VELOCITY_THRES
                    or (
                        self._xy_velocity < XY_THRES
                        and self._z_velocity < Z_THRES
                        and self._rot_velocity < ROT_THRES
                    )
                )
                if (not self.cur_mode) and self.precise_mode and len(self.action_queue) > 0:
                    self.if_replan = True
            self.cur_mode = self.precise_mode

        if self.if_replan:
            self.if_replan = False
            self.action_queue.clear()
