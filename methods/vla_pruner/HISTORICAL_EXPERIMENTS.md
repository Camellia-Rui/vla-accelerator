# VLA-Pruner 接入说明

> 工具箱接口已更新：挂载已有模型、启停、状态重置、恢复卸载和固定后端依赖见
> [INTEGRATION.md](INTEGRATION.md)。以下保留历史接入与实验记录；旧接口示例以新文档为准，
> 历史性能数字不代表当前封装已经完成真实模型验证。

面向 Codex 与使用者的可复现接入文档。方法本体在 `methods/vla_pruner/accelerator.py`，adapter 在 `pruners/vla_pruner.py`。

## 1. 方法简介

VLA-Pruner 是免训练的视觉 Token 剪枝方法，通过语义-动作双级重要性评估，在推理时动态剔除冗余视觉 Token。

- 论文：https://arxiv.org/abs/2511.16449
- 官方代码：https://github.com/MINT-SJTU/VLA-Pruner
- 是否需要训练：否

核心洞察：VLA 推理分 prefill（语义理解）和 action-decode（动作生成）两阶段，两者注意力分布重叠率仅约 50%。只按语义剪枝会丢掉动作关键 token，所以 VLA-Pruner 用历史 action attention 的 EMA 估计动作重要性，再和语义评分融合。

## 2. 环境要求

| 类别 | 要求 | 说明 |
|------|------|------|
| Python | 3.10 | 必须 |
| PyTorch | 2.2.0 | 必须 |
| CUDA | 12.1 | 必须 |
| transformers | 4.47.0 | 必须用 vendored 版本，不能用官方版 |
| tokenizers | 0.21.1 | 与 transformers 匹配 |
| timm | 0.9.10 | 必须 < 1.0.0 |
| robosuite | 1.4.0 | LIBERO 依赖 |
| mujoco | 2.3.0 | 与 robosuite 匹配 |
| libero | 0.1.4 | 从 lerobot-libero-main 安装 |

环境变量（必须设置）：

- LD_LIBRARY_PATH 指向 torch 自带的 cuDNN
- PYTHONPATH 指向 openvla-oft 根目录

缺失依赖时，重点检查 transformers 和 timm 版本。

## 3. 模型权重

| 项目 | 值 |
|------|-----|
| 基础模型 | openvla-7b-oft-libero-object |
| 下载来源 | https://hf-mirror.com/moojink/openvla-7b-oft-finetuned-libero-object |
| 必需文件 | 4 个 safetensors + config.json + tokenizer 相关文件 + action head + proprio projector |

## 4. 调用接口

原生接口（方法本体）：

python
from methods.vla_pruner.accelerator import VLAPrunerAccelerator

acc = VLAPrunerAccelerator.install(
    checkpoint="模型权重路径",
    task_suite_name="libero_object",
    fastv_r=0.9275,
)
acc.start_episode()
result = acc.step(image=图像, wrist_image=腕部图像, instruction="任务指令", proprio=本体感知)
action = result["action"]
stats = acc.end_episode(success=True)

统一接口（通过工具箱）：

python
from pruners.registry import build
pruner = build("vla_pruner", config={"checkpoint": "模型权重路径", "fastv_r": 0.9275})
pruner.attach(model=None)
pruner.reset()
pruner.step(image=..., wrist_image=..., instruction=..., proprio=...)
action = pruner.get_action()
pruner.end_episode(success=True)

## 5. 关键参数

| 参数 | 含义 | 默认值 | 可调范围 |
| fastv_r | 剪枝比例 | 0.9275 | 0.5~0.95 |
| vla_pruner_layer | 剪枝层 | 15 |10~20|
| vla_pruner_mode | 剪枝模式 | semantic_action | semantic_action / fastv |
| vla_pruner_av_hist_w | EMA 窗口 | 3 | 2~5 |
| vla_pruner_av_decay | EMA 衰减率 | 0.8 | 0.5~0.9

推荐配置：
- 追求极致加速 → fastv_r=0.9275（保留 7.25%）
- 追求精度 → fastv_r=0.875（保留 12.5%）

## 6. 已知问题（Codex 应提前告知用户）

| 问题 | 现象 | 原因 | 解决 |
|------|------|------|------|
| 剪枝统计显示 0% | Token Reusing Ratio: 0.00% | 指标混淆 | 以 pruning-report 为准 |

## 7. 验证标准（Codex 判断是否跑通）

跑通后，日志里应出现：

```
[pruning-report] mode=semantic_action layer=3
seq_len_before=598 seq_len_after=124
image_len=512 kept_visual=38
flop_saving=69.92%
```

**Codex 判断逻辑**：
- 看到 pruning-report → 剪枝生效
- flop_saving > 0 → 加速成功
- 成功率与基线接近 → 精度保持

## 8. 实测结果

环境：NVIDIA H20，OpenVLA-OFT，LIBERO-Object，seed 7，500 episode。

| 指标 | Baseline | VLA-Pruner 7.25% |
|------|----------|------------------|
| 成功数 | 493/500 | 480/500 |
| 成功率 | 98.6% | 96.0% |
| 成功率下降 | — | 2.6 个百分点 |
| 纯推理每步延迟 | 124 ms | 79 ms |
| 纯推理加速比 | 1× | 1.57× |
| 配对样本 | — | 500 |
| 行为一致 episode | — | 485/500（97%） |
| 平均步数差 | — | 1.96 步 |
| 权重指纹 | sha256 前后一致 | 免训练主张成立 |

准入判定：**PASS**（manifest 见 `evaluator/results/vla_pruner_object.json`，阈值全部通过）。

结论：加速 1.57 倍，成功率降 2.6 个百分点，任务执行轨迹 97% 一致。与论文报的 7.25% 保留时成功率 96.0% 一致。

## 9. Codex 指导用户时的建议话术

1. 用户问“怎么用” → “把工具箱文件夹上传，我读完 README 后指导你。”
2. 用户问“环境怎么配” → “需要 Python 3.10 + PyTorch 2.2.0 + transformers 4.47.0（vendored版），我帮你检查。”
3.  用户问“跑不起来” → “先检查 LD_LIBRARY_PATH 和 PYTHONPATH，这是最常见的坑。”
4. 用户问“效果如何” → “7.25% 保留时 FLOPs 省 70%，成功率 96%，延迟 73ms。”
5. 用户问“选哪个剪枝比例” → “追求加速选 0.9275，追求精度选 0.875。”
