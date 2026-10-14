# VLA-Pruner

## 方法定位

VLA-Pruner 是一个免训练的视觉 Token 剪枝方法，用于加速 VLA 模型推理。
它通过语义-动作双级重要性评估，在推理时动态剔除冗余视觉 Token。

- 来源：NeurIPS 2025
- 论文：https://arxiv.org/abs/2511.16449
- 代码：https://github.com/MINT-SJTU/VLA-Pruner
- 是否需要训练：否

**适用场景**：通用机器人操作任务，追求计算量降低且精度不显著下降。


## 加速效果

| 保留比例 | FLOPs 节省 | 成功率 | 推理延迟 |
|----------|-----------|--------|----------|
| 7.25% | 69.92% | 96.0% | 73 ms/step |
| 12.5% | ~60% | 98.0% | ~75 ms/step |
| 25% | ~40% | 97.4% | ~77 ms/step |

**基线（不剪枝）**：FLOPs 3.95T，成功率 98.6%，延迟 127 ms/step。

**推荐配置**：

- 追求极致加速 → fastv_r=0.9275（保留 7.25%），延迟 73ms，成功率 96.0%
- 追求精度 → fastv_r=0.875（保留 12.5%），成功率 98.0%，与基线仅差 0.6%


## 环境要求

| 类别 | 要求 | 说明 |
|------|------|------|
| Python | 3.10 | 必须 |
| PyTorch | 2.2.0 | 必须 |
| CUDA | 12.1 | 必须 |
| transformers | 4.47.0 | **必须用 vendored 版本**，不能用官方版 |
| tokenizers | 0.21.1 | 与 transformers 匹配 |
| timm | 0.9.10 | 必须 < 1.0.0 |
| robosuite | 1.4.0 | LIBERO 依赖 |
| mujoco | 2.3.0 | 与 robosuite 匹配 |
| libero | 0.1.4 | 从 lerobot-libero-main 安装 |

**环境变量**（必须设置，否则跑不起来）：

- LD_LIBRARY_PATH 指向 torch 自带的 cuDNN
- PYTHONPATH 指向 openvla-oft 根目录

**如果用户环境缺失依赖**：Codex 应提示用户按上表补装，重点检查 transformers 和 timm 版本。


## 模型权重

| 项目 | 值 |
|------|-----|
| 基础模型 | openvla-7b-oft-libero-object |
| 下载来源 | https://hf-mirror.com/moojink/openvla-7b-oft-finetuned-libero-object |
| 必需文件 | 4 个 safetensors + config.json + tokenizer 相关文件 |

**如果用户没有模型权重**：Codex 应指导用户从上述来源下载，并检查文件完整性。


## 调用接口

```python
from vla_pruner_accelerator import VLAPrunerAccelerator

acc = VLAPrunerAccelerator.install(
    checkpoint="模型权重路径",
    task_suite_name="libero_object",
    fastv_r=0.9275,
)

acc.start_episode()

result = acc.step(
    image=图像,
    wrist_image=腕部图像,
    instruction="任务指令",
    proprio=本体感知,
)
action = result["action"]

stats = acc.end_episode(success=True)
```

**Codex 应告知用户**：

- 四个接口：install / start_episode / step / end_episode
- step 的输入必须包含 image 和 instruction
- proprio 是 8 维数组


## 关键参数

| 参数 | 含义 | 默认值 | 用户可调范围 |
|------|-----|------|-----|
| fastv_r | 剪枝比例 | 0.9275 | 0.5~0.95 |
| vla_pruner_layer | 剪枝层 | 15 | 10~20 |
| vla_pruner_mode | 剪枝模式 | semantic_action | semantic_action / fastv |
| vla_pruner_av_hist_w | EMA 窗口 | 3 | 2~5 |
| vla_pruner_av_decay | EMA 衰减率 | 0.8 | 0.5~0.9|

**Codex 应根据用户需求推荐参数**：
- 追求极致加速 → fastv_r=0.9275
- 追求精度 → fastv_r=0.875


## 已知问题（Codex 应提前告知用户）

| 问题 | 现象 | 原因 | 解决 |
|------|------|------|------|
| 剪枝统计显示 0% | Token Reusing Ratio: 0.00% | 指标混淆 | 以 pruning-report 为准 |


## 验证标准（Codex 判断是否跑通）

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


## Codex 指导用户时的建议话术

1. 用户问“怎么用” → “把工具箱文件夹上传，我读完 README 后指导你。”
2. 用户问“环境怎么配” → “需要 Python 3.10 + PyTorch 2.2.0 + transformers 4.47.0（vendored版），我帮你检查。”
3.  用户问“跑不起来” → “先检查 LD_LIBRARY_PATH 和 PYTHONPATH，这是最常见的坑。”
4. 用户问“效果如何” → “7.25% 保留时 FLOPs 省 70%，成功率 96%，延迟 73ms。”
5. 用户问“选哪个剪枝比例” → “追求加速选 0.9275，追求精度选 0.875。”