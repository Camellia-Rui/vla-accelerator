# LightVLA Packaged Pruner：接入、评测与验证

本文档对应代码目录 `lightvla_clean/`，说明如何在 OpenVLA/LIBERO 中运行无剪枝基线（`baseline`）和独立插件剪枝（`packaged`），并记录当前已完成的环境、权重、接口、参数、实测数据和验证标准。

> 当前结论仅适用于 `lightvla_spatial_official + libero_spatial + NVIDIA H20`。不要将本文结果表述为其他 LIBERO suite 或其他硬件上的结果。

## 1. 项目与模式

LightVLA 通过视觉 token 剪枝减少 LLaMA decoder 的输入序列长度，从而降低推理计算量和延迟。本仓库将剪枝逻辑抽取为可复用插件：

```text
lightvla_clean/packaged_pruner/src/lightvla_pruner/
├── core.py
├── layout.py
└── adapters/prefix_llama.py
```

三种运行模式：

| 模式 | 行为 | 用途 |
|---|---|---|
| `baseline` | 将 decoder 恢复为标准 `LlamaModel.forward`，不做 token 剪枝 | 无剪枝对照 |
| `packaged` | 先关闭 builtin 剪枝，再挂载独立 `lightvla_pruner` adapter | 插件实验 |
| `builtin` | 使用 checkpoint 自带的 LightVLA pruner | 核对原实现，不是 baseline |

`baseline` 和 `packaged` 使用同一个 checkpoint，仅切换运行时 forward/adapter；不是重新训练，也不会把 packaged pruner 与 builtin pruner 叠加。

## 2. 环境

### 2.1 已实测服务器环境

| 项目 | 实测值 |
|---|---|
| 操作系统 | Linux x86_64 |
| Conda 环境 | `daisy_openvla` |
| Python | 3.10 |
| Python 路径 | `/home/wxy/miniconda3/envs/daisy_openvla/bin/python` |
| PyTorch | `2.3.0+cu121` |
| Torch CUDA | `12.1` |
| cuDNN Python 包 | `nvidia-cudnn-cu12==8.9.2.26` |
| GPU | NVIDIA H20 |
| robosuite | `1.4.1` |
| 图形后端 | EGL（无显示器服务器） |
| 项目目录 | `/home/wxy/daisy_work/lightvla_clean` |

仓库还依赖 LIBERO、MuJoCo、Transformers/OpenVLA-OFT、timm、peft、sentencepiece、TensorFlow 等；具体声明见：

```text
lightvla_clean/pyproject.toml
lightvla_clean/experiments/robot/libero/libero_requirements.txt
lightvla_clean/packaged_pruner/pyproject.toml
```

### 2.2 激活与动态库隔离

服务器曾出现来自其他 conda 环境的 `LD_LIBRARY_PATH` 污染：

```text
libcudnn_cnn_infer.so.8: undefined symbol ... libcudnn_ops_infer.so.8
```

运行本项目时使用当前 shell 级别的临时清理，不会修改其他用户或服务器文件：

```bash
conda activate daisy_openvla
cd ~/daisy_work/lightvla_clean

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
env -u LD_LIBRARY_PATH python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

不要把 CoppeliaSim 或其他 conda 环境的动态库永久写入本环境的 `LD_LIBRARY_PATH`。

### 2.3 离线服务器加载 packaged pruner

服务器无网络时，`pip install -e ./packaged_pruner` 可能因无法下载 `setuptools>=68` 失败。可以直接使用源码路径：

```bash
export PYTHONPATH="$PWD/packaged_pruner/src${PYTHONPATH:+:$PYTHONPATH}"

env -u LD_LIBRARY_PATH \
  "$CONDA_PREFIX/bin/python" -c \
  "import torch, lightvla_pruner; print(torch.__version__); print(lightvla_pruner.__file__)"
```

预期 `lightvla_pruner.__file__` 指向：

```text
/home/wxy/daisy_work/lightvla_clean/packaged_pruner/src/lightvla_pruner/__init__.py
```

## 3. 权重

### 3.1 本次使用的 checkpoint

```text
checkpoints/lightvla_spatial_official
```

对应任务：

```text
task_suite_name = libero_spatial
10 tasks × 50 trials/task = 500 episodes/mode
```

checkpoint 和 suite 必须匹配。Spatial checkpoint 的结果不能写成 Object、Goal、LIBERO-10 或 LIBERO-90 的结果。

### 3.2 权重指纹

对 checkpoint 中的权重文件（`.safetensors`、`.bin`、`.pt`、`.pth`、`.ckpt`）计算组合 SHA-256，不包含可能由加载逻辑调整的 `config.json`。

四次指纹：

| 时点 | SHA-256 |
|---|---|
| baseline 前 | `ec51ac34125bc18f1ebdbbb56462ab0769c0cf6ab59f31b77c5b8915718777c94` |
| baseline 后 | `ec51ac34125bc18f1ebdbbb56462ab0769c0cf6ab59f31b77c5b8915718777c94` |
| packaged 前 | `ec51ac34125bc18f1ebdbbb56462ab0769c0cf6ab59f31b77c5b8915718777c94` |
| packaged 后 | `ec51ac34125bc18f1ebdbbb56462ab0769c0cf6ab59f31b77c5b8915718777c94` |

判定结果：

```text
四次指纹是否完全一致: True
```

这证明 baseline 与 packaged 使用同一份模型权重，且两次评估均未修改 checkpoint 参数。运行时挂载 pruner 不等于训练或改写权重。

## 4. 接口

### 4.1 Python 包接口

`lightvla_pruner` 导出：

```python
from lightvla_pruner import (
    DifferentiableTokenPruner,
    PrefixLlamaPruningAdapter,
    PruningResult,
    SequencePruningLayout,
    StaticPrefixLayoutResolver,
    TokenSpan,
    build_position_ids,
)
```

适用的输入布局为：

```text
[prefix][连续 visual patch tokens][suffix]
```

它不直接适用于无法确定连续视觉 patch span 的架构，也不直接等价于 Q-Former 或 Flamingo cross-attention 模式。

### 4.2 仓库集成接口

统一入口位于：

```text
lightvla_clean/scripts/lightvla_pruning.py
```

调用方式：

```python
summary = configure_lightvla_pruning(
    vla=model,
    mode="packaged",
    patch_start=1,
    score_context="suffix",
)
```

运行时状态通过以下接口读取：

```python
get_pruning_runtime_summary(model)
```

packaged 模式应至少确认：

```text
mode = packaged
pruner_class = DifferentiableTokenPruner
pruned_forward_count > 0
```

baseline 模式应确认：

```text
mode = baseline
pruner_class = None
```

### 4.3 评估与 benchmark 接口

完整 LIBERO 评估：

```text
experiments/robot/libero/run_libero_eval.py
```

固定 observation benchmark：

```text
scripts/benchmark_libero_inference.py
```

每个 episode 的 JSONL 指标由修改后的 `run_libero_eval.py` 保存，字段包括：

```text
schema
task_id
episode_idx
success
episode_seconds
inference_times_ms
inference_seconds
inference_calls
mean_inference_ms
initial_state_hash
checkpoint
device
pruning_mode
error
```

benchmark JSON 的延迟节点同时保存统计量和原始数组：

```json
{
  "num_iterations": 50,
  "iteration_latencies_ms": [120.8, 125.6, 154.9],
  "avg_ms": 124.28,
  "min_ms": 109.27,
  "max_ms": 157.17,
  "p50_ms": 120.48,
  "p95_ms": 156.78
}
```

原始数组分别位于：

```text
full_query_latency_ms.iteration_latencies_ms
predict_action_latency_ms.iteration_latencies_ms
```

## 5. 参数

### 5.1 完整 500-episode 评估参数

| 参数 | 值 |
|---|---|
| `pretrained_checkpoint` | `checkpoints/lightvla_spatial_official` |
| `task_suite_name` | `libero_spatial` |
| `num_trials_per_task` | `50` |
| task 数量 | `10` |
| 总 episode 数 | `500` |
| `seed` | `7` |
| `model_family` | `openvla` |
| `use_l1_regression` | `true` |
| `use_diffusion` | `false` |
| `num_images_in_input` | `2` |
| `use_proprio` | `true` |
| `center_crop` | `true` |
| `num_open_loop_steps` | `8` |
| `env_img_res` | `256` |
| `initial_states_path` | `DEFAULT` |
| `save_rollout_video` | `false` |
| baseline 模式 | `--pruning_mode baseline` |
| packaged 模式 | `--pruning_mode packaged` |
| `pruning_patch_start` | `1` |
| `pruning_score_context` | `suffix` |

baseline 示例：

```bash
env -u LD_LIBRARY_PATH \
  "$CONDA_PREFIX/bin/python" -u experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/lightvla_spatial_official \
  --task_suite_name libero_spatial \
  --num_trials_per_task 50 \
  --pruning_mode baseline \
  --run_id_note baseline \
  --local_log_dir results/baseline_eval \
  --save_rollout_video false
```

packaged 示例：

```bash
env -u LD_LIBRARY_PATH \
  PYTHONPATH="$PWD/packaged_pruner/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$CONDA_PREFIX/bin/python" -u experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/lightvla_spatial_official \
  --task_suite_name libero_spatial \
  --num_trials_per_task 50 \
  --pruning_mode packaged \
  --run_id_note packaged \
  --local_log_dir results/packaged_eval \
  --save_rollout_video false
```

### 5.2 固定 observation benchmark 参数

两种模式必须使用相同参数和同一张 GPU：

| 参数 | 值 |
|---|---|
| `task_suite_name` | `libero_spatial` |
| `task_id` | `0` |
| `episode_idx` | `0` |
| `device` | `cuda:0` |
| `num_warmup` | `10` |
| `num_iters` | `50` |
| `seed` | `7` |
| `num_open_loop_steps` | `8` |

baseline：

```bash
env -u LD_LIBRARY_PATH \
  PYTHONPATH="$PWD/packaged_pruner/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$CONDA_PREFIX/bin/python" scripts/benchmark_libero_inference.py \
  --pretrained_checkpoint checkpoints/lightvla_spatial_official \
  --task_suite_name libero_spatial \
  --task_id 0 \
  --episode_idx 0 \
  --device cuda:0 \
  --num_warmup 10 \
  --num_iters 50 \
  --pruning_mode baseline \
  --output_json results/benchmark_baseline.json
```

packaged：

```bash
env -u LD_LIBRARY_PATH \
  PYTHONPATH="$PWD/packaged_pruner/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$CONDA_PREFIX/bin/python" scripts/benchmark_libero_inference.py \
  --pretrained_checkpoint checkpoints/lightvla_spatial_official \
  --task_suite_name libero_spatial \
  --task_id 0 \
  --episode_idx 0 \
  --device cuda:0 \
  --num_warmup 10 \
  --num_iters 50 \
  --pruning_mode packaged \
  --output_json results/benchmark_packaged.json
```

## 6. 实测数据

### 6.1 500 个配对 episode

指标文件：

```text
results/baseline_eval/EVAL-libero_spatial-openvla-2026_10_16-04_49_56--baseline.episodes.jsonl
results/packaged_eval/EVAL-libero_spatial-openvla-2026_10_16-18_33_31--packaged.episodes.jsonl
```

配对条件：按 task、episode 和 `initial_state_hash` 对齐。500 对初始状态哈希全部匹配，JSONL 中没有记录 episode runtime error。

| 配对结果 | episode 数 |
|---|---:|
| 两者都成功 | 482 |
| 仅 packaged 成功 | 8 |
| 仅 baseline 成功 | 8 |
| 两者都失败 | 2 |
| 总配对数 | 500 |

总体成功率：

| 模式 | 成功数 | 总数 | 成功率 |
|---|---:|---:|---:|
| baseline | 490 | 500 | 98.0% |
| packaged | 490 | 500 | 98.0% |

成功率相同不表示每个 episode 的结果逐项相同；本次有 16 对 discordant episodes，其中 packaged-only 成功 8 个、baseline-only 成功 8 个。

配对比较产物：

```text
results/comparison_20261016_192729_256818/episode_comparison.csv
results/comparison_20261016_192729_256818/task_comparison.csv
results/comparison_20261016_192729_256818/summary.json
```

完整 episode 日志中的推理调用平均值曾测得：

| 模式 | 平均推理延迟 | 相对结果 |
|---|---:|---:|
| baseline | 323.428 ms | 1.000× |
| packaged | 79.089 ms | 4.089× faster |

对应延迟下降 `75.55%`。但该组 500-episode 数据中有 `350/500` 对使用了不同 CUDA device ID，且没有剔除首次调用 warmup，因此只能作为真实服务器运行观测，不能作为严格受控硬件 benchmark。严格延迟结论应以第 6.3 节为准。

### 6.2 Action 一致性

测试设计：

- 10 个 LIBERO Spatial tasks；
- 每个 task 取 5 个固定初始状态；
- 共 50 个完全相同的 observations；
- 每个 observation 输出 8-action chunk；
- 共比较 400 个 7 维 action vectors；
- baseline 与 packaged 固定在同一 GPU 上分别推理。

结果：

| 指标 | 结果 |
|---|---:|
| Observation 数 | 50 |
| Action vector 数 | 400 |
| 总体 Action MAE | `0.00870194` |
| Gripper agreement | `100.0000%`（400/400） |
| Gripper mismatch | 0 |

`Gripper agreement` 表示 baseline 与 packaged 对机械臂末端夹爪开合方向的判断是否一致；100% 表示全部 400 个动作的夹爪决策一致，不代表 500 个任务全部成功。

结果文件：

```text
results/action_consistency/baseline_actions.json
results/action_consistency/packaged_actions.json
results/action_consistency/action_comparisons.json
results/action_consistency/summary.json
```

注意：旧脚本打印的 `Action MAE per dim: 0.00870194` 是标量重复，不是可靠的 7 维逐维 MAE。当前可正式引用的是总体 MAE `0.00870194` 和 gripper 一致率 `100%`；如需逐维 MAE，应从两份 action JSON 重新计算。

### 6.3 受控 benchmark 与原始 iteration 延迟

本组 benchmark 固定同一 checkpoint、同一 observation、同一 `cuda:0`、10 次 warmup 和 50 次正式 iterations。正式 iteration 的每次延迟均保存在输出 JSON 中，warmup 不计入统计。

#### `predict_action` 延迟

| 指标 | Baseline | Packaged | 变化 |
|---|---:|---:|---:|
| 平均延迟 | 214.67 ms | 124.28 ms | 降低 42.1% |
| 最小延迟 | 110.90 ms | 109.27 ms | 降低 1.5% |
| P50 | 250.70 ms | 120.48 ms | 降低 51.9% |
| P95 | 269.55 ms | 156.78 ms | 降低 41.8% |
| 最大延迟 | 270.19 ms | 157.17 ms | 降低 41.8% |
| Queries/s | 4.66 | 8.05 | 提升 1.73× |
| Control FPS | 37.27 | 64.37 | 提升 1.73× |

平均延迟加速比：

```text
214.67 / 124.28 = 1.73×
```

原始延迟分布显示 packaged 多数 iteration 位于约 `109–157 ms`；baseline 后段大量 iteration 位于约 `245–270 ms`。应保留完整 JSON 数组，而不是只保存平均值。

#### Full-query 延迟

根据输出的 queries/s：

| 模式 | Queries/s | 平均 full-query 延迟 |
|---|---:|---:|
| baseline | 3.5207 | 约 284.04 ms |
| packaged | 7.0998 | 约 140.85 ms |

完整查询约提升 `2.02×`。`full_query` 包含 observation clone、输入处理、模型调用和动作输出流程；`predict_action` 更接近核心模型调用，因此两类数字应分别报告，不能混写。

#### FLOPs

| 模式 | `torch.profiler` operator-reported FLOPs |
|---|---:|
| baseline | `8.809 × 10^12` |
| packaged | `3.406 × 10^12` |

Packaged FLOPs 约为 baseline 的 `38.7%`，减少约 `61.3%`。这是 `torch.profiler` 对一次 `predict_action` 的 operator-reported FLOPs，不应表述为硬件实际执行的全部指令数。

### 6.4 综合结果

| 维度 | 结果 |
|---|---|
| 成功率 | baseline 与 packaged 均为 98.0%（490/500） |
| 配对初始状态 | 500/500 哈希匹配 |
| Action 一致性 | MAE `0.00870194` |
| Gripper 一致性 | 100%（400/400） |
| 权重一致性 | 四次 SHA-256 完全一致 |
| 受控平均 `predict_action` 延迟 | 214.67 ms → 124.28 ms |
| 受控 `predict_action` 加速 | 1.73× |
| 受控平均 full-query 延迟 | 约 284.04 ms → 约 140.85 ms |
| 受控 full-query 加速 | 约 2.02× |
| FLOPs | 8.809T → 3.406T，降低 61.3% |

## 7. 验证标准

只有同时满足以下条件，才能认为 baseline/packaged 对比有效。

### 7.1 环境标准

- 使用同一 conda 环境和同一版本的 Python、PyTorch、CUDA、cuDNN；
- `torch.cuda.is_available()` 为 `True`；
- benchmark 固定同一 GPU，例如 `cuda:0`；
- 使用 `env -u LD_LIBRARY_PATH` 避免其他环境的 CUDA/cuDNN 动态库污染；
- 无显示器服务器使用 EGL，退出时出现的 `EGL_NOT_INITIALIZED` 析构警告不推翻已经完成并落盘的评估结果。

### 7.2 权重标准

- baseline 前后权重 SHA-256 相同；
- packaged 前后权重 SHA-256 相同；
- baseline 与 packaged 权重 SHA-256 相同；
- checkpoint 路径和 LIBERO suite 匹配；
- 不把 `config.json` 的运行时同步误判为模型参数变化。

### 7.3 模式标准

- baseline runtime summary 显示 `mode=baseline` 和 `pruner_class=None`；
- packaged runtime summary 显示 `mode=packaged` 和 `pruner_class=DifferentiableTokenPruner`；
- packaged 的 `pruned_forward_count > 0`；
- packaged 不与 builtin pruner 重复挂载。

### 7.4 Episode 配对标准

- 每种模式均完成 `500` episodes；
- 两份 JSONL 均恰好有 `500` 条有效记录；
- task ID、episode ID 能一一配对；
- `initial_state_hash` 为 `500/500` 匹配；
- `error` 字段为空；
- 成功率既报告总数，也报告 paired outcome：both success、packaged-only、baseline-only、both fail。

### 7.5 Action 一致性标准

- 两种模式输入完全相同的 observation 数据；
- 使用同一 checkpoint、GPU、dtype 和动作归一化流程；
- 报告 action vector 数、总体 MAE、gripper agreement 和 mismatch 数；
- 未正确计算 7 维逐维 MAE 时，不得把总体标量写成 per-dimension 结果。

### 7.6 延迟标准

- baseline 与 packaged 固定同一 GPU、task、episode、seed 和模型参数；
- warmup 次数相同，且 warmup 不计入正式统计；
- 正式 iteration 数相同，本次为 50；
- 使用 `torch.cuda.synchronize()` 包围计时段；
- 保存每次 iteration 的原始延迟，不只保存平均值；
- 至少报告 avg、min、max、P50、P95；
- 分开报告 `full_query` 和 `predict_action`；
- episode 级多 GPU 数据必须标记为非严格受控结果。

### 7.7 最小验收结论

本次实验满足以下核心证据链：

1. 四次权重 SHA-256 完全一致，排除训练或权重变化；
2. 500/500 初始状态哈希匹配，保证 episode 可配对；
3. 两种模式成功率均为 98.0%；
4. 400 个 action 的总体 MAE 为 `0.00870194`，gripper 决策 100% 一致；
5. 受控 benchmark 保存 50 次原始 iteration 延迟；
6. packaged 的 `predict_action` 平均延迟降低 42.1%，full-query 约提升 2.02×；
7. operator-reported FLOPs 降低约 61.3%。

因此，可以表述为：

> 在相同 `lightvla_spatial_official` 权重、相同 LIBERO Spatial 输入和受控 GPU 条件下，packaged pruner 在保持总体成功率和夹爪决策一致性的同时，显著降低了推理延迟与 profiler FLOPs；差异来自运行时 token 剪枝，而不是重新训练或权重修改。

## 8. 结果归档建议

建议服务器最终保留：

```text
results/
├── baseline_eval/
│   ├── *.episodes.jsonl
│   ├── weights_before.json
│   └── weights_after.json
├── packaged_eval/
│   ├── *.episodes.jsonl
│   ├── weights_before.json
│   └── weights_after.json
├── action_consistency/
│   ├── baseline_actions.json
│   ├── packaged_actions.json
│   ├── action_comparisons.json
│   └── summary.json
├── comparison_20261016_192729_256818/
│   ├── episode_comparison.csv
│   ├── task_comparison.csv
│   └── summary.json
├── benchmark_baseline.json
└── benchmark_packaged.json
```

每次新实验应同时归档命令、Git revision、环境版本、GPU 型号、checkpoint 指纹和原始 JSON/JSONL，避免只保留终端截图或平均值。
