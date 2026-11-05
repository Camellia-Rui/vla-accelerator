# VLA 工具箱测试说明

适用于 0.3.0a6。先确定测试对象，再选择命令；不要把普通 OpenVLA-OFT 和 LightVLA 专用训练权重混为一类。本文包含项目现有服务器的路径，外部用户需替换并重新检查资源。

## 1. 选择测试路线

| 你提供的模型 | 当前测试路线 |
|---|---|
| 匹配已锁定后端的 OpenVLA-OFT Object，连续 L1 动作头 | VLA-Pruner 原生基线 → 固定输入对照 → 目标任务闭环 |
| LightVLA 专用 Spatial 权重及匹配代码 | LightVLA 固定输入 → 配对闭环 |
| 其他 OFT 变体、基础 OpenVLA、其他 VLA | 先 doctor 和源码审查，不能直接承诺兼容 |

普通 OFT Object 直接使用 LightVLA 风格推理干预已有 1/6 的失败结果，不作为默认加速推荐。原始 OpenVLA 接入尚未完成。

## 2. 开始前准备

用户提供模型工作目录、checkpoint 路径、原生运行命令和 Python 环境。核对 processor/tokenizer、归一化统计、动作头、本体投影器、视图数和动作块形状。不要只传一个权重文件就假定资源齐全。

在工具箱根目录执行：

```bash
python -m core doctor --workspace /absolute/path/to/model-project --output runs/model-check-001
python -m core methods
```

检查不会加载模型。随后由 Codex 检查实际加载入口，确认原模型可以正常推理。

下面 GPU 命令在 Linux Bash 执行。先查看 `nvidia-smi`，选择自己有权使用且资源足够的 GPU。示例 GPU 2 不是预留资源。服务器上所有新增代码、缓存和结果放在 `/home/wxy/Rui_work`，其他目录只读；不得杀死其他成员的进程。

从仓库运行时，进入自己拉取的仓库根目录即可。若已解压 0.3.0a6 发布包，在解压目录执行完整性校验：

```bash
cd /实际解压位置/vla-express-toolbox-0.3.0a6
/home/wxy/miniconda3/envs/daisy_openvla/bin/python -B scripts/verify_product.py
```

只有包含 product-manifest.json 的完整发布包执行此校验。所有结果目录必须是新目录；重复测试更换后缀，不覆盖旧证据。

## 3. OpenVLA-OFT + VLA-Pruner

### 资源与环境

项目已验证环境为 `/home/wxy/miniconda3/envs/openvla-oft/bin/python`，PyTorch 2.2.0+cu121、自定义 Transformers 4.47.0。不能用普通 PyPI Transformers 替代。当前 tokenizers 0.21.4 与后端期望版本存在提示，需保留到环境记录，不擅自更改共享环境。

```bash
PYTHON=/home/wxy/miniconda3/envs/openvla-oft/bin/python
WORKSPACE=/home/wxy/Rui_work/VLA-Pruner/VLA-Pruner/src/openvla-oft
CHECKPOINT=/home/wxy/Rui_work/openvla-7b-oft-libero-object
OBS=/home/wxy/Rui_work/toolbox-native-check/capture-001
GPU=2
```

内置加载器检查官方固定提交、实际 import 来源与 checkpoint 模型代码。若检查失败，保留错误并审查差异，不通过覆盖 checkpoint 或移除检查强行运行。

OBS 必须包含经检查的 report.json 与 observation.npz：后者含 full_image、wrist_image、state，前者含 passed=true、observation_sha256、task_description。已有 capture-001 是真实采集输入。测试自己的任务时需按原生预处理重新采集，不能随意替换图片或复制其他任务的元数据。格式见 [统一运行指南](docs/UNIFIED_RUNTIME.md)。

### 原生配置基线

```bash
"$PYTHON" -B -m core run --method vla_pruner \
  --workspace "$WORKSPACE" --checkpoint "$CHECKPOINT" \
  --unnorm-key libero_object_no_noops --observation-dir "$OBS" \
  --gpu "$GPU" --prefer-env-cuda --mode baseline \
  --output /home/wxy/Rui_work/toolbox-native-check/pruner-user-baseline-001
```

预期退出码 0，report.json 中 passed=true、detached=true，actions.jsonl 含完整有限动作；当前目标形状应为 8×7。该命令使用工具箱的关闭剪枝配置；下一步还会与未经适配器改写的原生调用进行一致性对照。

### 配对加速评估

```bash
"$PYTHON" -B -m core evaluate --method vla_pruner \
  --workspace "$WORKSPACE" --checkpoint "$CHECKPOINT" \
  --unnorm-key libero_object_no_noops --observation-dir "$OBS" \
  --gpu "$GPU" --prefer-env-cuda --prune-ratio 0.5 \
  --pairs 6 --warmup 3 \
  --output /home/wxy/Rui_work/toolbox-native-check/pruner-user-eval-001
```

先检查 process.log、进程退出码、report.json 和 evaluation.json，再读 records.jsonl。应有原生/关闭剪枝一致性、真实层长度变化和卸载恢复结果。多个输入可重复传入 --observation-dir。

未指定质量门槛时 input_screen.decision=INCONCLUSIVE 是预期行为。可由用户确定 --max-action-mae、--max-action-error、--min-gripper-agreement、--min-speedup；全部满足也只代表输入筛查通过，不能替代闭环。INPUT_SCREEN_FAIL 返回 3，运行或恢复错误返回 2。

该统一 evaluate 目前只做输入级测试。现有 OFT 闭环实验脚本与六对证据见 [同权重对照说明](docs/OFT_METHOD_CHOICES.md)；其中包含普通 OFT + LightVLA 的失败实验，不要把它当成两种方法都质量合格。新模型正式验收需要目标任务的配对闭环。

## 4. LightVLA 专用 Spatial

使用已有 `daisy_openvla` 环境和专用训练权重，不在共享环境安装工具箱的 CPU 测试依赖。

```bash
PYTHON=/home/wxy/miniconda3/envs/daisy_openvla/bin/python
WORKSPACE=/home/wxy/daisy_work/lightvla_clean
CHECKPOINT=/home/wxy/daisy_work/lightvla_clean/checkpoints/lightvla_spatial_official
GPU=2
```

### 固定输入对照

```bash
"$PYTHON" -B -m core evaluate --method lightvla \
  --workspace "$WORKSPACE" --checkpoint "$CHECKPOINT" --gpu "$GPU" \
  --task 0 --trial 0 --warmup 3 --pairs 4 \
  --output /home/wxy/Rui_work/toolbox-native-check/lightvla-user-pair-001
```

预期 process.json 的 exit_code=0，report.json 中以下检查通过：

- passed、mechanism_observed、detached 为 true。
- native_vs_packaged.matches、restored_vs_native.matches 为 true。
- parameter_identity_samples_preserved、module_forward_restored 为 true。
- records.jsonl 保存每次完整 8×7 动作与同步计时；observation.npz 保存实际输入。

baseline 指同一专用训练权重关闭裁剪；builtin 指原项目实现；packaged 指工具箱实现。不能将该 baseline 当成普通 OpenVLA 的另一套权重。

### 配对闭环

先做一对最小回归：

```bash
"$PYTHON" -B -m core evaluate --method lightvla --closed-loop \
  --workspace "$WORKSPACE" --checkpoint "$CHECKPOINT" --gpu "$GPU" \
  --tasks 0 --trials 0 --seed 7 \
  --output /home/wxy/Rui_work/toolbox-native-check/lightvla-user-loop-001
```

检查 report.json 中执行完成、无错误、参数与模块恢复；对照逐回合文件的 initial_state_hash、seed、完整动作与 success。完成运行不代表任务一定成功，应单独报告两模式成功数。

扩大到四任务二十对时，用后台启动器避免终端关闭中断任务：

```bash
bash scripts/launch_lightvla_closed_loop.sh "$PYTHON" "$WORKSPACE" "$CHECKPOINT" \
  /home/wxy/Rui_work/toolbox-native-check/lightvla-user-loop-020 "$GPU" \
  --tasks 0 1 2 3 --trials 0 1 2 3 4
```

每回合原子保存。需要续跑时先确认旧父子进程都结束，再用相同配置和统一闭环命令加 --resume；不跨代码、权重或环境版本续跑，也不无条件删除锁文件。

独立重算报告：

```bash
"$PYTHON" -B scripts/summarize_lightvla_closed_loop.py \
  /home/wxy/Rui_work/toolbox-native-check/lightvla-user-loop-001/report.json \
  --output /home/wxy/Rui_work/toolbox-native-check/lightvla-user-loop-001/audit.json
```

LightVLA 使用 quality_decision 字段，不使用 VLA-Pruner 的 input_screen 阈值参数。当前小规模评估保持 INCONCLUSIVE；成功率、质量门槛和正式统计需要另行审查。

## SpecPrune 实验评估

使用匹配的 SpecPrune Object 后端和模型环境。运行 `python -m core evaluate --method specprune --help` 查看入口，单回合使用 `--closed-loop`，批量和续跑使用 `scripts/batch_specprune.py`。服务器资源路径、实际命令与三个已测初始状态见 [SpecPrune 使用说明](docs/SPECPRUNE_WORKFLOW.md)。

## 5. 常见问题

| 现象 | 如何处理 |
|---|---|
| 输出目录已存在 | 改用新的目录；仅明确的闭环续跑使用 --resume |
| 找不到动作头、processor 或归一化键 | 检查权重变体和配套文件，不借用其他模型的统计数据 |
| 固定后端或 checkpoint 代码校验失败 | 保存差异，核对版本和 import 来源，不覆盖原文件 |
| CUDA/cuDNN 加载错误 | 核对实际 Python、GPU 可见性和库路径；OFT 示例使用 --prefer-env-cuda |
| 没有 report.json 或进程非零退出 | 查看 process.log 和启动日志，按失败处理，不能宣称已完成 |
| 速度更快但任务成功率下降 | 保留失败样本，不推荐该配置；不能只展示速度 |
| 原生也无法运行 | 先解决模型环境、资源或输入问题，再接入加速 |

## 6. 测试后回传什么

回传工具箱 commit/版本、模型源码版本、checkpoint 身份、实际命令、环境与 GPU、输入/初始状态、完整报告与日志、动作和原始计时。注明任务数量、初始状态数量、成功数、错误数、质量门槛、是否与历史样本重叠。不要上传模型权重、密码或访问令牌。

当前可参考的原始结果：

- [OFT Object：原生和 VLA-Pruner 各 6/6，LightVLA 风格干预 1/6](docs/OFT_METHOD_CHOICES.md)。
- [LightVLA Spatial：四任务二十对，两模式各 20/20](docs/LIGHTVLA_CLOSED_LOOP.md)。
- [新统一入口：四对固定输入和一对闭环回归](docs/LIGHTVLA_UNIFIED_RESULT.md)。新闭环与上述样本重叠，不相加计算独立样本。

本地代码测试与发布包校验见 [总 README](README.md)。本地 CPU 测试通过不等于真实模型质量验收通过。
