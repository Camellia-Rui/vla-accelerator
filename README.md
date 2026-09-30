# VLA Acceleration Evaluator

`vla-acceleration-evaluator` 是一个用于评估、调优和接入 **免训练 VLA 推理加速模块** 的 Codex skill。

它把加速模块视为需要实验验证的系统改动，而不是仅凭 token 数、FLOPs 或局部 prefill 时间判断效果。最终目标是形成可复现的证据，并对当前模型、任务、硬件和参数组合给出 `PASS`、`TUNE`、`REJECT` 或 `INCONCLUSIVE` 决策。

## 适用场景

可以使用本 skill 完成以下工作：

- 检查 VLA 模型和加速仓库，定位安全的接入点；
- 区分单帧无状态裁剪与跨帧有状态缓存；
- 构建 baseline、加速模式和零效果 control 的配对实验；
- 编写可丢弃、可恢复的运行时探测、临时 wrapper、配对 benchmark 和闭环验证脚本；
- 验证 action、gripper、KV cache、位置编码和 attention mask 的一致性；
- 分解 vision、projector、调度器、LLM prefill 和 decode 延迟；
- 扫描加速参数，同时评估速度、显存和行为偏差；
- 生成标准结果 manifest，并执行确定性的准入判定；
- 仅在验证通过后生成可启停、可恢复的部署适配器。

它不适合只做理论 FLOPs 估算、只比较 synthetic input，或在没有真实模型运行证据时直接宣称部署加速。

## 快速使用

在 Codex 中显式调用：

```text
使用 $vla-acceleration-evaluator 检查这个 VLA 加速模块，完成配对基准、动作一致性验证和准入判断。
```

建议同时提供：

- VLA 实现或仓库路径；
- 模型 checkpoint 路径；
- 加速模块路径或仓库版本；
- 评测任务、episode、seed 或保存的真实 observation；
- 结果输出目录；
- 期望的最低加速比与可接受行为误差；
- GPU、CUDA、PyTorch 和 Transformers 环境信息。

示例：

```text
使用 $vla-acceleration-evaluator 验证 /workspace/accelerator 对 OpenVLA 的免训练加速效果。
checkpoint 位于 /models/openvla，使用 LIBERO-Spatial，结果写入 /workspace/results。
要求进行零裁剪 control、30 次配对延迟测试、动作误差检查和闭环 reset 验证。
```

## 核心评测流程

Skill 按以下顺序执行门控：

1. **审计**：记录代码提交、checkpoint、依赖版本、默认参数和本地修改。
2. **运行时探测**：先运行未修改模型，确认真实模块结构、视觉 token 布局、prefill/decode 调用和 cache 格式。
3. **接入与 control**：使用最小、可逆的 wrapper 或 hook；零效果模式必须复现 baseline。
4. **配对测量**：相同输入、充分预热、交替顺序、CUDA 同步并重复采样，报告 median 和 p95。
5. **语义验证**：检查完整 action、gripper、确定性、顺序效应、状态 reset 和闭环成功率。
6. **归因与调优**：定位收益与开销所在阶段，一次只改变一个参数族，并保留默认回归配置。
7. **准入决策**：生成 manifest，验证字段，然后输出确定性决策。

详细流程见 [`references/evaluation-protocol.md`](references/evaluation-protocol.md)。

编写临时验证脚本时，还应遵循 [`references/temporary-script-playbook.md`](references/temporary-script-playbook.md)。其中规定了训练权重不变检查、prefill/decode 分流、真实 vision prior、hook 清理、配对原始样本和远程结果回传等要求。

## 模块分类

### 无状态模块

典型形式是当前帧内的视觉 token 或 patch 裁剪。它不保存跨 observation 状态，但可能改变 action。

必须验证：

- patch 起点与数量来自真实 prefill，而不是硬编码猜测；
- 非视觉上下文、position IDs、attention mask 和 cache 语义保持正确；
- 零裁剪 control 与 baseline 一致；
- 固定画面动作相似性之外，还要验证闭环成功率。

已知的 LightVLA 风格风险和历史证据见 [`references/lightvla-pruner.md`](references/lightvla-pruner.md)。

### 有状态模块

典型形式是跨帧 patch/KV cache 复用。它必须定义 first frame、cached frame、episode reset、任务切换和异常恢复生命周期。

必须验证：

- 每次推理前清除无效的帧相关配置；
- 只有在 previous image、attention 和 cache 都有效时才启用复用；
- reset 后能够恢复首帧 baseline 行为；
- 不把局部 prefill 收益误认为端到端收益。

VLA-Cache 的适用环境、reset 约束和历史结果见 [`references/vla-cache.md`](references/vla-cache.md)。

## 结果 manifest

评测结果应整理为 schema v2 JSON manifest。完整字段和示例见 [`references/result-schema.md`](references/result-schema.md)。v2 强制记录训练权重未变化、原始证据路径，以及配对均值/中位节省时间置信区间的下界。

在 skill 目录中运行：

```bash
python scripts/validate_manifest.py /path/to/result.json
python scripts/admission_decision.py /path/to/result.json --pretty
```

第一条命令检查字段完整性和 schema 约束；第二条命令根据 manifest 内保存的阈值输出：

- `PASS`：结构、control、reset、行为、闭环和端到端速度门控全部通过；
- `TUNE`：结构与正确性成立，但性能或非关键行为指标仍有可信调优空间；
- `REJECT`：结构不兼容、状态泄漏、行为/闭环失败，或已无可信加速空间；
- `INCONCLUSIVE`：缺少必需证据或 manifest 无效。

阈值保存在每一份 manifest 中，因此决策只对该模型、checkpoint、任务分布、硬件、软件环境、适配器和参数组合有效。

## 内置脚本

### `validate_manifest.py`

检查 manifest 必填字段、数值类型、状态类型、closed-loop 证据和 stateful reset hook。

### `admission_decision.py`

在 manifest 有效后执行确定性的准入判断。非 `PASS` 结果使用非零退出码，便于接入自动化流程。

### `profile_vla_cache_stages.py`

面向 OpenVLA/VLA-Cache 的参考 profiler，可测量：

- 端到端 wall/CUDA 时间；
- vision backbone 与 projector；
- LLM prefill 与 decode；
- 静态 patch 检测、任务相关选择和 layer schedule；
- action 差异与实际复用 patch 数。

示例：

```bash
python scripts/profile_vla_cache_stages.py \
  --checkpoint /path/to/checkpoint \
  --frames-dir /path/to/real_frames \
  --output /path/to/profile.json
```

该脚本依赖 CUDA、PyTorch、Pillow、NumPy 以及目标 OpenVLA 工程中的 `experiments.robot.openvla_utils`。不同仓库应调整模型加载 import，不应假定它是通用 profiler。

### `self_test.py`

不依赖模型或 GPU，验证 manifest 校验器以及 `PASS / TUNE / REJECT / INCONCLUSIVE` 四类决策：

```bash
python scripts/self_test.py
```

## 目录结构

```text
vla-acceleration-evaluator/
├── SKILL.md
├── README.md
├── agents/
│   └── openai.yaml
├── references/
│   ├── evaluation-protocol.md
│   ├── temporary-script-playbook.md
│   ├── result-schema.md
│   ├── lightvla-pruner.md
│   └── vla-cache.md
└── scripts/
    ├── validate_manifest.py
    ├── admission_decision.py
    ├── self_test.py
    └── profile_vla_cache_stages.py
```

## 重要边界

- 不修改用户原始 checkpoint；需要修改 remote-code 文件时使用独立 checkpoint 目录。
- 仓库、权重、数据集、日志和实验结果应保存在 skill 目录之外。
- synthetic frame 只能作为 smoke test，不能替代真实 observation 和闭环任务。
- token、patch 或 cache 数量下降不等于端到端加速。
- 固定画面 action 接近不等于机器人任务成功率不下降。
- 内置 LightVLA 与 VLA-Cache 结论是特定 OpenVLA/LIBERO/RTX 4090 环境的历史证据，不代表对这些方法的普遍否定。
- 只有通过全部准入门控的配置才能生成部署适配器。
