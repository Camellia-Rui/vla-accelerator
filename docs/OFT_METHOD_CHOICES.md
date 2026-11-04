# 同一份 OpenVLA-OFT 权重的两种剪枝入口

本次使用同一份 `/home/wxy/Rui_work/openvla-7b-oft-libero-object` 权重、固定 VLA-Pruner OFT 后端、同一个已加载模型实例，分别运行 baseline、LightVLA 风格前端剪枝和 VLA-Pruner。没有重新训练，也没有替换为 LightVLA 的 Spatial 专用权重。

这里的 LightVLA 选项是**已有 OFT 权重上的推理干预**，不等同于包含微调的原始 LightVLA 方法复现。Spatial 专用权重的 20/20 结果不能迁移到本次 Object 权重。

同权重闭环已观察到原生成功而 LightVLA 风格剪枝失败的回合。因此该 LightVLA 入口当前仅供实验，不作为这份 Object 权重的默认质量推荐。失败回合必须纳入最终报告。

## 同权重闭环结果（2026-11-04）

2 个任务、每任务 3 个初始状态、三种模式共 18 个 episode，独立审核结果：

| 模式 | 成功数 | 当前建议 |
|---|---:|---|
| 原生 OFT | 6/6 | 对照 |
| VLA-Pruner，50% 剪枝 | 6/6 | 可继续扩大验证，尚非正式部署验收 |
| LightVLA 风格前端剪枝 | 1/6 | 不推荐用于这份 Object 权重 |

没有运行异常；LightVLA 风格剪枝的 5 次失败是任务未完成，不是导入、环境或形状报错。每组初始状态和 seed 一致，最终原生动作恢复、模块身份及参数抽样检查通过。原始动作和失败记录见 [审核报告](server-evidence/2026-11-04/oft-choice-closedloop-001/audit.json) 及同目录逐回合 JSON。

目前实现了可执行的双方法接入和比较，**没有得到两种都可靠的同权重选择**。该结果仅针对本次权重、实现和配置，不否定 LightVLA 专用训练权重的已验证结果。要使普通 OFT 上的 LightVLA 路线可推荐，需要另行研究剪枝强度或训练适配，并重新验收；不能通过更换名称把失败配置列为稳定支持。

## 已完成的固定输入检查

3 个已有真实输入，每输入 3 轮、每轮三种模式轮换次序，每模式预热 3 次。两种方法均有实际 decoder 序列缩短，完整动作有限，切换后的原生输出恢复，参数对象和抽样值检查通过。

正式记录配置的一轮使用 VLA-Pruner `fastv_r=0.5`；LightVLA 风格剪枝按其算法自适应选择 token，不能把两者称为同 token 预算的算法比较。LightVLA 在这三份输入中保留约 41、37、52 个视觉 token，原始为 512。

| 输入 | LightVLA 风格动作 MAE | VLA-Pruner 50% 动作 MAE |
|---|---:|---:|
| task 0 初始输入 | 0.09366 | 0.00765 |
| task 1 初始输入 | 0.15136 | 0.01155 |
| task 2 初始输入 | 0.04627 | 0.00899 |

原始记录在 `server-evidence/2026-11-04/oft-choice-002/`。首轮 `oft-choice-001` 使用 VLA-Pruner 默认 `0.9275`，保留用于来源追踪，不与 50% 配置混算。当前延迟仅为共享服务器条件下的描述性结果；attention 路径和硬件负载尚未完成严格控制。

## 使用同一后端和权重

VLA-Pruner 沿用 `python -m core run --method vla_pruner` 的固定后端加载器，见 `UNIFIED_RUNTIME.md`。

LightVLA 风格实验入口为已有的统一工厂协议：

```bash
python -B -m core run --method lightvla \
  --factory methods.lightvla.pruner_oft_factory:build \
  --factory-kwargs '{"workspace":"/实际/VLA-Pruner/src/openvla-oft","checkpoint":"/实际/OFT权重","observation_dir":"/实际/采集输入","unnorm_key":"libero_object_no_noops"}' \
  --mode packaged --output /实际/新动作报告.json
```

必须在匹配的模型环境、GPU 和缓存配置下运行；上述占位路径需替换。该工厂读取本地权重并保留完整动作，来源字段明确标为推理干预。它不是默认质量推荐。

服务器上的受控对照入口（自动隔离子进程缓存）为：

```bash
/home/wxy/miniconda3/envs/openvla-oft/bin/python -B scripts/server_oft_compare.py \
  --workspace /home/wxy/Rui_work/VLA-Pruner/VLA-Pruner/src/openvla-oft \
  --checkpoint /home/wxy/Rui_work/openvla-7b-oft-libero-object \
  --observation-dir /home/wxy/Rui_work/toolbox-native-check/capture-001 \
  --gpu 2 --prune-ratio 0.5 \
  --output /home/wxy/Rui_work/toolbox-native-check/oft-new-comparison
```

`--observation-dir` 可重复。输出必须是新目录，GPU 需要按实际资源选择。

## 同模型生命周期接口

`methods.oft_choices.OFTChoice` 接收已经审查加载的固定后端 bundle 与 AcceleratorConfig。调用 `select("baseline"|"lightvla"|"vla_pruner")` 切换；它先卸载旧方法，再挂载新方法。`query(observation)` 保留完整动作块，episode 开始时调用 `reset()`，结束使用 `finally: close()`。不允许同时叠加两种方法。

## 闭环协议和判定

同权重闭环使用 `server_oft_compare.py --closed-loop --tasks 0 1 --trials 0 1 2`。每个 episode 新建 Object 环境、相同初始状态、三模式轮换次序、10 步静置、最多 280 步控制，每次执行完整 8 动作块。脚本记录逐次动作、实际裁剪和退出状态；`scripts/audit_oft_choices.py` 负责核对三模式配对。

报告的 passed 仅表示执行和恢复检查，不等于两种方法均达到了质量要求。正式推荐必须依据当前 checkpoint 的闭环证据；不得用更快的固定输入推理掩盖任务失败。
