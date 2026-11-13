# SpecPrune · OpenVLA-OFT / LIBERO-Object

SpecPrune 利用动作预测和跨步历史信息做剪枝。工具箱提供实验性 `evaluate`、配对闭环和批量续跑。已验证的项目 profile 为 `specprune_object`。

## 准备与运行

使用 SpecPrune 的 OpenVLA-OFT 源码与匹配环境，准备 [Object 检查点](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-object)、配套的动作头和本体投影器，以及包含 `observation.npz` 和 `report.json` 的已验证采集目录。闭环还需要 LIBERO Object 资产与初始状态。从工具箱根目录，在该模型环境中运行：

```bash
python -m core evaluate --method specprune --workspace "$WORKSPACE" --checkpoint "$CHECKPOINT" --observation "$OBS" --gpu "$GPU" --closed-loop --task 0 --trial 0 --output runs/specprune-001
```

将变量设为实际路径和 GPU 编号，输出目录须为新目录。先检查原生与保留全部 token 的动作是否一致、是否观察到剪枝，以及卸载后的恢复检查。再比较同一初始状态的两份 episode 记录；策略调用中只有 `execute_action=true` 的动作实际送入环境。[环境、批量命令和续跑](../../docs/SPECPRUNE_WORKFLOW.md)

## 已验证范围

当前封装已完成 **10 个 Object 任务、50 对初始状态**的配对闭环，基线与 SpecPrune 均成功 **49/50（98%）**。平均查询策略耗时 275.47 → 210.03 ms，降低 **23.75%**；平均每回合策略耗时 4900.44 → 4301.93 ms，降低 **12.21%**。这是共享 H20、单 seed 下的观测结果，不保证普遍无损加速。[原始记录、区间和展示命令](../../docs/SPECPRUNE_DISPLAY.md)

项目选择器使用批量闭环结果筛查成功率，再用同输入动作块配对计时比较速度；本轮在示例质量门槛下已选出 SpecPrune。固定输入计时与闭环查询耗时分开报告。[选择口径](../../docs/METHOD_SELECTION.md) · [来源记录](provenance.json)

早期独立实验的性能表、Object/Goal 权重清单及开发接口保存在[历史实验记录](HISTORICAL_EXPERIMENTS.md)。其中的 Goal 和大样本结果应按原工程条件引用。

历史 3 对烟雾验证用于追踪接入过程，不与本轮 50 对混合计算成功率。

## 运行与评估一次完成

使用项目配置启动 `python -m core select --project project.json --output runs/new-evaluation`，即可在运行中检查记录、结束后自动生成评估报告，不需要第二条评估指令。未给质量门槛时仅报告证据。检查时机、计时口径和本版验证范围见 [运行中评估](../../docs/LIVE_EVALUATION.md)。
