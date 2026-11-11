# 选择成功率达标且平均推理延迟最低的配置

最新的 [Object 同模型评分对照](OFT_OBJECT_SCORING.md)包含 50 对留出状态，支持直接读取压缩原始报告。零成功率下降选 VLA-Pruner，允许下降两百分点选同后端 FastV 评分。旧 LightVLA 风格对照仍按其原协议保留。

在工具箱根目录运行。第一次演示可直接复用仓库随附的真实 H20 对照记录，无需模型或 GPU：

```bash
python -B -m core select --report docs/server-evidence/2026-11-04/oft-choice-closedloop-001/report.json --min-success-pct 90 --max-drop-pp 0 --min-pairs 6 --output results/selection-demo
```

在这六个相同的任务初始状态上，基线成功 6/6，平均每次模型查询 826.16 ms。普通 OFT 权重上的 LightVLA 风格干预成功 1/6，381.35 ms，未达到成功率门槛。VLA-Pruner 成功 6/6，710.90 ms，相对基线 1.16×，因此被选中。以上门槛仅是演示示例。

要用自己的模型重新测量，准备 [项目配置](PROJECT_WORKFLOW.md) 和该后端的运行环境，替换以下门槛。每次选择一个新的输出目录。

```bash
python -B -m core select --project project.json --tasks 0 1 --trials-per-task 3 --min-success-pct 90 --max-drop-pp 0 --min-pairs 6 --output results/new-selection
```

| 项目 profile | 同一次评测中的候选 | 计时口径 |
|---|---|---|
| oft_object | 原生、同后端 FastV 评分、VLA-Pruner 语义＋动作评分；可用 `--oft-comparison lightvla-style` 复现旧候选组合 | 闭环模型查询 |
| lightvla_spatial | 专用训练权重关闭剪枝、启用 LightVLA | 闭环模型查询 |
| fastv_openvla_spatial / fastv_oft_spatial | 原生、FastV K=2 剪枝 50% | 闭环单动作 / 8 动作块，剔除各模式开头五次预测 |
| specprune_object | 原生、SpecPrune | 同输入配对的动作块推理；质量来自同初始状态闭环 |

FastV 原始 OpenVLA 一次覆盖十个任务，请省略 `--tasks`。[环境和权重说明](FASTV_RESULTS.md)。SpecPrune 复核模式使用批量根报告及其配套的每对原始报告。各方法的外部源码和权重要求与各自评估入口一致。

11 月 9 日在 H20 上用 `--project` 重新运行了 FastV OFT 的任务 0、初始状态 0。两种模式均成功，门槛设为 1 对、成功率 100%、零下降时选中 FastV。[选择结果](server-evidence/2026-11-09/fastv-selection/selection.json) 和 [原始报告](server-evidence/2026-11-09/fastv-selection/measurement/report.json) 随仓库提供。复核用 `--report` 指向原始报告，保留同目录的 CSV 和 metadata。

`selection.json` 保存所选模式、原始报告路径和摘要哈希、各候选成功数、平均延迟、加速比和筛除理由。从项目启动时原始数据在 `measurement/`。如果只有基线达标则选择 baseline；所有模式均未达标或配对数不足时 selected 为 null。运行失败、记录不完整或配对检查失败时返回非零退出码。结果仅适用于这次配置和样本，成功率门槛按观测值判断。不同权重、后端或独立运行报告的延迟不能在这里混排。
