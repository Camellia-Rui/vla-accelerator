# VLA-Cache · 隔离实验

VLA-Cache 尝试复用相邻推理步骤的计算。工具箱保留其历史评估记录，当前状态为 `quarantined`。在历史的 20 对闭环中，原生成功 **14/20**，启用缓存后成功 **0/20**。虽然单次查询延迟中位数的比值约为 1.036×，缓存配置未能保持任务成功率。

运行 `python -m core methods` 可查看隔离状态。当前没有 VLA-Cache 的 `run`、`evaluate` 或自动选择 profile。研究缓存方案时，可从历史配对条件、缓存刷新策略及其对动作的影响开始复核。[证据说明](../../docs/EVIDENCE.md) · [评估记录与重新接入条件](../../workbench/skill/vla-acceleration-evaluator/references/vla-cache.md)
