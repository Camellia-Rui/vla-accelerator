# 证据与可信范围

## 当前证据

| 内容 | 来源 | 能支持的结论 |
|---|---|---|
| 根 tests 中 LightVLA CPU 模型测试及 VLA-Pruner 契约测试 | 本仓库源码、实际测试输出 | 封装配置、生命周期及对应测试范围；不证明真实 VLA 性能 |
| `python -m core demo` 的 report.json | 当前执行生成，标记 synthetic | 随机模型发生 token 裁剪，卸载后参数与原生输出恢复；不证明速度或控制效果 |
| LightVLA 既有实验说明 | `methods/lightvla/EXPERIMENTS.md`、`workbench/EVIDENCE_INDEX.json` | 历史结论；LightVLA 关键阶段原始记录尚未随包提供，不能据此重新裁定 PASS |
| VLA-Pruner 既有 README/复现说明 | 方法目录和历史资料 | 既有复现工作记录；当前真实后端尚未在本环境验收 |
| VLA-Cache 历史 JSON 和日志 | `workbench/archive/excluded-vla-cache-evidence`，哈希见证据索引 | 特定旧设置的失败证据，不是当前可部署方法 |

VLA-Cache 历史闭环样本中 native 为 14/20、cache 为 0/20，尽管查询延迟中位数有约 1.036 倍的比值，也不能宣称方法可用。后续历史探测没有找到满足条件的 refresh 候选。这些是历史数据解释，不是本次重新实验。

## 真实模型应补什么

1. 固定源码、checkpoint、环境、输入哈希及配置；确认模型原生动作前向正常。
2. 同一输入的原生、加速和零效果对照；实际 decoder 输入形状、保留 token/复用数量，不能只看日志里期望参数。
3. 完整 query/action 延迟和端到端 wall 延迟，记录 warmup、设备同步、样本数与配对 bootstrap 区间；不把局部 prefill 提速当端到端提速。
4. 动作误差、gripper 一致性、任务切换/reset，以及卸载后原生数值与参数恢复。
5. 需要控制质量结论时，补闭环任务、初始状态与 seed 成对结果，以及样本不确定性。真机结论另需真实机器人实验。

严格评估入口：

```bash
python -B workbench/skill/vla-acceleration-evaluator/scripts/validate_manifest.py 实际manifest.json
python -B scripts/compare_methods.py 实际manifest.json --pretty
```

两条命令消费已有 schema 3 证据，不执行模型、不凭空计算缺失数据。schema 模板和字段语义以评估 Skill 为准；原始路径相对 manifest 文件所在目录。校验通过只是格式及已定义门槛通过，仍需核对原始记录确实支撑摘要；manifest 中的指标不等于重新计算出的独立证明。材料缺失应标记 INCONCLUSIVE，不补造结果。

详细原始文件哈希见 `workbench/EVIDENCE_INDEX.json`。`workbench/evaluator/common/` 提供成对匹配、统计、原子记录、hook 形状跟踪和采样参数检查；公共组件完成不等于每个方法完整 evaluator 已完成。
