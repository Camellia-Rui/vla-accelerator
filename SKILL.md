---
name: vla-express-toolbox
description: Inspect a user's VLA model workspace, select compatible acceleration candidates, guide environment and checkpoint setup, integrate LightVLA or the pinned VLA-Pruner backend, and assess reproducible inference evidence. Use when a user adds this toolbox and a VLA project to Codex and asks to accelerate, reproduce, or evaluate its inference.
---

# VLA Express Toolbox

用户目标是给现有 VLA 推理项目提供可执行的接入与评估指导；不局限于 OpenVLA-OFT。先读 [用户指南](docs/USER_GUIDE.md) 和 [发布状态](docs/RELEASE_STATUS.md)。用户只询问方案时不自动启动实验；请求执行时在已有授权内继续。

## 检查与选择

在工具箱根目录使用 `python -m core doctor --workspace <实际目录> --output <独立新报告目录>`，再阅读用户的原生加载及推理入口。检查输出是源码线索，不能作为兼容性证明。明确 checkpoint、processor、归一化统计、动作头、视图数、动作维度与 chunk、decoder 结构、KV cache、依赖版本和已有 patch。

- LightVLA：已验证 Spatial/H20 接入及当前封装 4 任务 20 对闭环，两种模式均 20/20 成功；读 [方法说明](methods/lightvla/README.md)、[服务器结果](docs/LIGHTVLA_SERVER_RESULT.md) 和 [来源](methods/lightvla/provenance.json)。原方法含微调，不标为完全免训练。小规模闭环不等于正式质量验收，历史 500 对记录也不能替代当前版本验收。
- VLA-Pruner：读 [接口说明](methods/vla_pruner/INTEGRATION.md)，必须核实固定官方 backend；普通 PyPI Transformers 不替代该 backend。

已验证的窄范围：OpenVLA-OFT LIBERO-Object、连续 L1 动作头、双视图、8D 本体、8×7 动作块、H20，自定义 Transformers 4.47.0。在 3 个任务各一个初始输入上完成关闭剪枝对照、实际裁剪、卸载恢复。复现入口见 [服务器对比说明](docs/SERVER_PRUNER_PAIR.md)；结果见 [多输入验证](docs/MULTITASK_RESULT.md)。此证据不迁移为任意 checkpoint 支持或闭环质量承诺。
- SpecPrune：已有实验 evaluate、事务式安装/卸载、Object 两任务三个初始状态的闭环验证；尚未完成正式部署验收。读 [当前接入说明](docs/SPECPRUNE_WORKFLOW.md)，旧开发资料不能替代本次结果。
- VLA-Cache：隔离，不提供集成或部署建议。AutoPrune 已退出项目，不恢复旧路线。

无可验证的兼容方法时直接说明缺口，不强行分配方法。环境按所选模型与后端分别配置，不将本工具箱的 CPU 测试环境推广到真实 VLA。权重从用户已有记录或核实的上游取得；检查模型和任务变体及归一化键，不编造下载地址或公开可用性。

同一普通 OFT LIBERO-Object 权重的最新对照见 [双方法验证](docs/OFT_METHOD_CHOICES.md)：原生与 VLA-Pruner 50% 各 6/6，LightVLA 风格推理干预 1/6。不推荐把该 LightVLA 配置作为此权重的稳定选择；它与专用 LightVLA Spatial 训练权重的 20/20 是不同目标。用户明确做实验时可用对应工厂或 OFTChoice 接口，并保留失败结果。

## 执行与报告

使用 [统一运行指南](docs/UNIFIED_RUNTIME.md) 中的 `python -m core run --method lightvla|vla_pruner`。固定 OFT 后端可直接用 VLA-Pruner 内置加载器；其他兼容项目根据原生入口在独立目录编写真实加载工厂，避免修改原权重。VLA-Pruner 输入级对照使用 `python -m core evaluate --method vla_pruner`。保持原生 action API，在 `finally` 卸载并检查恢复。工厂导入会执行用户代码，先阅读其行为。

`report.json` 的 passed/integration_passed 只代表运行或接入检查通过。必须另读 input_screen 和 closed_loop_tested；缺少用户质量门槛时 INCONCLUSIVE，不能自行指定阈值冒充用户验收。输入级报告不是 schema 3 正式验收清单，不能直接提交正式 admission 工具。

LightVLA 专用 Spatial 统一评估使用 `python -m core evaluate --method lightvla`，闭环增加 `--closed-loop`；环境与完整命令见 [LightVLA 使用指南](docs/LIGHTVLA_QUICKSTART.md)。其 report 使用 quality_decision，不套用 VLA-Pruner 的 input_screen 字段或阈值参数。

给用户完整回答：选哪个方法及依据、适用环境、权重来源、按实际路径填好的命令、成功输出、实测效果与可信范围。无法完成的项指出实际阻塞资源。不得把带占位符的命令描述为已能直接运行。

`python -m core demo` 只展示 tiny LLaMA CPU 的 8→5→8 生命周期；不是速度基准或真实 VLA 成功率。真实性标准见 [证据说明](docs/EVIDENCE.md)。需要评估真实模型时读 [评估 Skill](workbench/skill/vla-acceleration-evaluator/SKILL.md) 并使用其 schema 3 校验器；旧根目录 evaluator 是历史版本。

工作目录与其他成员并行时按 [协作规则](docs/USER_GUIDE.md) 回传源码差异、环境、命令、原始记录和校验值，不覆盖他人成果。用户的新指令优先于历史交接包的任务。
