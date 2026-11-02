---
name: vla-express-toolbox
description: Inspect a user's VLA model workspace, select compatible acceleration candidates, guide environment and checkpoint setup, integrate LightVLA or the pinned VLA-Pruner backend, and assess reproducible inference evidence. Use when a user adds this toolbox and a VLA project to Codex and asks to accelerate, reproduce, or evaluate its inference.
---

# VLA Express Toolbox

用户目标是给现有 VLA 推理项目提供可执行的接入与评估指导；不局限于 OpenVLA-OFT。先读 [用户指南](docs/USER_GUIDE.md) 和 [发布状态](docs/RELEASE_STATUS.md)。用户只询问方案时不自动启动实验；请求执行时在已有授权内继续。

## 检查与选择

在工具箱根目录使用 `python -m core doctor --workspace <实际目录> --output <独立新报告目录>`，再阅读用户的原生加载及推理入口。检查输出是源码线索，不能作为兼容性证明。明确 checkpoint、processor、归一化统计、动作头、视图数、动作维度与 chunk、decoder 结构、KV cache、依赖版本和已有 patch。

- LightVLA：有 CPU 验证的封装；读 [方法说明](methods/lightvla/README.md) 和 [来源](methods/lightvla/provenance.json)。原方法含微调，不标为完全免训练。真实模型兼容和效果待验证。
- VLA-Pruner：读 [接口说明](methods/vla_pruner/INTEGRATION.md)，必须核实固定官方 backend；普通 PyPI Transformers 不替代该 backend。
- SpecPrune：实验路线，不承诺完整统一接入；读 [开发评估说明](workbench/evaluator/specprune/README-ZH.md)。
- VLA-Cache：隔离，不提供集成或部署建议。AutoPrune 已退出项目，不恢复旧路线。

无可验证的兼容方法时直接说明缺口，不强行分配方法。环境按所选模型与后端分别配置，不将本工具箱的 CPU 测试环境推广到真实 VLA。权重从用户已有记录或核实的上游取得；检查模型和任务变体及归一化键，不编造下载地址或公开可用性。

## 执行与报告

根据用户原项目在独立目录编写真实加载工厂，避免修改原权重。LightVLA 使用 `scripts/run_pruning.py --method lightvla` 或方法 CLI；已有模型按其生命周期接口接入。VLA-Pruner 使用固定后端接口。保持原生 action API，在 `finally` 卸载并检查恢复。工厂导入会执行用户代码，先阅读其行为。

给用户完整回答：选哪个方法及依据、适用环境、权重来源、按实际路径填好的命令、成功输出、实测效果与可信范围。无法完成的项指出实际阻塞资源。不得把带占位符的命令描述为已能直接运行。

`python -m core demo` 只展示 tiny LLaMA CPU 的 8→5→8 生命周期；不是速度基准或真实 VLA 成功率。真实性标准见 [证据说明](docs/EVIDENCE.md)。需要评估真实模型时读 [评估 Skill](workbench/skill/vla-acceleration-evaluator/SKILL.md) 并使用其 schema 3 校验器；旧根目录 evaluator 是历史版本。

工作目录与其他成员并行时按 [协作规则](docs/USER_GUIDE.md) 回传源码差异、环境、命令、原始记录和校验值，不覆盖他人成果。用户的新指令优先于历史交接包的任务。
