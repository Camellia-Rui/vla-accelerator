# VLA Acceleration Evaluator Skill

这是本发布包供 Codex 自动发现和执行的 Skill。详细的人类使用说明位于发布包根目录 `README-ZH.md`。

本 Skill 覆盖 VLA 加速模块的源码审计、真实运行时探测、临时接入脚本、严格配对测量、闭环评估、权重审计、调优与准入决策。可复用实现位于 `assets/toolbox/`，需要先通过 `scripts/export_toolbox.py` 导出到实验工作区再修改。

当前发布边界：LightVLA、SpecPrune 与 VLA-Pruner 可以按各自参考文档进入评估流程。AutoPrune 已永久退役，VLA-Cache 已隔离；两者均没有可执行资产，也不得从旧文件、旧 ZIP 或历史描述中恢复。

安装后可显式调用：

```text
$vla-acceleration-evaluator
```

先运行：

```bash
python scripts/self_test.py
```

Skill 的判定脚本只负责确定性检查 manifest；它不能替代真实 GPU、真实观测和闭环证据。

