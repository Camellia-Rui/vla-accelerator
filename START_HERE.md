# 开始使用

这是统一工具箱入口；`workbench/` 是评估组件和历史交接资料，旧文档中的“当前任务”不会自动成为用户任务。

成品入口是 `python -m core doctor`、`python -m core run` 和 `python -m core evaluate`。可执行测试步骤见 [README_TESTING.md](README_TESTING.md)，接口见 `docs/UNIFIED_RUNTIME.md`。OFT Object 与专用 LightVLA Spatial 已有小规模闭环记录；正式质量与性能验收仍未完成，证据见 `docs/EVIDENCE.md`。

1. 读根目录 `README.md` 和 `SKILL.md`，确认本次目标与方法状态。
2. 在工具箱根目录运行 `python -m core doctor --workspace "实际用户目录" --output "runs/新检查目录"`。它只读检查源码，不启动模型或 GPU。
3. 读 `docs/USER_GUIDE.md`。根据用户真实项目和原生加载入口选择方法，明确环境、checkpoint、输入布局与阻塞项；需要运行时沿用用户已授权范围。
4. 查看 `docs/EVIDENCE.md`，区分合成测试、原始证据和历史结论。评估真实效果时再读 `workbench/skill/vla-acceleration-evaluator/SKILL.md`。
5. 展示成品按 `docs/MIDTERM_DEMO.md`；多人协作采用独立工作目录和输出目录，见用户指南。

收到完整发布包时先运行 `python -B scripts/verify_product.py`。源码 checkout 尚未生成 `product-manifest.json`，不运行此发布包校验，改为测试后构建候选包。
