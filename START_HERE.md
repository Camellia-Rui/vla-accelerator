# 开始使用

这是统一工具箱入口；`workbench/` 是评估组件和历史交接资料，旧文档中的“当前任务”不会自动成为用户任务。

成品入口是 `python -m core doctor`、`python -m core run` 和 `python -m core evaluate`。可执行测试步骤见 [README_TESTING.md](README_TESTING.md)，接口见 `docs/UNIFIED_RUNTIME.md`。OFT Object 与专用 LightVLA Spatial 已有小规模闭环记录；正式质量与性能验收仍未完成，证据见 `docs/EVIDENCE.md`。

1. 读根目录 `README.md` 和 `SKILL.md`，确认本次目标与方法状态。
2. 在工具箱根目录运行 `python -m core doctor --workspace "实际用户目录" --output "runs/新检查目录"`。它只读检查源码，不启动模型或 GPU。
3. 读 `docs/USER_GUIDE.md`。根据用户真实项目和原生加载入口选择方法，明确环境、checkpoint、输入布局与阻塞项；需要运行时沿用用户已授权范围。
4. 查看 `docs/EVIDENCE.md`，区分合成测试、原始证据和历史结论。评估真实效果时再读 `workbench/skill/vla-acceleration-evaluator/SKILL.md`。
5. 展示成品按 `docs/MIDTERM_DEMO.md`；多人协作采用独立工作目录和输出目录，见用户指南。

用户要求“实际测试并评估”时，资源就绪后优先执行 `core select --project 项目.json --output 新目录`，同一次命令自动运行闭环、审核证据并输出 assessment.json/assessment.md 与 selection.json；无需再等用户发“请评估”。可同时传入完整质量门槛；未提供门槛也可运行并报告证据，但不自动判定达标。OFT、LightVLA、FastV 在回合结束时检查记录；SpecPrune 子任务按回合检查、批量入口按配对汇总。运行状态与限制见 [运行中评估](docs/LIVE_EVALUATION.md)。

只有工具箱、尚无模型源码或环境的初次使用者，按 [OpenVLA-OFT 从零准备与录制](docs/FIRST_USE_OFT_DEMO.md) 检查资源并建立独立目录；资源未就绪时不得把随包实验写成新模型实测。

收到完整发布包时先运行 `python -B scripts/verify_product.py`。源码 checkout 尚未生成 `product-manifest.json`，不运行此发布包校验，改为测试后构建候选包。
