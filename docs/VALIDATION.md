# 本地验证记录：2026-11-02

运行环境：Windows；现有 `.venv/Scripts/python.exe`，Python 3.12.14、torch 2.3.0+cpu、transformers 4.40.1。无 GPU 实验、无真实 VLA 模型加载。

实际执行与结果：

| 检查 | 结果 |
|---|---|
| `python -B -m unittest discover -s tests -v` | 47 项通过：LightVLA 22、VLA-Pruner 21、统一工具箱 4 |
| `python -B workbench/evaluator/common/tests/test_infrastructure.py` | 19 项通过 |
| `python -B workbench/scripts/verify_bundle.py` | 原样迁入的评估组件 186 文件校验通过 |
| `python -B workbench/scripts/validate_handoff.py` | 文档、证据和源码检查通过 |
| `python -B workbench/skill/vla-acceleration-evaluator/scripts/self_test.py` | 严格评估自检通过 |
| skill-creator 的 `quick_validate.py .` | Skill 有效；Windows 需以 `python -X utf8` 运行校验器 |
| `python -m core demo` | 8→5→8，恢复检查通过，原始报告在 `docs/demo/` |
| `python -m core doctor --workspace .` | 已实际完成工具箱自检；不作为用户真实模型兼容证据 |

本记录描述开发环境中的验证，不证明在未测试的模型、硬件、依赖版本下可用。发布包还需在新解压目录运行 `python -B scripts/verify_product.py` 逐文件校验。
