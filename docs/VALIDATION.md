# 验证记录

## 2026-11-03：当前 LightVLA 封装闭环

- 本地共 66 项测试通过，新增错误/缺失 episode 处理、原子结果保存与配对审核检查。
- 首轮 2 任务 6 对：两模式各 6/6；扩展 4 任务 20 对：两模式各 20/20，无运行错误，配对及恢复检查通过。扩展包含首轮初始状态，不累加为独立样本。
- 原始动作、计时、环境记录及源码清单均已直接下载，结果见 `LIGHTVLA_CLOSED_LOOP.md`。正式质量判定仍为 INCONCLUSIVE。

## 2026-11-03：LightVLA 实际模型补充

- 根目录 unittest 共 62 项通过，包括新增完整动作保存、非有限动作拒绝及历史配对数据审核。
- daisy Spatial 权重在 H20 上通过当前封装 4 对调用、实际序列裁剪和卸载恢复检查；统一 run 入口输出完整 8×7 动作。
- 历史 500 对 episode 从原始 JSONL 重算，两组各 490 次成功；350 对设备标签不同，未接受为正式速度证据。
- 首次 EGL 编号错误及成功重试均归档，详见 [LightVLA 结果](LIGHTVLA_SERVER_RESULT.md)。

## 2026-11-03：统一执行入口

- 根目录 unittest：60 项通过。新增完整动作、异常恢复、重复输出拒绝、缓存隔离、动作形状及非有限值、质量门槛缺失、重复输入 ID 检查。
- H20 上真实 `core run` 和 `core evaluate` 均退出 0；3 输入共 12 对的实际结果见 [统一入口验证](UNIFIED_RUNTIME_RESULT.md)。这验证内置 OFT 加载路径；不代表任意用户工厂或闭环已验收。
- 下方保留历史版本记录。

## rc2 新增验证

- 当前根测试共 53 项通过，包含新增服务器 GPU 选择、动作差异和诊断 hook 清理测试。
- 服务器真实 OFT / VLA-Pruner 验收：3 个任务初始输入、18 对调用均通过接入检查，详见 MULTITASK_RESULT.md。
- SHOWCASE.html 从原始 NPZ/JSONL/report 生成：输入哈希、记录一致性、静态链接、PNG 数据及三个任务的 JS 切换通过检查。
- 内置浏览器无法连接本机预览服务；没有把 DOM 模拟检查当作浏览器视觉验收。离线页面无需该预览服务。
- 下面保留 rc1 的原始测试记录，不能把其“无 GPU 实验”描述扩展到后续 rc2 工作。

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
