# VLA Express Toolbox

**0.3.0a7 · 研究预览版**。面向已有 VLA 模型工程、希望降低推理延迟的研究者和机器人开发者。把工具箱和模型工作目录添加给 Codex，它可以检查模型资源、运行原生与加速对照，并按你的任务成功率要求选择本次实测延迟最低的配置。权重保存在用户可访问的目录即可。

## 按模型选择方法

| 你的模型与后端 | 方法 README | 已有 LIBERO 闭环证据（基线 → 加速） |
|---|---|---|
| 原始 OpenVLA 或 OpenVLA-OFT，Spatial 权重，对应的 FastV 工程 | [FastV](methods/fastv/README.md) | 原始模型 421/500 → 434/500；OFT 492/500 → 497/500 |
| OpenVLA-OFT，Object 权重，固定的 VLA-Pruner 后端 | [VLA-Pruner](methods/vla_pruner/README.md) | 两个任务六对初始状态，50% 剪枝时 6/6 → 6/6 |
| LightVLA 专用训练权重，Spatial 后端 | [LightVLA](methods/lightvla/README.md) | 四个任务二十对初始状态，20/20 → 20/20 |
| OpenVLA-OFT，Object 权重，SpecPrune 工程 | [SpecPrune](methods/specprune/README.md) | 两个任务三个初始状态，3/3 → 3/3；提供实验评估入口 |
| 查看历史缓存实验 | [VLA-Cache](methods/vla_cache/README.md) | 历史对照 14/20 → 0/20；已从可选方案中隔离 |

每行结果只适用于对应的权重、后端和任务。比如在普通 OFT Object 权重上，LightVLA 风格的推理干预仅成功 1/6，与专用 Spatial 权重的 20/20 属于不同实验。[同权重对照](docs/OFT_METHOD_CHOICES.md)

## 第一次使用

在工具箱根目录执行命令。源码目录检查需要 Python 3.10 或更高版本；真实评测使用对应方法的模型环境、GPU、完整权重和 LIBERO 资产。根据方法 README 准备相机输入、动作头、本体投影器、归一化统计或采集的 observation。

```bash
git clone https://github.com/Camellia-Rui/vla-accelerator.git
cd vla-accelerator
python -m core methods
python -m core doctor --workspace /absolute/path/to/model-project --output runs/first-check
```

把示例路径换成自己的模型工作目录，输出写入新目录。打开 `runs/first-check/report.md` 查看候选方法和缺失资源。doctor 根据文件发现线索，兼容性还需通过模型加载和运行验证。

把两个目录交给 Codex 后，可以这样说：

> 请读工具箱 START_HERE.md，检查我的模型工程、权重和运行环境。告诉我哪些方法兼容、还缺什么；资源齐全后，按我给出的最低成功率、允许下降的百分点和最低配对数运行比较，把选择结果和原始记录写到新目录。

## 运行评估与自动选择

只测试一种方法时，从上表的 README 进入，按对应的 `core evaluate --method …` 命令操作。要重新比较项目支持的候选方案，先生成配置，填写实际的 profile、模型源码、权重、Python 解释器和输出路径，然后检查：

```bash
python -m core project init --project project.json
# 编辑 project.json 后执行
python -m core project check --project project.json --probe-env
```

也可以先用随仓库交付的真实记录体验选择流程，无需 GPU。以下门槛是示例：最低成功率 90%、相对基线最多下降 0 个百分点、至少六对初始状态。

```bash
python -m core select --report docs/server-evidence/2026-11-04/oft-choice-closedloop-001/report.json --min-success-pct 90 --max-drop-pp 0 --min-pairs 6 --output runs/selection-demo
```

打开 `runs/selection-demo/selection.json`，预期选择 `vla_pruner`：六对中基线与 VLA-Pruner 都成功六次，平均查询延迟从 826.16 ms 降到 710.90 ms。更快的 LightVLA 风格干预因成功率过低被排除。换成自己的工程时，使用 `--project project.json` 启动新实验。不同后端支持的比较范围、任务参数和结果字段见[选择指南](docs/METHOD_SELECTION.md)。

## 验证与交付

- **模型接入**：按方法检查真实动作、剪枝是否发生、闭环成功数和原始记录。报告里的 `passed` 只表示对应检查通过。按观测成功率选出的配置仍需在目标任务和设备上验证。[测试手册](README_TESTING.md) · [证据索引](docs/EVIDENCE.md)
- **Codex 与协作**：每人使用独立工作和输出目录，回传模型版本、执行命令及原始报告。[Codex 入口](START_HERE.md) · [项目配置](docs/PROJECT_WORKFLOW.md) · [用户指南](docs/USER_GUIDE.md)
- **源码测试与发行**：运行 `python -B -m unittest discover -s tests -v`。用 `python scripts/build_product.py` 构建新包，再对解压目录执行 `python -B scripts/verify_product.py <目录>`。发布包包含源码、文档和小型证据，模型权重与环境另行准备。[发行状态](docs/RELEASE_STATUS.md)
- **来源与许可**：工具箱许可证及第三方源码、权重的来源见 [LICENSE](LICENSE) 与各方法 README。
