# VLA Express Toolbox

面向 VLA 研究者、机器人开发者和科研团队的 **Codex 推理加速工具箱**。将工具箱和模型工作目录交给 Codex，可以检查模型资源、筛选兼容方法、配置运行入口，并执行推理对照、闭环测试和结果审查。

**当前版本：0.3.0a4（研究预览版）**。已提供真实模型运行代码和可追溯实验记录；支持范围以具体模型、权重和后端为准，不承诺任意 VLA 模型自动加速或无损精度。

[测试说明](README_TESTING.md) · [Codex 入口](START_HERE.md) · [使用与协作](docs/USER_GUIDE.md) · [发布状态](docs/RELEASE_STATUS.md) · [实验依据](docs/EVIDENCE.md)

## 工具箱能做什么

- 检查用户模型工作目录，识别源码、权重和推理资源线索。
- 按模型变体、动作头、视觉输入和后端选择兼容候选方法。
- 使用真实模型加载与动作接口，执行加速、关闭加速和卸载恢复。
- 保存完整动作、配对计时、实际裁剪和恢复检查记录。
- 区分“代码跑通”“输入筛查通过”和“闭环质量达标”，保留失败证据。

项目主要交付源码、Codex 指令、运行入口、评估组件和文档。当前不提供通用训练服务、任意模型自动转换或机器人硬件驱动；LightVLA 原方法包含微调，不能把本工具箱称为所有方法均免训练。

## 当前支持范围

| 模型与方法 | 已验证能力 | 使用边界 |
|---|---|---|
| OpenVLA-OFT / LIBERO-Object + VLA-Pruner | H20 上真实推理、生命周期恢复、两任务六对闭环；原生与 50% 剪枝各 6/6 | 固定官方 OFT 后端、双视图、连续 L1 动作头、8D 本体、8×7 动作块；更广质量与正式性能验收待完成 |
| LightVLA 专用 LIBERO-Spatial 权重 + LightVLA | 统一推理及固定输入/闭环评估；四任务二十对闭环，两模式各 20/20 | 需要专用训练权重与已检查的后端；不是任意 OFT 权重的通用插件 |
| 普通 OFT Object 权重 + LightVLA 风格推理干预 | 已实现实验接口并保留失败结果，闭环仅 1/6 | 当前不推荐该组合 |
| 原始 OpenVLA | 资源与接口接入工作尚未完成 | 不能套用 OFT 支持结论 |
| SpecPrune / VLA-Cache | 前者保留实验资料，后者隔离历史证据 | 不属于当前统一可用方法 |

小规模闭环结果不是准确率保证；共享 GPU 的观测延迟不是稳定加速承诺。最新依据见 [OFT 同权重对照](docs/OFT_METHOD_CHOICES.md)、[LightVLA 闭环](docs/LIGHTVLA_CLOSED_LOOP.md) 和 [LightVLA 统一入口回归](docs/LIGHTVLA_UNIFIED_RESULT.md)。

## 开始使用

### 1. 获取源码

```bash
git clone https://github.com/Camellia-Rui/vla-accelerator.git
cd vla-accelerator
```

目录检查只需要 Python 3.10 或以上，不需要模型依赖或 GPU。在仓库根目录即可运行，无需先安装工具箱：

```bash
python -m core methods
python -m core doctor --workspace /absolute/path/to/model-project --output runs/first-check
```

将模型路径替换为实际目录，输出目录必须是新目录。检查结果在 report.json、report.md 和 report.html 中。doctor 不加载模型，发现文件不代表模型已兼容。

### 2. 将目录添加给 Codex

同时添加工具箱和自己的模型工作目录，可以直接使用：

> 请读取工具箱 START_HERE.md 和 README_TESTING.md。检查我的模型源码、checkpoint、processor、动作头、归一化统计和推理环境，推荐有证据支持的加速方法。先说明缺口，再按我的授权执行原生推理、加速对照和恢复检查。结果写入独立新目录，不修改原权重；报告实测效果和可信范围。

没有可验证的兼容方法时，应报告缺口，而不是强行套用方法。无需把大型权重上传聊天，提供可访问的模型目录即可。

### 3. 在模型环境中测试

真实推理使用模型对应的 Python 环境。VLA-Pruner OFT 和 LightVLA 的定制 Transformers 不能当作同一套环境混装；不要使用工具箱 CPU 测试依赖覆盖已有模型环境。

```bash
python -m core run --method vla_pruner --help
python -m core evaluate --method vla_pruner --help
python -m core run --method lightvla --help
python -m core evaluate --method lightvla --help
python -m core evaluate --method lightvla --closed-loop --help
```

实际可执行服务器命令、输入准备与结果判据见 **[README_TESTING.md](README_TESTING.md)**。外部用户必须替换为自己的路径并先核对后端。

## 模型与环境准备

工具箱不包含模型权重、训练数据或 Python 虚拟环境。真实测试至少需要：

- 模型源码及可追溯版本，原生推理入口。
- 完整 checkpoint、processor/tokenizer、动作归一化统计。
- 模型需要的动作头、本体投影器等组件。
- 对应环境、可用 GPU；闭环测试还需要匹配任务的 LIBERO 资源。
- 与模型一致的真实 observation、任务指令和明确的质量要求。

权重来源应按 [方法来源记录](methods/lightvla/provenance.json)、[VLA-Pruner 接入说明](methods/vla_pruner/INTEGRATION.md) 和具体上游模型说明核实。基础模型权重不能代替任务微调权重，归一化键相同也不能单独证明模型适配。

如需将工具箱安装到已有模型环境，可使用 `python -m pip install -e . --no-deps`；在仓库根目录执行命令则不必安装。工具箱基础包无强制模型依赖。

## 如何判断效果可信

1. **运行检查**：真实模型输出合法动作，日志无运行错误。
2. **接入检查**：关闭剪枝对照一致，实际计算发生裁剪，卸载后模块、参数抽样与动作恢复。
3. **输入对照**：相同输入、预热、同步计时、交替顺序，保留每次完整动作。
4. **闭环评估**：同任务、同初始状态与 seed，统计成功率、错误和轨迹。
5. **质量验收**：按照用户允许的质量下降和统计要求做判断。

report 的 passed 只表示对应执行/接入检查通过。INCONCLUSIVE 表示证据或质量门槛不足，不等于失败，也不等于允许部署。输入动作 MAE 不能替代机器人任务成功率。

## 项目结构

| 路径 | 内容 |
|---|---|
| core/ | 方法目录、工作目录检查与统一运行入口 |
| methods/lightvla/ | 可恢复 LightVLA 适配器、工厂、固定输入与闭环评估 |
| methods/vla_pruner/ | 固定 OFT 后端适配、加载与输入级评估 |
| scripts/ | 服务器启动、证据审核、构建和校验 |
| docs/ | 使用说明、结果与原始证据 |
| tests/ | 本地配置、生命周期与协议测试 |
| workbench/ | 深入评估 Skill、严格验收组件及历史资料 |

多人协作使用独立代码目录和输出目录，不覆盖其他人的模型或实验。回传代码版本、环境、命令、原始结果和校验值，详见 [协作说明](docs/USER_GUIDE.md)。

## 测试与发行

本地已验证环境：Python 3.12、torch 2.3.0+cpu、transformers 4.40.1；73 项测试通过。它只用于代码测试，不是 GPU 模型环境规范。

```bash
python -B -m unittest discover -s tests -v
python scripts/build_product.py
python -B scripts/verify_product.py dist/vla-express-toolbox-0.3.0a4
```

构建脚本拒绝覆盖同版本包。发布包使用逐文件哈希检查完整性；源码 checkout 没有 product-manifest.json，不能直接进行发布包校验。ZIP 不包含权重、虚拟环境和临时运行结果。

## 来源与许可

本仓库许可证见 [LICENSE](LICENSE)。第三方源码、模型和数据各自的许可独立适用，根许可证不覆盖它们。LightVLA 继承实现尚未完成与官方逐字快照的对应核验，相关来源核验与独立用户验收仍列在 [发布状态](docs/RELEASE_STATUS.md)。当前以研究预览版发布，不将尚未完成的正式验收描述为已完成。
