# VLA Express Toolbox

给 VLA 研究者和机器人开发者使用的 Codex 加速接入工具箱。把本目录和模型工作目录添加给 Codex，说明要加速的模型，它将检查源码、选择候选方法、说明环境与权重需求，并根据实际加载接口生成接入代码和评估方案。

**0.3.0a1 提供统一的实际推理与输入级评估入口。** VLA-Pruner 已通过 H20 / OpenVLA-OFT / LIBERO-Object 的真实运行验证；LightVLA 支持用户加载工厂。闭环任务质量和正式速度验收仍待完成，不能将它称为四种方法均可一键加速的正式版。

主要交付是源码、Codex 指令、运行入口及可追溯证据。最新真实运行结果见 [统一入口验证](docs/UNIFIED_RUNTIME_RESULT.md)。

从 [START_HERE.md](START_HERE.md) 开始。给 Codex 的指令在 [SKILL.md](SKILL.md)。

## 五分钟体验

Python >= 3.10。在工具箱根目录执行，`用户模型目录` 换成已有项目路径：

```bash
python -m core doctor --workspace "用户模型目录" --output "runs/first-check"
python -m core methods
```

检查不需要安装模型依赖，不导入用户代码、不下载权重、不运行 GPU。打开输出目录中的 `report.html` 或 `report.md` 查看方法状态、源码线索和缺口；`report.json` 用于后续协作。只确认权重路径存在，不保证文件齐全或能加载。

LightVLA 合成演示需要独立环境中的 `torch` 和 `transformers`；本机已验证的是 torch 2.3.0+cpu、transformers 4.40.1，不能视为其他 VLA 后端的通用环境：

```bash
python -m pip install -e ".[lightvla]"
python -m core demo --output "runs/first-demo"
```

预期：`sequence_lengths: [8, 5, 8]`、`restored: true`、`passed: true`。这是随机 tiny LLaMA 的 CPU 演示，**没有真实动作、速度或成功率结论**。输出目录必须是新目录。

## 方法支持情况

| 方法 | 现在能做什么 | 尚未完成什么 |
|---|---|---|
| LightVLA | 生命周期封装、真实 CPU 张量测试、用户工厂单次调用、合成演示 | 指定真实 checkpoint 的原生与加速成对测试、闭环证据及来源核验 |
| VLA-Pruner | 固定后端挂载、启停、reset、卸载；OFT/H20 三个真实输入的对照、实际裁剪与恢复验证 | 多初始状态、attention 路径核对、闭环质量和正式性能验收 |
| SpecPrune | 历史实验实现、运行前资源检查和评估公共组件 | 完整统一接入及卸载、真实实验、再分发来源核验 |
| VLA-Cache | 隔离历史证据供分析 | 不属于当前部署路线 |

不是任意 VLA 都支持这些方法。目录关键词只能提供审查线索，必须检查 decoder、视觉 token 布局、动作头、视图数、KV cache 和已有 patch。AutoPrune、GRIDs、OpenPI 不属于当前范围。

## 使用自己的模型

按 [接入指南](docs/USER_GUIDE.md) 先确定模型版本、原生运行入口及 checkpoint；让 Codex 在独立工作目录编写加载工厂。已有的真实调用入口：

```bash
python scripts/run_pruning.py --method lightvla --workspace "用户模型目录" --factory toolbox_entry:build --mode packaged --output "runs/query.json"
```

此命令依赖用户目录中真实存在且已审核的 `toolbox_entry.py:build`，不是开箱即有的文件。统一入口为 `python -m core run --method lightvla` 或 `--method vla_pruner`。VLA-Pruner 还有无需另写工厂的固定 OFT 后端加载器：

```bash
python -m core run --method vla_pruner --help
python -m core evaluate --method vla_pruner --help
```

完整参数、输入采集格式和工厂约定见 [统一运行指南](docs/UNIFIED_RUNTIME.md)。运行保存完整动作块；评估保存原生对照、配对延迟、动作差异、实际裁剪和恢复检查。未提供完整质量门槛时返回 INCONCLUSIVE，不宣称准确率合格。

## 评估、协作与发布

实际证据以 [证据说明](docs/EVIDENCE.md) 为准。严格的 schema 3 校验、成对统计和恢复检查位于 `workbench/`；其中的阶段脚本保留历史环境假设，不能直接当作通用推理命令。根目录旧 `evaluator/` 仅为历史兼容，发布包不包含它。

- [中期展示流程及完成标准](docs/MIDTERM_DEMO.md)
- [用户接入、并行协作与模型来源](docs/USER_GUIDE.md)
- [当前缺口与发布边界](docs/RELEASE_STATUS.md)

```bash
python -B -m unittest discover -s tests -v
python scripts/build_product.py
python -B scripts/verify_product.py "dist/vla-express-toolbox-0.3.0a1"
```

构建包含白名单源码和已归档的小型验收证据，不包含本地虚拟环境、模型权重、训练数据、临时运行目录或 Git。校验文件用于检查发布包完整性，不是第三方签名。正式公开版仍需完成闭环质量、独立用户验收和来源/许可证审查。
