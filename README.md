# VLA Express Toolbox

给 VLA 研究者和机器人开发者使用的 Codex 加速接入工具箱。把本目录和模型工作目录添加给 Codex，说明要加速的模型，它将检查源码、选择候选方法、说明环境与权重需求，并根据实际加载接口生成接入代码和评估方案。

**0.2.0-rc1 是可演示的发布候选版。** 已有统一检查入口、LightVLA 接入/卸载演示与证据评估组件；尚未完成真实 VLA 模型环境验收。不能将它称为四种方法均可一键加速的正式版。

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
| VLA-Pruner | 固定官方后端的已有模型挂载、启停、reset、卸载与契约测试 | 官方模型环境实际前向、机制验证和闭环验收 |
| SpecPrune | 历史实验实现、运行前资源检查和评估公共组件 | 完整统一接入及卸载、真实实验、再分发来源核验 |
| VLA-Cache | 隔离历史证据供分析 | 不属于当前部署路线 |

不是任意 VLA 都支持这些方法。目录关键词只能提供审查线索，必须检查 decoder、视觉 token 布局、动作头、视图数、KV cache 和已有 patch。AutoPrune、GRIDs、OpenPI 不属于当前范围。

## 使用自己的模型

按 [接入指南](docs/USER_GUIDE.md) 先确定模型版本、原生运行入口及 checkpoint；让 Codex 在独立工作目录编写加载工厂。已有的真实调用入口：

```bash
python scripts/run_pruning.py --method lightvla --workspace "用户模型目录" --factory toolbox_entry:build --mode packaged --output "runs/query.json"
```

此命令依赖用户目录中真实存在且已审核的 `toolbox_entry.py:build`，不是开箱即有的文件。VLA-Pruner 使用 [固定后端接口](methods/vla_pruner/INTEGRATION.md)，不使用该 LightVLA 命令。

## 评估、协作与发布

实际证据以 [证据说明](docs/EVIDENCE.md) 为准。严格的 schema 3 校验、成对统计和恢复检查位于 `workbench/`；其中的阶段脚本保留历史环境假设，不能直接当作通用推理命令。根目录旧 `evaluator/` 仅为历史兼容，发布包不包含它。

- [中期展示流程及完成标准](docs/MIDTERM_DEMO.md)
- [用户接入、并行协作与模型来源](docs/USER_GUIDE.md)
- [当前缺口与发布边界](docs/RELEASE_STATUS.md)

```bash
python -B -m unittest discover -s tests -v
python scripts/build_product.py
python -B scripts/verify_product.py "dist/vla-express-toolbox-0.2.0-rc1"
```

构建只包含白名单源码，不包含本地虚拟环境、模型、数据、运行输出、Git 或交接快照。校验文件用于检查发布包完整性，不是第三方签名。开源前仍需完成真实环境验收和来源/许可证审查。
