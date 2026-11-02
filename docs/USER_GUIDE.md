# 用户接入与协作

## 面向谁

主要面向已经有 VLA 模型代码、模型资源和推理需求的研究生、本科研究团队及机器人开发者。工具箱帮助用户检查和接入加速实现，并组织可信的评估；模型训练、任意模型自动转换和机器人硬件驱动不是当前成品能力。

用户给 Codex 添加工具箱目录和自己的模型目录，可以这样说：

> 请读取工具箱 START_HERE.md，检查我的模型工作目录。我想加速现有 VLA 推理。先告诉我兼容方法、环境、权重和缺口，再按我的授权完成接入和测试。

无需安装 Skill 也可以显式请 Codex 阅读入口；是否自动发现 Skill 取决于用户的 Codex 配置。先用根目录 `python -m core doctor` 检查，报告不会自动加载模型。目录名字中出现 OpenVLA 不等于支持任意 OpenVLA 变体。

## 环境、权重和配置

检查工具只需 Python >= 3.10。模型运行用原项目的独立环境，在该环境安装工具箱基础包。LightVLA 当前 CPU 测试环境和可选依赖见总 README；VLA-Pruner 的自定义 Transformers 不能与普通 PyPI Transformers 混装代替。

用户可以在模型目录新建 `toolbox-project.json`，配置已知信息，例如：

```json
{
  "model_family": "用户确认的模型变体",
  "checkpoint": "weights/my-checkpoint",
  "factory": "toolbox_entry:build"
}
```

这只是发现元数据，不会触发下载或工厂调用。相对 checkpoint 路径相对模型目录；模型、factory 等文字是用户声明，仍需审核源码和真实加载测试。没有此文件也能检查。不要在配置中填写令牌或密码。

工具箱不捆绑模型权重。LightVLA 来源记录在 `methods/lightvla/provenance.json`，其中 checkpoint 示例是历史记录，尚未核实当前下载可用性，不能当作已验证下载入口。VLA-Pruner 的固定官方仓库及 commit 见 `methods/vla_pruner/backend.lock.json` 与 `INTEGRATION.md`，checkpoint、processor、动作头及归一化统计应匹配该后端的任务变体。需要下载时 Codex 应核实所选上游版本的说明与真实权重地址。

真实模型运行至少要知道：模型源码位置和版本、checkpoint 及必要附属文件、Python 解释器、设备、预处理函数、动作头、视图布局、动作维度与 chunk、推理函数及一份合法输入。闭环评估还需要 LIBERO 或实际任务环境。缺少信息时先列出缺口；单纯封装和演示不需要先跑训练数据或真机。

## LightVLA 真实调用的工厂契约

由 Codex 根据用户项目的原生代码编写 `toolbox_entry.py`。`build(**kwargs)` 返回以下对象之一：

- `{"model": model.eval(), "inputs": prepared_inputs_dict}`：CLI 调用原 `predict_action`。
- `{"model": model.eval(), "run": zero_argument_callable}`：闭包捕获 processor、动作头等资源，内部使用模型原生接口；必须真实执行模型，不能返回缓存结果。

不提供通用假 loader。工厂模块会被导入并执行，执行前要阅读加载代码及设备选择。`--factory-kwargs` 是 JSON 对象，路径按工厂自己的约定处理；CLI 的相对路径沿用当前工作目录。命令例子见根 README，必须先创建真实工厂再运行。

成功输出包含调用类型、输出形状、剪枝统计和 `detached: true`。该 JSON 是单次接入摘要，不是原始动作或性能证据。完整动作、配对延迟及环境记录需在评估脚本另行保存。CLI `finally` 尝试卸载，卸载失败应排查，不能继续宣称恢复成功。

## 和队友并行工作

复用自己的 checkout；需要隔离时给每位成员独立工作目录，例如 `成员A-lightvla`、`成员B-specprune`，每项实验用新的输出目录。共享 checkpoint 仅只读，禁止直接改权重、其他成员工作目录或已发布快照。不同自定义后端使用不同虚拟环境。

开始前记录源码 commit、依赖版本、方法配置和负责范围。回传时提交源文件差异、运行命令及工作目录、解释器/设备、完整日志、原始 JSON、失败信息、结果摘要和 SHA256。不要只回传截图或手填加速倍数。主分支合并前复核接口、来源和证据，不能把别人历史“当前任务”当成新指令。

当前本地发布候选没有附带服务器登录配置、真实 VLA checkpoint 或仿真资产。服务器工作需要用户提供实际可用的路径/访问方式；不假设包内示例 `/cloud/...` 在用户机器存在。
