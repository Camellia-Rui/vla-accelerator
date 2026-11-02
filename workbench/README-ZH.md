# VLA Codex Workbench r6 — 评估基础设施与只读预检开发包

本轮开发成果见 [开发进度](DEVELOPMENT.md)。新增两方法配置与预检、公共统计/记录/审计组件。
完整真实评估链尚未完成；这些组件不是在目标 VLA 上的实测证据。

本版基于 r4，补齐状态、证据索引、协作约定和严格校验。新 Codex 先只读盘点；
SpecPrune/VLA-Pruner 真实五阶段评估器仍是待办，不是本包已有功能。上传后可发送：

```text
请执行附件中的 START_HERE.md。
```

Codex 将校验并读取交接资料，完成只读环境探测；不会自动开始开发或实验。后续开发仍依赖
服务器上可访问的目标 VLA 源码、方法上游源码/固定后端、checkpoint、LIBERO 资产、GPU 和
对应 Python 环境。缺失材料应先列出，不能把脚手架或历史成绩标成真实验收。

先读 [项目状态](PROJECT_STATUS.md)、[证据索引](EVIDENCE_INDEX.json) 和
[协作约定](COLLABORATION.md)。检查工具：

```bash
python -B scripts/verify_bundle.py
python -B scripts/validate_handoff.py
python -B scripts/discover_environment.py
python -B skill/vla-acceleration-evaluator/scripts/self_test.py
```

前三项不加载模型，最后一项只使用合成测试 fixture 检查门控；都不启动 GPU 计算。

这是一个给“全新 Codex 页面”使用的自包含交接包。所有初始化对话、项目指示、信息收集模板和实验交付格式均在 ZIP 内。新 Codex 先读取 `START_HERE.md`，再以 `CODEX_INSTRUCTIONS.md` 为项目权威入口，不依赖旧聊天。

## 结论先行

本地项目中，单帧真实观测评估器的最高版本是 `lightvla-updated-evaluator-2026-10-26-r6`。但完整最新版流程不止 r6，还包括后续的闭环 smoke、多任务配对、定向复核和权重数值审计。本包将这五个阶段统一收录，并把脚本改为可从当前目录运行、允许用环境变量覆盖云端根目录。

“最新版”在这里指本地项目截至 2026-10-31 的最高修订和完整验收链；不是对互联网仓库未来版本的声明。

VLA-Cache 当前不属于可接入实现：它的内部计算减少和固定观测约 1.025x 加速成立，但测试过的跨帧复用在闭环中没有速度/正确性交集。包内不携带 VLA-Cache overlay，只在归档区保留 09-13 原始结果用于追溯。

AutoPrune 已按用户决定永久归档。r2 已从 Skill 和工具箱删除其源码、注册器、OpenVLA prior bridge、示例与专项测试；新 Codex 不得从旧目录、旧 ZIP、历史文本或同义命名中恢复它。

## 包内容

```text
skill/vla-acceleration-evaluator/       可安装 Codex Skill
evaluator/lightvla/                     五阶段最新版 LightVLA 验收链
templates/remote-experiment/            后台、PID、防重入、状态脚本模板
START_HERE.md                           自动初始化入口
ONE_LINE_TO_SEND.txt                    上传后唯一需要发送的一句话
conversation/                           全部初始化/收集/附件/实验对话协议
CODEX_INSTRUCTIONS.md                   新 Codex 必须遵守的完整项目指示
history/                                原始项目交接摘要
archive/excluded-vla-cache-evidence/    隔离的 VLA-Cache 失败证据
scripts/verify_bundle.py                发布包完整性校验
NEW_CODEX_PROMPT.md                     新页面首条消息模板
RELEASE_NOTES.md                        版本与边界
SHA256SUMS.txt                          文件哈希清单
```

## 在新的 Codex 页面使用

1. 把整个 ZIP 附加到新页面。
2. 只发送：`请执行附件中的 START_HERE.md。`

之后由新 Codex 自行解压、校验、读取规范、确认边界并回复初始化回执。用户不需要粘贴长提示词，也不需要再次介绍历史。若同一条消息还附带结果或源码，新 Codex 会在初始化后直接路由处理；只有真正缺少关键输入时，才会一次性询问全部缺项。

## 推荐的新页面工作顺序

### 1. 审计与接入定位

记录源仓库 commit/dirty state、依赖版本、checkpoint 身份、官方推理入口、动作归一化、真实 decoder/vision/projector/cache 类和张量形状。先跑一个不带加速的真实输入。

### 2. 编写临时脚本

临时脚本写到新的实验目录，不写入 Skill、checkpoint 或原仓库。优先用可恢复 wrapper/hook；必须有 baseline、可行时的 retain-all control、accelerated 路径、`finally` 清理、原始逐样本 JSON 和环境记录。

### 3. 固定观测筛选

相同真实观测、交替顺序、所有模式预热、CUDA 同步、完整 action、夹爪一致率、真实裁剪/跳过计数、显存、wall time 和配对置信区间。固定观测只决定是否值得进入闭环。

### 4. 闭环验证

每 episode 新环境；同任务、初态和 seed 配对；模式顺序交替；有状态方法每 episode/reset/异常后清状态；逐 episode 原子写结果。先小规模 smoke，通过后再多任务扩大。

### 5. 权重与准入

包装启用时按原始参数对象比较数值；卸载后检查参数名称和拓扑恢复。最终 manifest 交给 `validate_manifest.py` 和 `admission_decision.py`，得到 PASS/TUNE/REJECT/INCONCLUSIVE。只有 PASS 才编写长期部署 adapter。

## LightVLA 最新评估链

| 阶段 | 目录 | 用途 | 何时运行 |
|---|---|---|---|
| 01 | `01-fixed-observation-r6` | 真实单观测、官方配置、结构/动作/速度筛选 | 首先 |
| 02 | `02-closed-loop-smoke-r1` | 任务 0/1 小规模闭环 | 01 通过后 |
| 03 | `03-multitask-paired-r1` | 任务 0-3 多任务配对 | 02 PASS 后 |
| 04 | `04-adjudication-r1` | 指定单一逐任务失败的定向复核 | 只有脚本规定的失败形态 |
| 05 | `05-weight-audit-r1` | 活跃 wrapper 下数值审计与 detach 恢复 | 最终判定前 |

每个阶段都保留原始 README。新 Codex应读取阶段依赖，不得跳过前置门槛，也不得为了“跑完流程”强行执行 04。

## 后台运行

以下仅供后续明确授权的实验使用。初始化不执行；先验证服务器路径、环境、GPU 排期和已有结果。

在上传并解压到云端后，可用统一启动器运行一个阶段：

```bash
WORKBENCH=/cloud/cloud-ssd1/vla-codex-workbench-2026-11-02-r6
ROOT=/cloud/cloud-ssd1/lightvla_updated_acceptance_2026-10-26
ROOT="$ROOT" bash "$WORKBENCH/evaluator/lightvla/launch-stage.sh" 01
sleep 5
ROOT="$ROOT" bash "$WORKBENCH/evaluator/lightvla/status-stage.sh" 01
echo TERMINAL_REMAINS_OPEN
```

阶段编号为 `01`、`02`、`03`、`04`、`05`。脚本后台运行、保存 PID 并阻止相同阶段重复启动。已有有效结果应先检查并复用；启动器不会替你决定是否应该重跑。

## 新模块实验包

`templates/remote-experiment/` 是运行外壳，不是通用算法。新 Codex应先复制到新目录，再根据真实仓库编写 `benchmark.py`，补齐结果 schema 和 resume key，并验证 `bash -n`、Python 编译和依赖无关自测。不要把模板空壳当成已完成评估器。

## 核心判定

- 权重、checkpoint、任务、初态、seed 和协议可核验且配对。
- 有真实机制证据，而不是计算后再筛选。
- 固定观测的 wall speedup 达到项目门槛且配对 CI 为正。
- 闭环总体成功率下降不超过阈值，并检查逐任务退化。
- 有状态方法通过 reset、异常恢复和新 episode 测试。
- 无错误记录；错误 episode 必须重试，不能算完成。
- 只对已记录的模型、任务、硬件、软件和参数组合下结论。

## 已知限制

- 本包不含 checkpoint、LIBERO 数据、远端 Python 环境或历史 LightVLA 原始结果 JSON。
- LightVLA 评估脚本仍是环境固定的真实脚本，不是任意 VLA 的即插即用评估器。
- Skill 能指导新 Codex编写新的临时脚本，但不能在未知模型上自动保证正确接入。
- VLA-Cache 原始日志被归档并隔离，不构成可用实现。
- AutoPrune 已永久退役，包内不存在可执行实现。

