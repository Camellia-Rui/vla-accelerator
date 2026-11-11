# SpecPrune 新接入工作流

**最新结果（2026-11-11）**：已扩大到 Object 全部十个任务、每任务五个初始状态，共 50 对；基线与加速均成功 49/50。真实查询计数、闭环耗时、区间、选择器演示和原始证据见 [中期展示说明](SPECPRUNE_DISPLAY.md)。下文 11 月 5 日的 3 对记录保留为接入历史，不代表最新覆盖规模。

当前入口是实验评估，不是部署验收通过。已核实继承实现与 cloris 的冻结工程封装在统一换行符后完全一致；不是核实为官方逐字快照，来源与许可见 methods/specprune/provenance.json。

新增事务式安装、失败回滚、完整原注意力模块与前向恢复。80 项本地测试通过。首次真实 Object 输入验证中，保留全部 token 对照和卸载后动作 MAE 均为 0，观察到实际剪枝，参数身份/抽样与模块恢复通过。剪枝动作 MAE 约 0.0328、最大差约 0.131，不据此判定任务质量。

在当前工具箱根目录，使用匹配的 SpecPrune 环境：

```bash
/home/wxy/cloris_work/env/bin/python -B -m core evaluate --method specprune \
  --workspace /home/wxy/cloris_work/SpecPrune-VLA/openvla-oft \
  --checkpoint /home/wxy/cloris_work/models/openvla-7b-oft-finetuned-libero-object \
  --observation /home/wxy/Rui_work/toolbox-native-check/capture-001 \
  --gpu 4 --output /home/wxy/Rui_work/toolbox-native-check/specprune-user-001
```

GPU 需先确认可用；输出必须是新目录。子进程隔离缓存，只读加载原权重，不调用会同步 checkpoint 文件的 get_vla。固定输入包含原生、保留全部 token、剪枝、恢复及 4 对预热同步计时。报告保留完整动作，质量保持 INCONCLUSIVE。

可加 --closed-loop 做 Object task 0/trial 0/seed 7 的配对烟雾测试。可用 --task/--trial 选择 Object 初始状态，不支持任意任务集批量验收。每回合新环境、相同初始状态，显式记录 execute_action，仅为 true 时执行动作。队列中的每步输出和完整固定输入动作分别保存。执行成功与任务成功分开报告。

core run --method specprune 暂不开放；正式部署仍需要更多任务、初始状态和质量门槛。原始结果位于 docs/server-evidence/2026-11-05。历史 README 的 500 回合统计不能代替当前生命周期修改的验证。

## 2026-11-05 新验证结果

独立真实闭环：Object task 0 / trial 0 / seed 7，原生和加速都成功。原生 155 次环境动作；加速 141 次策略调用，其中 132 次执行环境动作，前 9 次仅内部消费队列。相同初始状态哈希、恢复动作 MAE=0。4 对固定输入计时中位数约 775.75 ms 与 451.15 ms，比值约 1.72；不是受控性能基准，不作为跨机器承诺。

原始报告、逐步动作和实际脚本位于 server-evidence/2026-11-05/specprune-pair-loop-002。CLI 可通过 --task/--trial 选择其他 Object 初始状态，--pairs/--warmup 设置配对次数与预热；只有上述一个初始状态已完成此次闭环验证，不宣称完整任务集通过。

项目配置可显式使用 profile=specprune_object，并提供恰好一个 observation_dirs 输入目录；执行仍限于实验 evaluate，不自动推荐为优于 VLA-Pruner 的方法。该项目 profile 新增路由尚未独立做服务器端端到端验证，实际统一 evaluate 回归与项目路由要区分。

统一入口已在服务器以真实模型回归通过，退出码 0，保留全部 token 对照、实际剪枝、卸载和参数/模块检查通过；证据见 specprune-unified-001。可用 `scripts/audit_specprune.py report.json --output 新审核文件.json` 独立核对配对计时、动作合法性和闭环状态配对。82 项本地测试通过。统一入口回归快照之后新增 task/trial/pairs/warmup 参数及卸载错误报告分支，默认成功路径未改变；这些新增参数的边界已本地检查，未逐组合进行 GPU 实验。

## 批量验证与续跑

服务器上已验证的批量脚本位于 `/home/wxy/Rui_work/specprune-batch-20261105`。执行下方命令前先进入该目录。此次 Object 任务 0、1 各取 trial 1，原生和加速各成功 2/2。任务 0 的环境动作数分别为 128 和 153，任务 1 分别为 170 和 127。重复使用 `--resume` 复用了两对已审核记录。加上此前任务 0 的 trial 0，目前两个任务共覆盖三个不同的任务初始状态，两种模式各成功 3/3。[本轮配对报告](server-evidence/2026-11-05/specprune-batch-001/report.json)

新增 scripts/batch_specprune.py，在匹配的模型 Python 环境执行。参数沿用统一入口的 workspace、checkpoint、observation、gpu、output，增加 `--tasks 0 1 --trials 1`。每个任务/初始状态在独立子进程运行固定输入门槛和配对闭环，闭环模式顺序按 task+trial 的奇偶交替。

```bash
/home/wxy/cloris_work/env/bin/python -B scripts/batch_specprune.py \
  --workspace /home/wxy/cloris_work/SpecPrune-VLA/openvla-oft \
  --checkpoint /home/wxy/cloris_work/models/openvla-7b-oft-finetuned-libero-object \
  --observation /home/wxy/Rui_work/toolbox-native-check/capture-001 \
  --gpu 4 --tasks 0 1 --trials 1 \
  --output /home/wxy/Rui_work/toolbox-native-check/specprune-my-batch
```

每个子目录保存原始报告、日志和完整动作；根目录 progress.json 记录进度，report.json 只在所有配对完成并审核后生成。缺失报告、运行错误或恢复失败都不计入完成。

中断后先确认旧父子进程均已结束，再使用相同参数加 --resume。批次配置会比较工具箱源码哈希、模型环境元数据和 checkpoint 文件大小/修改时间清单；它不是权重全量哈希，不用于安全认证。不要跨模型、依赖或源码变化续跑。残留 batch.lock 必须先检查对应进程，不能无条件删除。有效配对复用，失败尝试保留在原目录，新尝试放入 retry 子目录。任务失败但正常完成的回合属于有效质量证据，不自动重跑直到成功。

更广任务覆盖仍不自动产生部署许可；批量报告默认 INCONCLUSIVE，用户质量门槛与充分统计尚待明确。
