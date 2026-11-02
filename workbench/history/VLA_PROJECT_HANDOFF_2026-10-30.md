# VLA 加速模块评估项目交接摘要

更新时间：2026-10-30（Asia/Shanghai）

## 1. 项目目标

在同一套可复现评估框架中，对 VLA 加速模块进行真实性验证，重点区分：

- 模型内部指标改善：decoder prefill CUDA、FLOPs、token/patch 复用率。
- 单次动作查询延迟改善。
- LIBERO 闭环端到端延迟改善。
- 任务成功率是否不低于 baseline。
- 加速是否来自真实跳过计算，而不是计算后再筛选。
- 权重、任务初始状态、seed、评估协议是否保持一致。

涉及模块：LightVLA、VLA-Pruner、SpecPrune、VLA-Cache。评估器可以为这些模块生成临时调用脚本；当前重点是 VLA-Cache。

## 2. 用户工作方式与约束

- 本地工作目录：`D:\packaged_pruner`
- 本地产物目录：`D:\packaged_pruner\dist`
- 用户通过 JupyterLab 手动上传 ZIP 到 `/cloud/cloud-ssd1/`。
- 尽量每轮只提供一个可直接粘贴的完整命令块，避免多轮零散命令。
- 长测试必须后台运行、支持断点续跑，并在命令结束时输出 `TERMINAL_REMAINS_OPEN`。
- 不要使用会导致 Jupyter 终端退出的 `exit`，也不要让主命令长期占据前台。
- 常用后台形式：`nohup setsid bash ... >log 2>&1 </dev/null &`，保存 PID，并提供状态脚本。
- 用户希望先做小规模、严格配对验证，通过后再扩大样本。
- 不应把不同硬件、不同依赖版本或不同任务协议的数字直接混为正式对比。

## 3. 云端关键路径

```text
Python:
/cloud/cloud-ssd1/lightvla_workspace/envs/vla-cache/bin/python

OpenVLA / LIBERO 工作区：
/cloud/cloud-ssd1/lightvla_workspace/repos/LightVLA
/cloud/cloud-ssd1/lightvla_workspace/repos/LIBERO

当前 VLA-Cache OpenVLA 源码：
/cloud/cloud-ssd1/vla_cache_retest_2026-10-28/source/vla-cache/src/openvla

当前 VLA-Cache Transformers 源码：
/cloud/cloud-ssd1/vla_cache_retest_2026-10-28/source/transformers-vla-cache

OpenVLA LIBERO Object checkpoint：
/cloud/cloud-ssd1/vla_cache_acceptance_2026-10-23/models/openvla-7b-finetuned-libero-object

既有真实配对帧：
/cloud/cloud-ssd1/vla_cache_acceptance_2026-10-23/results/paired_real_frames_r2
```

环境大致为 Python 3.10.16、PyTorch 2.2.0+cu121、Transformers 4.47.0、RTX 4090。VLA-Cache 论文补丁与 OpenVLA 0.0.3、Transformers 4.47.0 组合已通过兼容性修正；不要重新使用旧的 OpenVLA 0.0.1 源树。

## 4. 已确认的 VLA-Cache 结论

### 4.1 早期 500-episode LIBERO Object 结果

- Baseline：415/500，83.0%。
- VLA-Cache：430/500，86.0%。
- Decoder prefill CUDA：46.67 ms → 40.23 ms，约下降 13.8%。
- Decoder prefill FLOPs：1.857e12 → 1.442e12，约下降 22.3%。
- Prompt layer reuse：22.15%。
- 但动作查询端到端平均延迟：285.16 ms → 327.81 ms，反而变慢。

结论：论文机制确实减少 decoder 内部计算，但原始实现的注意力请求、相似度比较、调度和同步开销抵消了收益。

### 4.2 开销归因

五条路径的代表性 wall 中位数：

```text
native              209.401 ms
control             233.550 ms
cache_reference     227.130 ms
cache_no_visual     227.166 ms
cache_gpu_helpers   232.782 ms
```

- `output_attentions=True` 带来约 24.16 ms 的控制开销。
- reference cache 相对 native 仍慢，约 0.922x。
- 删除视觉管理部分没有明显改善。
- 将 helper 搬到 GPU 反而更慢；CPU/GPU 同步和小算子调度不是简单搬运即可解决。
- 真正的 decoder skip 已观察到，通常跳过第 2～31 层，但管理税吞掉了收益。

### 4.3 Attention 摊销与 memoization

Attention 摊销实验代表性 wall 中位数：

```text
native            231.423 ms
control           255.883 ms
cache_reference   253.355 ms
cache_once        228.993 ms
cache_refresh8    229.942 ms
```

`cache_once` 和 `cache_refresh8` 的端到端速度略快，但部分动作一致性/置信区间门槛没有通过。

Memoized refresh 代表性结果：

```text
native              234.296 ms
cache_reference     258.146 ms
cache_once_memo     228.597 ms   (~1.025x)
cache_refresh8      234.891 ms
cache_refresh8_memo 229.715 ms   (~1.020x)
cache_event_memo    230.490 ms   (~1.017x)
```

Memoization 行为正确且有命中，但较小样本下速度置信区间仍不稳定，event 模式动作一致性未过。

### 4.4 Fused-hit 与 clean fixed-observation

Fused-hit 测试 72 对样本：

- native → fused-hit 平均节省约 3.39 ms，95% CI 约 `[1.15, 5.22]` ms。
- 与 `cache_once_memo` 相比没有显著额外收益。
- 动作 exact 检查通过，真实 skip 与命中均存在。

Clean fixed-observation 最重要的已确认结果：

- native wall median：约 231.703 ms。
- `cache_once_memo` wall median：约 226.111 ms。
- 速度比：约 1.02473x。
- 120 个严格配对样本，mean saved CI 为正。
- fused 版本没有比 `cache_once_memo` 更好。

结论：已经存在约 2.5% 的干净固定观测查询加速证据，但必须通过闭环任务验证成功率和端到端稳定性。

## 5. 当前待执行任务：VLA-Cache 配对闭环验证

本地包：

```text
D:\packaged_pruner\dist\vla-cache-closed-loop-paired-2026-10-30-r1.zip
SHA256: B12384FADD5D64CF81A3B6644D1F2BF798C565BED14013BF44055C82395A422F
```

设计：

- 比较 `native` 与优化后的 `cache_once_memo`。
- 默认 LIBERO Object 任务 0、1、2、3，每任务 5 个配对 trial。
- 共 20 对、40 个 episode。
- 每对使用相同任务、初始状态和 seed，并交替执行先后顺序。
- 每 episode 使用新环境，并清空 cache、retained attention、helper memo 状态。
- 每个 episode 后原子写入结果，支持断点续跑。
- 带错误的 episode 不视为完成，下次启动会自动重试。
- 检查 success、端到端查询延迟、动作一致性、真实 decoder skip、memo hits、权重指纹和 episode 错误。
- 启动脚本有 PID 防重入保护。

上传 ZIP 到 `/cloud/cloud-ssd1/` 后运行：

```bash
ROOT=/cloud/cloud-ssd1/vla_cache_closed_loop_2026-10-30
ZIP=/cloud/cloud-ssd1/vla-cache-closed-loop-paired-2026-10-30-r1.zip
PKG=/cloud/cloud-ssd1/vla-cache-closed-loop-paired-2026-10-30-r1

mkdir -p "$PKG" "$ROOT/logs"
python -m zipfile -e "$ZIP" "$PKG"
chmod +x "$PKG"/*.sh

ROOT="$ROOT" bash "$PKG/launch.sh"
sleep 5
ROOT="$ROOT" bash "$PKG/status.sh"
echo TERMINAL_REMAINS_OPEN
```

结果与日志：

```text
/cloud/cloud-ssd1/vla_cache_closed_loop_2026-10-30/results/09_closed_loop_paired.json
/cloud/cloud-ssd1/vla_cache_closed_loop_2026-10-30/logs/09_closed_loop_paired.log
```

预计耗时约 1～3 小时。后台运行，Jupyter 终端关闭不应中断。

闭环判定重点：

- 所有配对完成、无 episode 错误。
- cache 的整体成功率下降不超过 5 个百分点。
- 每任务成功率没有不可接受的下降。
- 查询端到端速度至少约 1.01x，最好给出配对 bootstrap CI。
- 权重不变，确实发生 decoder skip 与 memo 命中。
- 如果固定观测加速而闭环不加速，需要检查环境、图像预处理、动作解码和 refresh/memo 生命周期开销。

## 6. 其他模块状态

### LightVLA

- 更新版源码已完成兼容修复。
- 真实 fixed-observation benchmark 通过。
- LIBERO paired closed-loop smoke 通过。
- 权重审计通过，打包/解绑前后权重未变化。
- 已观察到明显查询加速；后续正式报告仍应使用严格配对、多任务闭环数据。

### VLA-Pruner / SpecPrune

- 云端已有各自 acceptance 工作区。
- 评估器能够为它们创建临时调用脚本。
- 新聊天若转向这些模块，应先做 source manifest、权重指纹、真实 skip/pruning 证明，再做 fixed-observation 和闭环配对测试。

### GRIDs / OpenPI

- 曾进行过独立测试，但用户确认不是当前需要的方向，已停止；不要把 GRIDs 结果混入 VLA-Cache、LightVLA、VLA-Pruner 或 SpecPrune 的正式结论。

## 7. 下一位助手应采取的工作方式

1. 先让用户运行或上传 `09_closed_loop_paired.json`，不要在没有结果时宣称闭环通过。
2. 收到结果后，优先报告 success、端到端 speedup、配对 CI、错误数、真实 skip/memo 证据。
3. 如果通过，再扩大到更多任务/更多 trials；如果失败，按 stage timing 和 episode 日志定位，不要立即扩大样本。
4. 每次尽量制作一个 ZIP，包含 `run.sh`、`launch.sh`、`status.sh`、分析脚本和中文 README。
5. 给用户一段完整上传运行代码；脚本应后台运行、可恢复、PID 防重入，并保持终端打开。
6. 不要重复下载已有 checkpoint，也不要覆盖用户已有源码或结果目录。
7. 不要只用 FLOPs 或 decoder CUDA 证明端到端加速；最终结论必须结合真实 wall time 与闭环成功率。

## 8. 新聊天的首条消息建议

```text
请读取我附加的 VLA_PROJECT_HANDOFF_2026-10-30.md，并把它作为本项目的唯一历史上下文继续工作。当前任务是完成 VLA-Cache 的 native vs cache_once_memo 配对闭环验证。请先确认交接摘要中的当前待执行包、云端路径、判定标准和终端运行约束；不要重做已完成的旧测试，也不要把 GRIDs 结果混入本项目。后续我会上传运行截图或 09_closed_loop_paired.json，请基于实际结果决定扩大验证还是继续优化。
```
