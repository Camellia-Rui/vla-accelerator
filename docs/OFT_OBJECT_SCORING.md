# 同一份 OFT Object 权重上的评分策略对照

比较限定在 VLA-Pruner 的固定 OpenVLA-OFT 后端上。一次加载 Object checkpoint，在同一模型实例上依次切换原生、后端内置的 FastV 评分（`fastv_scoring`）和语义＋动作评分（`vla_pruner`）。此处的 FastV 评分与独立 Spatial 工程的 500 回合评测分开记载。

## 留出实验结果

10 个 Object 任务各取状态 5–9，共 50 对、150 回合，全部执行完成，模型与参数恢复检查通过。以下结果来自共享 H20 上的本轮测量。

| 模式 | 成功回合 | 平均查询延迟 | 相对基线 | 平均每回合策略耗时 |
|---|---|---|---|---|
| 原生 | 50/50 | 176.63 ms | 1.00× | 2988.52 ms |
| 同后端 FastV 评分 | 49/50 | 126.48 ms | 1.40× | 2228.52 ms |
| VLA-Pruner 语义＋动作评分 | 50/50 | 152.26 ms | 1.16× | 2615.75 ms |

FastV 评分的失败出现在任务 8 的一个状态。预设的零下降门槛选中 VLA-Pruner；同一结果若允许下降 2 个百分点，则选中 FastV 评分。后一种门槛是结果上的敏感性展示。50 回合中一个成败即占 2 个百分点，这些观测还需更大样本和多 seed 验证。

[完整压缩报告](server-evidence/2026-11-10/oft-scoring-object/holdout/report.json.gz) · [审核与逐任务结果](server-evidence/2026-11-10/oft-scoring-object/holdout/audit.json) · [零下降选择](server-evidence/2026-11-10/oft-scoring-object/holdout/selection-drop-0.json) · [允许下降两百分点](server-evidence/2026-11-10/oft-scoring-object/holdout/selection-drop-2.json)

从仓库根目录可以直接复算；将 `--max-drop-pp 0` 改为 `2` 并换一个输出目录，即可展示另一个选择。

```bash
python -B -m core select --report docs/server-evidence/2026-11-10/oft-scoring-object/holdout/report.json.gz --min-success-pct 90 --max-drop-pp 0 --min-pairs 50 --output runs/object-scoring-selection
```

最终 `core select --project` 入口也用单独的任务 0、状态 0 做了真实回归。[入口回归报告](server-evidence/2026-11-10/oft-scoring-object/final-entry-check/measurement/report.json.gz)

## 冻结的评测条件

- 固定 K=3、请求剪去 50% 的视觉 token、seed=7，按任务和初始状态轮换三种模式的运行顺序。语义＋动作评分会先积累历史，实际每次裁剪的数量见逐次记录。
- 预试验覆盖任务 0、1，各用初始状态 0、1、2。留出组覆盖全部十个 Object 任务，各用状态 5–9。每个状态建立独立仿真环境并核对哈希；每回合先静置 10 步，控制上限 280 步。
- 逐次查询前后同步 CUDA。报告平均和中位查询毫秒数、每回合策略耗时（只累加模型查询）、调用数、执行动作数与逐任务成功数。共享 GPU 的外部负载可变，查询包含对应模式的模型调用封装。
- 中期演示使用示例门槛：至少 50 对、观测成功率至少 90%、比基线最多下降 0 个百分点。按同一次留出实验的平均查询延迟排序，若原生最快且质量达标，则选择原生。

## 复现

从工具箱根目录，在正确的模型环境执行。发布包不含模型、后端、LIBERO 资产及 Python 环境。项目配置使用 `oft_object`，新实验默认比较两种评分；`--oft-comparison lightvla-style` 保留旧的对照路线。

```bash
python -B scripts/server_oft_compare.py --closed-loop --scoring \
  --workspace "$WORKSPACE" --checkpoint "$CHECKPOINT" --observation-dir "$OBS" \
  --gpu "$GPU" --prune-ratio 0.5 --tasks 0 1 2 3 4 5 6 7 8 9 --trials 5 6 7 8 9 \
  --output "$OUTPUT"
```

本轮使用 H20 物理卡 0、`/data2/wxy/VLA-Adapter/conda_envs/openvla-oft/bin/python`（PyTorch 2.2.0+cu121），通过 PYTHONPATH 加载 VLA-Pruner 工程的 `transformers/src`（4.47.0）、后端根目录、团队 LIBERO 和隔离的 tokenizers 0.21.4。后端建议 tokenizers 0.21.1，本次保留版本差异。运行器额外传入 `--cublas-dir /data2/wxy/openvla_oft_runtime/cublas-12.4.5.8/nvidia/cublas/lib`；早期尝试使用环境默认库时子进程收到 SIGFPE。LIBERO_CONFIG_PATH 指向 `/home/wxy/daisy_work/openvla-fastv-code/.libero-config`。三种模式共用这些设置，未修改模型工程或安装环境。

## 预试验与第三方后端核对

在 `project.json` 中指定 `cublas_dir` 后，`core project execute` 与 `core select --project` 都将其传给子进程。在 Object 后端复用选择结果时，取 `selection.json` 的 `configuration` 对象保存为单独的 JSON，再作为 `core run --method vla_pruner --config` 的配置；保持同样的模型、环境和输入。若选择结果为 baseline，请用 `--mode baseline`。

预试验三种模式各成功 6/6。采集输入预热后，原生第一层和最后一层序列长度均为 596，两种剪枝均为 596 → 340。闭环中 FastV 评分第一步已裁去 256/512 个视觉 token；语义＋动作路线在历史不足时可保持完整输入。[预试验报告](server-evidence/2026-11-10/oft-scoring-object/pilot/report.json) · [审核摘要](server-evidence/2026-11-10/oft-scoring-object/pilot/audit.json)

另外比较了 VLA-Pruner 与 SpecPrune 工程的 Object checkpoint。四个主权重分片、LoRA 权重、动作头、本体投影器及归一化统计的 SHA-256 分别相同，`modeling_prismatic.py` 不同。在同一个 capture-001 上，两份已存原生 8×7 动作的平均绝对差为 0.000590、最大绝对差为 0.0078125。这个结果支持继续对齐后端；当前选择器不会把两个独立报告的延迟合并排名。[文件校验值，列表顺序为 VLA-Pruner、SpecPrune](server-evidence/2026-11-10/oft-scoring-object/specprune-checkpoint-sha256.json) · [OFT 原生动作](server-evidence/2026-11-04/oft-choice-002/report.json) · [SpecPrune 原生动作](server-evidence/2026-11-05/specprune-unified-001/report.json)
