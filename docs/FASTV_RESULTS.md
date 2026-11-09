# FastV 在 LIBERO-Spatial 上的结果

首次使用工具箱请按下方“从工具箱运行”执行。团队的[完整实验记录](README-OpenVLA-OFT-FastV.md)保留了独立工程的安装步骤、逐任务结果、FLOPs 采集命令和故障排查。独立工程的启动脚本各运行一种模式；工具箱的 `evaluate` 一次顺序执行两种模式，并将结果写入指定的新目录。

## 选择权重与后端

| | 原始 OpenVLA | OpenVLA-OFT |
|---|---|---|
| Spatial 权重 | [openvla，实验 revision](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial/tree/962318cec55ac10993ff0f5f43eda9a270b4c873) | [moojink，已合并权重](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial)，实验权重 revision 尚未冻结 |
| 注意力与输出 | eager 因果注意力，动态 KV cache，自回归生成一个 7 维动作 | eager 双向注意力，关闭 KV cache，一次前向生成 8×7 动作块 |
| K=2、剪去 50% | 单视图 256 → 128 个视觉 token | 双视图 512 → 256；剪枝后需修正动作特征位置 |
| 实验环境 | Transformers 4.40.1，Protobuf 3.20.3 | Transformers OFT fork 提交 `bc339d9ad707454c0c115970db43c260067c61ab`，Protobuf 4.25.3 |

两组实验均采用 BF16、PyTorch 2.2.0+cu121 和独立预加载的 cuBLAS 12.4.5.8。按所选后端使用对应的 Conda 环境。OFT 原始实验曾在错误的因果掩码下得到基线 290/500；即使零剪枝和基线输出相同，也可能两者都使用了错误的掩码。迁移 OFT 后端应检查实际加载了团队的 `oft_attention.py` 双向适配，识别启动信息 `OFT attention: eager, bidirectional, no padding, no KV cache`，并核对输出形状为 `(8, 7)`。

## 正式结果

每行使用对应模型的同一检查点，比较原生基线与 FastV。两种模式各运行 500 回合，seed 为 7，使用 BF16 和 eager attention。FastV 设置 K=2，剪去 50% 的视觉 token。OFT 使用双向注意力，两路图像共 512 个视觉 token，剪枝后保留 256 个。

| 模型及一次预测的输出 | 成功回合（基线 → FastV） | 平均预测延迟（基线 → FastV） | 固定输入矩阵乘法与卷积 GFLOPs（基线 → FastV） |
|---|---|---|---|
| OpenVLA，1 个动作 | 421 → 434（84.2% → 86.8%） | 242.32 → 215.57 ms（降低 11.04%） | 4390.880 → 2773.818（降低 36.83%） |
| OpenVLA-OFT，8 个动作 | 492 → 497（98.4% → 99.4%） | 291.68 → 222.73 ms（降低 23.64%） | 9029.542 → 5733.774（降低 36.50%） |

延迟来自共享 NVIDIA H20 上的闭环运行，每种模式剔除整次运行最初 5 次预测。成功率差值为本批回合的观测值。FLOPs 仅统计固定输入的一次完整预测中的矩阵乘法和卷积。

计时包含动作入口内部的策略预处理与推理，排除仿真步进和视频保存。OFT 的 222.73 ms 是一次动作块的响应时间，除以 8 只能得到每个动作的摊销耗时。两条实验线各用自己的固定输入做 FLOPs 对照，乘加计 2 FLOPs。成功率和共享 GPU 延迟均应限定在本次 seed 与测试条件下解释。

## 服务器证据

- OpenVLA 工程为 `/home/wxy/daisy_work/openvla-fastv-code`，Conda 环境为 `openvla-fastv`。在 `experiments/logs/` 下，从 `baseline-vs-fastv-n50-seed7-per-episode.csv` 复算成功数，延迟和 FLOPs 核对自同前缀的 `summary.csv`。
- OpenVLA-OFT 工程为 `/home/wxy/daisy_work/openvla-oft`，Conda 环境为 `openvla-oft`。结果位于 `/home/wxy/daisy_work/results/openvla_oft/`。基线目录为 `formal_baseline_20261108_205330_480201`，FastV 目录为 `formal_fastv_20261108_222808_420499`。从两份 `episodes.csv` 复算成功数，按任务、回合编号和初始状态 SHA-256 匹配到 500 对。从两份 `actions.csv` 跳过各自前 5 条数据后复算平均延迟。计算量来自 `flops_comparison_20261109_005614_597939/summary.csv`。

## 从工具箱运行

11 月 9 日通过统一入口另做了 H20 闭环回归。OpenVLA 十个任务各取一个初始状态，基线成功 8/10，FastV 成功 10/10。OFT 取任务 0 的一个初始状态，两者均成功 1/1，状态哈希相同。服务器原始输出在 `/home/wxy/Rui_work/fastv-adapter-smoke-1109/` 下的 `openvla-002` 和 `oft-002`，各自含有 `report.json`。这些小样本用于检查工具箱接入，上表性能数据来自此前的完整实验。

项目配置支持 `fastv_openvla_spatial` 和 `fastv_oft_spatial` 两个 profile。使用 `project check/execute` 时可在 JSON 填入 `cublas_dir`，并通过下述环境变量指定 LIBERO。项目入口默认十个任务各一个初始状态；指定 OFT 任务子集用下方 evaluate 命令。

进入工具箱根目录，使用与后端匹配的 Python 环境。以下命令适用于团队服务器，GPU 编号和全新的输出目录由使用者选择。LIBERO 需要同时提供 Python 源码及指向资产、初始状态和 BDDL 的配置。

```bash
daisy=/home/wxy/daisy_work
envs=/data2/wxy/VLA-Adapter/conda_envs
export PYTHONPATH="$daisy/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$daisy/openvla-fastv-code/.libero-config"
gpu=2
out=/home/wxy/Rui_work

"$envs/openvla-oft/bin/python" -m core evaluate --method fastv --backend oft \
  --workspace "$daisy/openvla-oft" --checkpoint "$daisy/checkpoints/openvla-7b-oft-finetuned-libero-spatial" \
  --gpu "$gpu" --tasks 0 --trials 1 --output "$out/fastv-oft-demo" \
  --cublas-dir /data2/wxy/openvla_oft_runtime/cublas-12.4.5.8/nvidia/cublas/lib

"$envs/openvla-fastv/bin/python" -m core evaluate --method fastv --backend openvla \
  --workspace "$daisy/openvla-fastv-code" --checkpoint "$daisy/checkpoints/openvla-7b-finetuned-libero-spatial" \
  --gpu "$gpu" --trials 1 --output "$out/fastv-openvla-demo" \
  --cublas-dir "$daisy/openvla-fastv-code/runtime-libs/cublas-12.4/nvidia/cublas/lib"
```

入口依次执行 baseline 和 FastV，使用同一权重和 seed。原始 OpenVLA 默认覆盖十个任务并保存视频。OFT 支持指定任务。完整规模为全部十个任务、每任务 `--trials 50`。两个子目录保存原始记录，两组均完成后生成 `report.json`。报告中的成功回合和平均延迟用于比较本次运行，质量验收仍由用户设置门槛。
