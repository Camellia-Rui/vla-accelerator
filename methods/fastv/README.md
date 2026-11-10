# FastV · OpenVLA 与 OpenVLA-OFT / LIBERO-Spatial

FastV 在前两层处理完整序列（K=2），再按注意力保留一半视觉 token。两条路线分别用同一个任务检查点比较原生与剪枝，无需重新训练。先按模型选择后端：

| 工程 / profile | 权重 | 输入与预测 |
|---|---|---|
| 原始 OpenVLA / `fastv_openvla_spatial` | [Spatial 权重，已测 revision](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial/tree/962318cec55ac10993ff0f5f43eda9a270b4c873) | 因果注意力，256 → 128 个视觉 token，一次预测一个动作 |
| OpenVLA-OFT / `fastv_oft_spatial` | [OFT Spatial 权重](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial) | 双向注意力，512 → 256 个视觉 token，一次预测 8 个动作 |

## 运行与结果

使用相应的 FastV 后端源码、模型环境和 LIBERO-Spatial 资源。从工具箱根目录运行 `python -m core evaluate --method fastv --help` 查看参数。传入 `--backend openvla` 或 `--backend oft`、源码、权重、GPU 和新输出目录后，工具箱依次运行 baseline 和 FastV。OFT 可先用 `--tasks 0 --trials 1`；原始 OpenVLA 的 `--trials 1` 会覆盖十个任务。[两套完整命令与环境](../../docs/FASTV_RESULTS.md)

完成后检查 `baseline/`、`fastv/` 与 `report.json`。团队的 H20 正式实验在 seed=7 下为每种模式运行 500 回合。原始 OpenVLA 的成功数为 **421 → 434**，平均单动作延迟 **242.32 → 215.57 ms**。OFT 为 **492 → 497**，平均动作块延迟 **291.68 → 222.73 ms**。每种模式仅剔除整次运行的前五次预测，两条路线各用自己的基线计算加速比。[原始实验、逐任务统计与 FLOPs 口径](../../docs/README-OpenVLA-OFT-FastV.md)
