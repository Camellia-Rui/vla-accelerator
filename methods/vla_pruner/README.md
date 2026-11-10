# VLA-Pruner · OpenVLA-OFT / LIBERO-Object

VLA-Pruner 在推理中结合语义注意力与历史动作注意力裁剪视觉 token。工具箱已验证固定后端、双视图、8 维本体输入、连续动作头和 8×7 动作块。项目 profile 为 `oft_object`。

## 准备与运行

准备 [Object 微调权重](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-object)、动作头、本体投影器和 processor。使用 VLA-Pruner 工程内的 OpenVLA-OFT 与定制 Transformers 4.47.0；源码检查以提交 `84d4b7192c77abf1585610e2f12393319b7ebff9` 为基准。[后端安装与接口](INTEGRATION.md)

从工具箱根目录，在模型环境中设置 `WORKSPACE` 为后端源码目录、`CHECKPOINT` 为权重目录、`OBS` 为匹配任务的采集目录、`GPU` 为设备编号。采集目录须含 `observation.npz` 及记录其哈希和指令的 `report.json`。使用新输出目录：

```bash
python -m core evaluate --method vla_pruner --workspace "$WORKSPACE" --checkpoint "$CHECKPOINT" --unnorm-key libero_object_no_noops --observation-dir "$OBS" --gpu "$GPU" --prefer-env-cuda --prune-ratio 0.5 --pairs 6 --warmup 3 --output runs/pruner-001
```

检查 `evaluation.json` 和逐次记录：关闭剪枝时与原生动作一致，加速时实际序列缩短，卸载后恢复。固定输入动作误差用于初筛；要回答“还能否完成任务”，使用同权重闭环比较。[测试命令与判据](../../README_TESTING.md) · [闭环选择](../../docs/METHOD_SELECTION.md)

## 已验证范围

H20 上两个 Object 任务、六对初始状态，原生与 50% 剪枝各成功 **6/6**。这批闭环的平均查询延迟为 826.16 → 710.90 ms，约 **1.16×**。更多任务与其它权重应重新测量。[同权重实验](../../docs/OFT_METHOD_CHOICES.md)

早期参数、接口示例和实验表见[历史实验记录](HISTORICAL_EXPERIMENTS.md)；当前接入以本页和 INTEGRATION.md 为准。
