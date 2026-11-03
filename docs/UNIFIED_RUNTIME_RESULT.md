# 统一入口真实运行验证（2026-11-03）

本次验证的是可执行工具箱代码。直接通过 SSH 执行并下载原始报告，没有用合成动作替代真实模型结果。

## 实际执行

- 单次推理：`/home/wxy/Rui_work/vla-runtime-20261103-r1`，输出 `toolbox-native-check/unified-run-20261103-01`，退出码 0，保存完整动作块并卸载恢复。
- 输入级评估：`/home/wxy/Rui_work/vla-runtime-20261103-r2`，输出 `toolbox-native-check/unified-eval-20261103-01`，退出码 0。
- Python：`/home/wxy/miniconda3/envs/openvla-oft/bin/python`；torch 2.2.0+cu121、自定义 Transformers 4.47.0、H20 GPU 2。
- 后端：`/home/wxy/Rui_work/VLA-Pruner/VLA-Pruner/src/openvla-oft`；加载前核对核心代码与锁定提交一致。
- 权重：`/home/wxy/Rui_work/openvla-7b-oft-libero-object`，归一化键 `libero_object_no_noops`。
- 新源码、日志、缓存及临时文件都放在 Rui_work 中，未安装或升级环境包。

完整参数和退出状态保存在 [process.json](server-evidence/2026-11-03/unified-evaluate/process.json)。重跑时必须更换 output 新目录。使用步骤见 [运行指南](UNIFIED_RUNTIME.md)。

## 结果与含义

3 个真实初始输入，每输入 4 对调用、每模式预热 3 次，共 12 对。它们复用上一轮采集的 3 个输入，不是 12 个独立任务或新增初始状态。

- `integration_passed: true`：原生与关闭剪枝一致、观察到实际 decoder 序列缩短、卸载后的原生动作恢复。
- 参数对象、每个参数前 64 个值的抽样以及模块/forward 身份检查通过；不是完整权重内容哈希审计。
- 配对动作 MAE 分别约 0.00765、0.01155、0.00899；最大绝对差异分别约 0.03282、0.06046、0.04869。
- 合并调用延迟中位数之比约 **1.162×**，只作该共享服务器、该输入流程下的描述性结果。attention 路径差异尚未完成归因，不能当作正式算法加速结论。
- `input_screen: INCONCLUSIVE`：没有设置用户可接受的误差和速度门槛。
- `closed_loop_tested: false`：没有机器人/仿真闭环成功率，不能说准确率达标或可部署。

原始 [evaluation.json](server-evidence/2026-11-03/unified-evaluate/evaluation.json)、[records.jsonl](server-evidence/2026-11-03/unified-evaluate/records.jsonl)、[report.json](server-evidence/2026-11-03/unified-evaluate/report.json) 随源码归档。原始日志保留依赖警告，包括 tokenizers 版本不一致及 attention 回退提示。

## 尚需完成

成对闭环任务评估、更多初始状态、质量门槛约定、独立用户从发布包接入，以及其他方法的真实模型验收。当前版本可用于固定后端的研发与输入级筛查，不是任意 VLA 的自动部署服务。
