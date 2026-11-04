# 2026-11-04 LightVLA 统一入口回归

本次补齐 `core evaluate --method lightvla`，将固定输入与闭环实现移入 methods.lightvla，确保 Python 包包含运行代码。旧 scripts 入口转为兼容调用；增加子进程绝对路径处理、Spatial 统计预检与中文使用指南。原始 OpenVLA 扩展按用户要求暂停，未混入此验证范围。

真实测试用 H20 GPU 2、daisy_openvla 环境及既有 lightvla_spatial_official 权重，原项目只读。执行目录 `/home/wxy/Rui_work/lightvla-unified-20261104`，输出在 Rui_work/toolbox-native-check 中。源码快照和哈希保存在固定输入证据目录。

- 固定输入：预热 3 次、4 对调用；原生自带实现与 packaged 动作 MAE=0，卸载后 MAE=0；实际序列缩短、参数身份与抽样、模块恢复通过。本次描述性延迟中位数比约 1.85，不作为稳定速度承诺。
- 新闭环入口：task 0 / trial 0 / seed 7，baseline 与 packaged 各 1/1 成功；状态配对、模块与参数恢复审核通过，无运行错误。此输入与此前 20 对证据重叠，不能相加为 21 个独立样本。
- 本地 73 项测试通过；统一固定输入和闭环命令帮助均可运行。

原始证据：

- [固定输入报告](server-evidence/2026-11-04/lightvla-unified-pair/report.json)、同目录 records.jsonl、observation.npz、process.log、source-snapshot.zip 与 source-manifest.json。
- [闭环报告](server-evidence/2026-11-04/lightvla-unified-loop/report.json)、[配对审核](server-evidence/2026-11-04/lightvla-unified-loop/audit.json)，同目录逐回合完整动作和日志。

这是统一入口的回归检查，quality_decision 仍为 INCONCLUSIVE。已有较大范围的当前封装证据为此前 4 任务 20 对闭环、两模式均 20/20；正式质量门槛、更广任务覆盖与受控性能评估仍待完成。使用方法见 [LightVLA 指南](LIGHTVLA_QUICKSTART.md)。
