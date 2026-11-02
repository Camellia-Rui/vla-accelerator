# 更新版 LightVLA：权重指纹复核包（2026-10-27 r1）

之前 50 组配对闭环结果中，成功率与速度门槛满足，但旧 `weights_unchanged` 检查返回 false，因此该次验收**尚未通过**。旧检查把参数名称也纳入哈希；packaged 模式给解码器增加包装层，参数名称会变化，即使参数数值没变也会触发 false。

本包不重跑闭环，也不修改旧结果。它在真实 LIBERO 固定观测上完成 10 组交替配对推理，并独立检查：

1. 保留原始参数对象引用，抽样 16 个张量（各前 4096 个数值），在 packaged 包装仍启用时逐项比较数值；
2. 记录包装状态下的旧式名称敏感指纹；
3. 卸载包装，检查参数名称完整恢复，且卸载后的旧式指纹与启动前一致。

结果写入 `/cloud/cloud-ssd1/lightvla_updated_acceptance_2026-10-26/results/05_weight_audit_r1.json`，日志为同一根目录下的 `logs/05_weight_audit_r1.log`。这个复核只证明抽样权重在本次测试期间没有变化；它不能倒推证明先前整场闭环测试的所有权重均未变化。请勿把旧结果的 false 直接改为 true。

## 测试基底

本包针对同一个本地 `LightVLA-libero-spatial` checkpoint 比较：

- `baseline`：关闭 checkpoint 自带裁剪 forward；
- `packaged`：挂载用户提供的更新版 LightVLA 裁剪器。

r3 从 checkpoint 根目录的 `dataset_statistics.json` 显式加载并校验 `libero_spatial_no_noops`，防止误用 `config.json` 中继承的基础 OpenVLA 预训练统计。

r4 同时兼容 LightVLA/OFT 仓库中一参数和两参数版本的 `get_libero_image` API。

r5 严格采用官方 LightVLA-Spatial 推理配置：主视角和腕部两张图、proprio 输入、L1 action head，以及 8-action chunk；每次配对推理都复制同一观测，避免原地归一化污染后续样本。

r6 显式将服务器 VLA-Cache Transformers fork 的 `reusable_patches` 与 `proportion_attn_var` 清空为 `None`，确保 baseline 与 packaged 均不继承其他加速模块的运行时状态。

步骤包括源码清单、环境与仓库记录、轻量配置契约、真实 LIBERO-Spatial 固定观测上的交替配对测量、权重抽样指纹、动作误差、夹爪符号一致率、真实输入/输出 token 长度和保留 patch 数。通用微型 LLaMA smoke 在定制 Transformers fork 缺少通用配置字段时只标记为跳过；真实 checkpoint 测试才是权威结果。

这一步通过后才进入闭环 episode 测试。单张固定观测不能作为部署验收结论。
