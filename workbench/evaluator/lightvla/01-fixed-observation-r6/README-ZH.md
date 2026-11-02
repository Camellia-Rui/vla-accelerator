# 更新版 LightVLA r6：运行时预检与真实观测微基准

本包针对同一个本地 `LightVLA-libero-spatial` checkpoint 比较：

- `baseline`：关闭 checkpoint 自带裁剪 forward；
- `packaged`：挂载用户提供的更新版 LightVLA 裁剪器。

r3 从 checkpoint 根目录的 `dataset_statistics.json` 显式加载并校验 `libero_spatial_no_noops`，防止误用 `config.json` 中继承的基础 OpenVLA 预训练统计。

r4 同时兼容 LightVLA/OFT 仓库中一参数和两参数版本的 `get_libero_image` API。

r5 严格采用官方 LightVLA-Spatial 推理配置：主视角和腕部两张图、proprio 输入、L1 action head，以及 8-action chunk；每次配对推理都复制同一观测，避免原地归一化污染后续样本。

r6 显式将服务器 VLA-Cache Transformers fork 的 `reusable_patches` 与 `proportion_attn_var` 清空为 `None`，确保 baseline 与 packaged 均不继承其他加速模块的运行时状态。

步骤包括源码清单、环境与仓库记录、轻量配置契约、真实 LIBERO-Spatial 固定观测上的交替配对测量、权重抽样指纹、动作误差、夹爪符号一致率、真实输入/输出 token 长度和保留 patch 数。通用微型 LLaMA smoke 在定制 Transformers fork 缺少通用配置字段时只标记为跳过；真实 checkpoint 测试才是权威结果。

这一步通过后才进入闭环 episode 测试。单张固定观测不能作为部署验收结论。
