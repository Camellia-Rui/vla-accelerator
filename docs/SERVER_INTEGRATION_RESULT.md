# VLA-Pruner 真实模型接入验证：2026-11-02

本次由 Codex 使用专用 SSH 密钥直接连接服务器、上传独立检查包并执行。未覆盖旧工具箱仓库，未终止其他 GPU 进程。之前用户运行的真实输入采集与原生前向报告也已直接下载核对。

**结论：INTEGRATION_PASS。** 当前输入下，关闭剪枝与原生输出完全一致，剪枝确实缩短 decoder 层输入，卸载后原生输出完全恢复。尚不构成模型任务质量或正式速度验收。

## 实际配置与结果

| 项目 | 实测记录 |
|---|---|
| 模型/输入 | OpenVLA-OFT LIBERO-Object；task 0、trial 0；双视图 224×224、8D 本体；输出 8×7 |
| 方法 | VLA-Pruner semantic_action，剪除比例 0.5，fastv_k=3，vla_pruner_layer=15 |
| 设备/环境 | H20，物理 GPU 0；torch 2.2.0+cu121，cuDNN 8.9.2，自定义 Transformers 4.47.0 |
| 后端 | HEAD 3d156ee29e3d15172abbb69dc3495c47e826e460；受检查的核心源码与工具箱固定提交一致 |
| 模型代码 | 实际 HF 缓存、checkpoint 与后端 modeling_prismatic.py 的 SHA256 一致 |
| 关闭剪枝对照 | 与原生动作的 MAE、最大绝对差均为 0 |
| 实际机制 | 视觉 token 512→256；序列 596→340，零起始编号第 4 层起实际输入缩短 |
| 卸载恢复 | 与原生动作的 MAE、最大绝对差均为 0；模块/forward 身份恢复 |
| 参数检查 | 所有注册参数的对象身份及各参数前 64 个数值采样保持一致；不是全量参数内容审计 |
| 文件检查 | checkpoint 小文件哈希和文件大小未变化；没有全量权重内容哈希验证 |
| 动作变化 | 开启剪枝相对关闭：MAE 0.0076543，最大绝对差 0.0328159；该动作块的原始夹爪符号一致率 100% |
| 描述性延迟 | 关闭剪枝中位数 131.496 ms；开启 112.719 ms；两中位数比约 1.167 |

6 对交替顺序调用，每个模式开始时清理状态并重复同一输入预热 3 次。全部使用同一份 observation；它们不是 6 个独立场景或 6 次闭环任务。timing 采用 CUDA 同步，诊断 hook 不在计时调用中。

## 结论边界

- 单个动作块的误差不能确定任务成功率；尚未约定用户可接受的质量下降门槛，也未执行策略闭环。
- GPU 非独占，首次选择时利用率 0 不能证明全程独占。约 1.167 的比值仅为本次重复输入的描述性结果，不作为正式加速宣传指标。
- 配置为 LlamaSdpaAttention，但运行日志出现 eager/manual 回退提示；未对两条路径逐调用审计实际 attention 内核，正式基准前需处理。
- tokenizers 0.21.4 与后端提示的 0.21.1 不同，本次没有修改现有环境；后续正式基准需固定并记录该偏差。
- 输入来自 lerobot-libero-main 的 benchmark 模块，资产配置指向另一个 LIBERO-master 目录，本次能正常采集；可复现发行前仍需固定两者版本与资产来源。

## 原始证据

完整下载文件位于 [verified-pruner-pair](server-evidence/2026-11-02/verified-pruner-pair/EVIDENCE_MANIFEST.json)，包含输入 NPZ、原生动作、配对动作/延迟 JSONL、生命周期动作 NPZ、报告和启动器日志。输入哈希已独立复核为 `95031c05767a3910fbc41d4abd39b3a28cbd4e54a2dc372c8f61d37d8254b9f8`。

服务器结果目录：`/home/wxy/Rui_work/toolbox-native-check/pruner-pair-001`。脚本与封装版本由独立包 `vla-pruner-server-check-r1.zip` 的发布清单标识，包 SHA256 为 `daf023d763a8f68012ad8fe9ba5d5917fb866e31fab1beaee23a826da8f5639a`。

下一阶段：多任务/多初始状态输入对比、attention 路径核对、明确质量容忍范围，随后进行成对闭环验证。该结果不能替代这些门槛。
