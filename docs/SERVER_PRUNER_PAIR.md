# 首次 VLA-Pruner 工具箱接入对比

前置结果：用户回传 LIBERO-Object task 0 / trial 0 采集通过，OFT 原生前向输出 8×7 有限动作；指定当前环境 cuDNN 路径后解决了本次动态库符号错误。原始 observation 的 SHA256 为 `95031c05767a3910fbc41d4abd39b3a28cbd4e54a2dc372c8f61d37d8254b9f8`。

独立包包含实际工具箱 VLA-Pruner adapter 和本次脚本，不使用服务器旧版未实现的 run_real。解压到新目录运行，不覆盖旧仓库。

```bash
/home/wxy/miniconda3/envs/openvla-oft/bin/python -B scripts/launch_pruner_check.py \
  --gpu auto \
  --observation-dir /home/wxy/Rui_work/toolbox-native-check/capture-001 \
  --output /home/wxy/Rui_work/toolbox-native-check/pruner-pair-001
```

默认 6 对交替顺序调用，每种模式每次清理状态后重复同一 observation 预热 3 次，剪除比例 0.5。每次调用复制原始输入以避免重复归一化。复用一次模型加载，不执行环境动作。诊断 hook 只用于额外机制检查调用，不计入配对延迟。

检查原生输出与关闭剪枝的工具箱输出（rtol=1e-5、atol=1e-6）、开启剪枝后的实际 decoder 层输入缩短、卸载后的模块/forward 身份及原生输出恢复。参数检查覆盖所有注册参数的对象身份和每个参数前 64 个数值，是采样检查，不是全量权重哈希。

`INTEGRATION_PASS` 只表示当前输入的接入、机制与上述恢复门槛通过；不表示动作质量达标、闭环成功或性能基准通过。动作误差及夹爪原始符号一致性只作报告，不使用任意阈值替用户做质量决策。默认 `quality_accepted=false`、`benchmark_valid=false`。

GPU 可能与其他任务共享；记录时间仅供初步诊断。固定 observation 与重复历史不是机器人真实轨迹，也不是多任务样本。后续必须补多输入和闭环、同条件 attention 路径核对及正式配对统计。

输出目录外的 `.launcher.log` 与 `.launcher.json` 记录原生进程中止；目录内 `report.json` 给出检查详情，`records.jsonl` 每次调用立即落盘，`lifecycle_actions.npz` 保存原生和卸载后动作。进程级崩溃时不能期待 Python finally 一定完成，按 launcher 非零退出处理失败。重试使用新的输出名称。

若 loaded/cache/checkpoint/backend 的 modeling_prismatic.py 哈希不一致，脚本停止，先审查差异，不自动覆盖模型代码。当前配置来自已验证的服务器：两视图、8D 本体、连续 L1 动作头、8×7 chunk、Object 归一化统计、跳过 checkpoint 自动同步。

该脚本已做本地协议检查，真实服务器上的剪枝接入尚待执行。包内发布清单可供追溯脚本与封装版本。
