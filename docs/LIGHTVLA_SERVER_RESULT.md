# LightVLA 当前封装与历史证据（2026-11-03）

已实际读取 `/home/wxy/daisy_work/lightvla_clean`，使用其 Spatial 训练权重和 `daisy_openvla` 环境完成当前工具箱封装的真实输入验证。不是随机小模型，也没有重新训练。

## 当前封装

原项目 HEAD 为 `a4680fd`，有本地修改，不能称为未修改的官方快照。新增加载器 `methods/lightvla/oft_factory.py` 直接读取本地权重，避开上游会同步改写 checkpoint 的 get_vla；使用 checkpoint 的 Spatial 归一化统计、双图、本体状态、L1 动作头和完整 8×7 动作块。模型缓存与结果置于 Rui_work，没有改写 daisy 原项目或安装包。

在 H20 GPU 2 上，Spatial task 0 / trial 0，settle 10 步后固定输入、预热 3 次、4 对交替调用：

- 原生内置 LightVLA 与当前 packaged 封装动作完全一致（本输入 MAE 0）。原生路径本身已带剪枝，不是无剪枝 baseline。
- 相对关闭剪枝 baseline，packaged 的动作 MAE 约 0.00893，最大差异约 0.03844，夹爪符号一致。
- 实际 decoder 序列缩短；卸载后的动作、模块/forward 身份恢复；参数对象及每参数前 64 个值的抽样检查通过。
- 描述性延迟比约 1.560×。共享 GPU、单输入、少量调用，不能当作正式速度结论。
- 统一 `core run` 调用也通过，保存完整 8×7 动作及来源记录。

[配对原始报告](server-evidence/2026-11-03/lightvla-current-002/report.json) · [统一入口动作](server-evidence/2026-11-03/lightvla-unified-001/query.json)。首次检查因 EGL 设备编号不匹配失败，也保留在 `lightvla-current-001` 中；修正为物理 GPU 编号后通过。

## 使用命令

在包含最新源码的工具箱根目录，用原模型环境的 Python 运行下面的受控入口。它配置独立子进程缓存，实际调用统一运行入口：

```bash
/home/wxy/miniconda3/envs/daisy_openvla/bin/python -B scripts/server_lightvla_pair.py \
  --workspace /home/wxy/daisy_work/lightvla_clean \
  --checkpoint /home/wxy/daisy_work/lightvla_clean/checkpoints/lightvla_spatial_official \
  --output /home/wxy/Rui_work/toolbox-native-check/lightvla-my-new-run \
  --gpu 2 --unified-run
```

输出必须是新的绝对路径，GPU 需要按实际资源选择；不保证上述 GPU 始终空闲。去掉 `--unified-run` 即执行有界对照检查，可加 `--task`、`--trial`、`--pairs`、`--warmup`。当前工厂只针对已审核的 LightVLA Spatial OFT 接口；其他工程需另行检查。`core evaluate --method lightvla` 尚未统一，不能冒充已支持。

## 历史闭环记录已找回

从 daisy 的 results 下载了两组原始 episode JSONL 和对应比较文件，并使用 `scripts/audit_lightvla_history.py` 独立重算：

- 500 对 episode，配对初始状态哈希、seed、checkpoint 路径和任务说明一致。
- 两组各成功 490/500，即 98%；baseline 独有成功 8 次，packaged 独有成功 8 次。
- 350 对的 GPU 设备标签不同，不能将其历史延迟差当作受控性能结论。

见 [重算报告](server-evidence/2026-11-03/lightvla-historical/recomputed.json)。这些是历史实现的记录，checkpoint 路径一致不等于内容哈希一致，也尚未建立历史实现与当前封装的完整等价关系。不能用这一组 98% 直接给当前代码判定部署 PASS。

## 当前封装的新增闭环

首轮 2 个任务、每任务 3 个初始状态的闭环检查完成：baseline 与 packaged 各 6/6 成功，12 个 episode 无运行错误，初始状态配对和卸载恢复通过。源码与历史 500 对实现分开记录，详见 [闭环协议](LIGHTVLA_CLOSED_LOOP.md) 和 [独立审核](server-evidence/2026-11-03/lightvla-closedloop-001/audit.json)。

随后扩展至 4 个任务、每任务 5 个初始状态：baseline 与 packaged 各 20/20 成功，40 个 episode 无运行错误，配对与恢复检查通过，见 [扩展审核](server-evidence/2026-11-03/lightvla-closedloop-002/audit.json)。扩展阶段包含首轮状态，不把两轮样本重复相加。该结果仅支持所测范围内未观察到成功率下降，不是完整 Spatial 套件或真机部署保证。

当前结论：真实模型接入、统一调用及小规模闭环已验证；充分质量验收、正式性能和来源审查仍待完成。
