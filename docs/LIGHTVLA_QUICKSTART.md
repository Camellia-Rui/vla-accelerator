# LightVLA 实际使用入口

当前可用目标是经过检查的 LightVLA 专用 LIBERO-Spatial 训练权重、双视图、连续动作头和 8×7 动作块。不是把任意 OpenVLA/OFT 权重交给 LightVLA 就能保证有效。普通 OFT Object 的推理干预已出现 1/6 成功，不推荐该组合。

## 环境与资源

服务器已核对的环境是 `/home/wxy/miniconda3/envs/daisy_openvla/bin/python`，源码为 `/home/wxy/daisy_work/lightvla_clean`，权重为该目录的 `checkpoints/lightvla_spatial_official`。沿用其定制 Transformers、processor、动作头、本体投影器、LIBERO 资源；不要用工具箱的 CPU 测试依赖覆盖这个环境。

队友需要有上述资源的读取权限。下载工具箱不包含权重，不代表另一台机器已配置完成。首次接入先让 Codex 检查实际路径、环境和统计文件；当前加载器会拒绝缺少 Spatial 归一化统计的权重。新缓存和结果全部放在指定 output 内，原项目与权重只读。

## 固定输入对照

在工具箱根目录执行。输出目录必须尚不存在，GPU 编号先按资源使用情况确认。

```bash
/home/wxy/miniconda3/envs/daisy_openvla/bin/python -B -m core evaluate \
  --method lightvla \
  --workspace /home/wxy/daisy_work/lightvla_clean \
  --checkpoint /home/wxy/daisy_work/lightvla_clean/checkpoints/lightvla_spatial_official \
  --gpu 2 --task 0 --trial 0 --warmup 3 --pairs 4 \
  --output /home/wxy/Rui_work/toolbox-native-check/lightvla-my-pair
```

检查 `process.json` 的 exit_code 为 0，并检查 `report.json`：passed、mechanism_observed、detached、module_forward_restored、parameter_identity_samples_preserved 为 true；native_vs_packaged 和 restored_vs_native 的 matches 为 true。完整动作和每次同步计时在 records.jsonl，实际输入在 observation.npz。

baseline 是同一训练权重关闭裁剪；builtin 是原项目自带裁剪；packaged 是工具箱实现。不能将 baseline 描述为另一套原始 OpenVLA 权重。

## 闭环验证

同一入口增加 `--closed-loop`，使用 tasks/trials，而不是单输入的 task/trial：

```bash
/home/wxy/miniconda3/envs/daisy_openvla/bin/python -B -m core evaluate \
  --method lightvla --closed-loop \
  --workspace /home/wxy/daisy_work/lightvla_clean \
  --checkpoint /home/wxy/daisy_work/lightvla_clean/checkpoints/lightvla_spatial_official \
  --gpu 2 --tasks 0 1 --trials 0 1 2 --seed 7 \
  --output /home/wxy/Rui_work/toolbox-native-check/lightvla-my-loop
```

长实验使用已有 `scripts/launch_lightvla_closed_loop.sh` 后台启动器，参见 [闭环说明](LIGHTVLA_CLOSED_LOOP.md)。旧脚本调用同一个包内实现。需要恢复时用同一参数加 `--resume`，并先确认旧父子进程已结束；不能跨源码、环境或权重版本续跑。

闭环报告应包括每模式成功数、完整动作、配对状态哈希、错误记录、参数与模块恢复结果。重新审核：

```bash
python scripts/summarize_lightvla_closed_loop.py /path/to/report.json --output /path/to/new-audit.json
```

## 能展示什么

已有专用 Spatial 的 4 任务、20 对当前封装闭环证据，两种模式各 20/20；见 [原始结果说明](LIGHTVLA_CLOSED_LOOP.md)。现场可展示 Codex 检查资源、运行固定输入对照、读取动作与实际裁剪记录，再展示保存的闭环证据。查询时间比仅是特定环境的观测；小样本成功不能证明任意模型可用。当前 quality_decision 保持 INCONCLUSIVE，需明确的质量容忍度和更充分评测才能作正式部署结论。
