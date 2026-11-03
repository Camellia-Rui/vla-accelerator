# H20 服务器：采集输入与原生前向

适用用户已核验的 `/home/wxy/Rui_work` 资源。仅做真实输入采集和一次原生 OFT 前向，不运行剪枝、训练、闭环策略或性能验收。脚本尚需在服务器实际验证。

将 `scripts/server_native_smoke.py` 放入服务器的独立目录，例如 `/home/wxy/Rui_work/toolbox-native-check/`。脚本独立运行，不依赖服务器旧工具箱，不必拉取或修改旧仓库。

```bash
cd /home/wxy/Rui_work/toolbox-native-check
/home/wxy/miniconda3/envs/openvla-oft/bin/python -B server_native_smoke.py capture --gpu auto --output capture-001
```

成功应显示 `PASS`，保存 `observation.npz` 和 `report.json`。采集使用已有 LIBERO-Object 的 task 0、初始状态 0、seed 7、10 个仿真等待步，调用后端相机处理和四元数转换函数。不会生成或执行策略动作。若失败，读取同目录报告和日志，先解决采集问题。

确认采集通过后：

```bash
/home/wxy/miniconda3/envs/openvla-oft/bin/python -B server_native_smoke.py baseline --gpu auto --observation-dir capture-001 --output baseline-001
```

成功应保存有限的 `8 × 7` 动作数组和 `report.json`。`first_query_ms` 仅是有同步的首次调用时间，不能用于宣称加速比或任务成功率。checkpoint 小文件哈希/文件大小检查不等价于全部权重内容校验。

`auto` 从显存满足最低条件的 GPU 中选择当前利用率较低的一张，采集至少 4,000 MiB，加载至少 32,000 MiB；这些是选择门槛，不是显存充足保证。选择与实际分配之间仍可能有其他进程变化。不会清理或终止任何已有进程。可用 `--gpu 2` 显式指定物理卡号。多个作业共享卡时，本次结果不能作为正式延迟基准。

默认开启 EGL 渲染、TensorFlow 显存按需分配、HF 离线模式和 `OPENVLA_SKIP_CHECKPOINT_SYNC=1`。原生模型加载仍会执行用户已有 checkpoint 的模型代码；脚本会记录实际类文件位置。独立输出目录必须不存在，失败后重试也使用新名字。

当前本地只验证脚本语法、命令行和 GPU 选择逻辑。不能将本地检查写成服务器模型验收成功。
