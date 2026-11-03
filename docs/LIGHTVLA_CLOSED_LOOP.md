# 当前 LightVLA 封装闭环验证

入口为 `scripts/server_lightvla_closed_loop.py`，通过独立子进程加载 daisy 的 Spatial 权重；原项目只读，缓存与结果放在指定输出目录。`launch_lightvla_closed_loop.sh` 提供后台启动、PID 检查和重复目录拒绝，`status_lightvla_closed_loop.sh` 显示进度。

协议沿用核对过的原项目逻辑：每个 episode 新建环境、相同初始状态和 seed、10 步静置、最多 220 步控制、8 个动作一组、按原生接口做夹爪转换。模式顺序按 pair 交替，每模式预热 3 次。模型只加载一次，结束后卸载并核对原生动作、模块身份和参数抽样。每回合原子写入独立 JSON，含完整动作和同步查询计时。

首轮使用 tasks 0、1 与 trials 0、1、2：两模式各 6/6 成功，无运行错误，配对和恢复审核通过。原始记录见 `server-evidence/2026-11-03/lightvla-closedloop-001`。这是当前源码的新增闭环证据，与之前找回的历史 500 对记录区分。

扩展测试使用 tasks 0、1、2、3 与 trials 0、1、2、3、4。不能将两个阶段相加宣称 26 个独立初始状态，扩展阶段包含首轮的 6 个状态。

扩展测试已完成并独立审核：baseline 和 packaged 各 20/20 成功，40 个 episode 没有运行错误，初始状态配对一致，卸载后的原生动作、模块及参数抽样检查通过。原始报告、每回合动作、日志和源码清单见 `server-evidence/2026-11-03/lightvla-closedloop-002/`，重算结果见该目录 `audit.json`。

该轮查询延迟中位数分别约 124.24 ms 和 74.29 ms，仅为描述性观测；两种模式轨迹可能不同，且 GPU 与其他用户共享。首轮同类延迟明显更高，也说明环境负载不受控。不能合并这两轮计时宣称稳定加速倍数。

```bash
bash scripts/launch_lightvla_closed_loop.sh \
  /home/wxy/miniconda3/envs/daisy_openvla/bin/python \
  /home/wxy/daisy_work/lightvla_clean \
  /home/wxy/daisy_work/lightvla_clean/checkpoints/lightvla_spatial_official \
  /home/wxy/Rui_work/toolbox-native-check/lightvla-new-closedloop 2 \
  --tasks 0 1 2 3 --trials 0 1 2 3 4
```

工具箱目录需包含当前源码。输出路径必须是新目录，GPU 编号应根据实际资源选择。进度通过同目录 status 脚本查看；完整结果必须同时核对 process.json 的退出码和 report.json。`passed` 表示执行与恢复检查，不代表任务质量部署许可。

恢复中断任务时先确认旧父子进程都已结束。脚本 `--resume` 保留已完成回合、重试错误回合，拒绝配置不同的输出目录；不要在代码、权重或依赖变更后复用旧目录。异常终止遗留锁文件时先核实对应进程，不能无条件删除锁并重启。

用 `scripts/summarize_lightvla_closed_loop.py report.json --output 新审核文件.json` 重算配对结果。小样本与共享 GPU 计时只作探索性证据，quality_decision 保持 INCONCLUSIVE。闭环轨迹和查询数可能不同，查询中位数之比不能当成固定输入成对速度结论。
