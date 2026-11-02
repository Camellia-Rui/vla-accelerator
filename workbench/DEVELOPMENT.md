# 本轮开发交付

## 已实现

- `evaluator/specprune/probe_runtime.py` 和 `evaluator/vla-pruner/probe_runtime.py`：
  检查配置、绝对路径、checkpoint 元数据、解释器身份；VLA-Pruner 另检查固定官方 commit。
  不导入模型，不连接 SSH，不安装依赖，不运行 GPU。名字沿用交付契约，但目前仅是 metadata
  preflight，不能替代一次真实原生 forward probe。
- `evaluator/common/paired_stats.py`：按输入、初态、任务、seed、checkpoint、协议和设备身份
  配对，拒绝重复/错误/不配对记录；保留原始差值，输出确定性配对 bootstrap CI。
- `result_io.py`：同文件系统原子替换，已完成证据不覆盖，错误记录可重试，配置身份隔离。
  单目录由一个 worker 持有；它不是跨主机的多写者数据库或 GPU 调度器。
- `mechanism.py`：在实际 block 前向计算前捕获真实输入 shape，移除所有注册的 hook。
  较短输入只是一部分机制证据，不能替代方法特定的 EMA/draft/precise/回退计数。
- `lifecycle.py`：抽样检查原参数对象数值，检查卸载后参数/模块/函数/hook 恢复。
  原参数对象不再注册时明确显示，不能据原对象未变就认证替换后新参数未变。
  buffers/config/cache 和原生输出恢复仍需方法特定测试。

## 本地检查

本轮新增 19 项基础设施测试已通过，其中 2 项使用真实 CPU torch 模块；
没有 checkpoint、真实 observation 或 GPU 工作负载参与这些测试。

```bash
python -B evaluator/common/tests/test_infrastructure.py -v
python -B evaluator/specprune/probe_runtime.py --config evaluator/specprune/config.example.json --dry-run
python -B evaluator/vla-pruner/probe_runtime.py --config evaluator/vla-pruner/config.example.json --dry-run
```

两份示例使用未填写的目标路径，预检非零退出并列出缺项是预期结果，不是算法失败。
无 torch 的环境会明确跳过 CPU tensor 测试。CPU 模块仅用于测试 hook/审计逻辑，不是 VLA。

## 为什么此处不生成虚假的完整链

Skill 要求先核验真正的加载、推理、动作队列和 reset API，并运行原生真实观测探测。
当前只能访问 Windows 工具箱和快照，目标 Linux 源码、checkpoint、LIBERO 未被确认。
盲猜 loader、用随机图像代替真实输入，或复制 LightVLA 脚本换方法名，都不满足交付契约。
完整 adapter 和 01—05 实验阶段仍标 INCOMPLETE_REAL_RUNTIME_VALIDATION。

下一步需要一次提供/确认目标 VLA、SpecPrune、固定 VLA-Pruner 后端、checkpoint、LIBERO、
各 Python 环境及输出目录。历史 /cloud 路径可作为候选，但必须在真实远程工作区验证。
确认原生 API 后才能实现并检查两个方法的真实调用和五阶段链；GPU 实验仍需相应授权。
