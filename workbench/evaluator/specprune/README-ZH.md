# specprune 评估链开发区

状态：**INCOMPLETE_REAL_RUNTIME_VALIDATION**。

已交付：独立配置、无模型资源预检，以及 evaluator/common 的配对统计、原子写盘、记录级恢复、真实 block 输入 hook、参数与卸载审计。
尚未交付：目标服务器的原生真实观测探测、已验证方法 adapter 和五阶段完整运行脚本。未创建虚假的空阶段来冒充完成。

在当前机器执行（没有模型或 GPU 计算）：

```bash
python -B probe_runtime.py --config config.example.json --stage 01 --dry-run
```

示例路径必须替换为服务器真实绝对路径。缺项返回码 2；这不是运行失败或算法 REJECT。预检没有模型导入，即使所有路径存在也不证明 API 兼容。

后续先按 builder/SERVER_DISCOVERY_CHECKLIST.md 核对原生入口，获得授权后录制真实 baseline 探测，再构建方法特定 adapter 和五阶段链。
SpecPrune wrapper 会替换 forward、predict_action 和 attention 模块；独立 baseline/加速进程是默认方案。必须记录 execute_action、动作队列和 draft/precise 状态。

公共统计不能用来自不同设备、输入、初态或配置的记录拼成配对。参数抽样与结构审计不是全量权重和原生输出恢复的替代证明。

