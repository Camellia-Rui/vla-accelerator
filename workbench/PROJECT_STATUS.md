# 项目状态：2026-11-02

## 目标与交付状态

用户提供工具箱和模型工作目录后，Codex 根据实际模型指导方法选择、环境、权重、运行、
排错和结果判断，并用原始证据评估推理加速与行为质量。支持范围按模型/方法组合记录。

已交付：Skill、schema-v3 工具、方法接入资产、LightVLA 五阶段脚本和远程运行模板。
本轮新增：SpecPrune/VLA-Pruner 独立配置与资源预检；公共配对统计、记录级恢复、
原子写盘、计算前 block 输入探针、参数抽样与卸载结构审计。测试只使用合成数据和微型 CPU 模块。
未交付：服务器原生真实观测探测、已验证方法 adapter、完整五阶段评估器。
builder 是待办契约，不是完成证明。
缺少真实栈验证时标记 INCOMPLETE_REAL_RUNTIME_VALIDATION。

| 方法 | 状态 |
| --- | --- |
| LightVLA | 固定观测 r6、闭环 smoke、多任务、条件复核、权重审计脚本齐全；完整历史结果未随包提供，不能宣布最终 PASS |
| SpecPrune | 实际工程 wrapper；有跨帧/动作队列状态，完整卸载尚未证明，优先独立 baseline/加速进程 |
| VLA-Pruner | 适配器与后端锁文件；官方模型/剪枝内核不在包内，须核对外部版本和 EMA reset |
| VLA-Cache | 隔离 09—13 JSON/log；被测实现闭环失败，不接入部署 |

AutoPrune 永久退役；GRIDs/OpenPI 排除。LightVLA 原方法包含微调；这里的固定权重推理
改动无需再训练，不代表原论文方法免训练。

## 证据与缺口

见 EVIDENCE_INDEX.json。LightVLA 05 是后续独立抽样审计，不能倒推整场旧闭环权重未变，
不能改写旧 weights_unchanged=false。SpecPrune/VLA-Pruner README 数字仅作历史记录。
测试 fixture 不属于实验结果。缺少原始数据时不以文档或截图替代。

## 本次实际环境与后续条件

本次在 Windows D:/大创/vla-express-toolbox 盘点，不是学校 Remote-SSH 终端。
本地 RTX 4050 Laptop 约 6 GB；已有 .venv 为 Python 3.12、torch 2.3.0+cpu、
Transformers 4.40.1。当前工程没有发现 checkpoint/LIBERO 资产及有效远程连接。
这是本次机器观测，不是所有使用者的环境；新机器须运行 discover_environment.py。
历史 /cloud 路径只作待核验线索。

后续先取得目标 VLA 工程、方法源码/版本、checkpoint、LIBERO 和环境路径。
授权开发后，在独立目录完成两条评估链；真实探测与 GPU/闭环运行另需相应授权。
本地检查通过不等于模型已跑通或加速有效。
