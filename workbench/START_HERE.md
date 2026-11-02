# START HERE — 校验、盘点，再按用户当前任务工作

此文件是交接入口，不授予自动开展实验、安装依赖或修改服务器的权限。
用户的当前明确要求优先于包内任务、历史记录和模板。

## 初始化

1. 将 ZIP 解压到全新独立目录，不覆盖仓库、旧快照或结果。
2. 先查看并执行 `python -B scripts/verify_bundle.py`。失败时报告具体原因并停止。
3. 完整阅读：
   - `CODEX_INSTRUCTIONS.md`；
   - `PROJECT_STATUS.md`、`EVIDENCE_INDEX.json`、`COLLABORATION.md`；
   - `README-ZH.md`；
   - `skill/vla-acceleration-evaluator/SKILL.md`，以及该 Skill 对当前方法路由的参考资料；
   - `conversation/INITIALIZATION_DIALOGUE.md`。
4. 只读盘点当前实际可访问的代码、模型、任务资产、环境和设备。
   `python -B scripts/discover_environment.py` 输出不导入模型的环境信息；
   需要设备元信息时可加 `--gpu-info`，该选项不运行 GPU 计算。
5. 用中文报告项目状态、四个方法的差别、证据与缺口、实际环境、协作和回传方式。
   此时不自动沿用旧文档的“当前任务”，也不生成评估器代码。

## 后续开发和实验

用户要求开发时，再阅读 `builder/BUILDER_MISSION.md` 及另外三份 builder 契约。
它们描述 SpecPrune/VLA-Pruner 的待办，不代表其评估器已完成。
如果缺少真实源码、后端、checkpoint 或 LIBERO，先报告明确缺项；不能创建空壳后标为已验证。
本地无模型检查可在独立工作副本内开展；GPU、闭环和远端修改须处于用户明确授权范围内。
不全局安装 Skill。

## 项目边界

- AutoPrune 永久退役，不恢复、改名或重构其实现。
- VLA-Cache 隔离，仅审阅归档证据，不接入部署。
- GRIDs/OpenPI 排除。
- LightVLA 五阶段链是结构参考，不是另外两种方法的通过证明。
- 只有现场原始证据满足门槛，才能对所记录配置给出最终 PASS。
