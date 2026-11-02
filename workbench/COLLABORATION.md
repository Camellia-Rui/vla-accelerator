# 并行协作与成果回传

发布包、原始证据、权重、数据和上游仓库保持只读。成员各自使用
`work/<成员>/<方法>/<日期-任务>/`、独立 Git 分支、环境及输出根目录。
SpecPrune/VLA-Pruner 可分开开发，共享基础组件由指定集成人合并。
不直接修改 Skill assets，不把旧工具箱和新快照混入同一运行路径。

每次记录 Python、PYTHONPATH、设备、源码 commit/dirty state、experiment_id 和配置哈希。
同卡正式测速串行预约，单阶段 PID 并不是全局 GPU 锁，不同 ROOT 也不自动互斥。
团队应记录设备和时间预约，避免并发干扰。

结果区分 planned/running/complete/failed，记录稳定 resume key；错误记录需重试，
不计为完成。旧结果不覆盖，重跑先归档并注明理由。历史 LightVLA 阶段不全具有
统一 record-level resume，新实验须实现并验证。

回传一个版本化 ZIP 或明确 commit，包含代码、配置、中文说明、source manifest、哈希、
许可证来源、运行命令、环境、PID/状态工具和预期输出路径。
附本地验证日志；运行过实验时附原始 JSON/log、错误记录、schema-v3 manifest、
配对协议、机制计数、权重/卸载审计及结论对应的证据路径。
未运行标 NOT_RUN，缺失标 MISSING，不填写虚构数值或 PASS。
集成人先校验完整性，再审代码与证据；其他模型或成员通过不代表当前配置通过。
