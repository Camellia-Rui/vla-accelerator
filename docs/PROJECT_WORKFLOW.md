# 以用户项目为入口

此入口集中管理路径和运行计划，尚不自动识别任意模型，也不自动安装环境。模型 profile 必须由 Codex 阅读实际源码及配置后确认，不能凭目录名填写。

```bash
python -m core project init --project toolbox-project-run.json
```

编辑生成的 JSON。所有相对路径都相对 JSON 所在目录；python 填模型环境解释器文件，output_root 填有写权限的独立结果根目录。profile 目前支持 oft_object、lightvla_spatial 和 specprune_object，其他值明确阻止执行。specprune_object 需要一个采集目录；oft_object 需要 observation_dirs；LightVLA 专用 Spatial 使用内置采集流程。

```bash
python -m core project check --project toolbox-project-run.json --name evaluation-001
python -m core project execute --project toolbox-project-run.json --name evaluation-001
```

check 不导入模型、不运行 GPU、不安装依赖，输出候选方法、原因、缺失项及参数列表。BLOCKED 时 execute 不启动子进程。READY_FOR_RUNTIME_CHECK 仅表示资源初检通过，不证明权重完整或结构兼容。实际加载与评估仍可能拒绝后端、输入或模型；相应错误保存在评估器日志。

execute 使用指定模型环境，从工具箱根目录调用已有评估入口，结果目录必须新建。结束后在结果目录补充 project-plan.json，记录计划、命令和边界。现阶段执行固定输入评估，不自动做闭环或认定质量合格；闭环、门槛及报告判据见根 README_TESTING.md。

初检只读元数据并检查文件存在，不导入用户代码。profile 是明确声明，不是兼容性证据；即使文件存在也保持 compatibility_verified=false。后续需要补充环境探测、动作头和权重分片完整性检查，以及实际运行结果驱动的兼容状态。

## 环境和权重初检

`project check --probe-env` 在指定解释器中读取安装包版本，不导入模型、不初始化 GPU。execute 自动执行此检查；缺少基础依赖或探测超时会阻止启动。版本记录不代表自定义 Transformers 实际导入来源已经通过验证，真实评估器仍负责后端检查。

权重索引中的分片必须存在且非空，索引不得指向 checkpoint 目录外。检查不做十几 GB 权重全量哈希，不能检测所有内容损坏。运行结束后的 project-plan.json 增加退出码、接入结果、输入筛查与是否闭环信息，保留质量结论与执行状态的区别。

## 已完成的真实入口验证（2026-11-04）

使用专用 Spatial 权重，经 `project execute` 完成环境元数据检查、4 对真实固定输入评估和 project-plan.json 留档。退出码 0、integration_passed=true、closed_loop_tested=false；原始报告和代码快照见 `server-evidence/2026-11-04/project-workflow/`。这不是新增闭环证据。

本地 78 项测试通过。配置样例在 configs/project-oft-object.example.json 和 configs/project-lightvla-spatial.example.json，示例路径必须替换。新项目入口本次实际验证了 LightVLA 路线，OFT 项目配置路线尚需单独端到端验证，不能用旧底层评估结果代替。
