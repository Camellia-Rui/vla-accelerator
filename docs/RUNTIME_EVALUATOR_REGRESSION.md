# 2026-11-13 运行中评估回归

本版在服务器 `wxy@172.23.148.117` 的 `/home/wxy/Rui_work/live-evaluator-20261113` 独立目录完成五条路线。复用只读模型和环境，不属于从零安装实验。所有成功路线退出码为 0，运行结束自动产生 assessment.json、assessment.md 和 selection.json，无需用户第二次发出评估指令。

| 路线 | 本次成功回合，原生→加速 | 平均查询 ms，原生→加速 |
|---|---|---|
| OFT Object / VLA-Pruner | 1/1 → 1/1 | 137.62 → 128.56 |
| 同一 OFT Object / FastV 评分 | 1/1 → 1/1 | 137.62 → 105.57 |
| SpecPrune Object | 1/1 → 1/1 | 185.85 → 119.95 |
| 原始 OpenVLA Spatial / FastV | 8/10 → 10/10 | 240.78 → 203.25 |
| OFT Spatial / FastV | 1/1 → 1/1 | 215.77 → 193.64 |
| LightVLA Spatial | 1/1 → 1/1 | 133.06 → 77.04 |

表格来自统一描述统计，包含全部已记录查询；FastV 选择器仍沿用剔除最初五次查询的口径。SpecPrune 表格为闭环查询，选择器使用其同输入动作块计时。不同路线不可据此直接排名。

本次 VLA-Pruner 查询虽快，回合策略总耗时却由 2889.96 ms 增至 2956.85 ms，墙钟由 6.601 s 增至 7.213 s；报告保留这一结果。OFT FastV 的墙钟受运行顺序和初始化等影响，不应将小样本墙钟比例直接解释为稳定推理加速。

未指定正式质量门槛，所有结论均为 INCONCLUSIVE，未授予部署 PASS。单 seed、共享 GPU，除原始 OpenVLA 为十任务各一个状态外，其余仅任务 0、状态 0。此次验证的是工程链路，不替代多 seed、大样本和预先规定门槛的质量验收。

## 修复与审计

- 修复项目配置解引用虚拟环境 Python 软链接导致误用系统解释器的问题。
- LightVLA 增加实际 decoder 序列缩短检查；保留参数和卸载恢复检查。
- FastV 新增逐回合状态哈希与动作记录；SpecPrune 区分真实模型查询和队列调用。
- 旧环境缺失 requests、pyparsing、cycler，以及 Transformers 不兼容、CUDA SIGFPE 均被记录为失败，没有计入有效数据。仅在独立目录增加依赖层，切换兼容环境和显式 cuBLAS 配置，未改原环境。
- 最终代码对五份成功报告重新复核通过；105 项本地测试通过，包括损坏记录、缺失回合、子进程失败、无机制证据和虚拟环境路径测试。

## 原始证据

[证据包](server-evidence/2026-11-13/runtime-evaluator/evidence.zip)包含源码快照、执行命令、原始报告、动作、成功及失败日志；[汇总](server-evidence/2026-11-13/runtime-evaluator/summary.json)保存各路线原始自动报告。

证据包 SHA256：`9346c278e74f76da547c37bdaec9ba953f6850f81fa2426f7c895dca584e6fb2`。解包后 234 个清单文件哈希均已核对。权重未打包；权重清单的大小和修改时间不等于重新计算完整权重哈希。
