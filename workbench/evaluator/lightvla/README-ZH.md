# LightVLA 统一评估链

本目录合并本地项目截至 2026-10-31 的最高修订：`fixed-observation r6`、`closed-loop smoke r1`、`multitask paired r1`、`targeted adjudication r1` 和 `weight audit r1`。

`r6` 是固定观测评估器的最高版本；后四个目录是后续验收阶段，不是 r7-r10。阶段脚本保留原始测试逻辑，只把包路径改为可移动，并增加统一后台启动器。

阶段 04 不是必跑项：只有 03 恰好出现它所接受的单一逐任务失败模式时才能运行。阶段 05 不得把先前名称敏感指纹的 false 直接改成 true；它提供独立的数值证据和卸载恢复证据。

统一启动：

```bash
ROOT=/cloud/cloud-ssd1/lightvla_updated_acceptance_2026-10-26 \
  bash launch-stage.sh 01
ROOT=/cloud/cloud-ssd1/lightvla_updated_acceptance_2026-10-26 \
  bash status-stage.sh 01
```

启动器对 live PID 防重入，并默认拒绝覆盖已有非空结果。确有理由重跑时，必须显式设置 `FORCE=1`，并先把旧结果复制到新的归档位置。原始历史阶段脚本并非统一的 episode 级断点续跑实现；新的长闭环实验应按 Skill 的 `cloud-operations.md` 生成可恢复脚本，而不是盲目复用这些历史脚本。

