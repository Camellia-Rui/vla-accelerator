# 更新版 LightVLA：配对闭环 smoke r1

在通过 r6 固定真实观测预检后，本包对 LIBERO-Spatial 任务 0、1 各运行三个相同初态的 baseline/packaged 配对 episode，共 12 个 episode。

每个 episode 创建全新环境；模式顺序交替；推理使用官方双图、proprio、L1 action head 和 8-action chunk。结果记录成功率、逐次查询延迟、裁剪结构、显存、错误与权重指纹。该阶段是闭环 smoke，而不是 500-episode 最终验收。
