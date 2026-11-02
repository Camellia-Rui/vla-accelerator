# 更新版 LightVLA：多任务配对闭环 r1

只有 r6 固定真实观测预检和前一阶段闭环 smoke 均通过后，本包才会启动。它对 LIBERO-Spatial 任务 0、1、2、3 各运行五个相同初态的 baseline/packaged 配对 episode，共 40 个 episode。

每个 episode 创建全新环境；模式顺序交替；推理使用官方双图、proprio、L1 action head 和 8-action chunk。结果记录总体及逐任务成功率、逐次查询延迟、裁剪结构、显存、错误与权重指纹。总体和每个任务的 packaged 成功数均不得低于 baseline。
