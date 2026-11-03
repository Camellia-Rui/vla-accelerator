# 服务器资源核验记录（用户回传输出）

来源：用户在 2026-11-02 从服务器终端回传的检查文本。当前本地会话 SSH 认证未通过，没有直接登录服务器；以下不是本地实际运行的 GPU 验收结果。

## 已确认资源

- 主机：`wxy@172.23.148.117`，工作目录 `/home/wxy/Rui_work`。
- 工具箱：`/home/wxy/Rui_work/vla-accelerator`，提交 `16e4055`；`scripts/run_pruning.py` 有本地修改，其 `run_real` 明确未实现。不要覆盖该工作区。
- 后端仓库：`/home/wxy/Rui_work/VLA-Pruner/VLA-Pruner`，HEAD `3d156ee29e3d15172abbb69dc3495c47e826e460`。
- 后端入口：`src/openvla-oft`。评估脚本和 `vla_pruner_srcipts/run_vla_pruner/run_object.sh` 有本地修改，另有未跟踪的实验辅助文件和日志。
- 候选环境：`/home/wxy/miniconda3/envs/openvla-oft`，Python 3.10.16、torch 2.2.0、transformers 4.47.0、accelerate 1.5.2、libero 0.1.4、robosuite 1.4.1。
- Transformers 的 editable 安装来源已确认是该后端的 `src/openvla-oft/transformers`。尚未验证实际 import 路径或运行兼容性。
- checkpoint：`/home/wxy/Rui_work/openvla-7b-oft-libero-object`，四个索引分片路径均存在，另有动作头、本体投影器、processor、tokenizer、配置及统计。未验证分片哈希或加载。
- 归一化键：`libero_object_no_noops`。checkpoint README 的 Spatial 示例不能直接用于该 Object 任务。
- checkpoint 存在多个配置备份和模型代码备份，当前远程代码需核验；不能认定为原样官方版本。
- GPU：回传快照为 8 张 H20，均有进程；没有约定可使用的 GPU。

## 已发现的接入问题

1. 工具箱当前锁定后端 `84d4b7192c77abf1585610e2f12393319b7ebff9`，与服务器 HEAD 不同。先比较关键实现，不能直接改锁文件或宣称兼容，也不能重置服务器旧 checkout。
2. `get_vla` 默认会调用 `update_auto_map` 和 `check_model_logic_mismatch`。该源码提供 `OPENVLA_SKIP_CHECKPOINT_SYNC=1` 开关，可避免这两处同步调用；这不构成对其他代码无写入行为的全面证明。仍需检查实际加载的 checkpoint 远程代码。
3. 用户修改的闭环入口启动时清空 `./experiments/logs/episodes.jsonl`，随后追加 episode。不能用同一目录并行运行或直接覆盖既有记录。
4. `get_vla_action` 会原地更新 `obs['state']` 为归一化值。配对测试必须为每次调用提供独立原始 observation 副本，避免重复归一化。
5. 返回值为 `(action_list, last_caches, result_image, metrics)`，不是单独的 action list。
6. 内部 `time_elapsed` 使用 `time.time()` 且未见显式 CUDA 同步，计时区间也排除了前面的预处理。不能直接当完整 query/wall 延迟或可靠 GPU 基准。
7. 旧 `run_object.sh` 固定 GPU 0，串行执行三个剪枝率、每任务 50 次试验。它不是小规模验收命令，不直接运行。

## 下一步

先只读比较服务器 HEAD 与工具箱固定后端的模型、剪枝内核、加载工具及常量。根据差异决定独立固定版本环境或新增兼容适配。之后再准备独立工具箱目录、输出目录及 checkpoint 加载方案，并取得实际输入、明确 GPU 使用安排后进行原生前向验收。

尚未安装依赖、下载模型、修改服务器文件或启动 GPU 实验。未得到速度、动作质量或任务成功率的新结论。
