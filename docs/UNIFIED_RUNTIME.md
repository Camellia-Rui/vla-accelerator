# 统一推理和输入级评估

统一入口现在为 `python -m core run` 和 `python -m core evaluate`。支持真实执行的 run 方法是 `lightvla`、`vla_pruner`；evaluate 当前支持 `vla_pruner`。SpecPrune 未完成统一恢复接口，VLA-Cache 已隔离，不伪装成可运行选项。

LightVLA 的实际 Spatial 权重工厂、隔离启动命令和真实验证结果见 [LightVLA 接入](LIGHTVLA_SERVER_RESULT.md)，成对仿真闭环使用 [闭环入口](LIGHTVLA_CLOSED_LOOP.md)。

## VLA-Pruner 内置 OFT 加载器

此加载器只针对当前固定官方 OFT 后端，不是任意 VLA 自动加载器。无需修改用户原项目或在原环境执行 pip install。在工具箱根目录使用原模型环境的 Python：

```bash
python -B -m core run --method vla_pruner \
  --workspace /实际/VLA-Pruner/src/openvla-oft \
  --checkpoint /实际/checkpoint \
  --unnorm-key libero_object_no_noops \
  --observation-dir /实际/capture-directory \
  --gpu 0 --prefer-env-cuda --prune-ratio 0.5 \
  --output /实际/新输出目录
```

`--workspace` 为模型项目的 Python 导入根；相对 checkpoint 路径相对 workspace 解析。其他命令行路径相对执行命令时的工作目录。示例路径需替换成用户实际路径。`--gpu` 指定物理 GPU 编号；不指定时保留调用者的设备可见性。内置后端在可见设备内使用其原生设备配置。

`--observation-dir` 可重复传入多个目录，使用此前 capture 产生的 `observation.npz` 与 `report.json`。NPZ 包含 full_image、wrist_image、state；report 必须通过采集检查并包含输入文件 SHA256 和 task_description。它们是已经按原生流程处理的输入，不能随意用图片文件替代。

默认模式 accelerated；`--mode baseline` 关闭剪枝。run 将每个 observation 作为独立查询，在查询前 reset；不是机器人动作调度器。连续 episode 使用 Python 生命周期 API 管理状态。

内置路径先验证源码、checkpoint 代码一致性和输入哈希，再加载模型。输出 `actions.jsonl` 保存完整动作块和实际裁剪统计；`report.json` 记录成功或失败及恢复状态。错误、非有限动作和不兼容结构返回非零状态。输出目录必须是新的，避免覆盖他人记录。

## 用用户项目自己的加载函数

```bash
python -B -m core run --method vla_pruner \
  --workspace /实际/用户项目 --factory toolbox_entry:build \
  --factory-kwargs '{"checkpoint":"/实际/权重"}' \
  --config /实际/pruner.json --output /实际/新结果目录
```

这里 `toolbox_entry:build` 必须由 Codex 根据用户项目的真实加载入口编写并检查，不能凭空假定已经存在。函数返回：

```python
{
    "model": model.eval(),
    "processor": processor,
    "action_head": action_head,
    "proprio_projector": proprio_projector,
    "observations": [
        {"id": "input-0", "full_image": primary_image,
         "wrist_image": wrist_image, "state": raw_proprio,
         "instruction": task_description}
    ],
    # 以下两个函数在 evaluate 时必需：
    "native_query": native_query,  # 接收 observation，返回原生动作块
    "native_reset": native_reset,  # 无参数，清空原生时间状态
    "provenance": {"source_commit": "实际版本"}
}
```

可选 `action_runner` 遵循原生 get_vla_action 签名；不提供则使用固定后端实现。factory 与模型代码会真实执行，应先检查。评估的 native_query 必须走未经工具箱改写的原生调用路径，不能调用加速器伪造对照。

## 统一评估与阈值

将 run 换为 evaluate，保留加载参数，增加 `--pairs 6 --warmup 3` 即可。同一进程加载一次模型，对每份 observation 执行原生、关闭剪枝对照、开启剪枝配对和卸载恢复。每次查询复制输入，防止原地归一化污染下一次调用；每个模式清理状态后用同一输入预热。

可选输入筛查门槛：`--max-action-mae`、`--max-action-error`、`--min-gripper-agreement`、`--min-speedup`。未提供完整门槛时，筛查结论为 INCONCLUSIVE；不替用户编造“可接受准确率”。达到门槛只返回 INPUT_SCREEN_PASS，未达到则 INPUT_SCREEN_FAIL（退出码 3）。运行或恢复错误退出码 2。关闭剪枝对照和卸载恢复的数值容差固定为 atol=1e-6、rtol=1e-5，这是接入一致性门槛，不是动作质量门槛。

`evaluation.json` 含每输入的原生动作、实际层长度、恢复检查和配对差异；`records.jsonl` 持久化每次调用的完整动作和同步计时。当前评估仍是输入级评估，不执行机器人闭环。无论筛查结论如何，`closed_loop_quality_accepted` 都为 false。正式部署仍需任务成功率、更多初始状态及性能环境控制。

## 环境与目录隔离

统一入口使用新子进程运行模型，把 Hugging Face 动态模块、Matplotlib、CUDA、Triton、Torch、Numba、临时文件等已配置缓存放入 `--cache-dir`，默认是结果目录的 cache 子目录。保持 HOME 不变，不修改已安装包。`--prefer-env-cuda` 在 Linux 子进程优先使用当前 Python 环境的 cuDNN/cuBLAS 路径，解决已遇到的混用问题；缺少对应目录时直接报错，不自动安装。

服务器操作必须将代码、输出及 cache 路径放在 `/home/wxy/Rui_work` 内；模型和环境按需只读访问。第三方或用户自定义 factory 若另有写入路径，必须先检查并改为用户授权目录。进程原生崩溃也会保留 process.log 和退出状态，不能把无 report.json 当成功。
