# SpecPrune-VLA Accelerator

## 方法定位

SpecPrune-VLA 通过视觉 token 剪枝和动作感知控制加速 VLA 推理，无需训练。本封装面向 OpenVLA-OFT，包含 `spec_prune_accelerator.py` 和本 README；本文提供用户与 Codex 接入所需的环境、调用方式、参数约束和评测结果。

- 论文：[SpecPrune-VLA: Accelerating Vision-Language-Action Models via Action-Aware Self-Speculative Pruning](https://arxiv.org/abs/2509.05614)
- 官方代码：[alexwhz-sjtu/SpecPrune-VLA](https://github.com/alexwhz-sjtu/SpecPrune-VLA)。本目录是工程封装，不是官方仓库本体。

## 最终性能（NVIDIA H20）

以下结果使用“关键参数”中的冻结配置。成功率来自 500-episode 闭环评测；单次查询延迟和加速比来自另行开展的三轮独立测速。两种计时口径不要混算。

| Suite | 方法 | 成功率（500 episodes） | 单次查询延迟（ms） | 加速比 | 单次查询计算量（GFLOPs） | FLOPs 降幅 |
|---|---|---:|---:|---:|---:|---:|
| Object | Baseline | 486/500（97.2%） | 137.619 | 1.000× | 8876.321 | — |
| Object | Original SpecPrune | 483/500（96.6%） | 91.624 | 1.587× | 7538.461 | 15.072% |
| Object | **Accelerator** | **483/500（96.6%）** | **91.306** | **1.507×** | **7538.461** | **15.072%** |
| Goal | Baseline | 496/500（99.2%） | 137.417 | 1.000× | 8834.801 | — |
| Goal | Original SpecPrune | 483/500（96.6%） | 92.069 | 1.493× | 7510.601 | 14.988% |
| Goal | **Accelerator** | **483/500（96.6%）** | **95.264** | **1.442×** | **7510.601** | **14.988%** |

- **成功率**：10 tasks × 50 trials，seed=7；逐 episode 初始状态匹配与成败对照见下方“配对与闭环结果”。
- **延迟**：另行进行独立测速；同一 H20 顺序运行三方法，三轮轮换顺序。每轮 10 tasks × 5 trials、seed=7；每个任务第 1 次试验不计入延迟，仍计入成功率。最终延迟按全 suite `总模型查询耗时 / 总查询次数` 计算。表中延迟为三轮结果的中位数。
- **加速比**：三轮 `baseline_ms / method_ms` 的中位数，不是两列延迟中位数之比。Accelerator 范围：Object **1.470～1.519×**，Goal **1.415～1.466×**。
- **FLOPs**：基于 5 组固定随机种子（7、8、9、10、11）生成的合成视觉输入进行统计，**由 PyTorch profiler 统计的 FLOPs**，仅计 PyTorch profiler 支持的算子，不是完整理论值或真实闭环轨迹平均值。

## 配对与闭环结果

使用上表冻结配置、seed=7 和 LIBERO 默认初始状态，每套任务运行 10 tasks × 50 episodes。六组记录各有 500 条，无缺失或重复；三种方法逐 episode 的任务描述和初始状态 SHA256 全部匹配。

| Suite | 初始状态匹配（三方法） | Original 与 Accelerator 同成败 | 仅 Baseline 成功 | 仅 Accelerator 成功 |
|---|---:|---:|---:|---:|
| Object 15/12 | 500/500 | 500/500 | 12 | 9 |
| Goal 40/31 | 500/500 | 500/500 | 16 | 3 |

Original 与 Accelerator 在全部 1000 个配对 episode 中，成败、环境步数、策略步数和模型查询次数均一致。相对 Baseline，Object 成功率下降 0.6 个百分点，Goal 下降 2.6 个百分点；该结果支持封装版复现 Original 的闭环表现，存在的成功率取舍见上表。

闭环统计覆盖每组全部 500 个 episode，包括各任务首个 episode：

| Suite | 方法 | 查询/episode | 环境步/episode | 推理秒/episode | 推理 ms/query | rollout 秒/episode |
|---|---|---:|---:|---:|---:|---:|
| Object | Baseline | 19.83 | 138.83 | 3.146 | 158.676 | 30.827 |
| Object | Original | 20.37 | 140.27 | 2.013 | 98.806 | 31.065 |
| Object | Accelerator | 20.37 | 140.27 | 2.066 | 101.420 | 30.004 |
| Goal | Baseline | 16.28 | 108.46 | 2.595 | 159.409 | 34.555 |
| Goal | Original | 17.65 | 117.00 | 1.789 | 101.348 | 36.854 |
| Goal | Accelerator | 17.65 | 117.00 | 1.801 | 102.078 | 35.162 |

- **环境步**：实际执行策略动作的环境步数，不含初始稳定等待。模型仅在动作队列需要更新时查询，因此步数与查询数不同。
- **推理时延 ms/query**：累计推理时间除以累计查询数。本轮 Accelerator 相对 Baseline 为 Object 1.565×、Goal 1.562×；README 顶部的三轮独立测速采用不同样本和汇总方式，两者分别报告。
- **rollout 时间**：每个 episode 的环境重置、观测处理、推理和动作执行等墙钟耗时。本次 Object 的整集耗时略有下降，Goal 略有上升；Original 与 Accelerator 的查询时延接近，尚未体现后者的额外速度优势。

逐任务结果（每任务 50 episodes）

下表 A 表示 Accelerator；“B独有/A独有”表示相同初始状态下仅 Baseline / 仅 Accelerator 成功的次数。Original 的环境步数与查询次数逐 episode 与 A 一致。

| Suite | 任务 ID / 目标 | Baseline / Original / A 成功数 | B独有 / A独有 | A 查询/episode | A 环境步/episode | A 推理秒/episode |
|---|---|---:|---:|---:|---:|---:|
| Object | 0 alphabet soup → basket | 50 / 50 / 50 | 0 / 0 | 20.38 | 140.32 | 2.06 |
| Object | 1 cream cheese → basket | 50 / 48 / 48 | 2 / 0 | 21.22 | 140.02 | 2.14 |
| Object | 2 salad dressing → basket | 48 / 43 / 43 | 5 / 0 | 20.42 | 144.32 | 2.08 |
| Object | 3 bbq sauce → basket | 47 / 50 / 50 | 0 / 3 | 17.64 | 124.34 | 1.83 |
| Object | 4 ketchup → basket | 50 / 50 / 50 | 0 / 0 | 19.32 | 134.50 | 1.99 |
| Object | 5 tomato sauce → basket | 48 / 44 / 44 | 4 / 0 | 21.02 | 146.64 | 2.17 |
| Object | 6 butter → basket | 50 / 50 / 50 | 0 / 0 | 23.58 | 156.72 | 2.44 |
| Object | 7 milk → basket | 48 / 49 / 49 | 0 / 1 | 19.28 | 134.94 | 1.90 |
| Object | 8 chocolate pudding → basket | 46 / 50 / 50 | 0 / 4 | 22.90 | 155.46 | 2.28 |
| Object | 9 orange juice → basket | 49 / 49 / 49 | 1 / 1 | 17.96 | 125.46 | 1.76 |
| Goal | 0 open middle drawer | 50 / 48 / 48 | 2 / 0 | 23.80 | 157.42 | 2.53 |
| Goal | 1 bowl → stove | 50 / 49 / 49 | 1 / 0 | 13.24 | 89.18 | 1.40 |
| Goal | 2 wine bottle → cabinet top | 49 / 50 / 50 | 0 / 1 | 12.78 | 86.02 | 1.37 |
| Goal | 3 open top drawer, put bowl inside | 49 / 45 / 45 | 4 / 0 | 28.82 | 204.08 | 3.02 |
| Goal | 4 bowl → cabinet top | 48 / 48 / 48 | 2 / 2 | 14.86 | 95.68 | 1.50 |
| Goal | 5 push plate to stove front | 50 / 49 / 49 | 1 / 0 | 18.12 | 129.32 | 1.78 |
| Goal | 6 cream cheese → bowl | 50 / 45 / 45 | 5 / 0 | 18.92 | 122.44 | 1.86 |
| Goal | 7 turn on stove | 50 / 50 / 50 | 0 / 0 | 11.48 | 70.72 | 1.12 |
| Goal | 8 bowl → plate | 50 / 49 / 49 | 1 / 0 | 12.28 | 79.58 | 1.21 |
| Goal | 9 wine bottle → rack | 50 / 50 / 50 | 0 / 0 | 22.18 | 135.60 | 2.23 |


## 环境与安装

已测平台：Ubuntu 22.04.5、Python 3.10.14、NVIDIA H20。下表仅记录已验证环境中的相关实装依赖，不能直接作为无冲突的一键安装配置。

| 组件 | 版本 / 要求 |
|---|---|
| PyTorch / CUDA runtime | `2.6.0+cu124` / `12.4`；驱动 `570.158.01` |
| torchvision / torchaudio | `0.21.0` / `2.6.0` |
| NumPy / OpenCV | `1.26.4` / `opencv-python==4.10.0.84` |
| transformers / tokenizers | `4.40.1` / `0.19.1`；使用 [OpenVLA-OFT fork 固定提交](https://github.com/moojink/transformers-openvla-oft/tree/bc339d9ad707454c0c115970db43c260067c61ab)，不要换成同版本官方包 |
| MuJoCo | `2.3.7`，使用 OSMesa 无显示渲染 |
| TensorFlow / protobuf | `2.15.0` / `6.31.1`；兼容问题见 FAQ |
| tensorflow-datasets / tensorflow-graphics / tensorflow-metadata | `4.9.3` / `2021.12.3` / `1.21.0` |
| LIBERO / robosuite | `0.1.0` / `1.4.1`；LIBERO 源码状态见下文 |
| gym / bddl | `0.26.2` / `3.6.0` |
| 模型加载依赖 | `timm==0.9.10`、`accelerate==1.15.0`、`peft==0.11.1`、`draccus==0.8.0` |
| 其他核心依赖 | `sentencepiece==0.1.99`、`diffusers==0.39.0`、`einops==0.8.2`、`huggingface-hub==0.36.2` |
| 图像与渲染 | `Pillow==12.3.0`、`PyOpenGL==3.1.10`、`imageio==2.38.0`、`imageio-ffmpeg==0.6.0` |
| 数据与工具 | `easydict==1.13`、`cloudpickle==3.1.2`、`json-numpy==2.1.1`、`jsonlines==4.0.0` |
| 日志与绘图 | `matplotlib==3.10.9`、`wandb==0.28.0`、`rich==15.0.0` |
| dlimp | [OpenVLA fork](https://github.com/moojink/dlimp_openvla/tree/040105d256bd28866cc6620621a3d5f7b6b91b46)，提交 `040105d` |

环境还装有 `fastapi==0.141.1`、`uvicorn==0.54.0`，本地示例不启动服务；`hydra-core`、`robomimic` 未安装。

**复现边界**：已测 LIBERO 为 [`8f1084e`](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01) 加本地修改（`dirty`）。本封装已在上述环境中完成验证。性能结果对应已验证的代码与依赖组合；更换上游代码或依赖版本后，请重新运行 smoke test；若要继续引用表中的性能结果，还需按相同口径重新测评。


**代码版本**：H20 评测副本与当前交付代码仅有两处参数护栏差异：当前版拒绝 `num_open_loop_steps != 8` 和 `layer_skip=True`。正式配置使用 `8` 和 `False`，对应的推理路径相同；逐字节复现以原始运行清单中的服务器代码哈希为准。

新环境先按 [fork SETUP](https://github.com/alexwhz-sjtu/SpecPrune-VLA/blob/main/openvla-oft/SETUP.md) 和 [LIBERO 安装说明](https://github.com/alexwhz-sjtu/SpecPrune-VLA/blob/main/openvla-oft/LIBERO.md) 在独立 Python 3.10 环境准备依赖；核对上表，勿直接用旧 requirements 覆盖核心版本。用 `python -m pip check` 检查约束：除 FAQ 中的 TF/protobuf 冲突外，已测环境的 `nvidia-cusparselt-cu12==0.7.1` 也不符合 torch 2.6 所要求的 `0.6.2`。不要直接改动已冻结环境，也不要将有冲突的版本表当作安装配方。

以下是**依赖已准备好后的接入命令**，不是空环境的一键安装。在放有 `spec_prune_accelerator.py` 的项目根目录执行；已有已验收仓库跳过 clone/checkout。`--no-deps` 只注册源码，不补装依赖。

```bash
git clone https://github.com/alexwhz-sjtu/SpecPrune-VLA.git
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
git -C LIBERO checkout --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01
python -m pip install --no-deps -e ./SpecPrune-VLA/openvla-oft -e ./LIBERO
sudo apt-get install -y libosmesa6 libosmesa6-dev
export PROJECT_ROOT="$PWD"
export OFT_ROOT="$PROJECT_ROOT/SpecPrune-VLA/openvla-oft"
export PYTHONPATH="$PROJECT_ROOT:$OFT_ROOT:$PROJECT_ROOT/LIBERO"
export PYTHONNOUSERSITE=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export PYTHONHASHSEED=7
export CUDA_VISIBLE_DEVICES=0
cd "$OFT_ROOT"
```

后续命令在同一终端、OpenVLA-OFT fork 根目录运行，供 loader 定位模型实现。LIBERO 须配置 BDDL、assets、初始状态路径，无需训练演示数据。剪枝依赖 eager attention 的权重及 token 位置。

## 模型权重

从下列 Hugging Face 仓库完整下载，保留原始文件名：

| Suite | 模型仓库 | action head / proprio projector 步数 |
|---|---|---:|
| Object | [moojink/openvla-7b-oft-finetuned-libero-object](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-object/tree/main) | `150000` |
| Goal | [moojink/openvla-7b-oft-finetuned-libero-goal](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-goal/tree/main) | `50000` |

```bash
hf download moojink/openvla-7b-oft-finetuned-libero-object --local-dir "$PROJECT_ROOT/models/openvla-7b-oft-finetuned-libero-object"
hf download moojink/openvla-7b-oft-finetuned-libero-goal --local-dir "$PROJECT_ROOT/models/openvla-7b-oft-finetuned-libero-goal"
```

`hf` 来自 `huggingface_hub`；以上命令将两个模型分别放入项目的 `models/` 目录。

每个模型必须完整包含：4 个 `model-*-of-00004.safetensors`、`model.safetensors.index.json`、`config.json`、`dataset_statistics.json`、tokenizer/processor 配置与文件、`configuration_prismatic.py`、`modeling_prismatic.py`、`processing_prismatic.py`、`lora_adapter/`，以及对应的 `action_head--步数_checkpoint.pt` 和 `proprio_projector--步数_checkpoint.pt`。**Object 与 Goal 的辅助权重不能互相复制或改名。**

## 最小调用示例

将下面代码保存为项目根目录的 `quickstart.py`，作为调用示例和 smoke 入口。默认运行 Object 第 0 个 task 的第 0 个初始状态，设置 `SUITE=libero_goal` 可切换 Goal。`step()` 接收主/腕 RGB 图像（`uint8 H×W×3`）和 8 维 proprio，返回 7 维动作及 `info`；图像方向由 LIBERO helper 处理。

```python
import os
import random
import numpy as np
import torch
from libero.libero import benchmark
from experiments.robot.openvla_utils import (
    get_vla, get_processor, get_action_head, get_proprio_projector,
)
from experiments.robot.libero.libero_utils import (
    get_libero_env, get_libero_dummy_action, get_libero_image,
    get_libero_wrist_image, quat2axisangle,
)
from spec_prune_accelerator import (
    AcceleratorConfig, SpecPruneAccelerator, get_final_precise_topk,
)
random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
assert torch.cuda.is_available(), "需要可用的 PyTorch CUDA 环境"
suite = os.environ.get("SUITE", "libero_object")
max_steps = {"libero_object": 330, "libero_goal": 380}[suite]
p, w = get_final_precise_topk(suite)
cfg = AcceleratorConfig(unnorm_key=suite, attn_topk_precise=p,
                        attn_topk_precise_wrist=w)
cfg.pretrained_checkpoint = os.path.join(
    os.environ["PROJECT_ROOT"], "models", "openvla-7b-oft-finetuned-" + suite.replace("_", "-"))
cfg.task_suite_name = suite
model = get_vla(cfg)
processor = get_processor(cfg)
action_head = get_action_head(cfg, model.llm_dim)
proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
acc = SpecPruneAccelerator.install(model, processor, action_head=action_head,
    proprio_projector=proprio_projector, config=cfg)
print(f"CONFIG suite={suite} precise_topk={cfg.attn_topk_precise}/{cfg.attn_topk_precise_wrist}")
acc.describe()
tasks = benchmark.get_benchmark_dict()[suite]()
env, text = get_libero_env(tasks.get_task(0), "openvla", resolution=256)
success = False
env_steps = 0
try:
    env.reset()
    obs = env.set_init_state(tasks.get_task_init_states(0)[0])
    acc.start_episode()
    for t in range(max_steps + 10):
        if t < 10:  # LIBERO 初始稳定等待
            obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
            continue
        proprio = np.concatenate((obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
        action, info = acc.step(get_libero_image(obs),
            get_libero_wrist_image(obs), text=text, proprio=proprio)
        assert action.shape == (7,) and np.isfinite(action).all()
        if info["execute_action"]:
            obs, _, done, _ = env.step(action.tolist())
            env_steps += 1
            if done:
                success = True
                break
    stats = acc.end_episode()
    assert stats["infer_steps"] > 0 and env_steps > 0
    print(stats)
finally:
    env.close()
print(f"SMOKE_COMPLETED suite={suite} success={success}")
print("EXIT_CODE=0")
```

每个 episode 都要调用 `start_episode()`；`install()` 接收已加载模型，不接受路径。

## 关键参数

> **Precise 模式参数：LIBERO-Object 使用 15/12，LIBERO-Goal 使用 40/31。**
> `AcceleratorConfig` 默认是 15/12，因此运行 Goal 时请通过`get_final_precise_topk(suite)` 获取并传入对应参数。

前八项属于 `AcceleratorConfig`，其余大写参数为模块常量。正式复现使用表中值；其他取值只用于另行验证的消融实验。

| 参数 | 含义 | 正式值 / 合理取值 |
|---|---|---|
| `token_prune` | 浅层文本引导与帧差剪枝 | `True`；布尔值 |
| `dynamic_prune` | 深层动态剪枝 | `True`；布尔值 |
| `layer_skip` | 跳层开关 | `False`；不推荐开启，未验证为可用加速选项 |
| `action_adapter` | draft/precise 动作感知切换 | `True`；关闭会改变控制时序 |
| `attn_topk_precise` | precise 主视图预算 | Object `15`，Goal `40`；非负整数，不超过可选 patch 数（每视图初始 256） |
| `attn_topk_precise_wrist` | precise 腕视图预算 | Object `12`，Goal `31`；同上 |
| `num_open_loop_steps` | 动作队列 chunk 长度 | 固定 `8`，与模型输出一致 |
| `image_resize` | policy 图像尺寸 | 固定 `224`，即 224×224 |
| `STATIC_PRUNE_RATIO` | 静态 Top-K 缩放系数 α | `0.8`；通常 `(0,1]`，不是全模型剪除比例 |
| `DYNAMIC_PRUNE_RATIO` | 动态剪枝保留比例 | `0.9`；通常 `(0,1]`，不是剪掉 90% |
| `PRIMARY_SIM_THRESHOLD` | 主视图帧间 patch 相似度阈值 | `0.986`；范围 `[0,1]` |
| `WRIST_SIM_THRESHOLD` | 腕视图帧间 patch 相似度阈值 | `0.98`；范围 `[0,1]` |
| `VELOCITY_THRES` | 进入 precise 的平移动作范数阈值 | `0.35`；保持正式值 |
| `XY_THRES` | XY 动作范数阈值 | `0.1`；保持正式值 |
| `Z_THRES` | Z 动作分量阈值（带符号） | `0.1`；保持正式值 |
| `ROT_THRES` | 旋转动作范数阈值 | `0.2`；保持正式值 |

Draft 模式使用固定 Top-K 预算：`主视图 24、腕视图 19`。Precise 模式采用 Object 15/12、Goal 40/31；如需调整，请在 Python 调用代码中设置 `attn_topk_precise` 和 `attn_topk_precise_wrist`。`libero_spatial` 和 `libero_10` 尚无正式配置，返回的默认值 15/12 不代表已验证的推荐参数。

控制器比较动作值而非实测物理速度：平移、XY、旋转取范数，Z 不取绝对值。进入 precise 用上述阈值；退出还需动作队列为空，且 Z > `0.2` 或旋转范数 > `0.3`。不要把进入与退出条件混为一个阈值。

## 常见问题

| 现象 | 原因 | 解决方法 |
|---|---|---|
| Goal 显示 `15/12` | 使用了 base 默认值或显式覆盖 | 按“关键参数”中的 Precise 模式配置选择 40/31，检查 `CONFIG` 和 `acc.describe()` |
| α 仍为 `0.6` | 混入旧版实验配置 | 使用冻结版 `0.8`；检查原版对照 fork 也一致 |
| Goal 找不到 `150000` action head | 套用了 Object 文件名 | Goal 使用 `50000`，下载完整对应模型，不改名凑文件 |
| `MessageFactory.GetPrototype` / TensorFlow CUDA 提示 | TF 2.15 要求 protobuf ≥3.20.3、<5（排除 4.21.0～4.21.5），与实装 6.31.1 冲突；CUDA 重复注册是另一类提示 | 在独立环境协调 TF、protobuf、tensorflow-metadata 等约束，运行 `pip check` 和 smoke；不能仅凭版本能打印就忽略错误 |
| MuJoCo 无显示渲染失败 | OSMesa 缺失或后端未在 import 前设置 | 安装 `libosmesa6`，启动前设置两项 `osmesa` 环境变量；同时检查 LIBERO assets 路径 |
| PyTorch 2.6 读取初始状态报 `weights_only` 错误 | `torch.load` 默认行为变化 | 仅对可信 LIBERO 初始状态文件，在加载处显式 `torch.load(path, weights_only=False)`；无需全局关闭安全加载 |
| 出现 `config.json.back.*` 或并发加载互相影响 | `get_vla()` 会备份/修改 config，并可能同步模型实现文件 | 每个进程使用独立 checkpoint shadow；仅权重可 hardlink，config、模型 `.py` 等可写文件须独立复制 |
| 每次 `step()` 后直接 `env.step()` 导致结果不一致 | action_adapter 前 9 个 acting steps 只消耗动作队列 | 仅在 `info["execute_action"]` 为真时推进环境 |

## 验证与成功标准

完成环境设置，并将上面的示例保存为 `quickstart.py` 后，在同一终端执行：

```bash
python -m py_compile "$PROJECT_ROOT/spec_prune_accelerator.py" "$PROJECT_ROOT/quickstart.py"
SUITE=libero_object python "$PROJECT_ROOT/quickstart.py"
SUITE=libero_goal python "$PROJECT_ROOT/quickstart.py"
```

`py_compile` 应无报错并退出 0。两套 smoke 分别检查以下输出：

| Suite | 必须出现的输出 |
|---|---|
| Object | `CONFIG suite=libero_object precise_topk=15/12`；`SMOKE_COMPLETED suite=libero_object` |
| Goal | `CONFIG suite=libero_goal precise_topk=40/31`；`SMOKE_COMPLETED suite=libero_goal` |

成功标准：参数正确、模型加载及 LIBERO 运行正常、动作检查通过，有实际模型查询和环境动作；程序正常运行结束，没有 Python 异常报错，且退出码为 0。`success=False` 只表示该 episode 未完成任务。失败时依次检查导入路径、依赖、权重、渲染和生效参数。

封装版已通过普通模式和精细模式下与原版的数值一致性验证。本示例仅检查模型加载、推理和环境交互是否正常，不重复进行数值对比，也不用于测量正式成功率或加速比。
