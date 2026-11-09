# FastV 在 OpenVLA 与 OpenVLA-OFT 上的加速实验

本文合并原始 `README-oprnvla_fastV.md` 与 OpenVLA-OFT 的实测记录，包含方法、正式结果、环境、执行方式和结果路径。第一部分保留原 OpenVLA 文档的环境与复现细节，第二部分补充 OFT。两套 Conda 环境独立使用，不能把第一部分的依赖版本直接套到第二部分。

FastV 在各自相同模型检查点下进行 Baseline / 剪枝对比，不需要重新训练。项目是基于 [FastV](https://github.com/pkunlp-icler/FastV) 的工程适配，不是官方发布的 OpenVLA/OFT 插件。源码和权重位于服务器，此 README 不附带它们。

## 正式结果总览

共同条件：共享 NVIDIA H20、BF16、seed=7、LIBERO-Spatial 共 10 个任务，每任务 50 回合。每条实验线的两种方法各运行 500 回合。FastV 使用 K=2、视觉 token 剪枝 50%。

| 指标 | OpenVLA Baseline | OpenVLA + FastV | OFT Baseline | OFT + FastV |
|---|---:|---:|---:|---:|
| 成功回合数 | 421/500 | 434/500 | 492/500 | 497/500 |
| 成功率 | 84.20% | 86.80% | 98.40% | 99.40% |
| 成功率差 | — | +2.60 个百分点 | — | +1.00 个百分点 |
| 平均推理延迟，排除整次最初 5 次调用 | 242.32 ms/动作 | 215.57 ms/动作 | 291.68 ms/动作块 | 222.73 ms/动作块 |
| 延迟降低 | — | 11.04% | — | 23.64% |
| 加速比 | 1.000× | 1.124× | 1.000× | 1.310× |
| 固定输入矩阵乘法与卷积计算量 | 4,390.880 GFLOPs/动作 | 2,773.818 GFLOPs/动作 | 9,029.542 GFLOPs/动作块 | 5,733.774 GFLOPs/动作块 |
| FLOPs 降幅 | — | 36.83% | — | 36.50% |
| 峰值 PyTorch 已分配显存 | 14.45 GiB | 14.33 GiB | 14.719304 GiB | 14.604370 GiB |

**计量单位必须区分：OpenVLA 每次预测一个 7 维动作，OFT 每次预测包含 8 个 7 维动作的动作块。** 两条实验线的检查点、输入、预测机制与固定 FLOPs 样例不同，不能直接相除跨模型延迟或 FLOPs 作为加速比。动作块耗时除以 8 只能作为摊销计算，不能称为实测单动作响应延迟或控制周期。

原版使用 eager 因果注意力，OFT 使用 eager **双向注意力**；必须保持各自模型所需语义。仅设置 eager 不能保证符合 OFT。

成功率提升为本次单 seed 观察，未证明 FastV 普遍提升成功率。两种方法顺序运行于共享 GPU，轨迹和外部负载可不同，延迟结论限定于本次测试条件。FLOPs 仅计 profiler 支持的矩阵乘法和卷积，乘加为 2 FLOPs，不是全部算子总量或全任务平均值。

---

# 第一部分：原版 OpenVLA + FastV

## 方法定位

本项目将 FastV 视觉 token 剪枝适配到标准 OpenVLA，无需重新训练模型，通过减少语言模型中参与计算的视觉 token 降低推理开销。本文提供环境、调用方式、参数约束与已验证的 LIBERO-Spatial 结果。

方法来源：[FastV 官方代码](https://github.com/pkunlp-icler/FastV)。基础模型：[OpenVLA](https://github.com/openvla/openvla)。评测基准：[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)。本项目是工程适配，不是 FastV 官方发布的 OpenVLA 插件；本部分只描述原版 OpenVLA；OpenVLA-OFT 的适配与结果见第二部分。当前不包含动作感知控制。

## 性能结果（NVIDIA H20）

正式配置：BF16、eager attention、seed=7；LIBERO-Spatial 共 10 个任务，每任务 50 回合。

| 方法 | 成功率（500 回合） | 平均动作延迟 | 加速比 | 固定输入计算量/动作 | FLOPs 降幅 |
|---|---:|---:|---:|---:|---:|
| Baseline | 421/500（84.2%） | 242.32 ms | 1.000× | 4,390.880 GFLOPs | — |
| FastV，K=2，剪枝 50% | 434/500（86.8%） | 215.57 ms | 1.124× | 2,773.818 GFLOPs | 36.83% |

本次 FastV 多成功 13 回合，成功率提高 **2.6 个百分点**，平均动作延迟降低 **11.04%**。

**统计口径：**

- 成功率来自单个 seed 的两组 500 回合实验，未进行多 seed 或显著性检验。
- 延迟来自共享 GPU 上的闭环评测，包含动作入口内的图像处理、处理器和模型推理；计时前后同步 CUDA，不含环境执行和视频保存。表中排除整次运行最初 5 个动作，属于事后汇总。两组轨迹和 GPU 外部负载可能不同，不是隔离环境下的固定输入测速。
- FLOPs 使用相同图像与指令，统计一次完整 7 维动作预测，包括视觉编码、投影、语言模型预填充及 6 次后续解码。采用 PyTorch profiler 支持的矩阵乘法和卷积计数，乘加为 2 FLOPs；不包含全部算子，也不是整个任务集的平均值。统计时两种方法均展开视觉注意力矩阵运算，该过程不用于测延迟。

| 补充指标 | Baseline | FastV |
|---|---:|---:|
| 动作记录数 | 62,851 | 61,444 |
| 原始平均延迟 | 242.33 ms | 215.58 ms |
| 原始中位数延迟 | 201.59 ms | 179.37 ms |
| 原始 P95 延迟 | 462.93 ms | 428.71 ms |
| 峰值 PyTorch 已分配显存 | 14.45 GiB | 14.33 GiB |

显存为 `torch.cuda.max_memory_allocated()`，不是 `nvidia-smi` 的进程占用。

### 逐任务成功数

以下任务均要求将指定位置的黑碗放到盘子上，每任务 50 回合。

| ID | 黑碗位置 | Baseline | FastV |
|---|---|---:|---:|
| 0 | 盘子与小烤碗之间 | 45/50 | 42/50 |
| 1 | 小烤碗旁 | 46/50 | 45/50 |
| 2 | 桌子中央 | 44/50 | 46/50 |
| 3 | 饼干盒上 | 48/50 | 49/50 |
| 4 | 木柜顶部抽屉中 | 34/50 | 35/50 |
| 5 | 小烤碗上 | 45/50 | 46/50 |
| 6 | 饼干盒旁 | 47/50 | 48/50 |
| 7 | 炉子上 | 43/50 | 40/50 |
| 8 | 盘子旁 | 39/50 | 46/50 |
| 9 | 木柜上 | 30/50 | 37/50 |

## 实现与文件

`K=2` 表示前两层处理完整序列，后续层使用剪枝后的序列。剪枝依据前一层最后一个 query 对视觉 token 的注意力评分，跨 head 平均后选择并按原顺序保留 token。

本模型融合 DINOv2 与 SigLIP 特征，projector 输出 **256 个视觉 token**。剪枝 50% 后保留 128 个；示例预填充缓存由前两层的 291 缩为后 30 层的 163，后续解码各层缓存逐次增长 1。总序列长度随任务指令变化。视觉编码和投影仍完整执行，因此视觉 token 减半不等于全模型计算量减半。

| 文件 | 用途 |
|---|---|
| `experiments/robot/fastv_openvla.py` | FastV forward 与因果掩码适配 |
| `run_libero_fastv.py` / `.sh` | 模式选择、模型适配与运行环境设置 |
| `experiments/robot/openvla_utils.py` | 本地模型和处理器加载，BF16 eager attention |
| `prismatic/extern/hf/modeling_prismatic.py` | prompt token 与 attention mask 对齐修复 |
| `experiments/robot/libero/run_libero_eval.py` | 评测及逐动作延迟、显存记录 |
| `profile_openvla_flops.py` | 固定输入 FLOPs 对比 |
| `summarize_libero_n50.py` | 整体、逐任务、逐回合汇总与校验 |

这些文件位于服务器工程，本文档不附带源码或模型权重。

## 环境与模型

已验证平台为 Linux、Python 3.10.22、NVIDIA H20，驱动 570.158.01。下表是已测版本记录，不是空环境的一键安装配方。

| 组件 | 版本 / 设置 |
|---|---|
| PyTorch / CUDA runtime | 2.2.0+cu121 / 12.1 |
| torchvision / torchaudio | 0.17.0+cu121 / 2.2.0+cu121 |
| 推理 cuBLAS | 隔离安装 12.4.5.8，启动时预加载 |
| NumPy | 1.26.4 |
| transformers / tokenizers | 4.40.1 / 0.19.1 |
| timm / accelerate / peft | 0.9.10 / 0.28.0 / 0.11.1 |
| TensorFlow / protobuf | 2.15.0 / 3.20.3 |
| tensorflow-datasets / tensorflow-metadata | 4.9.3 / 1.15.0 |
| tensorflow-graphics / tensorflow-probability | 2021.12.3 / 0.23.0 |
| LIBERO / robosuite / MuJoCo | 0.1.0 / 1.4.1 / 2.3.7 |
| gym / bddl / future | 0.25.2 / 1.0.1 / 0.18.2 |
| 无显示渲染 / attention | EGL / eager；未安装 FlashAttention |

OpenVLA 基础提交为 `c8f03f48af692657d3060c19588038c7220e9af9`，包含上述本地修改；dlimp 使用 `moojink/dlimp_openvla` 提交 `040105d256bd28866cc6620621a3d5f7b6b91b46`。FastV 和 LIBERO 的精确提交未在本轮冻结，发布可复现代码包时需补存版本与 diff。

模型：[openvla/openvla-7b-finetuned-libero-spatial](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial)，revision 为 `962318cec55ac10993ff0f5f43eda9a270b4c873`。完整保留 4 个权重分片、索引、配置及 tokenizer/processor 文件，反归一化 key 为 `libero_spatial`。

| 资源 | 本次服务器路径 |
|---|---|
| Conda 环境 | `/data2/wxy/VLA-Adapter/conda_envs/openvla-fastv` |
| 工程 | `/home/wxy/daisy_work/openvla-fastv-code` |
| LIBERO | `/home/wxy/daisy_work/LIBERO` |
| 模型 | `/home/wxy/daisy_work/checkpoints/openvla-7b-finetuned-libero-spatial` |

启动器目前固定使用上述模型和 LIBERO-Spatial，迁移机器需要修改相应路径。LIBERO 使用工程的 `.libero-config/config.yaml`，配置 BDDL、assets、初始状态等目录，无需训练演示数据。本次另通过 site-packages 中的 `libero_source_path.pth` 注册 LIBERO 仓库根目录。

## 安装与运行

以下命令适用于依赖已准备好的环境；`--no-deps` 仅注册源码。

```bash
conda activate /data2/wxy/VLA-Adapter/conda_envs/openvla-fastv
cd ~/daisy_work/openvla-fastv-code
python -m pip install -e . --no-deps
python -m pip install -e ~/daisy_work/LIBERO --no-deps
```

本环境原有 cuBLAS 12.1 在 BF16 单 token Linear 中出现 SIGFPE，隔离使用 cuBLAS 12.4 后通过验证。已准备好的工程无需重复安装：

```bash
python -m pip install --no-deps \
  --target ./runtime-libs/cublas-12.4 \
  'nvidia-cublas-cu12==12.4.5.8'
```

启动脚本配置隔离 cuBLAS、离线模型、EGL 和 LIBERO 路径，并禁用 TensorFlow GPU，仅由 PyTorch 使用 GPU。

### 小规模检查

```bash
EVAL_GPU=3 bash run_libero_fastv.sh --mode baseline --trials 1 --seed 7
EVAL_GPU=3 bash run_libero_fastv.sh \
  --mode fastv --trials 1 --seed 7 --k 2 --prune-ratio 0.5
```

### 正式评测：每种方法 500 回合

```bash
EVAL_GPU=3 bash run_libero_fastv.sh \
  --mode baseline --trials 50 --seed 7

EVAL_GPU=3 bash run_libero_fastv.sh \
  --mode fastv --trials 50 --seed 7 --k 2 --prune-ratio 0.5
```

按顺序执行，不要在同一 GPU 上并发测速。`EVAL_GPU` 为物理 GPU 编号，按实际资源修改。

## 关键参数

| 参数 | 含义 | 正式值 / 约束 |
|---|---|---|
| `--mode` | 推理模式 | 必填：`baseline` 或 `fastv` |
| `--trials` | 每任务回合数 | 50；范围 1～50 |
| `--seed` | 随机种子 | 7 |
| `--k` | 剪枝层索引 | 2；FastV 要求 1 ≤ K < 层数 |
| `--prune-ratio` | 剪去的视觉 token 比例 | 0.5；范围 [0,1)，至少保留 1 个 token |
| `image_token_start_index` | 视觉 token 起点 | 当前为 1，即 BOS 后 |
| `image_token_length` | 视觉 token 数量 | 动态捕获；本模型为 256 |
| `center_crop` | 中心裁剪 | 当前启动器固定为 True |
| 8bit / 4bit 加载 | 量化 | 正式实验均关闭 |

保留数量为 `round(V × (1-r))`。当前只验证 batch size 1、无 padding 和动态 KV cache；不支持缓存后的多 token 输入。训练、静态缓存和其他 attention 后端未验证。更改模型、套件或参数后需重新评测。

## FLOPs 与结果汇总

确认已有测试图 `experiments/logs/libero_render_check.png` 后，在已激活的环境中执行：

```bash
env -u LD_PRELOAD \
  PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
  LD_PRELOAD="$PWD/runtime-libs/cublas-12.4/nvidia/cublas/lib/libcublasLt.so.12:$PWD/runtime-libs/cublas-12.4/nvidia/cublas/lib/libcublas.so.12" \
  LD_LIBRARY_PATH="$PWD/runtime-libs/cublas-12.4/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=2 PYTHONFAULTHANDLER=1 \
  python -u profile_openvla_flops.py --k 2 --prune-ratio 0.5

python summarize_libero_n50.py
```

FLOPs 成功标记为 `OpenVLA FLOPs comparison: OK`；汇总成功标记为 `Formal evaluation data validation: OK`。汇总脚本绑定本次两组运行文件名，新评测后应更新 `runs`。FLOPs 重复运行会覆盖同名输出。

结果保存在工程的 `experiments/logs/`：

| 文件 | 内容 |
|---|---|
| `EVAL-…-actions.csv` / `EVAL-….txt` | 原始动作指标 / 回合成败日志 |
| `baseline-vs-fastv-n50-seed7-summary.csv` | 整体成功率、延迟、显存及固定输入 FLOPs |
| `baseline-vs-fastv-n50-seed7-per-task.csv` | 20 条逐任务统计 |
| `baseline-vs-fastv-n50-seed7-per-episode.csv` | 1,000 条逐回合统计 |
| `openvla-flops-k2-r0.5-summary.csv` | FLOPs 对比 |
| `openvla-flops-k2-r0.5-operators.csv` | 算子计数明细 |
| `openvla-flops-k2-r0.5-metadata.json` | 统计范围、输入及缓存记录 |

动作 CSV 字段为 `task_id`、`episode_idx`、`action_idx`、`latency_s`、`peak_allocated_gib`。成功率从日志读取，FLOPs 从独立统计文件读取。视频位于 `rollouts/`。

## 常见问题与验证

| 现象 | 处理 |
|---|---|
| BF16 推理出现 Floating point exception | 检查隔离 cuBLAS 12.4 是否实际加载；本环境仅改系统 nvcc 无法解决。 |
| 离线仍下载 `processing_prismatic.py` | 使用工程已修改的本地模型和处理器加载入口，检查文件完整性。 |
| 找不到 `libero` | 检查 namespace package 和 `libero_source_path.pth` 是否指向仓库根目录。 |
| BDDL 缺少 `future` | 本次使用 `future==0.18.2`。 |
| attention mask 长度不匹配 | 追加 prompt token 时同步扩展 mask，使用已修复的模型代码。 |
| EGL 或资源加载失败 | 检查渲染环境变量、GPU、LIBERO 配置和 assets。 |
| TF 重复注册、Gym 或 macro 警告 | 本次未阻止评测完成；依据实际异常与退出状态判断，不为消除提示随意升级冻结依赖。 |

已通过小型 Llama 无剪枝数值对齐、剪枝缓存及连续解码检查、实际 OpenVLA 单动作检查、两组 500 回合记录校验和固定输入 FLOPs 重复测量。尚未做逐回合初始状态 SHA256 配对验证。

迁移后先做语法和小规模闭环检查：

```bash
python -m py_compile experiments/robot/fastv_openvla.py \
  run_libero_fastv.py profile_openvla_flops.py summarize_libero_n50.py
bash -n run_libero_fastv.sh
```

成功标准：模型与环境正常加载，剪枝参数和缓存符合预期，输出有限的 7 维动作，指标文件生成且程序正常退出。`Success: False` 仅表示该回合未完成任务；软件异常应停止排查，不应计为普通任务失败。


---

# 第二部分：OpenVLA-OFT + FastV

## 方法与正式配置

基础项目：[moojink/openvla-oft](https://github.com/moojink/openvla-oft)。检查点：[moojink/openvla-7b-oft-finetuned-libero-spatial](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial)。

| 参数 | 本次正式设置 |
|---|---|
| 任务 | LIBERO-Spatial，10 个任务，每任务 50 回合 |
| seed | 7 |
| 模型精度 | BF16，不使用 4bit / 8bit 量化 |
| Attention | eager，双向注意力 |
| 相机 | 第三人称 + 腕部，2 路图像 |
| 本体状态 | use_proprio=True |
| 动作头 | L1 连续动作回归，use_l1_regression=True |
| diffusion / FiLM | 均关闭 |
| 中心裁剪 | center_crop=True |
| 开环步数 | num_open_loop_steps=8 |
| 动作输出 | 一次前向输出 `(8, 7)`，56 个动作相关 token |
| KV cache | 关闭，不进行自回归单 token 解码 |
| FastV | K=2，prune_ratio=0.5 |
| GPU | 物理 GPU 3，NVIDIA H20 |
| 渲染 | EGL，已验证设备编号 3 |
| 正式视频 | 不保存，避免磁盘增长和编码开销 |

## OFT 正式结果

| 指标 | Baseline | FastV |
|---|---:|---:|
| 成功回合 | 492/500 | 497/500 |
| 成功率 | 98.40% | 99.40% |
| 平均动作块延迟，排除最初 5 次调用 | 291.682139 ms | 222.732673 ms |
| 中位动作块延迟，同上 | 296.356369 ms | 227.699276 ms |
| P95 动作块延迟，同上 | 300.444501 ms | 240.067264 ms |
| 原始平均动作块延迟 | 291.772502 ms | 222.833454 ms |
| 原始中位动作块延迟 | 296.357250 ms | 227.699330 ms |
| 原始 P95 动作块延迟 | 300.451680 ms | 240.086318 ms |
| 动作块推理调用数 | 6,659 | 6,584 |
| 排除前 5 次后的调用数 | 6,654 | 6,579 |
| 峰值 PyTorch 已分配显存 | 14.719304 GiB | 14.604370 GiB |
| 固定输入矩阵乘法与卷积 FLOPs/动作块 | 9,029.542 GFLOPs | 5,733.774 GFLOPs |

平均延迟降低 **23.64%**，加速比 **1.310×**；FLOPs 降低 **36.50%**；显存减少约 **0.115 GiB**。成功率增加 **1.00 个百分点**，多完成 5 回合；这不等于已证明策略能力普遍提升。

### OFT 逐任务成功数

| 任务 ID | Baseline | FastV |
|---|---:|---:|
| 0 | 50/50 | 50/50 |
| 1 | 50/50 | 50/50 |
| 2 | 50/50 | 50/50 |
| 3 | 49/50 | 50/50 |
| 4 | 48/50 | 48/50 |
| 5 | 48/50 | 49/50 |
| 6 | 49/50 | 50/50 |
| 7 | 49/50 | 50/50 |
| 8 | 50/50 | 50/50 |
| 9 | 49/50 | 50/50 |
| 合计 | 492/500 | 497/500 |

正式两组结果已经核验：共同配置、记录范围内的源码 SHA256、全部 500 个回合覆盖与无重复回合、动作记录数、对应初始状态 SHA256 一致。

正式脚本源码哈希清单没有覆盖全部仓库，例如 `oft_attention.py` 未纳入当时正式脚本的清单，其哈希在后续 FLOPs 元数据中保存。不能把这些核验称为整个环境和所有源码已完整冻结。

## 适配原理

### 视觉剪枝与动作位置

两路图像共 512 个视觉 token，K=2 时前两个解码层处理完整序列，在进入第 3 层前将视觉 token 剪到 256 个。视觉区间从 BOS 后索引 1 开始，数量按模型动态读取。本体状态、文本和动作 token 均在剪枝区间之外保留。

评分沿用旧 FastV：第 2 层最后一个 query 对视觉 token 的注意力跨 head 平均，选择 top-k，再按原顺序保留；不是对全部动作 query 聚合。保留 token 的 RoPE 位置使用原始位置。

官方样例序列总长度 605 → 349；前两层输入长度 605，其余 30 层为 349。总长度随任务指令改变。视觉编码与投影仍完整执行，减少一半视觉 token 不代表全模型计算量减半。

`_regression_or_discrete_prediction()` 根据原输入长度减去解码器最终输出长度，得到实际删除数，修正动作特征切片位置。Baseline 删除数为 0，保持原位置。OFT 语言模型只前向一次，不使用原 OpenVLA 的 7 次自回归调用或 KV cache 检查。

### 必须修正 eager 双向注意力

OFT 项目声明使用官方 Transformers fork 支持双向并行动作解码。本次 fork 的 SDPA 分支把因果掩码最后一行广播到全部 query，设置 `is_causal=False`；eager 分支本身不做相同转换。

本地新增 `experiments/robot/oft_attention.py`，只修改当前模型实例：在 batch size 1、无 padding、无历史 cache 的范围内，用 `attention_mask=None` 为 eager 提供全双向注意力；输入约束不满足时拒绝运行。FastV 剪枝前后同步使用同样的双向语义。不修改已安装 Transformers 的全局文件，不改变原版 OpenVLA 的因果注意力。

**已验证**：全部 32 层没有因果屏蔽；零剪枝与修正后 Baseline 的样例动作最大绝对差为 0.0；50% 剪枝的层位置与 token 数正确；输出为有限的 `(8, 7)`。

上述检查不是 eager 与官方 SDPA 的全模型跨后端数值一致性测试；这种对照尚未在本次记录中完成。

### 排查阶段结果与正式结果分开

早期 OFT 在错误因果注意力下 Baseline 为 290/500（58.0%），固定输入延迟 131.33 → 99.19 ms、降低 24.47%。这些记录不代表正确 OFT 配置，不混入正式结果。

修正后覆盖全部任务的 50 回合检查：Baseline 49/50，FastV 50/50；随后才运行正式两组 500 回合。本 README 的 OFT 正式指标仅采用修正后的指定目录。

## OFT 统计口径

- 延迟包裹评测脚本的 `get_action()`，包含内部策略预处理与动作生成；前后同步 CUDA。
- 外部 `prepare_observation()`、`env.step()`、日志写入与视频保存不计入；正式运行不保存视频。
- 每种方法排除整次运行最初 5 次推理，不是每回合排除 5 次，也不是动作块中的前 5 个动作。
- 原 `run_episode()` 会捕获异常返回失败；正式记录脚本将 `Episode error:` 转为停止运行，不将程序错误静默当作任务失败。
- 显存为模型加载后重置峰值计数、评测期间的 `torch.cuda.max_memory_allocated()`，不等于全进程、整卡或其他用户显存。
- FLOPs 固定输入为官方 `sample_libero_spatial_observation.pkl`，与第一部分旧项目测试图不同。
- FLOPs 覆盖两路视觉编码、投影器、本体状态投影器、语言模型及动作头中的矩阵乘法与卷积，一次输出完整 8 动作块。
- 总计纳入 profiler 的 `aten::mm`、`aten::addmm`、`aten::bmm`、`aten::conv2d`，乘加计 2 FLOPs；排除其他算子、softmax、归一化、激活、排序、预处理与仿真。
- profiling 时两种方法均展开视觉注意力矩阵运算，仅作用于统计进程；不将其耗时混入正式延迟。

## OFT 环境记录

这是一套独立环境，不沿用第一部分的 Protobuf、Metadata 和 BDDL 版本。以下为对话中已核实或明确安装的版本；未完整导出的项不补造版本号。

| 组件 | OFT 设置 |
|---|---|
| Python | 3.10 系列，完整补丁号需环境导出 |
| GPU / 驱动 | NVIDIA H20 / 570.158.01 |
| PyTorch | 2.2.0+cu121 |
| torchvision / torchaudio | 按 0.17.0 / 2.2.0 cu121 安装，完整版本以导出为准 |
| CUDA runtime | 12.1 |
| CUDA toolkit | `/usr/local/cuda-12.1`，nvcc 12.1.105 |
| cuDNN | nvidia-cudnn-cu12 8.9.2.26 |
| 独立 cuBLAS | 12.4.5.8，仅本次进程预加载 |
| NumPy / OpenCV | 1.26.4 / opencv-python 4.10.0.84 |
| Transformers | 官方 OFT fork，版本 4.40.1 |
| FlashAttention | 2.5.5；正式对比实际使用 eager |
| TensorFlow | 2.15.0，禁止其使用 GPU |
| tensorflow-datasets / metadata | 4.9.3 / 1.17.3 |
| Protobuf | 4.25.3 |
| W&B | 0.16.6，补齐 appdirs / docker-pycreds / setproctitle，关闭上传 |
| robosuite / MuJoCo / BDDL | 1.4.1 / 2.3.7 / 3.6.0 |
| 渲染 | EGL |

Transformers 的 `direct_url.json` 来源：`https://github.com/moojink/transformers-openvla-oft.git`，提交 `bc339d9ad707454c0c115970db43c260067c61ab`。仅检查版本号 4.40.1 不能确认安装来源或注意力语义。

OFT 源码与模型权重的精确 revision、LIBERO/FastV 的提交和全部本地 diff 尚需补存。官方 OFT 论文使用 A100、多 seed，本次使用 H20、单 seed，不能称为完全相同论文条件。本次使用已合并权重，未在 H20 重新合并 LoRA。

## OFT 路径与实现文件

| 资源 | 服务器路径 |
|---|---|
| 环境 | `/data2/wxy/VLA-Adapter/conda_envs/openvla-oft` |
| 工程 | `/home/wxy/daisy_work/openvla-oft` |
| 复用 LIBERO | `/home/wxy/daisy_work/LIBERO` |
| 模型访问入口 | `/home/wxy/daisy_work/checkpoints/openvla-7b-oft-finetuned-libero-spatial` |
| 模型实际存储 | `/data2/wxy/openvla_oft_checkpoints/openvla-7b-oft-finetuned-libero-spatial` |
| HF 缓存 | `/data2/wxy/openvla_oft_checkpoints/.hf_home` |
| 独立 cuBLAS | `/data2/wxy/openvla_oft_runtime/cublas-12.4.5.8/nvidia/cublas/lib` |
| 结果根目录 | `/home/wxy/daisy_work/results/openvla_oft` |

模型访问入口是指向 /data2 的软链接，文件只保存一份。模型完整下载约 14.84 GiB，保留 4 个分片、索引、配置、tokenizer/processor、动作头、本体状态投影器和 LoRA adapter。Spatial 反归一化使用 `libero_spatial_no_noops`。

`/home/wxy` 与 `/data2/wxy` 属于不同文件系统，路径都有 wxy 不代表使用相同磁盘。结果保存在 daisy_work，大权重放在 /data2。

OFT 通过 PYTHONPATH 复用 LIBERO 源码，仿真资产和初始状态路径已核实有效；环境依赖仍单独安装。不需要复制第二份 LIBERO 或下载训练演示数据。

| 文件 | 用途 |
|---|---|
| `experiments/robot/fastv_oft.py` | OFT FastV 启用、视觉区间与序列长度检查 |
| `experiments/robot/oft_attention.py` | 模型实例级双向 eager，拒绝 padding/cache |
| `experiments/robot/openvla_utils.py` | BF16 eager 加载并自动应用双向适配 |
| `prismatic/extern/hf/modeling_prismatic.py` | 剪枝后动作特征切片修正 |
| `experiments/robot/libero/run_libero_eval.py` | 评测，finally 中关闭环境 |
| `run_oft_comparison.py` | 逐回合、逐推理、显存与元数据记录 |
| `profile_oft_flops.py` | 完整 8 动作块 FLOPs 对比 |

## OFT 运行方式

以下用于已准备好的服务器环境。仅从官方克隆不会包含新增适配器和脚本。GPU 3 是本次验证配置，换卡应同时核实 CUDA 与 EGL 映射；使用前遵循共享服务器分配规则。

### 正式闭环对比

```bash
conda activate /data2/wxy/VLA-Adapter/conda_envs/openvla-oft
cd /home/wxy/daisy_work/openvla-oft

(
ulimit -c 0
oft_cublas_dir=/data2/wxy/openvla_oft_runtime/cublas-12.4.5.8/nvidia/cublas/lib
export LD_PRELOAD="$oft_cublas_dir/libcublasLt.so.12:$oft_cublas_dir/libcublas.so.12"
export LD_LIBRARY_PATH="$oft_cublas_dir:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
export PYTHONPATH="/home/wxy/daisy_work/openvla-oft:/home/wxy/daisy_work/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_VISIBLE_DEVICES=3
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=3
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export WANDB_MODE=disabled
export HF_HOME=/data2/wxy/openvla_oft_checkpoints/.hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for oft_mode in baseline fastv; do
    python -u -X faulthandler run_oft_comparison.py \
        --mode "$oft_mode" --trials 50 --seed 7 --gpu 3
    oft_status=$?
    printf '\n%s exit code: %s\n' "$oft_mode" "$oft_status"
    if [ "$oft_status" -ne 0 ]; then exit "$oft_status"; fi
done
)
```

环境变量只影响子 shell 与其进程，不修改系统设置。启动应打印 `OFT attention: eager, bidirectional, no padding, no KV cache`；FastV 应显示 `512 -> 256`。

OFT 脚本的 K=2、r=0.5 写在代码中，没有 `--k` 或 `--prune-ratio` 参数，不要照搬原 OpenVLA 启动器的全部参数。不传 `--tasks` 为全部任务，`--trials` 是每任务回合数。

### 小规模与 FLOPs

在上述相同子 shell 环境配置内，替换正式循环，先做小规模，再正式运行；不要同时执行全部命令造成重复评测：

```bash
# 单任务、单回合。
python -u -X faulthandler run_oft_comparison.py --mode baseline --tasks 0 --trials 1 --seed 7 --gpu 3
python -u -X faulthandler run_oft_comparison.py --mode fastv --tasks 0 --trials 1 --seed 7 --gpu 3

# 全部任务，每任务 5 回合。
python -u -X faulthandler run_oft_comparison.py --mode baseline --trials 5 --seed 7 --gpu 3
python -u -X faulthandler run_oft_comparison.py --mode fastv --trials 5 --seed 7 --gpu 3

# FLOPs 独立统计，不混入闭环计时。
python -u -X faulthandler profile_oft_flops.py
```

OFT 结果使用新时间戳目录、不覆盖旧结果；正式评测不保存视频。当前不支持断点续跑，中断保留 CSV，新运行从头开始；运行时保持终端/SSH 会话有效。

### 迁移前导出环境与源码

```bash
python -m pip freeze > oft-environment-freeze.txt
conda list --explicit > oft-conda-explicit.txt
git rev-parse HEAD > oft-source-commit.txt
git diff > oft-tracked-changes.patch
git status --short > oft-source-status.txt
```

`git diff` 不包含未跟踪文件；另行保存新增的 `fastv_oft.py`、`oft_attention.py`、`run_oft_comparison.py`、`profile_oft_flops.py`。只保存 patch 不足以恢复实现。迁移机器后替换绝对路径并先做语法、样例、全任务小规模检查。

## OFT 正式结果与数据字段

根目录：`/home/wxy/daisy_work/results/openvla_oft/`。

| 实验 | 已核验目录 |
|---|---|
| 双向 Baseline 500 回合 | `formal_baseline_20261108_205330_480201` |
| 双向 FastV 500 回合 | `formal_fastv_20261108_222808_420499` |
| 固定输入 FLOPs | `flops_comparison_20261109_005614_597939` |

闭环结果内包含：

| 文件 | 内容 |
|---|---|
| `metadata.json` | 配置、状态、记录范围内的源码哈希、库设置 |
| `nvidia-smi-start.txt` | 启动时共享 GPU 占用快照 |
| `eval.log` | 任务日志 |
| `actions.csv` | 每次动作块推理耗时与预热标记 |
| `episodes.csv` | 回合成功、初始状态 SHA256、推理次数 |
| `summary.json` | 成功率、原始/排除前 5 次后的延迟、峰值显存 |

`actions.csv` 字段：`task_id, episode_id, call_id, latency_ms, excluded_as_warmup`。

`episodes.csv` 字段：`task_id, episode_id, initial_state_sha256, success, inference_calls, mean_chunk_ms`。单回合 mean 包含其全部调用；排除整次前 5 次后的统计以 summary 对应项为准。

FLOPs 结果内包含 `summary.csv`、`operators.csv`、`metadata.json`、`baseline_actions.npy`、`fastv_actions.npy`；元数据保存输入校验值、源码哈希、统计范围与序列长度。

不能仅凭目录含 formal 判断为正式实验；应确认每任务 50 回合、10 个任务、completed 状态，并排除修正前运行。本文正式指标只采用上表指定目录。

## OFT 常见问题与修复记录

| 现象 | 本次定位与处理 |
|---|---|
| Baseline 约 58% | eager 使用因果掩码，不符合 OFT 双向语义；实例级修正后重测。 |
| 零剪枝一致但成功率低 | 两条路径可共同使用错误语义；需检查实际掩码。 |
| BF16 Linear 的 SIGFPE | GDB 定位 cuBLAS 12.1.3.1 的 `cublasLtMatmulAlgoGetHeuristic()`；独立 cuBLAS 12.4.5.8 解决。改 nvcc 或单纯禁用 addmm 路径未解决。 |
| cuDNN undefined symbol | 原 LD_LIBRARY_PATH 混入 effnet/CoppeliaSim；为本次进程使用当前环境库。 |
| Metadata / Protobuf 导入失败 | OFT 已验证 TF 2.15.0、TFDS 4.9.3、Metadata 1.17.3、Protobuf 4.25.3。 |
| W&B 缺包 | 0.16.6，补 appdirs、docker-pycreds、setproctitle，关闭上传。 |
| NumPy 与 OpenCV 冲突 | NumPy 1.26.4、OpenCV 4.10.0.84，避免 OpenCV 5 要求 NumPy 2。 |
| 退出时 EGL_NOT_INITIALIZED | run_task 使用 try/finally 显式 env.close()，后续正常退出。 |
| 加载更新 config / 创建备份 | OFT 加载器同步模型代码与 auto_map，可能写入本份检查点；加载并非完全只读。 |
| 找不到 LIBERO | 检查 PYTHONPATH 和 namespace package，代码复用不等于依赖共享。 |
| TF 注册、Gym、macro 提示 | 按实际异常和退出状态判断，不为去除提示盲目升级依赖。 |

## 共享服务器约束

只写自己的代码、环境、权重和结果，不使用 sudo，不修改系统 CUDA 或其他用户文件。独立库预加载与设备设置仅对本次进程生效。遵循 GPU 分配规则，利用率为 0% 不代表别人已经释放卡；不结束其他用户进程。下载、编译、推理均消耗共享资源，不能保证零资源影响。先检查磁盘空间，不清理他人环境或共享缓存；GPU 空闲不能解决磁盘不足。

## 可用于报告的 OFT 结果描述

在 LIBERO-Spatial 上，使用相同 OpenVLA-OFT 检查点、BF16 和 eager 双向注意力，在 seed=7、相同的 500 个任务初始状态下配对评测。FastV 在 K=2、视觉 token 剪枝 50% 时，将两路图像的视觉 token 从 512 减至 256，成功率为 99.4%（497/500），Baseline 为 98.4%（492/500）。共享 NVIDIA H20 上，排除整次运行最初 5 次推理后，平均动作块推理延迟由 291.68 ms 降至 222.73 ms，降低 23.64%，对应 1.310× 加速，每动作块包含 8 个动作。固定输入下，完整动作块预测的矩阵乘法与卷积计算量由 9,029.542 GFLOPs 降至 5,733.774 GFLOPs，降低 36.50%。

成功率差为本次单 seed 观察，不能宣称普遍提高成功率。FLOPs 只覆盖声明算子，闭环延迟受共享负载影响。后续多 seed、隔离 GPU 测速或参数扫描需新建结果目录，保留本次记录。
