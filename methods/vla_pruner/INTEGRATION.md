# VLA-Pruner 工具箱接口

本文件描述当前封装接口。历史说明保留在 `README.md`，其中旧版
`attach(model=None)` 示例已由本文接口替代。

## 支持范围和依赖边界

本工具箱封装 **官方 VLA-Pruner OpenVLA-OFT 分支**，目前支持连续 L1 动作头、
一视图或双视图输入。普通 OpenVLA、原版 OpenVLA-OFT、SpecPrune 修改版模型，
以及未修改的 PyPI Transformers 不在此接口支持范围内。

核心剪枝实现作为外部依赖提供，固定源码版本记录在 `backend.lock.json`：

- 仓库：<https://github.com/MINT-SJTU/VLA-Pruner>
- commit：`84d4b7192c77abf1585610e2f12393319b7ebff9`
- 模型：`src/openvla-oft/prismatic/extern/hf/modeling_prismatic.py`
- 剪枝内核：`src/openvla-oft/transformers/src/transformers/models/llama/modeling_llama.py`
- observation 预处理：`src/openvla-oft/experiments/robot/openvla_utils.py`

工具箱不自动下载或替换这些源码。`attach()` 会检查模型与 decoder 的剪枝入口、
层数和视图数；结构检查通过不代表已经验证真实模型的数值兼容性。
`describe()` 中的 backend commit 是接口开发所依据的版本，安装者应自行核对
外部 checkout 的 `git rev-parse HEAD`。

## 安装

工具箱本身只依赖 Python 标准库，Python >= 3.10：

```bash
python -m pip install -e /path/to/vla-express-toolbox
```

真实模型推理还需要单独安装官方后端。在独立模型环境中执行：

```bash
git clone https://github.com/MINT-SJTU/VLA-Pruner.git /path/to/VLA-Pruner
git -C /path/to/VLA-Pruner checkout 84d4b7192c77abf1585610e2f12393319b7ebff9
cd /path/to/VLA-Pruner/src/openvla-oft
python -m pip install -e transformers
python -m pip install -e .
```

完整模型依赖安装遵循这个版本的上游 `SETUP.md` / `README_VLA_Pruner.md`。
把 `src/openvla-oft` 加入当前进程的 `PYTHONPATH`，使 `experiments.robot.*` 可导入。
后端使用自己的设备配置；工具箱不接受一个实际未生效的 `device` 参数。
无需下载训练数据或开展 LIBERO/真机评测来安装此封装。

模型权重、processor、动作头、本体投影器和归一化统计由调用者提供。
本仓库不打包权重和数据，也不会在 `attach()` 时重新加载模型。

## 挂载已有模型（推荐）

```python
import torch
from pruners import build

# model 是使用上述官方后端加载的模型；先按需求配置其视图数量。
model.eval()
pruner = build("vla_pruner", {
    "fastv_r": 0.9275,
    "fastv_k": 3,
    "vla_pruner_layer": 15,
    "num_images_in_input": 2,
})
pruner.attach(model)
try:
    pruner.reset()  # 新 episode 或任务切换前调用
    # 保留模型原有接口，预处理和推理上下文由调用者管理。
    with torch.inference_mode():
        output = model.predict_action(**prepared_inputs)
    print(pruner.get_stats())

    pruner.disable()  # 切换到无剪枝路径，同时清理本次历史
    pruner.enable()   # 重新启用，从空历史开始
finally:
    pruner.detach()
```

上例中的 `torch`、`model`、`prepared_inputs` 来自调用方模型工程。
直接使用模型原生接口时，`get_stats()` 能读取最近一次 decoder 的剪枝信息，
但 `steps` 和时间只统计经过下面 `step()` 的查询。原生推理异常后须显式 `reset()`。

## 可选 observation 推理接口

需要工具箱代为调用官方预处理时，挂载额外资源：

```python
pruner = build("vla_pruner", {"task_suite_name": "libero_object"})
pruner.attach(model, processor=processor, action_head=action_head,
              proprio_projector=proprio_projector)
try:
    pruner.reset()
    result = pruner.step(image=image, wrist_image=wrist_image,
                         instruction="pick up the object", proprio=proprio)
    action = result["action"]       # 首个动作，兼容原工具箱调用
    chunk = result["actions"]       # 完整 chunk，调用者负责依次消费
    stats = result["info"]
finally:
    pruner.detach(model)
```

`image`/`wrist_image` 遵循上游预处理的 RGB uint8 H×W×3 格式；默认需要双视图
和 LIBERO 8 维 proprio。`unnorm_key` 默认使用 `task_suite_name`，支持
`_no_noops` 后缀回退，也可显式指定。

`step()` 的一次调用代表一次模型查询。它不维护动作执行队列，不做夹爪坐标约定转换，
不执行机器人动作。不要在仍有待执行 chunk 时无意中逐步重新查询。
返回动作已使用上游 checkpoint 统计反归一化；是否二值化或反转夹爪由环境适配层决定。
查询失败后会清空本次历史，避免复用不完整状态。

## 可选 checkpoint 加载

```python
from methods.vla_pruner import VLAPrunerAccelerator

# 调用前在进程环境设置 OPENVLA_SKIP_CHECKPOINT_SYNC=1。
acc = VLAPrunerAccelerator.from_pretrained("/path/to/compatible-checkpoint")
try:
    result = acc.step(image=image, wrist_image=wrist_image,
                      instruction=instruction, proprio=proprio)
finally:
    acc.detach()
```

此入口按上游 L1 动作头配置加载资源，不导入 LIBERO evaluation runner。
上游默认会同步本地 checkpoint 的模型代码；工具箱要求关闭该行为，且不会自动
设置全局环境变量。checkpoint 的 remote-code 必须事先与固定后端兼容，
不兼容的模型会在挂载检查时被拒绝。
旧式 `VLAPrunerAccelerator.install(checkpoint=...)` 保留为此入口的兼容写法。

## 配置与统计

可以把 `configs/vla_pruner.yaml` 用 YAML 库读成 dict 传入 `build()`。
`method`、`state_type` 元数据可保留；未知键、非法类型和范围会报错。

| 参数 | 含义 |
| --- | --- |
| `fastv_r` | 剪除比例，范围 `[0, 1)`；0 是零剪枝配置 |
| `fastv_k` | 实际执行 token 剪枝的 decoder 层，索引从 0 开始；本接口要求至少为 1 |
| `vla_pruner_layer` | 历史动作注意力评分层，索引从 0 开始 |
| `vla_pruner_mode` | `semantic_action`、`semantic`、`action` 或 `fastv` |
| `vla_pruner_av_hist_w` | 动作注意力历史窗口，至少为 1 |
| `vla_pruner_av_decay` | 历史权重衰减，范围 `[0, 1]` |
| `vla_pruner_semantic_weight` / `vla_pruner_action_weight` | 非负重要性权重，总和须大于 0 |
| `vla_pruner_action_horizon` | 参与评分的动作步数，0 表示全部，最大为 8 |
| `use_vla_pruner` | 初始是否启用；后续用 `enable()` / `disable()` 切换 |

`get_stats()` 区分请求比例与实际剪枝信息，提供 sequence/visual token 数和剪枝层。
未取得后端统计时返回 `None`，不会用配置比例伪造实际数量。
上游时间平滑模式需要积累历史，最初查询可能不剪枝。

`wall_time_ms` 是封装调用耗时；`backend_time_elapsed` 是上游原始秒数。
这些日常统计没有增加 CUDA 同步，不能直接作为严格加速基准。

## 生命周期和验证状态

- `attach()`：检查后端，保存被管理的属性，分配独立历史 deque。
- `reset()` / `start_episode()`：清理工具箱持有的历史、帧相关配置和统计。
- `disable()` / `enable()`：切换模式并重置历史。
- `detach()`：恢复原配置和原历史对象，清除工具箱新增属性；可重复调用。

原模型和配置的 forward 方法不会被替换。一个模型只允许一个本封装实例管理；
同一模型上的并发推理/切换以及与其他剪枝器叠加不受支持。
`detach()` 恢复的是挂载前状态，挂载前若已启用剪枝，卸载后会恢复它。

无需模型、GPU、数据的封装契约检查：

```bash
python -m unittest tests.test_vla_pruner -v
```

这些检查使用模拟后端验证资源归属、配置、动作 chunk 和异常清理。
真实模型数值兼容性、加速比和机器人行为未由本次封装验证。
历史 README 和 manifest 的性能数字不作为当前封装已验证的结论。
