# LightVLA 工具箱接入

本目录提供已有模型上的视觉 token 裁剪封装，支持统一注册、模式切换、reset 和
卸载恢复，不加载或修改 checkpoint，不执行机器人动作。

## 适用范围与方法来源

本封装要求模型在 `model.language_model.model` 暴露 LLaMA decoder，且视觉 token
在 prefill 输入中是可确认的连续区间。它可以接入符合这个结构的 VLA 模型；
不能据此声称任意 VLA、交叉注意力架构或任意权重都已经兼容。
`patch_start` 和 `patch_count` 应来自用户模型的实际输入结构。

官方 [LightVLA](https://github.com/LiAutoAD/LightVLA) / [论文](https://arxiv.org/abs/2509.12594)
包含微调阶段。本封装可以直接使用已有训练权重进行推理，不代表 LightVLA 原方法免训练，
也不代表把本插件挂到任意未适配权重上仍能复现官方成功率。
源码入口与维护状态见 [来源记录](provenance.json)，第三方声明见 [UPSTREAM_LICENSE.md](UPSTREAM_LICENSE.md)。

## 环境怎么配

轻量配置和注册器仅需要 Python >= 3.10；模型依赖在挂载时导入。
已有模型环境优先保留原项目验证过的 torch/Transformers fork，在工具箱目录执行：

```bash
python -m pip install --no-deps -e .
python -c "from pruners import available; print(available())"
```

输出应包含 `lightvla`。独立测试环境可执行：

```bash
python -m pip install -e ".[lightvla]"
python -m methods.lightvla --smoke
```

可选依赖面向 Transformers 4.40.x；并不代替用户模型工程的完整环境安装。
完整模型依赖遵循官方 [SETUP.md](https://github.com/LiAutoAD/LightVLA/blob/main/SETUP.md)。
本次 CPU 封装测试使用 Python 3.12、torch 2.3.0+cpu、Transformers 4.40.1；
完整模型环境应遵循官方工程的 Python 和依赖要求。
原服务器环境与当前封装测试环境分别记录，不把历史版本表作为通用安装配方。
模拟 smoke 使用微型随机 CPU LLaMA，无需 GPU、权重、LIBERO 或数据集。

## 模型权重从哪下

官方 checkpoint 入口由 [LightVLA 仓库](https://github.com/LiAutoAD/LightVLA) 提供。
例如 Spatial 是 [TTJiang/LightVLA-libero-spatial](https://huggingface.co/TTJiang/LightVLA-libero-spatial)：

```bash
hf download TTJiang/LightVLA-libero-spatial --local-dir /your/workspace/checkpoints/lightvla-spatial
```

`hf` 来自模型环境中的 Hugging Face Hub 客户端。发布实验时应固定下载 revision，
记录文件哈希，并保留完整快照中的配置、tokenizer/processor、归一化统计、模型分片、
action head、proprio projector 和 remote-code 文件。不要跨 suite 混用组件。
权重加载、图像与本体状态预处理仍使用用户模型工程的实现。
本次没有下载这些权重，也没有验证历史服务器 checkpoint 与当前 Hub 快照逐字节相同。

## 怎么挂载和运行

```python
import torch
from pruners import build

# model 和 prepared_inputs 来自用户模型工程。先确认实际 patch 起点与数量。
model.eval()
pruner = build("lightvla", {
    "mode": "packaged",
    "patch_start": 1,
    "patch_count": verified_patch_count,
    # 只有确认原 decoder 的自带 pruner 是 LightVLA 时才显式允许接管。
    "allow_existing_pruner": True,
})
pruner.attach(model)
try:
    with torch.inference_mode():
        raw_output = model.predict_action(**prepared_inputs)
    print(pruner.get_stats())

    pruner.disable()  # baseline：使用同一个 decoder 和权重，关闭 token 裁剪
    pruner.enable()   # 恢复先前启用的 packaged/builtin 模式
    pruner.reset()    # 无跨帧状态；清理本封装统计
finally:
    pruner.detach()
```

也可以用 `pruner.predict_action(**prepared_inputs)`，它保留原始返回值并启用
inference mode；`pruner.step(**prepared_inputs)` 返回 `{"output": 原始返回值, "info": 统计}`。
这里的输入是模型预处理后的关键字参数，不是统一的 image/proprio observation 接口。
原始返回值中的完整 action chunk 不会被截断。动作队列、夹爪转换、episode 调度由调用者负责。

旧函数入口继续支持，并增加恢复接口：

```python
from methods.lightvla import configure_lightvla_pruning, restore_lightvla_pruning

configure_lightvla_pruning(model, "packaged", patch_count=verified_patch_count,
                           allow_existing_pruner=True)
try:
    # 使用模型原有接口推理。
    pass
finally:
    restore_lightvla_pruning(model)
```

低层类从 `methods.lightvla.pruner` 导入；本仓库没有名为 `lightvla_pruner` 的顶层包。

## 用命令接入用户工作目录

本仓库交付的命令入口是 `python -m methods.lightvla`（安装后也可用 `vla-lightvla`）。
它接受用户工程的 factory，避免猜测不同模型的加载与预处理方式。
在用户工作目录的 `my_lightvla_entry.py` 中提供：

```python
def create(checkpoint, observation):
    # 按你的模型工程实现加载和预处理；这里不规定一个虚构的通用 loader。
    from my_policy_project import load_policy, prepare_observation
    model = load_policy(checkpoint).eval()
    inputs = prepare_observation(model, observation)
    return {"model": model, "inputs": inputs}
```

factory 也可以返回 `{"model": model, "run": 一个无参数推理函数}`，用于原工程的
完整 observation→action 流程。Codex 应读取工作目录，将上面的两个项目函数替换成
真实入口，不能原样运行这个模板。

```bash
python -m methods.lightvla --workspace /your/workspace --factory my_lightvla_entry:create \
  --factory-kwargs '{"checkpoint":"/absolute/checkpoint","observation":"/absolute/observation"}' \
  --config /path/to/toolbox/configs/lightvla.yaml --mode packaged --output /your/workspace/lightvla-query.json
```

`--workspace` 仅加入模块搜索路径；相对路径仍按当前工作目录解析。`--output` 不覆盖已有文件。
在 PowerShell 中可把命令写成一行。CLI 完成一次查询后会恢复原模型并写出统计，
不是数据集评测器，也不据一次查询宣称加速。

## 模式、配置和跑通判据

| 模式 | 行为 |
| --- | --- |
| `packaged` | 使用本目录的裁剪模块，调用标准 LLaMA forward |
| `baseline` | 使用标准 LLaMA forward，关闭本次裁剪；保持原权重 |
| `builtin` | 恢复挂载前自带 pruner 的原始 forward；要求它确实存在 |

`allow_existing_pruner=False` 默认拒绝接管自带 pruner。确认是 LightVLA 后可设为
`True`，本封装会绕过原裁剪 forward 再挂载 packaged，避免同一 forward 重复裁剪。
如果其他模型逻辑在 decoder 外也裁剪 token，需要先在模型工程中关闭。
模式切换失败会保留先前有效模式；卸载恢复原 decoder、forward 和原 summary 属性。
不支持同一模型上并发切换或与其他剪枝器叠加。

标准配置位于根目录 `configs/lightvla.yaml`。未知参数和非法值会报错。
`patch_count=null` 时优先由 vision backbone 的每视图 patch 数×视图数推断；
视觉布局不同的模型必须显式提供探测后的数量。
`score_context` 支持 `suffix` 和 `non_patch`。本方法按输入自适应选择，不承诺固定保留比例。
`noise_scale` 传入底层模块，但只影响其训练分支；本封装的 eval 推理选择不注入噪声。

`--smoke --mode packaged` 应输出：

```text
check = synthetic-smoke-only
stats.mode = packaged
stats.pruned_forward_count = 1
stats.last_input_sequence_length = 8
stats.last_output_sequence_length = 5
stats.last_patch_keep_counts = [1]
detached = true
```

这是确定性的合成布局：1 个 prefix + 4 个相同 patch + 3 个 suffix。
baseline 对同一布局保持长度 8，不能把这些数字当作真实模型的 token 数。
真实模型须确认 prefill 裁剪路径被调用、非视觉 token 被保留、输出 action 合法，
并在既有 KV cache 的 decode 阶段绕过二次裁剪。
cached decode 必须使用后端实际返回的 cache，按裁剪后的长度构造 attention mask，
并保留正确的绝对 position IDs。本封装不自动修改外层通用 `generate()` 的状态管理。
一个 token 都没剪除也可能是合法自适应结果，应查看长度和保留数量，不能只看调用计数。
本封装接受 2D `[B,S]` attention mask；不支持在该入口传入已展开的 4D causal mask。

## 效果和可信度

历史服务器记录保留在 [EXPERIMENTS.md](EXPERIMENTS.md)。其中 1.73×/2.02× 等数字
对应原服务器、原权重与原脚本，原始证据没有随当前仓库交付。本次封装修改不能继承
它们作为已验证性能，也不能推断对未适配模型的动作精度和闭环成功率。

当前验证分层记录：配置与生命周期契约、CPU 张量裁剪、微型 LLaMA 前向与 cache 测试；
用户 checkpoint 的 GPU 推理、闭环和真机表现仍需在目标环境测量。
只有同一输入、同一设备、适当预热与同步的配对测量，才能报告实际加速比。

```bash
python -m unittest tests.test_lightvla -v
```

没有 torch/Transformers 的环境仅执行轻量配置测试，张量测试会明确标为 skipped；
完整 CPU 测试使用已安装的 `.[lightvla]` 环境。
