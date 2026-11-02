# 远端实验运行外壳

复制本目录到新实验包后，由 Codex 根据真实模型实现 `benchmark.py`。模板只负责后台运行、PID 防重入和状态查看，不包含通用加速算法。

`benchmark.py` 至少接受 `--output`，逐步原子写入 JSON，并支持从已有有效 records 续跑。完成后把 `status` 写成 `complete`；错误 episode 不得计入完成。

运行前必须执行：

```bash
python -m py_compile benchmark.py summarize.py
bash -n run.sh launch.sh status.sh
```

