# 实际合成演示记录

本目录三个 report 文件由 2026-11-02 的实际本地运行生成，命令：

```bash
.venv/Scripts/python.exe -X utf8 -B -m core demo --output runs/midterm-demo-final-20261102
```

随后原样复制到本目录，方便解压后直接打开 `report.html`。环境、UTC 时间和随机 seed 在 JSON 内。只有随机 tiny LLaMA 的 CPU 生命周期结果，不包含真实 VLA checkpoint、动作、成功率或加速倍数。重新运行时新建输出目录，不覆盖这份记录。
