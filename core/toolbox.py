"""Read-only project discovery and an explicitly synthetic lifecycle demonstration."""
import argparse
import html
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

CATALOG = {
    "lightvla": {"status": "conditional", "label": "专用 Spatial 权重 20/20；普通 OFT Object 剪枝干预 1/6，不推荐迁移",
                 "guide": "methods/lightvla/README.md"},
    "vla_pruner": {"status": "conditional", "label": "固定 OFT Object/H20 两任务六对闭环 6/6；正式质量和性能验收待完成",
                   "guide": "methods/vla_pruner/INTEGRATION.md"},
    "specprune": {"status": "experimental", "label": "Object 两任务三个初始状态，原生与加速各 3/3；支持实验 evaluate，正式质量验收待完成",
                  "guide": "docs/SPECPRUNE_WORKFLOW.md"},
    "vla_cache": {"status": "quarantined", "label": "隔离，仅保留历史证据，不提供部署建议",
                  "guide": "workbench/PROJECT_STATUS.md"},
}
EXCLUDE = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build",
           "handoff-work", "handoff-snapshots", "models", "checkpoints", "data", "datasets", "runs", "results"}


def inspect_workspace(workspace):
    root = Path(workspace).resolve()
    if not root.is_dir():
        raise ValueError("工作目录不存在或不是目录")
    files, hits, scanned = [], {}, 0
    truncated = False
    # Never import user code or follow directory links during discovery.
    for parent, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE and not d.startswith('.')
                         and not (Path(parent) / d).is_symlink())
        if len(Path(parent).relative_to(root).parts) >= 5:
            dirs[:] = []
        for name in sorted(names):
            scanned += 1
            if scanned > 3000:
                truncated = True
                break
            path = Path(parent) / name
            if path.is_symlink() or path.suffix.lower() not in {".py", ".json", ".toml", ".yaml", ".yml"}:
                continue
            if path.stat().st_size > 256_000:
                continue
            rel = path.relative_to(root).as_posix()
            files.append(rel)
            source = path.read_text(encoding="utf-8-sig", errors="replace").lower()
            for marker in ("openvla", "predict_action", "lightvla", "vla_pruner", "specprune"):
                if marker in source:
                    hits.setdefault(marker, []).append(rel)
        if truncated:
            break
    config_path = root / "toolbox-project.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig")) if config_path.exists() else {}
    if not isinstance(config, dict):
        raise ValueError("toolbox-project.json 必须是 JSON 对象")
    checkpoint = config.get("checkpoint")
    checkpoint_info = {"configured": bool(checkpoint), "exists": False,
                       "validation": "仅检查路径，不确认权重完整性或模型兼容性"}
    if checkpoint:
        location = Path(checkpoint)
        if not location.is_absolute():
            location = root / location
        checkpoint_info.update(path=str(location.resolve()), exists=location.exists())
    versions = {}
    for name in ("torch", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    issues = ["真实模型、权重、输入布局及运行环境需要单独验证；源码关键词不能证明兼容。",
              "加速效果尚未由本报告测量；需要相同输入和配置的原生/加速成对测试。"]
    if not checkpoint_info["exists"]:
        issues.append("未确认可访问的模型权重路径。")
    if not config.get("factory"):
        issues.append("未提供模型加载工厂；Codex 需根据原项目加载代码编写接入文件。")
    # Discovery is a lead for review, never automatic compatibility admission.
    candidates = ["lightvla", "vla_pruner"] if "openvla" in hits else []
    return {"schema_version": 1, "check": "read-only-source-discovery", "workspace": str(root),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.executable, "python_version": platform.python_version(),
            "platform": platform.platform(), "packages": versions,
            "scan": {"files_read": len(files), "entries_seen": scanned, "truncated": truncated,
                     "max_depth": 5, "max_entries": 3000, "max_file_bytes": 256000},
            "source_signals": hits, "project_config": config, "checkpoint": checkpoint_info,
            "candidates_for_review": candidates, "compatibility_verified": False,
            "methods": CATALOG, "next_steps": issues,
            "execution": "未导入用户代码、未加载模型、未联网、未启动 GPU"}


def synthetic_demo():
    import torch
    from methods.lightvla.cli import _smoke_bundle
    from methods.lightvla import LightVLAAccelerator
    bundle = _smoke_bundle()
    model, inputs = bundle["model"], bundle["inputs"]
    decoder = model.language_model.model
    original_forward = decoder.forward
    parameters = {name: (value, value.detach().clone()) for name, value in model.named_parameters()}
    with torch.inference_mode():
        before = model.predict_action(**inputs)
    session = LightVLAAccelerator.install(model)
    try:
        during = session.predict_action(**inputs)
        stats = session.get_stats()
    finally:
        session.detach()
    with torch.inference_mode():
        after = model.predict_action(**inputs)
    restored = (model.language_model.model is decoder and decoder.forward == original_forward
                and all(dict(model.named_parameters()).get(n) is p and torch.equal(p, value)
                        for n, (p, value) in parameters.items())
                and torch.equal(before, after))
    lengths = [before.shape[1], during.shape[1], after.shape[1]]
    if lengths != [8, 5, 8] or not restored:
        raise RuntimeError("合成演示未通过：序列长度或卸载恢复异常")
    return {"schema_version": 1, "check": "synthetic-cpu-lifecycle-only", "passed": True,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "transformers": importlib.metadata.version("transformers"), "device": "cpu", "seed": 0},
            "sequence_lengths": lengths, "restored": restored, "stats": stats,
            "real_vla_validated": False, "speedup_measured": False,
            "explanation": "随机 tiny LLaMA，4 个相同视觉 token 合并为 1；展示接入及卸载，不代表机器人成功率或真实推理加速。"}


def write_report(report, directory):
    directory = Path(directory)
    # A fresh output directory avoids overwriting a teammate's reports.
    directory.mkdir(parents=True, exist_ok=False)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    (directory / "report.json").write_text(text + "\n", encoding="utf-8")
    if report.get("check") == "read-only-source-discovery":
        summary = "# VLA 工具箱检查报告\n\n"
        summary += "工作目录：" + report["workspace"] + "\n\n"
        summary += "兼容性：待验证。候选方法仅供审查，不表示可直接运行。\n\n"
        summary += "\n".join("- " + item for item in report["next_steps"])
        summary += "\n\n| 方法 | 当前状态 |\n|---|---|\n"
        summary += "\n".join(f"| {name} | {info['label']} |" for name, info in CATALOG.items())
        body = '<h2>工作目录检查</h2><p>' + html.escape(report["workspace"]) + '</p>'
        body += '<p>兼容性：待验证。候选仅供审查。</p><ul>'
        body += ''.join('<li>' + html.escape(item) + '</li>' for item in report["next_steps"]) + '</ul>'
        body += '<table><tr><th>方法</th><th>当前状态</th></tr>'
        body += ''.join('<tr><td>' + html.escape(name) + '</td><td>' + html.escape(info['label']) + '</td></tr>'
                        for name, info in CATALOG.items()) + '</table>'
    else:
        summary = "# LightVLA 合成 CPU 演示\n\n序列长度：8 → 5 → 8。卸载恢复检查通过。\n\n" + report["explanation"]
        body = '<h2>LightVLA 合成 CPU 演示</h2><div class="flow">8 → 5 → 8</div>'
        body += '<p>原生序列 → 剪枝序列 → 卸载后原生序列</p><p>参数与原生输出恢复检查：通过</p>'
        body += '<p>' + html.escape(report['explanation']) + '</p>'
    (directory / "report.md").write_text(summary + "\n", encoding="utf-8")
    (directory / "report.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>VLA 工具箱报告</title><style>body{font:18px/1.7 system-ui;max-width:1000px;'
        'margin:48px auto;padding:0 24px;color:#172b4d}pre{white-space:pre-wrap;overflow-wrap:anywhere;'
        'background:#f1f5f9;padding:24px;border-radius:12px}h1{color:#075985}'
        '.flow{font-size:64px;font-weight:700;color:#047857}table{border-collapse:collapse;width:100%}'
        'td,th{text-align:left;padding:12px;border-bottom:1px solid #cbd5e1}</style>'
        '<h1>VLA Express Toolbox</h1>' + body + '<details><summary>完整检查记录</summary><pre>'
        + html.escape(text) + '</pre></details></html>', encoding="utf-8")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "project":
        from core.project import main as project
        return project(argv[1:])
    if argv and argv[0] in ("run", "evaluate"):
        from core.run import main as run
        return run(argv[1:], evaluate=argv[0] == "evaluate")
    parser = argparse.ArgumentParser(description="VLA 工具箱统一入口")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="只读检查用户工作目录")
    doctor.add_argument("--workspace", required=True, type=Path)
    doctor.add_argument("--output", required=True, type=Path, help="新建报告目录")
    demo = commands.add_parser("demo", help="tiny LLaMA CPU 接入/卸载演示")
    demo.add_argument("--output", required=True, type=Path)
    commands.add_parser("methods", help="方法状态目录")
    commands.add_parser("project", help="项目配置：init/check/execute")
    commands.add_parser("run", help="执行真实推理：--method lightvla 或 vla_pruner")
    commands.add_parser("evaluate", help="原生/关闭剪枝/加速/恢复的输入级评估")
    try:
        args = parser.parse_args(argv)
        if args.command == "methods":
            report = CATALOG
        else:
            if args.output.exists():
                raise ValueError("报告目录已存在，请使用新目录")
            report = inspect_workspace(args.workspace) if args.command == "doctor" else synthetic_demo()
            write_report(report, args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc), "type": type(exc).__name__}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
