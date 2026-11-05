"""Bounded metadata checks; no model imports or GPU initialization."""
import json
from pathlib import Path
import subprocess


def weights(checkpoint):
    root = Path(checkpoint).resolve()
    issues = []
    indexes = list(root.glob("*.index.json"))
    for index in indexes:
        try:
            mapping = json.loads(index.read_text())["weight_map"]
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("empty weight_map")
            for name in set(mapping.values()):
                path = root / name
                if not path.resolve().is_relative_to(root):
                    issues.append(f"Weight index escapes checkpoint: {name}")
                elif not path.is_file() or path.stat().st_size == 0:
                    issues.append(f"Missing or empty weight shard: {name}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(f"Invalid weight index {index.name}: {exc}")
    files = list(root.glob("*.safetensors")) + list(root.glob("pytorch_model*.bin"))
    if not files or any(p.stat().st_size == 0 for p in files):
        issues.append("Missing or empty weight files")
    return issues


def environment(python):
    # Inspect installed distribution metadata only: no torch/TensorFlow import.
    code = """import importlib.metadata as m,json,sys
packages={}
for name in ('torch','transformers','tokenizers','numpy','timm','Pillow'):
    try: packages[name]=m.version(name)
    except m.PackageNotFoundError: packages[name]=None
print(json.dumps({'python':sys.version,'executable':sys.executable,'packages':packages}))
"""
    try:
        result = subprocess.run([python, "-I", "-B", "-c", code], capture_output=True,
                                text=True, timeout=20, check=True)
        report = json.loads(result.stdout)
        report["missing"] = [k for k, v in report["packages"].items() if v is None]
        report["runtime_imports_verified"] = False
        return report
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"error": str(exc), "runtime_imports_verified": False}
