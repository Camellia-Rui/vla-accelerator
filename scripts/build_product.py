"""Build one allowlisted, source-only product folder and ZIP; never include local assets."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    version = "0.3.0a2"
    name = "vla-express-toolbox-" + version
    destination = root / "dist" / name
    archive = destination.parent / (name + ".zip")
    if destination.exists() or archive.exists():
        raise FileExistsError("Release already exists; inspect it before making another version")
    folders = ("core", "pruners", "methods", "configs", "agents", "scripts", "tests", "workbench", "docs")
    files = ("README.md", "START_HERE.md", "SKILL.md", "LICENSE", "pyproject.toml")
    selected = [root / f for f in files]
    for folder in folders:
        selected.extend(p for p in (root / folder).rglob("*") if p.is_file()
                        and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"})
    excluded = {"showcase-template.html", "showcase-results.json", "PRESENTATION_SCRIPT.md", "build_showcase.py", "check_showcase.py"}
    for source in sorted(selected):
        if source.name in excluded:
            continue
        target = destination / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    rows = [{"path": p.relative_to(destination).as_posix(),
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(destination.rglob("*")) if p.is_file()]
    manifest = {"product": "vla-express-toolbox", "version": version, "files": rows,
                "scope": "source and small evidence artifacts; VLA-Pruner input checks and LightVLA closed-loop smoke; formal deployment acceptance pending"}
    (destination / "product-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                stream.write(path, name + "/" + path.relative_to(destination).as_posix())
    print(json.dumps({"directory": str(destination), "zip": str(archive), "files": len(rows),
                      "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
