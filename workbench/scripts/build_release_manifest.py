#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


EXCLUDED = {"release-manifest.json", "SHA256SUMS.txt"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in EXCLUDED or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        files.append({"path": relative, "size": path.stat().st_size, "sha256": digest(path)})

    manifest = {
        "schema_version": 1,
        "release": "vla-codex-workbench-" + json.loads((root / "VERSION.json").read_text(encoding="utf-8"))["version"],
        "file_count": len(files),
        "files": files,
    }
    (root / "release-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (root / "SHA256SUMS.txt").write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in files), encoding="utf-8"
    )
    print(f"WROTE_MANIFEST files={len(files)}")


if __name__ == "__main__":
    main()

