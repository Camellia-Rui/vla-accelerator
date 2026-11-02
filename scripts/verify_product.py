"""Verify a product release, rejecting changed, missing and unexpected files."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    data = json.loads((root / "product-manifest.json").read_text(encoding="utf-8"))
    errors, expected = [], {"product-manifest.json"}
    for row in data["files"]:
        name = row["path"]
        path = root / name
        if name in expected or not path.resolve().is_relative_to(root) or path.is_symlink():
            errors.append("invalid/duplicate path: " + name)
            continue
        expected.add(name)
        if not path.is_file():
            errors.append("missing: " + name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            errors.append("changed: " + name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    errors.extend("unexpected: " + name for name in sorted(actual - expected))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors = verify(args.directory)
    print(json.dumps({"valid": not errors, "errors": errors}, ensure_ascii=False, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
