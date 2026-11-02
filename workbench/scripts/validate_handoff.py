"""Validate handoff links/evidence hashes and parse Python without generating bytecode."""
import ast
import hashlib
import json
from pathlib import Path
import re


def check(root):
    errors = []
    for name in ("START_HERE.md", "CODEX_INSTRUCTIONS.md", "README-ZH.md", "PROJECT_STATUS.md",
                 "COLLABORATION.md", "conversation/INITIALIZATION_DIALOGUE.md",
                 "skill/vla-acceleration-evaluator/SKILL.md"):
        p = root / name
        if not p.is_file():
            errors.append(f"missing {name}")
            continue
        for target in re.findall(r"\]\(([^)]+)\)", p.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            if not (p.parent / target.split("#")[0]).exists():
                errors.append(f"broken link in {name}: {target}")
    index = json.loads((root / "EVIDENCE_INDEX.json").read_text(encoding="utf-8"))
    for item in index["files"]:
        p = (root / item["path"]).resolve()
        if not p.is_relative_to(root.resolve()):
            errors.append("unsafe evidence path")
        elif not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
            errors.append(f"missing/changed evidence: {item['path']}")
    for method in ("specprune", "vla-pruner"):
        manifest = json.loads((root / "evaluator" / method / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
        for item in manifest["files"]:
            p = (root / item["path"]).resolve()
            if not p.is_relative_to(root.resolve()) or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
                errors.append(f"missing/changed evaluator source: {item['path']}")
    for p in root.rglob("*.py"):
        try:
            ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"syntax error {p.relative_to(root)}: {exc}")
    return errors


def main():
    errors = check(Path(__file__).resolve().parents[1])
    for value in errors:
        print(value)
    print("HANDOFF_VALIDATE: " + ("FAIL" if errors else "PASS"))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
