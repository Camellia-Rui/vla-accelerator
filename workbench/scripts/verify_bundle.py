#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import sys


REQUIRED = {
    "DEVELOPMENT.md",
    "evaluator/common/tests/test_infrastructure.py",
    "evaluator/specprune/probe_runtime.py",
    "evaluator/specprune/SOURCE_MANIFEST.json",
    "evaluator/vla-pruner/probe_runtime.py",
    "evaluator/vla-pruner/SOURCE_MANIFEST.json",
    "PROJECT_STATUS.md",
    "EVIDENCE_INDEX.json",
    "COLLABORATION.md",
    "scripts/discover_environment.py",
    "scripts/validate_handoff.py",
    "START_HERE.md",
    "ONE_LINE_TO_SEND.txt",
    "CODEX_INSTRUCTIONS.md",
    "README-ZH.md",
    "NEW_CODEX_PROMPT.md",
    "VERSION.json",
    "conversation/INITIALIZATION_DIALOGUE.md",
    "conversation/PROJECT_INPUT_TEMPLATE.md",
    "conversation/ATTACHMENT_ROUTING.md",
    "conversation/EXPERIMENT_HANDOFF_DIALOGUE.md",
    "builder/BUILDER_MISSION.md",
    "builder/EVALUATOR_DELIVERY_CONTRACT.md",
    "builder/SERVER_DISCOVERY_CHECKLIST.md",
    "builder/METHOD_MECHANISM_REQUIREMENTS.md",
    "skill/vla-acceleration-evaluator/SKILL.md",
    "skill/vla-acceleration-evaluator/references/retired-methods.md",
    "skill/vla-acceleration-evaluator/scripts/self_test.py",
    "evaluator/lightvla/01-fixed-observation-r6/benchmark_real_observation.py",
    "evaluator/lightvla/02-closed-loop-smoke-r1/closed_loop_paired.py",
    "evaluator/lightvla/03-multitask-paired-r1/closed_loop_paired.py",
    "evaluator/lightvla/04-adjudication-r1/closed_loop_paired.py",
    "evaluator/lightvla/05-weight-audit-r1/benchmark_real_observation.py",
    "evaluator/lightvla/launch-stage.sh",
    "evaluator/lightvla/status-stage.sh",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def safe_relative(value: str) -> bool:
    path = PurePosixPath(value)
    return (not path.is_absolute() and ".." not in path.parts and value == path.as_posix()
            and "\\" not in value and ":" not in value)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    version = json.loads((root / "VERSION.json").read_text(encoding="utf-8"))
    if version.get("version") != "2026-11-02-r6" or manifest.get("release") != "vla-codex-workbench-" + version["version"]:
        errors.append("unexpected release identity")

    entries = manifest.get("files")
    if not isinstance(entries, list) or manifest.get("file_count") != len(entries):
        errors.append("manifest count mismatch")
        entries = entries if isinstance(entries, list) else []

    seen: set[str] = set()
    for entry in entries:
        relative = entry.get("path")
        if not isinstance(relative, str) or not safe_relative(relative):
            errors.append(f"unsafe manifest path: {relative!r}")
            continue
        if relative in seen:
            errors.append(f"duplicate manifest path: {relative}")
            continue
        seen.add(relative)
        path = root / Path(*PurePosixPath(relative).parts)
        if not path.is_file():
            errors.append(f"missing file: {relative}")
            continue
        if path.stat().st_size != entry.get("size"):
            errors.append(f"size mismatch: {relative}")
        if digest(path) != entry.get("sha256"):
            errors.append(f"hash mismatch: {relative}")

    missing_required = sorted(REQUIRED.difference(seen))
    errors.extend(f"required file not manifested: {item}" for item in missing_required)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    errors.extend(f"unmanifested file: {p}" for p in sorted(actual - seen - {"release-manifest.json", "SHA256SUMS.txt"}))
    expected_sums = "".join(f"{e['sha256']}  {e['path']}\n" for e in entries)
    if (root / "SHA256SUMS.txt").read_text(encoding="utf-8") != expected_sums:
        errors.append("SHA256SUMS differs from release manifest")

    forbidden = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.name == "__pycache__"
        or path.suffix == ".pyc"
        or "vla-cache-openvla" in path.as_posix()
        or path.name == "profile_vla_cache_stages.py"
        or "vla_accelerators" in {part.lower() for part in path.parts}
        or ("autoprune" in path.name.lower() and path.suffix.lower() in {".py", ".sh", ".zip"})
    ]
    errors.extend(f"forbidden executable/cache artifact: {item}" for item in forbidden)

    toolbox = root / "skill" / "vla-acceleration-evaluator" / "assets" / "toolbox"
    for source in toolbox.rglob("*.py"):
        if "autoprune" in source.read_text(encoding="utf-8", errors="ignore").lower():
            errors.append(f"AutoPrune executable reference remains: {source.relative_to(root).as_posix()}")

    instructions = (root / "CODEX_INSTRUCTIONS.md").read_text(encoding="utf-8")
    for phrase in ("AutoPrune is permanently retired", "VLA-Cache is quarantined", "GRIDs/OpenPI is outside"):
        if phrase not in instructions:
            errors.append(f"missing authoritative boundary: {phrase}")

    one_line = (root / "ONE_LINE_TO_SEND.txt").read_text(encoding="utf-8").strip()
    if one_line != "请执行附件中的 START_HERE.md。":
        errors.append("unexpected one-line bootstrap message")

    start = (root / "START_HERE.md").read_text(encoding="utf-8")
    for phrase in ("verify_bundle.py", "PROJECT_STATUS.md", "COLLABORATION.md", "EVIDENCE_INDEX.json"):
        if phrase not in start:
            errors.append(f"missing bootstrap invariant: {phrase}")

    if errors:
        print("BUNDLE_VERIFY: FAIL")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(2)

    print(f"BUNDLE_VERIFY: PASS files={len(entries)}")
    print("VLA_CACHE_EXECUTABLE_ASSETS: ABSENT")
    print("AUTOPRUNE_EXECUTABLE_ASSETS: ABSENT_PERMANENTLY_RETIRED")
    print("LIGHTVLA_ACCEPTANCE_STAGES: 5")
    print("TARGET_EVALUATOR_BLUEPRINTS: SPECPRUNE,VLA-PRUNER")
    print("BOOTSTRAP_DIALOGUE: SELF_CONTAINED_ONE_USER_MESSAGE")


if __name__ == "__main__":
    main()

