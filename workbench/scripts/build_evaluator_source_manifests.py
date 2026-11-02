"""Identify local developer tooling separately from still-unverified runtime sources."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    for method, package in (("specprune", "specprune"), ("vla-pruner", "vla_pruner")):
        candidates = list((root / "evaluator/common").rglob("*.py"))
        candidates += list((root / "evaluator" / method).glob("*.py"))
        candidates += list((root / "skill/vla-acceleration-evaluator/assets/toolbox/src" / package).glob("*.py"))
        files = [{"path": p.relative_to(root).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in sorted(candidates)]
        value = {"method": method, "status": "INCOMPLETE_REAL_RUNTIME_VALIDATION", "files": files,
                 "runtime_source_verified": False, "checkpoint_verified": False,
                 "five_stage_chain_delivered": False, "gpu_experiments_run": False}
        if method == "vla-pruner":
            value["required_backend_commit"] = "84d4b7192c77abf1585610e2f12393319b7ebff9"
        path = root / "evaluator" / method / "SOURCE_MANIFEST.json"
        path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print("EVALUATOR_SOURCE_MANIFESTS: WRITTEN")


if __name__ == "__main__":
    main()
