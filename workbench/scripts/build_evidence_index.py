"""Index bundled evidence without changing results or asserting new acceptance."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    files = []
    for path in sorted((root / "archive/excluded-vla-cache-evidence").glob("*")):
        if not path.is_file():
            continue
        item = {"path": path.relative_to(root).as_posix(), "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "kind": "archived_raw_result" if path.suffix == ".json" else "archived_log",
                "method": "vla-cache", "deployment": "quarantined"}
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            item.update(recorded_status=data.get("status"), recorded_decision=data.get("decision"))
            if path.name.startswith("09_"):
                item["recorded_summary"] = {"native_successes": data["native"]["successes"],
                    "cache_successes": data["cache_once_memo"]["successes"],
                    "episodes_per_mode": data["native"]["episodes"],
                    "query_median_speedup": data["comparison"]["query_median_speedup"]}
        files.append(item)
    missing = [{"method": "lightvla", "path_hint": f"results/{name}.json", "status": "NOT_BUNDLED"}
               for name in ("01_real_fixed_observation_r6", "02_closed_loop_smoke_r1",
                   "03_multitask_paired_r1", "04_targeted_adjudication_r1", "05_weight_audit_r1")]
    missing += [{"method": name, "status": "INCOMPLETE_REAL_RUNTIME_VALIDATION",
                 "needed": "runtime probe, mechanism records, paired timing/actions, closed-loop, active weight/detach audit"}
                for name in ("specprune", "vla-pruner")]
    index = {"schema_version": 1, "release": json.loads((root / "VERSION.json").read_text(encoding="utf-8"))["version"], "files": files,
             "missing_or_unvalidated": missing,
             "historical_context": ["history/VLA_PROJECT_HANDOFF_2026-10-30.md",
                 "skill/vla-acceleration-evaluator/references/project-state.md",
                 "skill/vla-acceleration-evaluator/assets/toolbox/SOURCE_MANIFEST.json"],
             "limits": ["Recorded summaries are not newly reproduced experiments.",
                 "LightVLA stage 04 is conditional, not universally required.",
                 "Stage 05 sampled weight audit cannot retroactively certify earlier closed-loop runs.",
                 "Unit-test synthetic-only.json is not experiment evidence."]}
    (root / "EVIDENCE_INDEX.json").write_text(json.dumps(index, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"EVIDENCE_INDEX_BUILT archived_files={len(files)}")


if __name__ == "__main__":
    main()
