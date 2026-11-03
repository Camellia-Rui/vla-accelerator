"""Recompute paired episode counts; never promote historical timing to current acceptance."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(baseline, packaged):
    def load(path):
        records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        result = {}
        for row in records:
            key = (row["task_suite"], row["task_id"], row["episode"])
            if key in result:
                raise ValueError("Duplicate episode")
            if type(row.get("success")) is not bool or row.get("error") is not None:
                raise ValueError("Invalid or failed execution record")
            result[key] = row
        return result
    a,b=load(baseline),load(packaged)
    if not a or a.keys()!=b.keys():
        raise ValueError("Missing or unmatched episode pairs")
    for key in a:
        for field in ("initial_state_hash", "seed", "checkpoint", "task_description"):
            if not a[key].get(field) and field != "seed":
                raise ValueError("Missing pairing metadata: "+field)
            if a[key][field]!=b[key][field]:
                raise ValueError("Mismatched pairing metadata: "+field)
    return {"paired_episodes":len(a), "baseline_successes":sum(r["success"] for r in a.values()),
        "packaged_successes":sum(r["success"] for r in b.values()),
        "baseline_only_success":sum(a[k]["success"] and not b[k]["success"] for k in a),
        "packaged_only_success":sum(b[k]["success"] and not a[k]["success"] for k in a),
        "device_mismatch_pairs":sum(a[k]["device"]!=b[k]["device"] for k in a),
        "current_adapter_acceptance":False, "formal_speedup_accepted":False,
        "source_sha256":{Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (baseline,packaged)}}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline",type=Path)
    parser.add_argument("packaged",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=audit(args.baseline,args.packaged)
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
