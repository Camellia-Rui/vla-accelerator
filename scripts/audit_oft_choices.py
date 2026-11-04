"""Verify common-checkpoint three-mode closed-loop records before comparing outcomes."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def audit(report):
    protocol=report["protocol"]
    modes=("baseline","lightvla","vla_pruner")
    expected={(t,i,m) for t in protocol["tasks"] for i in protocol["trials"] for m in modes}
    rows={}
    for row in report["records"]:
        key=row["task_id"],row["trial"],row["mode"]
        if key in rows or key not in expected:raise ValueError("Duplicate or unexpected episode")
        if not row["complete"] or row["error"] is not None or type(row["success"]) is not bool:
            raise ValueError("Incomplete/error record")
        if not row["queries"]:raise ValueError("No real queries")
        for query in row["queries"]:
            if not math.isfinite(query["ms"]) or query["ms"]<=0:raise ValueError("Invalid timing")
            if not query["actions"] or any(len(a)!=7 or not all(math.isfinite(v) for v in a) for a in query["actions"]):
                raise ValueError("Invalid action chunk")
        rows[key]=row
    if rows.keys()!=expected:raise ValueError("Missing episode")
    contrasts=[]
    for t in protocol["tasks"]:
        for i in protocol["trials"]:
            group=[rows[t,i,m] for m in modes]
            if not group[0]["initial_state_hash"] or len({r["initial_state_hash"] for r in group})!=1 or len({r["seed"] for r in group})!=1:
                raise ValueError("Initial state/seed mismatch")
            contrasts.append(dict(task=t,trial=i,**{m:rows[t,i,m]["success"] for m in modes}))
    return dict(initial_states=len(contrasts),episodes=len(rows),
        successes={m:sum(r["success"] for r in rows.values() if r["mode"]==m) for m in modes},
        baseline_only_successes={m:sum(r["baseline"] and not r[m] for r in contrasts) for m in modes[1:]},
        method_only_successes={m:sum(r[m] and not r["baseline"] for r in contrasts) for m in modes[1:]},
        contrasts=contrasts,restored=report["restored_vs_native"]["matches"],
        parameters_preserved=report["parameters_preserved"],modules_restored=report["modules_restored"],
        quality_decision="INCONCLUSIVE",formal_speedup_accepted=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("report",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=audit(json.loads(args.report.read_text()))
    result["source_sha256"]=hashlib.sha256(args.report.read_bytes()).hexdigest()
    with args.output.open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
