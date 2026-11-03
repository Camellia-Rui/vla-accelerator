"""Audit paired closed-loop evidence without inferring deployment acceptance."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def summarize(path):
    report=json.loads(Path(path).read_text())
    protocol=report["protocol"]
    expected={(t,i,m) for t in protocol["tasks"] for i in protocol["trials"] for m in ("baseline","packaged")}
    records={}
    for row in report["records"]:
        key=(row["task_id"],row["trial"],row["mode"])
        if key in records or key not in expected:
            raise ValueError("Duplicate or unexpected record")
        if not row["complete"] or row["error"] is not None or type(row["success"]) is not bool:
            raise ValueError("Incomplete/error episode cannot be accepted")
        records[key]=row
    if records.keys()!=expected:
        raise ValueError("Missing episode records")
    pairs=[]
    for t in protocol["tasks"]:
        for i in protocol["trials"]:
            a,b=records[t,i,"baseline"],records[t,i,"packaged"]
            if not a.get("initial_state_hash") or a["initial_state_hash"]!=b["initial_state_hash"] or a["seed"]!=b["seed"]:
                raise ValueError("Pairing mismatch")
            pairs.append(dict(task=t,trial=i,baseline_success=a["success"],packaged_success=b["success"],
                baseline_episode_seconds=a["episode_seconds"],packaged_episode_seconds=b["episode_seconds"]))
    outcomes=Counter((p["baseline_success"],p["packaged_success"]) for p in pairs)
    timing={m:[q["ms"] for r in records.values() if r["mode"]==m for q in r["queries"]] for m in ("baseline","packaged")}
    return dict(source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),paired_episodes=len(pairs),
        baseline_successes=sum(p["baseline_success"] for p in pairs),
        packaged_successes=sum(p["packaged_success"] for p in pairs),
        paired_outcomes={f"baseline_{a}_packaged_{b}":n for (a,b),n in outcomes.items()},
        descriptive_query_median_ms={m:statistics.median(v) for m,v in timing.items()},
        pairs=pairs,restored=report.get("restored_vs_native",{}).get("matches",False),
        parameters_preserved=report.get("parameters_preserved",False),modules_restored=report.get("modules_restored",False),
        quality_decision="INCONCLUSIVE",formal_speedup_accepted=False,
        note="Small smoke sample; query counts/trajectories may differ. Not fixed-observation paired speed evidence.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=summarize(args.report)
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
