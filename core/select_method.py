"""Choose the lowest measured latency among quality-qualified modes of one paired run."""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import mean
import subprocess
import sys

from core.project import plan, read_project

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def paired_metrics(report):
    if report.get("passed") is not True or report.get("closed_loop_tested") is not True:
        raise ValueError("Completed closed-loop comparison required")
    protocol = report["protocol"]
    modes = protocol.get("modes", ["baseline", "packaged"])
    if modes not in (["baseline", "packaged"], ["baseline", "lightvla", "vla_pruner"], ["baseline", "fastv_scoring", "vla_pruner"]):
        raise ValueError("Unknown comparison protocol")
    if "vla_pruner" in modes:
        from scripts.audit_oft_choices import audit
        checks = audit(report)
        if not all(checks[k] for k in ("restored", "parameters_preserved", "modules_restored")):
            raise ValueError("Model restoration checks failed")
    expected = set(itertools.product(protocol["tasks"], protocol["trials"]))
    if not expected:
        raise ValueError("Empty task protocol")
    groups = {m: {} for m in modes}
    timings = {m: [] for m in modes}
    for row in report["records"]:
        mode = row["mode"]
        pair = (row["task_id"], row["trial"])
        if (mode not in groups or pair not in expected or pair in groups[mode]
                or row.get("error") or row["complete"] is not True or type(row["success"]) is not bool
                or not row["initial_state_hash"] or row["seed"] != protocol["seed"]):
            raise ValueError("Invalid or duplicate episode")
        groups[mode][pair] = (row["initial_state_hash"], row["success"])
        times = [q["ms"] for q in row["queries"]]
        if not times or not all(math.isfinite(t) and t > 0 for t in times):
            raise ValueError("Invalid inference latency")
        timings[mode].extend(times)
    results = {}
    for mode, rows in groups.items():
        if rows.keys() != expected or any(rows[k][0] != groups["baseline"].get(k, (None,))[0] for k in rows):
            raise ValueError("Missing or mismatched initial states")
        name = {"packaged": "lightvla", "lightvla": "lightvla_style"}.get(mode, mode)
        results[name] = (sum(v[1] for v in rows.values()), len(rows), mean(timings[mode]))
    return results


def metrics(path):
    r = read(path)
    if r.get("backend") in ("oft", "openvla") and r.get("k") == 2 and r.get("prune_ratio") == 0.5:
        from methods.fastv.evaluation import openvla_summary
        result = {}
        keys = None
        runtime = None
        for mode in ("baseline", "fastv"):
            folder = path.parent / mode
            n = len(r["tasks"]) * r["trials_per_task"]
            if r["backend"] == "openvla":
                s = openvla_summary(folder, n)
                result[mode] = (s["successes"], n, s["latency_excluding_first_5_calls"]["mean_ms"])
                continue
            meta = read(folder / "metadata.json")
            if (meta["status"] != "completed" or meta["method"] != mode or meta["seed"] != r["seed"]
                    or meta["task_ids"] != r["tasks"] or meta["trials_per_task"] != r["trials_per_task"]):
                raise ValueError("FastV run metadata mismatch")
            signature = [meta[k] for k in ("checkpoint","task_suite","attention","precision","num_images",
                                            "actions_per_chunk","physical_gpu","source_sha256","timing_scope")]
            if runtime is not None and signature != runtime:
                raise ValueError("FastV model or runtime differs from baseline")
            runtime = signature
            with (folder / "episodes.csv").open() as stream:
                episodes = list(csv.DictReader(stream))
            states = {(int(e["task_id"]), int(e["episode_id"])): e["initial_state_sha256"] for e in episodes}
            expected = set(itertools.product(r["tasks"], range(r["trials_per_task"])))
            if (len(episodes) != n or states.keys() != expected or not all(states.values())
                    or keys is not None and keys != states or any(e["success"] not in ("True","False") for e in episodes)):
                raise ValueError("Incomplete FastV state pairing")
            keys = states
            with (folder / "actions.csv").open() as stream:
                times = [float(row["latency_ms"]) for row in csv.DictReader(stream)]
            if (len(times) != sum(int(e["inference_calls"]) for e in episodes) or len(times) <= 5
                    or not all(math.isfinite(t) and t > 0 for t in times)):
                raise ValueError("Incomplete FastV timing")
            result[mode] = (sum(e["success"] == "True" for e in episodes), n, mean(times[5:]))
        return result
    if r.get("execution_complete") and "paired_initial_states" in r:
        from scripts.audit_specprune import audit
        scores = {"baseline": [], "specprune": []}
        times = {m: [] for m in scores}
        seen = set()
        for item in r["records"]:
            key = (item["task"], item["trial"])
            if key in seen:
                raise ValueError("Duplicate SpecPrune pair")
            seen.add(key)
            child = path.parent / Path(item["directory"]).name / "report.json"
            checked = audit(child)
            if checked["source_sha256"] != item["audit"]["source_sha256"]:
                raise ValueError("SpecPrune evidence changed")
            for e in checked["episodes"]:
                if (e["task"], e["trial"]) != key:
                    raise ValueError("SpecPrune pair mismatch")
                scores["baseline" if e["mode"] == "baseline" else "specprune"].append(e["success"])
            for t in read(child)["timings"]:
                times["baseline" if t["mode"] == "baseline" else "specprune"].append(t["ms"])
        if not seen or len(seen) != r["paired_initial_states"]:
            raise ValueError("Incomplete SpecPrune batch")
        return {m:(sum(v), len(v), mean(times[m])) for m,v in scores.items()}
    return paired_metrics(r)


def decide(scores, min_success, min_pairs, max_drop):
    base_wins, n, base_ms = scores["baseline"]
    rows = {}
    for method, (wins, count, ms) in scores.items():
        if count != n or not n or not 0 <= wins <= n or not math.isfinite(ms) or ms <= 0:
            raise ValueError("Invalid metrics")
        reasons = []
        if n < min_pairs:
            reasons.append("too_few_pairs")
        if 100 * wins < min_success * n:
            reasons.append("below_success_threshold")
        if 100 * (base_wins - wins) > max_drop * n:
            reasons.append("excess_success_drop")
        if method != "baseline" and ms >= base_ms:
            reasons.append("no_speedup")
        rows[method] = dict(successes=wins, episodes=n, success_pct=100*wins/n, mean_ms=ms,
                            speedup=base_ms/ms, exclusions=reasons)
    eligible = [m for m in rows if not rows[m]["exclusions"]]
    winner = min(eligible, key=lambda m: rows[m]["mean_ms"], default=None)
    return dict(selected=winner, candidates=rows, scope="observed paired sample",
                policy=dict(min_success_pct=min_success, min_pairs=min_pairs, max_drop_pp=max_drop))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--report", type=Path)
    source.add_argument("--project", type=Path)
    p.add_argument("--output", required=True, type=Path, help="new directory")
    p.add_argument("--min-success-pct", required=True, type=float)
    p.add_argument("--max-drop-pp", required=True, type=float)
    p.add_argument("--min-pairs", required=True, type=int)
    p.add_argument("--trials-per-task", type=int, default=1)
    p.add_argument("--tasks", nargs="+", type=int, help="project task IDs; default all ten")
    p.add_argument("--oft-cublas-dir", type=Path, help="process-local cuBLAS override for oft_object")
    p.add_argument("--oft-comparison", choices=("scoring", "lightvla-style"),
                   help="OFT Object comparison, default scoring")
    args = p.parse_args(argv)
    if not (0 <= args.min_success_pct <= 100 and 0 <= args.max_drop_pp <= 100
            and args.min_pairs >= 1 and 1 <= args.trials_per_task <= 50):
        p.error("Invalid quality policy or trial count")
    out = args.output.resolve()
    tasks = args.tasks or list(range(10))
    if len(tasks) != len(set(tasks)) or not set(tasks) <= set(range(10)):
        p.error("tasks must be distinct IDs in 0..9")
    if args.report and (args.tasks or args.trials_per_task != 1 or args.oft_cublas_dir or args.oft_comparison):
        p.error("task options are only supported with --project")
    try:
        out.mkdir(parents=True, exist_ok=False)
        command = None
        path = args.report
        if args.project:
            data = read_project(args.project)
            data["output_root"] = str(out)
            planned = plan(data, "measurement")
            if planned["blockers"]:
                raise ValueError(str(planned["blockers"]))
            command = planned["command"]
            profile = data["profile"]
            if profile != "oft_object" and (args.oft_cublas_dir or args.oft_comparison):
                raise ValueError("OFT comparison options require the oft_object profile")
            trials = [str(i) for i in range(args.trials_per_task)]
            suite = ["--tasks", *map(str, tasks), "--trials", *trials]
            if profile == "oft_object":
                command = [data["python"], "-B", str(ROOT/"scripts/server_oft_compare.py"),
                           *command[7:]]
                command.remove("--prefer-env-cuda")
                command += ["--closed-loop", *suite]
                if args.oft_comparison != "lightvla-style":
                    command.append("--scoring")
                if args.oft_cublas_dir:
                    command += ["--cublas-dir", str(args.oft_cublas_dir.resolve())]
            elif profile == "specprune_object":
                command = [data["python"], "-B", str(ROOT/"scripts/batch_specprune.py"), *command[7:], *suite]
            elif profile == "lightvla_spatial":
                command += ["--closed-loop", *suite]
            else:
                command += ["--trials", str(args.trials_per_task)]
                if args.tasks and profile == "fastv_oft_spatial":
                    command += ["--tasks", *map(str, tasks)]
                elif tasks != list(range(10)):
                    raise ValueError("Original OpenVLA runner evaluates all ten tasks")
            subprocess.run(command, cwd=ROOT, check=True)
            path = out / "measurement/report.json"
        scores = metrics(path)
        result = decide(scores, args.min_success_pct, args.min_pairs, args.max_drop_pp)
        result.update(report=str(path.resolve()), report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      command=command)
        source_report = read(path)
        result["checkpoint"] = source_report.get("checkpoint") or source_report.get("provenance", {}).get("checkpoint")
        if result["selected"] in ("vla_pruner", "fastv_scoring"):
            result["configuration"] = dict(source_report["vla_pruner_config"])
            if result["selected"] == "fastv_scoring":
                result["configuration"]["vla_pruner_mode"] = "fastv"
        elif result["selected"] == "fastv":
            result["configuration"] = {k:source_report[k] for k in ("backend","k","prune_ratio")}
        dest = out / "selection.json"
        dest.write_text(json.dumps(result, indent=2, allow_nan=False))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, KeyError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        print(f"Selection failed: {exc}", file=sys.stderr)
        return 2

