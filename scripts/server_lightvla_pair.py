"""Bounded current-adapter LightVLA check, with isolated child caches and full actions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def worker(args):
    if args.unified_run:
        from core.run import main as run_main
        config = args.output / "config.json"
        config.write_text(json.dumps({"allow_existing_pruner":True}))
        return run_main(["--method","lightvla","--factory","methods.lightvla.oft_factory:build",
            "--factory-kwargs",json.dumps(dict(workspace=args.workspace,checkpoint=args.checkpoint,
                                               task_id=args.task,trial=args.trial)),
            "--config",str(config),"--mode","packaged","--output",str(args.output/"query.json")])
    import numpy as np
    import torch
    from methods.lightvla.oft_factory import build
    from methods.lightvla import LightVLAAccelerator
    from methods.vla_pruner.evaluation import actions_array, difference, trace
    report = {"passed":False,"closed_loop_tested":False,"formal_benchmark_valid":False,
              "quality_decision":"INCONCLUSIVE", "records":[]}
    adapter = None
    try:
        bundle = build(args.workspace, args.checkpoint, args.task, args.trial)
        model, run = bundle["model"], bundle["run"]
        report["provenance"] = bundle["provenance"]
        np.savez_compressed(args.output / "observation.npz", **bundle["observation"])
        parameters = dict(model.named_parameters())
        samples = {n:p.detach().flatten()[:64].cpu().clone() for n,p in parameters.items()}
        modules = {n:(m,m.forward) for n,m in model.named_modules()}
        def query():
            return actions_array(run())
        for _ in range(args.warmup):
            query()
        native, native_lengths = trace(model, query)
        report["native_actions"] = native.tolist()
        adapter = LightVLAAccelerator.install(model, mode="builtin", allow_existing_pruner=True)
        try:
            builtin = query()
            report["native_vs_builtin"] = difference(native, builtin)
            if not report["native_vs_builtin"]["matches"]:
                raise RuntimeError("Builtin mode differs from native")
            adapter.set_mode("packaged")
            packaged, lengths = trace(model, query)
            report["native_vs_packaged"] = difference(native, packaged)
            adapter.set_mode("baseline")
            baseline, baseline_lengths = trace(model, query)
            report["layer_lengths"] = dict(native=native_lengths, baseline=baseline_lengths, packaged=lengths)
            report["mechanism_observed"] = any(y<x for k,v in baseline_lengths.items() for x,y in zip(v,lengths.get(k,[])))
            for mode in ("baseline","packaged"):
                adapter.set_mode(mode)
                for _ in range(args.warmup):
                    query()
            for pair in range(args.pairs):
                for mode in (("baseline","packaged") if pair%2==0 else ("packaged","baseline")):
                    adapter.set_mode(mode)
                    torch.cuda.synchronize()
                    start=time.perf_counter()
                    actions=query()
                    torch.cuda.synchronize()
                    row=dict(pair=pair,mode=mode,query_ms=(time.perf_counter()-start)*1000,
                             actions=actions.tolist(),stats=adapter.get_stats())
                    report["records"].append(row)
                    with (args.output/"records.jsonl").open("a") as f:
                        f.write(json.dumps(row,allow_nan=False)+"\n")
                print(f"Pair {pair+1}/{args.pairs}",flush=True)
        finally:
            adapter.detach()
            report["detached"] = True
        report["restored_vs_native"] = difference(native,query())
        current = dict(model.named_parameters())
        report["parameter_identity_samples_preserved"] = current.keys()==parameters.keys() and all(
            current[n] is p and torch.equal(p.detach().flatten()[:64].cpu(),samples[n]) for n,p in parameters.items())
        report["module_forward_restored"] = dict(model.named_modules()).keys()==modules.keys() and all(
            dict(model.named_modules())[n] is m and m.forward==f for n,(m,f) in modules.items())
        times={mode:[r["query_ms"] for r in report["records"] if r["mode"]==mode] for mode in ("baseline","packaged")}
        report["descriptive_median_ratio"] = float(np.median(times["baseline"])/np.median(times["packaged"]))
        report["baseline_vs_packaged"] = difference(baseline,packaged)
        report["passed"] = all([report["mechanism_observed"], report["restored_vs_native"]["matches"],
            report["native_vs_packaged"]["matches"],
            report["parameter_identity_samples_preserved"],report["module_forward_restored"]])
    except Exception as exc:
        report["error"] = str(exc)
        traceback.print_exc()
    finally:
        (args.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    return 0 if report["passed"] else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--gpu",required=True)
    p.add_argument("--task",type=int,default=0)
    p.add_argument("--trial",type=int,default=0)
    p.add_argument("--pairs",type=int,default=4)
    p.add_argument("--warmup",type=int,default=3)
    p.add_argument("--worker",action="store_true")
    p.add_argument("--unified-run",action="store_true",help="exercise core run and save full action chunk")
    args=p.parse_args()
    if args.pairs<1 or args.warmup<1:
        p.error("positive pairs and warmup required")
    if args.worker:
        return worker(args)
    from core.runtime_env import runtime_environment
    args.output.mkdir(parents=True,exist_ok=False)
    env=runtime_environment(args.output/"cache",args.gpu)
    env.pop("LD_LIBRARY_PATH",None)
    env.pop("LD_PRELOAD",None)
    env.update(MUJOCO_GL="egl",PYOPENGL_PLATFORM="egl",MUJOCO_EGL_DEVICE_ID=str(args.gpu))
    with (args.output/"process.log").open("w") as log:
        result=subprocess.run([sys.executable,"-u","-B",str(Path(__file__).resolve()),*sys.argv[1:],"--worker"],env=env,stdout=log,stderr=subprocess.STDOUT,cwd=args.output)
    (args.output/"process.json").write_text(json.dumps({"exit_code":result.returncode}))
    print("RESULT:",args.output,"EXIT:",result.returncode)
    return result.returncode


if __name__=="__main__":
    raise SystemExit(main())
