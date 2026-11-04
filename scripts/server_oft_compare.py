"""Same-weight, same-input OFT baseline/LightVLA-style/VLA-Pruner comparison."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def worker(args):
    if args.closed_loop:
        from scripts.server_oft_closed_loop import worker as closed_worker
        return closed_worker(args)
    import numpy as np
    import torch
    from methods.vla_pruner.accelerator import AcceleratorConfig,VLAPrunerAccelerator
    from methods.vla_pruner.runtime import load_bundle
    from methods.vla_pruner.evaluation import actions_array,difference,trace,query
    from methods.lightvla import LightVLAAccelerator
    from scripts.server_lightvla_closed_loop import atomic
    report=dict(passed=False,closed_loop_tested=False,quality_decision="INCONCLUSIVE",records=[])
    try:
        cfg=AcceleratorConfig(checkpoint=args.checkpoint,unnorm_key=args.unnorm_key,fastv_r=args.prune_ratio)
        report["vla_pruner_config"]=asdict(cfg)
        bundle=load_bundle(SimpleNamespace(workspace=Path(args.workspace),factory=None,
            observation_dir=[Path(p) for p in args.observation_dir]),cfg)
        model=bundle["model"]
        report["provenance"]=bundle["provenance"]
        report["lightvla_scope"]="Inference intervention on unchanged OFT weights, not original fine-tuned method reproduction"
        report["environment"]=dict(torch=torch.__version__,transformers=__import__("transformers").__version__,
            cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))
        params=dict(model.named_parameters())
        samples={n:p.detach().flatten()[:64].cpu().clone() for n,p in params.items()}
        refs=[]
        report["inputs"]=[]
        for o in bundle["observations"]:
            bundle["native_reset"]()
            for _ in range(args.warmup): bundle["native_query"](o)
            value,lengths=trace(model,lambda:actions_array(bundle["native_query"](o)))
            refs.append(value)
            report["inputs"].append(dict(id=o["id"],native_actions=value.tolist(),native_lengths=lengths))
        modes=("baseline","lightvla","vla_pruner")
        for i,o in enumerate(bundle["observations"]):
            for pair in range(args.pairs):
                for mode in modes[pair%3:]+modes[:pair%3]:
                    adapter=None
                    try:
                        bundle["native_reset"]()
                        if mode=="lightvla":
                            adapter=LightVLAAccelerator.install(model,mode="packaged")
                        elif mode=="vla_pruner":
                            adapter=VLAPrunerAccelerator.install(model,config=cfg,
                                **{k:bundle[k] for k in ("processor","action_head","proprio_projector")})
                        def invoke():
                            return query(adapter,o) if mode=="vla_pruner" else actions_array(bundle["native_query"](o))
                        for _ in range(args.warmup): invoke()
                        torch.cuda.synchronize()
                        tick=time.perf_counter()
                        value=invoke()
                        torch.cuda.synchronize()
                        elapsed=(time.perf_counter()-tick)*1000
                        _,lengths=trace(model,invoke)
                        mechanism=any(y<x for k,v in report["inputs"][i]["native_lengths"].items() for x,y in zip(v,lengths.get(k,[])))
                        if mode!="baseline" and not mechanism:
                            raise RuntimeError(f"No actual computation shortening: {mode}")
                        if mode=="baseline" and not difference(refs[i],value)["matches"]:
                            raise RuntimeError("Native baseline drift")
                        row=dict(input=i,pair=pair,mode=mode,query_ms=elapsed,actions=value.tolist(),
                            difference=difference(refs[i],value),layer_lengths=lengths,mechanism_observed=mechanism,
                            stats=adapter.get_stats() if adapter else {})
                    finally:
                        if adapter is not None: adapter.detach()
                    bundle["native_reset"]()
                    restored=difference(refs[i],bundle["native_query"](o))
                    if not restored["matches"]: raise RuntimeError("Native output not restored")
                    row["restored"]=restored
                    report["records"].append(row)
                    atomic(args.output/"report.json",report)
                print(f"Input {i+1} pair {pair+1}/{args.pairs}",flush=True)
        current=dict(model.named_parameters())
        report["parameters_preserved"]=current.keys()==params.keys() and all(current[n] is p and
            torch.equal(p.detach().flatten()[:64].cpu(),samples[n]) for n,p in params.items())
        report["descriptive_median_ms"]={mode:float(np.median([r["query_ms"] for r in report["records"] if r["mode"]==mode])) for mode in modes}
        report["passed"]=report["parameters_preserved"]
    except Exception as exc:
        report["error"]=f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        atomic(args.output/"report.json",report)
    return 0 if report["passed"] else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("workspace","checkpoint","gpu"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--observation-dir",action="append",required=True)
    p.add_argument("--unnorm-key",default="libero_object_no_noops")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--pairs",type=int,default=3)
    p.add_argument("--warmup",type=int,default=3)
    p.add_argument("--prune-ratio",type=float,default=0.5)
    p.add_argument("--closed-loop",action="store_true")
    p.add_argument("--tasks",type=int,nargs="+",default=[0,1])
    p.add_argument("--trials",type=int,nargs="+",default=[0,1,2])
    p.add_argument("--worker",action="store_true")
    args=p.parse_args()
    if min(args.pairs,args.warmup)<1:p.error("positive pairs and warmup required")
    args.output=args.output.resolve()
    if args.worker:return worker(args)
    from core.runtime_env import runtime_environment
    args.output.mkdir(parents=True,exist_ok=False)
    env=runtime_environment(args.output/"cache",args.gpu,True)
    if args.closed_loop:
        env.update(MUJOCO_GL="egl",PYOPENGL_PLATFORM="egl",MUJOCO_EGL_DEVICE_ID=str(args.gpu))
    with (args.output/"process.log").open("w") as log:
        result=subprocess.run([sys.executable,"-u","-B",str(Path(__file__).resolve()),*sys.argv[1:],"--worker"],
            env=env,stdout=log,stderr=subprocess.STDOUT,cwd=args.output)
    (args.output/"process.json").write_text(json.dumps(dict(exit_code=result.returncode)))
    print("RESULT",args.output,"EXIT",result.returncode)
    return result.returncode


if __name__=="__main__":raise SystemExit(main())
