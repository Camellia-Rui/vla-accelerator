"""Paired LIBERO-Spatial smoke using the current reversible LightVLA adapter."""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def atomic(path, value):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")
    temp.replace(path)


def summarize(records, expected):
    valid=[r for r in records if r.get("error") is None and r.get("complete")]
    counts={mode:sum(r["success"] for r in valid if r["mode"]==mode) for mode in ("baseline","packaged")}
    return dict(expected_records=expected,valid_records=len(valid),error_records=len(records)-len(valid),
        successes=counts,execution_complete=len(valid)==expected,
        quality_decision="INCONCLUSIVE",formal_benchmark_valid=False,
        reason="Small smoke sample; no user quality tolerance or statistical acceptance supplied")


def worker(args):
    import numpy as np
    import torch
    from methods.lightvla.oft_factory import build
    from methods.lightvla import LightVLAAccelerator
    from methods.vla_pruner.evaluation import actions_array,difference
    report=dict(records=[],closed_loop_tested=True,passed=False)
    try:
        bundle=build(args.workspace,args.checkpoint,seed=args.seed)
        from libero.libero import benchmark
        from experiments.robot.libero.libero_utils import get_libero_env,get_libero_dummy_action
        from experiments.robot.robot_utils import normalize_gripper_action,invert_gripper_action,set_seed_everywhere
        suite=benchmark.get_benchmark_dict()["libero_spatial"]()
        model=bundle["model"]
        report["provenance"]=bundle["provenance"]
        report["protocol"]=dict(tasks=args.tasks,trials=args.trials,seed=args.seed,max_steps=220,
            settle_steps=10,action_chunk=8,mode_order="alternating",fresh_env_per_episode=True)
        report["environment"]=dict(python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
                                    visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
        parameters=dict(model.named_parameters())
        samples={n:p.detach().flatten()[:64].cpu().clone() for n,p in parameters.items()}
        modules={n:(m,m.forward) for n,m in model.named_modules()}
        native=actions_array(bundle["run"]())
        adapter=LightVLAAccelerator.install(model,mode="baseline",allow_existing_pruner=True)
        try:
            # Warm each mode on a fixed input outside episode timing.
            for mode in ("baseline","packaged"):
                adapter.set_mode(mode)
                for _ in range(3):
                    bundle["run"]()
            for task_id in args.tasks:
                task=suite.get_task(task_id)
                states=suite.get_task_init_states(task_id)
                for trial in args.trials:
                    state=np.asarray(states[trial],dtype="<f8")
                    state_hash=hashlib.sha256(json.dumps(list(state.shape)).encode()+state.tobytes()).hexdigest()
                    pair_index=args.tasks.index(task_id)*len(args.trials)+args.trials.index(trial)
                    modes=("baseline","packaged") if pair_index%2==0 else ("packaged","baseline")
                    for mode in modes:
                        path=args.output/f"{task_id}-{trial}-{mode}.json"
                        if path.exists():
                            previous=json.loads(path.read_text())
                            if previous.get("complete") and previous.get("error") is None:
                                report["records"].append(previous)
                                continue
                        row=dict(task_id=task_id,trial=trial,mode=mode,seed=args.seed,initial_state_hash=state_hash,
                            success=False,complete=False,error=None,queries=[],executed_actions=[],control_steps=0)
                        env=None
                        try:
                            set_seed_everywhere(args.seed)
                            adapter.set_mode(mode)
                            adapter.reset()
                            env,instruction=get_libero_env(task,"openvla",resolution=256)
                            env.reset()
                            obs=env.set_init_state(states[trial])
                            for _ in range(10):
                                obs,_,_,_=env.step(get_libero_dummy_action("openvla"))
                            queue=deque()
                            started=time.perf_counter()
                            for step in range(220):
                                if not queue:
                                    observation=bundle["prepare"](obs)
                                    torch.cuda.synchronize()
                                    tick=time.perf_counter()
                                    actions=actions_array(bundle["query"](observation,instruction))
                                    torch.cuda.synchronize()
                                    row["queries"].append(dict(ms=(time.perf_counter()-tick)*1000,
                                        actions=actions.tolist(),stats=adapter.get_stats()))
                                    queue.extend(actions)
                                action=invert_gripper_action(normalize_gripper_action(queue.popleft().copy(),binarize=True))
                                row["executed_actions"].append(action.tolist())
                                obs,_,done,_=env.step(action.tolist())
                                row["control_steps"]=step+1
                                if done:
                                    row["success"]=True
                                    break
                            row["episode_seconds"]=time.perf_counter()-started
                            row["complete"]=True
                        except Exception as exc:
                            row["error"]=f"{type(exc).__name__}: {exc}"
                            traceback.print_exc()
                        finally:
                            if env is not None:
                                env.close()
                            atomic(path,row)
                        report["records"].append(row)
                        atomic(args.output/"progress.json",summarize(report["records"],len(args.tasks)*len(args.trials)*2))
                        print(f"task={task_id} trial={trial} mode={mode} success={row['success']} error={row['error']}",flush=True)
                        if row["error"]:
                            raise RuntimeError("Episode runtime error; stop and inspect before retry")
        finally:
            adapter.detach()
        report["restored_vs_native"]=difference(native,bundle["run"]())
        current=dict(model.named_parameters())
        report["parameters_preserved"]=current.keys()==parameters.keys() and all(current[n] is p and
            torch.equal(p.detach().flatten()[:64].cpu(),samples[n]) for n,p in parameters.items())
        current_modules=dict(model.named_modules())
        report["modules_restored"]=current_modules.keys()==modules.keys() and all(current_modules[n] is m and m.forward==f for n,(m,f) in modules.items())
        report.update(summarize(report["records"],len(args.tasks)*len(args.trials)*2))
        report["passed"]=report["execution_complete"] and report["restored_vs_native"]["matches"] and report["parameters_preserved"] and report["modules_restored"]
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
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--tasks",type=int,nargs="+",default=[0,1])
    p.add_argument("--trials",type=int,nargs="+",default=[0,1,2])
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--worker",action="store_true")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if len(set(args.tasks))!=len(args.tasks) or len(set(args.trials))!=len(args.trials) or min(args.tasks+args.trials)<0:
        p.error("tasks and trials must be unique nonnegative indices")
    args.output=args.output.resolve()
    if args.worker:
        return worker(args)
    args.output.mkdir(parents=True,exist_ok=args.resume)
    config=dict(workspace=args.workspace,checkpoint=args.checkpoint,gpu=args.gpu,tasks=args.tasks,trials=args.trials,seed=args.seed)
    config_path=args.output/"run-config.json"
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        p.error("Resume configuration mismatch")
    atomic(config_path,config)
    lock=args.output/"worker.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    try:
        from core.runtime_env import runtime_environment
        env=runtime_environment(args.output/"cache",args.gpu)
        env.pop("LD_LIBRARY_PATH",None)
        env.pop("LD_PRELOAD",None)
        env.update(MUJOCO_GL="egl",PYOPENGL_PLATFORM="egl",MUJOCO_EGL_DEVICE_ID=str(args.gpu))
        with (args.output/"process.log").open("a") as log:
            result=subprocess.run([sys.executable,"-u","-B",str(Path(__file__).resolve()),*sys.argv[1:],"--worker"],
                env=env,stdout=log,stderr=subprocess.STDOUT,cwd=args.output)
        atomic(args.output/"process.json",dict(exit_code=result.returncode))
        return result.returncode
    finally:
        lock.unlink()


if __name__=="__main__":
    raise SystemExit(main())
