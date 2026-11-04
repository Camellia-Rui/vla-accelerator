"""Three-mode LIBERO-Object behavior check on one unchanged OFT checkpoint."""
from collections import deque
from dataclasses import asdict
import hashlib
import json
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


def worker(args):
    import numpy as np
    import torch
    from methods.vla_pruner.accelerator import AcceleratorConfig
    from methods.vla_pruner.runtime import load_bundle
    from methods.vla_pruner.evaluation import difference
    from methods.oft_choices import OFTChoice
    from scripts.server_lightvla_closed_loop import atomic
    report=dict(passed=False,quality_decision="INCONCLUSIVE",records=[],closed_loop_tested=True)
    session=None
    try:
        cfg=AcceleratorConfig(checkpoint=args.checkpoint,unnorm_key=args.unnorm_key,fastv_r=args.prune_ratio)
        bundle=load_bundle(SimpleNamespace(workspace=Path(args.workspace),factory=None,
            observation_dir=[Path(p) for p in args.observation_dir]),cfg)
        from libero.libero import benchmark
        from experiments.robot.libero import libero_utils as lu
        from experiments.robot.openvla_utils import resize_image_for_policy
        from experiments.robot.robot_utils import get_image_resize_size,set_seed_everywhere,normalize_gripper_action,invert_gripper_action
        suite=benchmark.get_benchmark_dict()["libero_object"]()
        size=get_image_resize_size(SimpleNamespace(model_family="openvla"))
        model=bundle["model"]
        params=dict(model.named_parameters())
        samples={n:p.detach().flatten()[:64].cpu().clone() for n,p in params.items()}
        modules={n:(m,m.forward) for n,m in model.named_modules()}
        native=bundle["native_query"](bundle["observations"][0])
        session=OFTChoice(bundle,cfg)
        report.update(provenance=bundle["provenance"],vla_pruner_config=asdict(cfg),
            protocol=dict(tasks=args.tasks,trials=args.trials,seed=7,max_steps=280,settle_steps=10,modes=list(session.modes)))
        for mode in session.modes:
            session.select(mode)
            for _ in range(3):session.query(bundle["observations"][0])
        for task_id in args.tasks:
            task=suite.get_task(task_id)
            states=suite.get_task_init_states(task_id)
            for trial in args.trials:
                state=np.asarray(states[trial],dtype="<f8")
                state_hash=hashlib.sha256(json.dumps(list(state.shape)).encode()+state.tobytes()).hexdigest()
                offset=(args.tasks.index(task_id)*len(args.trials)+args.trials.index(trial))%3
                for mode in session.modes[offset:]+session.modes[:offset]:
                    env=None
                    row=dict(task_id=task_id,trial=trial,mode=mode,seed=7,initial_state_hash=state_hash,
                        success=False,error=None,complete=False,queries=[],executed_actions=[])
                    try:
                        set_seed_everywhere(7)
                        session.select(mode)
                        env,instruction=lu.get_libero_env(task,"openvla",resolution=256)
                        env.reset()
                        obs=env.set_init_state(states[trial])
                        for _ in range(10):obs,_,_,_=env.step(lu.get_libero_dummy_action("openvla"))
                        queue=deque()
                        started=time.perf_counter()
                        for step in range(280):
                            if not queue:
                                observation=dict(full_image=resize_image_for_policy(lu.get_libero_image(obs),size),
                                    wrist_image=resize_image_for_policy(lu.get_libero_wrist_image(obs),size),
                                    state=np.concatenate((obs["robot0_eef_pos"],lu.quat2axisangle(obs["robot0_eef_quat"]),obs["robot0_gripper_qpos"])),instruction=instruction)
                                torch.cuda.synchronize()
                                tick=time.perf_counter()
                                actions=session.query(observation)
                                torch.cuda.synchronize()
                                row["queries"].append(dict(ms=(time.perf_counter()-tick)*1000,actions=actions.tolist(),stats=session.stats()))
                                queue.extend(actions)
                            action=invert_gripper_action(normalize_gripper_action(queue.popleft().copy(),binarize=True))
                            row["executed_actions"].append(action.tolist())
                            obs,_,done,_=env.step(action.tolist())
                            if done:
                                row["success"]=True
                                break
                        row.update(complete=True,control_steps=step+1,episode_seconds=time.perf_counter()-started)
                    except Exception as exc:
                        row["error"]=f"{type(exc).__name__}: {exc}"
                        traceback.print_exc()
                    finally:
                        if env is not None:env.close()
                        atomic(args.output/f"{task_id}-{trial}-{mode}.json",row)
                    report["records"].append(row)
                    atomic(args.output/"progress.json",dict(expected=len(args.tasks)*len(args.trials)*3,
                        completed=len(report["records"]),successes={m:sum(r["success"] for r in report["records"] if r["mode"]==m) for m in session.modes}))
                    print(f"task={task_id} trial={trial} mode={mode} success={row['success']} error={row['error']}",flush=True)
                    if row["error"]:raise RuntimeError("Episode execution error")
        session.close()
        report["restored_vs_native"]=difference(native,bundle["native_query"](bundle["observations"][0]))
        current=dict(model.named_parameters())
        report["parameters_preserved"]=current.keys()==params.keys() and all(current[n] is p and torch.equal(p.detach().flatten()[:64].cpu(),samples[n]) for n,p in params.items())
        current_modules=dict(model.named_modules())
        report["modules_restored"]=current_modules.keys()==modules.keys() and all(current_modules[n] is m and m.forward==f for n,(m,f) in modules.items())
        report["successes"]={m:sum(r["success"] for r in report["records"] if r["mode"]==m) for m in session.modes}
        report["passed"]=report["parameters_preserved"] and report["modules_restored"] and report["restored_vs_native"]["matches"]
    except Exception as exc:
        report["error"]=f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        if session is not None:session.close()
        atomic(args.output/"report.json",report)
    return 0 if report["passed"] else 2
