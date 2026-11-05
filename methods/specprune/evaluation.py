"""Experimental SpecPrune real-input lifecycle probe; not deployment admission."""
import argparse
import copy
import json
from pathlib import Path
import sys
import traceback
import time
import hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main(argv=None):
 argv=list(sys.argv[1:] if argv is None else argv)
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('workspace','checkpoint','observation','output'): p.add_argument('--'+n,required=True)
 p.add_argument('--closed-loop',action='store_true')
 p.add_argument('--task',type=int,default=0)
 p.add_argument('--trial',type=int,default=0)
 p.add_argument('--pairs',type=int,default=4)
 p.add_argument('--warmup',type=int,default=3)
 p.add_argument('--gpu',default=None)
 p.add_argument('--_worker',action='store_true')
 args=p.parse_args(argv)
 if not 0<=args.task<10 or args.trial<0 or args.pairs<2 or args.warmup<1:
  p.error('task must be 0..9, trial >= 0, pairs >= 2, warmup >= 1')
 for key in ('workspace','checkpoint','observation','output'): setattr(args,key,str(Path(getattr(args,key)).resolve()))
 out=Path(args.output)
 if not args._worker:
  import os,subprocess
  from core.runtime_env import runtime_environment
  out.mkdir(parents=True,exist_ok=False)
  env=runtime_environment(out/'cache',args.gpu)
  env.pop('LD_LIBRARY_PATH',None); env.pop('LD_PRELOAD',None)
  env.update(PYTHONNOUSERSITE='1',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa')
  command=[sys.executable,'-u','-B',str(Path(__file__).resolve()),*argv,'--_worker']
  for key in ('workspace','checkpoint','observation','output'): command.extend(['--'+key,getattr(args,key)])
  with (out/'process.log').open('x') as log:
   result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=out)
  (out/'process.json').write_text(json.dumps({'exit_code':result.returncode}))
  return result.returncode
 from core.runtime_env import runtime_environment
 import os
 os.environ.update(runtime_environment(out/'cache'))
 sys.path.insert(0,args.workspace)
 report={'passed':False,'quality_decision':'INCONCLUSIVE','closed_loop_tested':False}
 adapter=None
 try:
  import torch
  import numpy as np
  import inspect
  from experiments.robot import openvla_utils as vu
  from methods.specprune.accelerator import AcceleratorConfig,SpecPruneAccelerator
  from methods.vla_pruner.runtime import load_observation
  from methods.vla_pruner.evaluation import difference,trace,actions_array
  report['environment']=dict(python=sys.version,torch=torch.__version__,transformers=__import__('transformers').__version__,gpu=torch.cuda.get_device_name(0))
  report['provenance']=dict(workspace=args.workspace,checkpoint=args.checkpoint,observation=args.observation,
      source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
      (Path(__file__),Path(inspect.getfile(SpecPruneAccelerator)),Path(vu.__file__),Path(args.checkpoint)/'config.json')})
  cfg=AcceleratorConfig(pretrained_checkpoint=args.checkpoint,unnorm_key='libero_object_no_noops',token_prune=False,dynamic_prune=False,action_adapter=False)
  # Explicit local class loader bypasses upstream get_vla's checkpoint sync.
  config=vu.OpenVLAConfig.from_pretrained(args.checkpoint,local_files_only=True)
  model=vu.OpenVLAForActionPrediction.from_pretrained(args.checkpoint,config=config,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,local_files_only=True,attn_implementation='eager').to('cuda:0').eval()
  model.vision_backbone.set_num_images_in_input(2)
  model.norm_stats=json.loads((Path(args.checkpoint)/'dataset_statistics.json').read_text())
  processor=vu.get_processor(cfg)
  head=vu.get_action_head(cfg,model.llm_dim).eval()
  proprio=vu.get_proprio_projector(cfg,model.llm_dim,8).eval()
  obs=load_observation(args.observation)
  obs['prev_images']=[obs['full_image'].copy(),obs['wrist_image'].copy()]
  original=SpecPruneAccelerator(model,processor,head,proprio,config=cfg)
  def query(acc):
   acc.start_episode()
   with torch.inference_mode(): return actions_array(acc._query_model(copy.deepcopy(obs),obs['instruction'])[0])
  native,lengths=trace(model,lambda:query(original))
  report.update(native_actions=native.tolist(),native_lengths=lengths)
  modules=dict(model.named_modules()); parameters=dict(model.named_parameters())
  samples={n:v.detach().flatten()[:64].cpu().clone() for n,v in parameters.items()}
  adapter=SpecPruneAccelerator.install(model,processor,head,proprio,config=copy.deepcopy(cfg))
  control,control_lengths=trace(model,lambda:query(adapter))
  report['control']=difference(native,control)
  if not report['control']['matches']: raise RuntimeError('Retain-all control differs from native')
  adapter.cfg.token_prune=True; adapter.cfg.dynamic_prune=True
  actions,pruned_lengths=trace(model,lambda:query(adapter))
  report.update(accelerated_actions=actions.tolist(),action_difference=difference(native,actions),control_lengths=control_lengths,pruned_lengths=pruned_lengths)
  report['mechanism_observed']=any(y<x for k,v in control_lengths.items() for x,y in zip(v,pruned_lengths.get(k,[])))
  adapter.detach(); adapter=None
  report['restored']=difference(native,query(original))
  report['modules_restored']=dict(model.named_modules()).keys()==modules.keys() and all(dict(model.named_modules())[n] is m for n,m in modules.items())
  report['parameters_preserved']=all(dict(model.named_parameters())[n] is v and torch.equal(v.detach().flatten()[:64].cpu(),samples[n]) for n,v in parameters.items())
  report['passed']=all((report['restored']['matches'],report['modules_restored'],report['parameters_preserved'],report['mechanism_observed']))
  if not report['passed']: raise RuntimeError('Lifecycle gate failed')
  report['timings']=[]
  for pair in range(args.pairs):
   for mode in (('baseline','accelerated') if pair%2==0 else ('accelerated','baseline')):
    active=original
    if mode=='accelerated':
     accelerated_cfg=copy.deepcopy(cfg); accelerated_cfg.token_prune=True; accelerated_cfg.dynamic_prune=True
     adapter=SpecPruneAccelerator.install(model,processor,head,proprio,config=accelerated_cfg); active=adapter
    try:
     for _ in range(args.warmup): query(active)
     torch.cuda.synchronize(); started=time.perf_counter(); value=query(active); torch.cuda.synchronize()
     report['timings'].append(dict(pair=pair,mode=mode,ms=(time.perf_counter()-started)*1000,actions=value.tolist()))
    finally:
     if adapter is not None: adapter.detach(); adapter=None
  if args.closed_loop:
   sys.path.insert(0,str(Path(args.workspace).parent/'LIBERO'))
   from libero.libero import benchmark
   from experiments.robot.libero.libero_utils import get_libero_env,get_libero_dummy_action,get_libero_image,get_libero_wrist_image,quat2axisangle
   from experiments.robot.robot_utils import set_seed_everywhere
   suite=benchmark.get_benchmark_dict()['libero_object']()
   task=suite.get_task(args.task); initial=suite.get_task_init_states(args.task)[args.trial]
   state_hash=hashlib.sha256(np.asarray(initial).tobytes()).hexdigest()
   report['episodes']=[]
   modes=('baseline','accelerated') if (args.task+args.trial)%2==0 else ('accelerated','baseline')
   for mode in modes:
    env=None; active=original
    if mode=='accelerated':
     accelerated_cfg=copy.deepcopy(cfg); accelerated_cfg.token_prune=True; accelerated_cfg.dynamic_prune=True; accelerated_cfg.action_adapter=True
     adapter=SpecPruneAccelerator.install(model,processor,head,proprio,config=accelerated_cfg); active=adapter
    row=dict(mode=mode,task=args.task,trial=args.trial,seed=7,initial_state_hash=state_hash,success=False,records=[])
    try:
     set_seed_everywhere(7); active.start_episode()
     env,instruction=get_libero_env(task,'openvla',resolution=256)
     env.reset(); observation=env.set_init_state(initial)
     for _ in range(10): observation,_,_,_=env.step(get_libero_dummy_action('openvla'))
     executed=0
     for policy_step in range(350):
      state=np.concatenate((observation['robot0_eef_pos'],quat2axisangle(observation['robot0_eef_quat']),observation['robot0_gripper_qpos']))
      torch.cuda.synchronize(); tick=time.perf_counter()
      with torch.inference_mode(): action,info=active.step(get_libero_image(observation),get_libero_wrist_image(observation),instruction,state)
      torch.cuda.synchronize()
      row['records'].append(dict(policy_step=policy_step,action=np.asarray(action).tolist(),execute_action=bool(info['execute_action']),ms=(time.perf_counter()-tick)*1000))
      if info['execute_action']:
       observation,_,done,_=env.step(action.tolist()); executed+=1
       if done: row['success']=True; break
       if executed>=330: break
     row['environment_steps']=executed
    finally:
     if env is not None: env.close()
     if adapter is not None: adapter.detach(); adapter=None
    report['episodes'].append(row)
    (out/(mode+'-episode.json')).write_text(json.dumps(row,indent=2,allow_nan=False))
   report['closed_loop_tested']=True
   report['post_loop_restored']=difference(native,query(original))
   report['passed']=report['passed'] and report['post_loop_restored']['matches']
 except Exception as exc:
  report['passed']=False; report['error']=str(exc); report['traceback']=traceback.format_exc(); print(report['traceback'])
 finally:
  if adapter is not None:
   try: adapter.detach()
   except Exception as exc:
    report['passed']=False; report['detach_error']=str(exc)
  (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
 return 0 if report['passed'] else 2

if __name__=='__main__': raise SystemExit(main())
