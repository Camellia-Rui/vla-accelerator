"""Run independently isolated SpecPrune Object pairs with resumable evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_specprune import audit


def atomic(path,value):
 temp=path.with_suffix(path.suffix+'.tmp')
 temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def aggregate(rows,expected):
 if len(rows)!=expected or len({(r['task'],r['trial']) for r in rows})!=expected:
  raise ValueError('Missing or duplicate task/trial pairs')
 counts={m:0 for m in ('baseline','accelerated')}
 for row in rows:
  episodes=row['audit'].get('episodes',[])
  if len(episodes)!=2: raise ValueError('Missing closed-loop pair')
  for e in episodes:
   if (e['task'],e['trial'])!=(row['task'],row['trial']):raise ValueError('Task/trial mismatch')
   counts[e['mode']]+=e['success']
 return dict(execution_complete=True,paired_initial_states=expected,successes=counts,records=rows,
             quality_decision='INCONCLUSIVE',deployment_accepted=False)


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('workspace','checkpoint','observation','output','gpu'):p.add_argument('--'+k,required=True)
 p.add_argument('--tasks',type=int,nargs='+',default=[0,1])
 p.add_argument('--trials',type=int,nargs='+',default=[0])
 p.add_argument('--resume',action='store_true')
 args=p.parse_args(argv)
 if any(t<0 or t>=10 for t in args.tasks) or any(t<0 for t in args.trials) or len(set(args.tasks))!=len(args.tasks) or len(set(args.trials))!=len(args.trials):p.error('Unique valid task/trial indices required')
 root=Path(__file__).resolve().parents[1]; out=Path(args.output).resolve()
 cfg={k:str(Path(getattr(args,k)).resolve()) for k in ('workspace','checkpoint','observation')}
 cfg.update(tasks=args.tasks,trials=args.trials,gpu=args.gpu,python=sys.executable,
  source_sha256={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ('core','methods/specprune') for f in (root/folder).rglob('*.py')})
 from core.project_checks import environment
 for f in (Path(__file__),root/'scripts/audit_specprune.py'):
  cfg['source_sha256'][str(f.relative_to(root))]=hashlib.sha256(f.read_bytes()).hexdigest()
 cfg['environment']=environment(sys.executable)
 if cfg['environment'].get('error') or cfg['environment'].get('missing'): raise ValueError('Invalid model environment')
 cp=Path(args.checkpoint)
 cfg['checkpoint_inventory']={str(f.relative_to(cp)):[f.stat().st_size,f.stat().st_mtime_ns] for f in cp.rglob('*') if f.is_file()}
 out.mkdir(parents=True,exist_ok=args.resume)
 lock=out/'batch.lock'
 with lock.open('x') as f:f.write(str(os.getpid()))
 try:
  config_path=out/'batch-config.json'
  if args.resume:
   if not config_path.is_file() or json.loads(config_path.read_text())!=cfg:raise ValueError('Resume configuration/source/environment/checkpoint mismatch')
  else:atomic(config_path,cfg)
  rows=[]
  for task in args.tasks:
   for trial in args.trials:
    pair=out/f'{task}-{trial}'
    if pair.exists():
     if not args.resume:raise ValueError('Pair already exists')
     # Retain every failed attempt; never overwrite its evidence.
     attempts=[pair]+sorted(out.glob(f'{task}-{trial}-retry-*'))
     valid=None
     for attempt in attempts:
      try:
       if json.loads((attempt/'process.json').read_text())['exit_code']==0:
        valid=(attempt,audit(attempt/'report.json'))
      except (OSError,ValueError,KeyError):pass
     if valid is not None:
      rows.append(dict(task=task,trial=trial,directory=str(valid[0]),audit=valid[1]));continue
     pair=out/f'{task}-{trial}-retry-{len(attempts)}'
    command=[sys.executable,'-B','-m','core','evaluate','--method','specprune','--closed-loop',
     '--workspace',cfg['workspace'],'--checkpoint',cfg['checkpoint'],'--observation',cfg['observation'],
     '--output',str(pair),'--gpu',args.gpu,'--task',str(task),'--trial',str(trial)]
    result=subprocess.run(command,cwd=root)
    if result.returncode:raise RuntimeError(f'Pair {task}/{trial} failed; inspect {pair}')
    rows.append(dict(task=task,trial=trial,directory=str(pair),audit=audit(pair/'report.json')))
    atomic(out/'progress.json',dict(completed_pairs=len(rows),expected_pairs=len(args.tasks)*len(args.trials),records=rows))
  atomic(out/'report.json',aggregate(rows,len(args.tasks)*len(args.trials)))
 finally:lock.unlink()
 return 0

if __name__=='__main__':raise SystemExit(main())
