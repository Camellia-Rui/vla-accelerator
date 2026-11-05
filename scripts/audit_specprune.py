"""Audit current SpecPrune smoke records; never grant deployment admission."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def audit(path):
 r=json.loads(Path(path).read_text())
 if not r.get('passed') or not all(r.get(k,{}).get('matches') for k in ('control','restored')):
  raise ValueError('Missing/failed control or restoration')
 if not all(r.get(k) for k in ('mechanism_observed','modules_restored','parameters_preserved')):
  raise ValueError('Mechanism or preservation gate failed')
 rows=r['timings']; keys=[(x['pair'],x['mode']) for x in rows]
 pairs={x['pair'] for x in rows}
 if len(keys)!=len(set(keys)) or set(keys)!={(p,m) for p in pairs for m in ('baseline','accelerated')} or len(pairs)<2:
  raise ValueError('Unpaired timing records')
 for row in rows:
  if not math.isfinite(row['ms']) or row['ms']<=0: raise ValueError('Invalid time')
  if len(row['actions'])!=8 or any(len(a)!=7 or not all(math.isfinite(v) for v in a) for a in row['actions']): raise ValueError('Invalid actions')
 result={'source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'timing_pairs':len(pairs),
  'median_ms':{m:statistics.median(x['ms'] for x in rows if x['mode']==m) for m in ('baseline','accelerated')},
  'quality_decision':'INCONCLUSIVE','deployment_accepted':False}
 if r.get('closed_loop_tested'):
  episodes=r['episodes']
  if len(episodes)!=2 or {x['mode'] for x in episodes}!={'baseline','accelerated'}: raise ValueError('Missing paired episodes')
  a,b=episodes
  if not a.get('initial_state_hash'):
   raise ValueError('Missing initial state hash')
  if any(a[k]!=b[k] for k in ('task','trial','seed','initial_state_hash')): raise ValueError('State pairing mismatch')
  for e in episodes:
   if type(e['success']) is not bool or not e['records'] or e['environment_steps'] < 1:
    raise ValueError('Incomplete episode')
   if any(type(x['execute_action']) is not bool for x in e['records']):
    raise ValueError('Invalid action execution flag')
   episode_file = Path(path).parent / (e['mode'] + '-episode.json')
   if json.loads(episode_file.read_text()) != e:
    raise ValueError('Episode file differs from report')
   if sum(x['execute_action'] for x in e['records'])!=e['environment_steps']: raise ValueError('Executed action count mismatch')
   if any(len(x['action'])!=7 or not all(math.isfinite(v) for v in x['action']) for x in e['records']): raise ValueError('Invalid episode action')
  if not r.get('post_loop_restored',{}).get('matches'): raise ValueError('Post-loop restore failed')
  result['episodes']=[{k:e[k] for k in ('mode','success','environment_steps','task','trial')} for e in episodes]
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('report');p.add_argument('--output',required=True)
 args=p.parse_args(); result=audit(args.report)
 with Path(args.output).open('x',encoding='utf-8') as f: json.dump(result,f,indent=2)
 print(json.dumps(result,indent=2))
