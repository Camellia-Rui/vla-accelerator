"""Incremental evidence checks during OFT closed-loop execution."""
import math
import json


def atomic(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def publish(output, report, final=False):
    try:
        result = assess_progress(report)
        if final:
            if result['checked_episodes'] != result['expected_episodes']:
                raise ValueError('Missing episode at end of execution')
            result['status'] = 'AUDITED'
        atomic(output/'assessment.json', result)
        return result
    except (KeyError, TypeError, ValueError) as exc:
        atomic(output/'assessment.json', dict(status='FAILED',quality_decision='INCONCLUSIVE',error=str(exc)))
        raise


def assess_progress(report):
    protocol = report['protocol']
    modes = protocol['modes']
    if not modes or len(set(modes)) != len(modes) or not protocol['tasks'] or not protocol['trials']:
        raise ValueError('Empty or duplicate protocol')
    if len(set(protocol['tasks'])) != len(protocol['tasks']) or len(set(protocol['trials'])) != len(protocol['trials']):
        raise ValueError('Duplicate tasks or states')
    expected = {(t, i, m) for t in protocol['tasks'] for i in protocol['trials'] for m in modes}
    seen, states = set(), {}
    totals = {m:dict(episodes=0, successes=0, queries=0, query_ms=0., policy_ms=0., wall_seconds=0.) for m in modes}
    for row in report['records']:
        key = row['task_id'], row['trial'], row['mode']
        if key not in expected or key in seen:
            raise ValueError('Unexpected or duplicate episode')
        seen.add(key)
        if not row['complete'] or row['error'] is not None or type(row['success']) is not bool:
            raise ValueError('Incomplete or erroneous episode')
        if row['seed'] != protocol['seed'] or not row['initial_state_hash']:
            raise ValueError('Invalid seed or missing initial state')
        state_key = key[:2]
        if state_key in states and states[state_key] != row['initial_state_hash']:
            raise ValueError('Paired initial states differ')
        states[state_key] = row['initial_state_hash']
        if not row['queries']:
            raise ValueError('No model queries')
        for q in row['queries']:
            if type(q.get('queried',True)) is not bool:
                raise ValueError('Invalid query flag')
            if not math.isfinite(q['ms']) or q['ms'] <= 0:
                raise ValueError('Invalid query timing')
            if len(q['actions']) != protocol.get('action_chunk',8) or any(len(a)!=7 or not all(math.isfinite(v) for v in a) for a in q['actions']):
                raise ValueError('Invalid action chunk')
        if not any(q.get('queried',True) for q in row['queries']):
            raise ValueError('No actual model query')
        elapsed = sum(q['ms'] for q in row['queries'])
        if not math.isclose(row['policy_ms'], elapsed, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError('Policy timing differs from query records')
        if not math.isfinite(row['episode_seconds']) or row['episode_seconds'] <= 0:
            raise ValueError('Invalid episode wall time')
        if row['policy_ms'] > row['episode_seconds']*1000 + 1:
            raise ValueError('Policy timing exceeds containing episode wall time')
        if 'executed_actions' in row and (len(row['executed_actions']) != row['control_steps'] or row['control_steps'] < 1):
            raise ValueError('Executed action count mismatch')
        if any(len(a)!=7 or not all(math.isfinite(v) for v in a) for a in row.get('executed_actions',[])):
            raise ValueError('Invalid executed action')
        t = totals[row['mode']]
        t['episodes'] += 1
        t['successes'] += row['success']
        t['queries'] += sum(q.get('queried',True) for q in row['queries'])
        t['query_ms'] += sum(q['ms'] for q in row['queries'] if q.get('queried',True))
        t['policy_ms'] += row['policy_ms']
        t['wall_seconds'] += row['episode_seconds']
    paired = sum(all((t,i,m) in seen for m in modes) for t in protocol['tasks'] for i in protocol['trials'])
    return dict(status='RUNNING', expected_episodes=len(expected), checked_episodes=len(seen),
                complete_pairs=paired, metrics=totals, quality_decision='INCONCLUSIVE',
                reason='Incremental evidence only; final restoration and quality policy still required')


def specprune_progress(episodes, task, trial, seed):
    records=[]
    for e in episodes:
        queries=0
        for r in e['records']:
            if type(r.get('queried')) is not bool:raise ValueError('Missing query flag')
            queries+=r['queried']
            if r['infer_step']!=queries:raise ValueError('Query counter mismatch')
        records.append(dict(task_id=e['task'],trial=e['trial'],mode=e['mode'],seed=e['seed'],
            initial_state_hash=e['initial_state_hash'],complete=True,error=None,success=e['success'],
            queries=[dict(ms=r['ms'],actions=[r['action']],queried=r['queried']) for r in e['records']],
            policy_ms=sum(r['ms'] for r in e['records']),episode_seconds=e['episode_seconds'],
            executed_actions=[r['action'] for r in e['records'] if r['execute_action']],control_steps=e['environment_steps']))
    return dict(protocol=dict(tasks=[task],trials=[trial],seed=seed,action_chunk=1,
                modes=['baseline','accelerated']),records=records)
