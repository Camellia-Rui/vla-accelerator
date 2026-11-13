"""Unified progress supervision and descriptive timing reports (no GPU imports)."""
import json
from pathlib import Path
import subprocess
from core.live_evaluation import atomic


def run_monitored(command, cwd, output, initial):
    process=subprocess.Popen(command,cwd=cwd)
    latest=None
    while True:
        snapshot=output/'measurement/assessment.json'
        try:
            value=json.loads(snapshot.read_text(encoding='utf-8'))
        except (FileNotFoundError,json.JSONDecodeError):
            value=None
        if value is not None and value != latest:
            latest=value
            atomic(output/'assessment.json',dict(initial,live=value))
            print('运行中评估：'+str(value.get('status','RUNNING'))+
                  '；已检查回合 '+str(value.get('checked_episodes','见子任务报告'))+
                  '；完整配对 '+str(value.get('complete_pairs','见子任务报告')),flush=True)
        try:
            code=process.wait(timeout=1)
            if code:raise subprocess.CalledProcessError(code,command)
            return
        except subprocess.TimeoutExpired:
            continue


def timing_details(path, report):
    records=report.get('records',[])
    scope='Model-query wrapper, excluding simulator steps; per-episode sum uses all queries'
    if report.get('backend') in ('openvla','oft'):
        files=[path.parent/m/'live-episodes.json' for m in ('baseline','fastv')]
        if not all(p.exists() for p in files):
            return dict(scope='Legacy FastV records: no unified episode wall timing',modes={})
        records=[r for p in files for r in json.loads(p.read_text())['records']]
        scope='FastV query includes its preprocessing; wall timing scope is backend-specific (see method guide)'
    elif report.get('execution_complete') and 'paired_initial_states' in report:
        records=[]
        for item in report['records']:
            p=path.parent/Path(item['directory']).name/'report.json'
            for e in json.loads(p.read_text())['episodes']:
                calls=e['records']
                if not all('queried' in x for x in calls):continue
                records.append(dict(mode='baseline' if e['mode']=='baseline' else 'specprune',success=e['success'],
                    queries=[dict(ms=x['ms']) for x in calls if x['queried']],
                    policy_ms=sum(x['ms'] for x in calls),episode_seconds=e.get('episode_seconds')))
        scope='SpecPrune: query includes preprocessing/controller; strategy total includes queue-only calls'
    modes={}
    for mode in {r['mode'] for r in records}:
        rows=[r for r in records if r['mode']==mode]
        times=[q['ms'] for r in rows for q in r['queries']]
        walls=[r.get('episode_seconds') for r in rows]
        modes[mode]=dict(episodes=len(rows),successes=sum(r['success'] for r in rows),model_queries=len(times),
            mean_query_ms=sum(times)/len(times),
            mean_episode_policy_ms=sum(r.get('policy_ms',sum(q['ms'] for q in r['queries'])) for r in rows)/len(rows),
            mean_episode_wall_seconds=sum(walls)/len(walls) if all(x is not None for x in walls) else None)
    if 'baseline' in modes:
        for mode,row in modes.items():
            row['speedups']={k:modes['baseline'][k]/row[k] if row[k] and modes['baseline'][k] else None
                            for k in ('mean_query_ms','mean_episode_policy_ms','mean_episode_wall_seconds')}
    return dict(scope=scope,modes=modes)


def render(result):
    lines=['# 自动评估报告','',f"执行状态：{result['status']}",f"样本门槛：{result['sample_policy_status']}",
           f"选择：{result['selected'] or '未选择'}",'',
           '样本满足门槛不代表统计无损或部署验收通过。未提供门槛时仅报告证据。','',
           '|模式|成功回合|平均查询 ms|平均策略总耗时 ms|平均回合墙钟 s|查询/策略/墙钟加速比|','|---|---:|---:|---:|---:|---|']
    for mode,row in result['timing_details']['modes'].items():
        wall=row['mean_episode_wall_seconds']
        speedups=row.get('speedups',{})
        ratios=' / '.join(f'{speedups[k]:.3f}×' if speedups.get(k) is not None else '缺少记录'
                          for k in ('mean_query_ms','mean_episode_policy_ms','mean_episode_wall_seconds'))
        wall_text=f'{wall:.3f}' if wall is not None else '缺少记录'
        lines.append(f"|{mode}|{row['successes']}/{row['episodes']}|{row['mean_query_ms']:.2f}|{row['mean_episode_policy_ms']:.2f}|{wall_text}|{ratios}|")
    lines+=['',result['timing_details']['scope'],'','选择器各候选的排除原因：']
    for mode,row in result['candidates'].items():lines.append(f"- {mode}: {', '.join(row.get('exclusions',[])) or '未排除；未给门槛时不作达标判断'}")
    lines+=['',f"原始报告：{result['report']}",f"SHA256：{result['report_sha256']}"]
    return '\n'.join(lines)+'\n'
