"""Re-audit downloaded SpecPrune batches and summarize paired closed-loop evidence."""
import argparse
import json
from pathlib import Path
import random
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_specprune import audit
from scripts.batch_specprune import aggregate


def summarize(paths):
    rows = []
    protocol = None
    for path in paths:
        path = Path(path)
        report = json.loads(path.read_text())
        config = json.loads((path.parent / 'batch-config.json').read_text())
        comparable = {k:v for k,v in config.items() if k not in ('tasks', 'gpu')}
        if protocol is not None and comparable != protocol:
            raise ValueError('Different batch protocols cannot be pooled')
        protocol = comparable
        expected = {(t, i) for t in config['tasks'] for i in config['trials']}
        actual = {(r['task'], r['trial']) for r in report['records']}
        if actual != expected or len(report['records']) != len(expected):
            raise ValueError('Incomplete batch')
        for row in report['records']:
            # Remote absolute paths are evidence; resolve only their final component locally.
            pair = path.parent / Path(row['directory']).name / 'report.json'
            checked = audit(pair)
            if checked != row['audit']:
                raise ValueError('Stored audit differs from raw evidence')
            rows.append(dict(task=row['task'], trial=row['trial'], directory=str(pair.parent), audit=checked))
    result = aggregate(rows, len(rows))
    per_task = {}
    differences = []
    totals = []
    both_success = []
    for row in rows:
        episodes = {e['mode']: e for e in row['audit']['episodes']}
        b, a = episodes['baseline'], episodes['accelerated']
        differences.append(100 * (int(a['success']) - int(b['success'])))
        totals.append((b['policy_ms'], a['policy_ms']))
        if b['success'] and a['success']:
            both_success.append((b['policy_ms'], a['policy_ms']))
        task = per_task.setdefault(str(row['task']), dict(pairs=0, baseline_successes=0, accelerated_successes=0))
        task['pairs'] += 1
        task['baseline_successes'] += b['success']
        task['accelerated_successes'] += a['success']
    # Resample matched initial states, retaining all failures. Descriptive single-seed interval.
    rng = random.Random(7)
    deltas, ratios = [], []
    for _ in range(10000):
        sample = [rng.randrange(len(rows)) for _ in rows]
        deltas.append(statistics.mean(differences[i] for i in sample))
        ratios.append(sum(totals[i][0] for i in sample) / sum(totals[i][1] for i in sample))
    def interval(values):
        values.sort()
        return [values[249], values[9749]]
    result.update(per_task=per_task,
                  paired_successful_episodes=len(both_success),
                  paired_successful_episode_policy_speedup=(sum(b for b,a in both_success)/sum(a for b,a in both_success)) if both_success else None,
                  success_difference_pp=statistics.mean(differences),
                  success_difference_bootstrap95_pp=interval(deltas),
                  episode_policy_speedup=sum(b for b, a in totals)/sum(a for b, a in totals),
                  episode_policy_speedup_bootstrap95=interval(ratios),
                  interval_scope='Paired initial-state bootstrap, seed 7, 10000 resamples; not multi-seed or hardware uncertainty',
                  timing_scope='Synchronized policy step including image preparation and queue handling; excludes simulator step. Query subset includes preprocessing. All calls retained.')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reports', nargs='+')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    result = summarize(args.reports)
    with Path(args.output).open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}, indent=2))
