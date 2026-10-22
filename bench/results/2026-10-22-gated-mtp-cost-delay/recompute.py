"""Recompute paired C/comparator changes from retained DONE records (Python stdlib)."""
import json, statistics
from pathlib import Path
root = Path(__file__).resolve().parent
summary = []
for name, comparators in [('delay', ['U', 'A']), ('final', ['U', 'D', 'T4', 'T5', 'T6'])]:
    data = json.loads((root / f'{name}-measurements.json').read_text())
    grouped = {}
    for row in data['runs']:
        done = row['raw_done'].split()
        assert int(done[1]) == len(row['output_ids']) == 256
        assert done[5] == 'length'
        grouped.setdefault((row['block'], row['arm'], row['workload']), []).append((int(done[1]), float(done[4])))
    blocks = sorted({row['block'] for row in data['runs']})
    def rate(block, arm, workload):
        rows = grouped[block, arm, workload]
        return 1000 * sum(n for n, ms in rows) / sum(ms for n, ms in rows)
    for arm in comparators:
        for workload in ('english', 'repeat'):
            changes = [100 * (rate(b, 'C', workload) / rate(b, arm, workload) - 1) for b in blocks]
            mean = statistics.mean(changes)
            t = {8: 2.364624251, 16: 2.131449546}[len(changes)]
            half = t * statistics.stdev(changes) / len(changes) ** 0.5
            summary.append(dict(round=name, comparator=arm, workload=workload, n=len(changes), mean_percent=mean, ci95=[mean-half, mean+half]))
print(json.dumps(summary, indent=2))
