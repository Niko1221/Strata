#!/usr/bin/env python3
"""Compare complete matching Strata quality runs; refuse mismatched positions."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quant_quality as q

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('reference')
p.add_argument('candidates', nargs='+')
p.add_argument('--out', required=True)
a = p.parse_args()
ref = q.read_run(a.reference)
assert set(ref) == set(q.SEGMENTS), 'Missing reference segment'
assert all(len(rows) >= 999 for rows in ref.values()), 'Incomplete reference scoring'
result = {'reference': a.reference, 'metric': 'baseline-relative top-64 approximate KL', 'runs': {}}
for path in a.candidates:
    rows = q.read_run(path)
    assert set(rows) == set(ref), 'Mismatched segments'
    assert all(set(rows[s]) == set(ref[s]) for s in ref), 'Mismatched scored positions'
    result['runs'][path] = q.compare(ref, rows)
    print(path, json.dumps(result['runs'][path]['all']), flush=True)
Path(a.out).write_text(json.dumps(result, indent=2) + '\n')
