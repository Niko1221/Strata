"""Run each arm in isolation; do not run other GPU/CPU benchmarks concurrently."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--skip-ar', action='store_true')
o = p.parse_args()
out = Path(__file__).resolve().parent
arms = ([] if o.skip_ar else [('ar', 'ar', 1)]) + [
    ('mtp', 'mtp', 3), ('bf16', 'bf16', 3), ('Q4_0', 'Q4_0', 3),
    ('Q8_0', 'Q8_0', 3), ('Q5_0', 'Q5_0', 3), ('mtp', 'mtp-final', 3), ('mtp', 'mtp-default', 3)]
codes = {}
for arm, tag, repeat in arms:
    saved = out / f'{tag}.json'
    if saved.exists():
        data = json.loads(saved.read_text())
        complete = len(data.get('warmups', [])) == 4 and len(data.get('rows', [])) == 4*repeat
        if complete:
            codes[tag] = 0
            (out / 'exit-codes.json').write_text(json.dumps(codes, indent=2))
            print('SKIP complete', tag, flush=True)
            continue
    print('START', tag, flush=True)
    command = [sys.executable, str(out / 'run.py'), '--arm', arm, '--tag', tag,
               '--repeat', str(repeat)] + (['--spec-min-p', '0.5'] if tag == 'mtp-default' else [])
    with (out / f'{tag}-progress.log').open('w') as log:
        code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
    codes[tag] = code
    (out / 'exit-codes.json').write_text(json.dumps(codes, indent=2))
    print('FINISH', tag, code, flush=True)
    if code:
        sys.exit(code)
