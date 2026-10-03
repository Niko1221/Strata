#!/usr/bin/env python3
"""Measure actual Q8 PleTable load time, CPU usage and verified page residency."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import time


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--table', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    opt = parser.parse_args()
    opt.output.mkdir(parents=True, exist_ok=False)
    state = {'started': time.time(), 'table': str(opt.table), 'records': []}
    target = opt.output / 'startup.json'
    save(target, state)
    try:
        reference = None
        for threads in (0, 16, 1, 4, 8):
            for condition in ('cold', 'warm'):
                label = f'{condition}-threads{threads}'
                state['current'] = label; save(target, state)
                start = time.time()
                result = subprocess.run([str(opt.exe), str(opt.table), condition],
                    env={**os.environ, 'STRATA_PLE_PREFAULT_THREADS': str(threads)},
                    capture_output=True, text=True, timeout=600)
                (opt.output / (label + '.stdout')).write_text(result.stdout)
                (opt.output / (label + '.stderr')).write_text(result.stderr)
                if result.returncode:
                    raise RuntimeError(f'{label} failed: {result.stderr[-1000:]}')
                row = json.loads(result.stdout.strip().splitlines()[-1])
                row.update(label=label, threads=threads, requested_condition=condition,
                           process_seconds=time.time()-start)
                fraction = row['resident_before'] / row['pages']
                row['resident_fraction_before'] = fraction
                row['measured_condition'] = 'cold' if fraction < .001 else 'warm' if fraction > .999 else 'partial'
                row['table_GiB_per_open_second'] = row['table_bytes'] / 2**30 / row['open_seconds']
                state['records'].append(row); save(target, state)
                if not row['locked'] or row['resident_after'] != row['pages']:
                    raise RuntimeError(label + ' failed RAM-residency gate')
                reference = reference or row['row_checksum']
                if row['row_checksum'] != reference:
                    raise RuntimeError(label + ' row checksum differs')
        # Child-only zero mlock limit exercises the already-touched fallback.
        def no_lock():
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (0, 0))
        result = subprocess.run([str(opt.exe), str(opt.table), 'warm'],
            env={**os.environ, 'STRATA_PLE_PREFAULT_THREADS': '16'}, preexec_fn=no_lock,
            capture_output=True, text=True, timeout=300)
        (opt.output / 'lock-failure.stderr').write_text(result.stderr)
        if result.returncode:
            raise RuntimeError('lock failure fallback did not complete')
        fallback = json.loads(result.stdout.strip().splitlines()[-1])
        if fallback['locked'] or fallback['row_checksum'] != reference:
            raise RuntimeError('lock failure fallback corrupted state')
        state['lock_failure_fallback'] = fallback
        state['completed'] = True
    except BaseException as error:
        state['error'] = repr(error)
        raise
    finally:
        state['finished'] = time.time(); save(target, state)


if __name__ == '__main__':
    main()
