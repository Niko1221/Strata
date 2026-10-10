#!/usr/bin/env python3
"""Compare greedy token streams across draft/window configurations; standard library only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time


PIPE = re.compile(r"(\d+) speculative, (\d+) on.*?(\d+) rolled back")
SUFFIX = re.compile(r"suffix drafts: (\d+) windows")
SWITCH = re.compile(r"strata pipeline switch: pw=(\d+) theta=([\d.]+) force_miss=(\d+) short_read=(\d+)")


def difference(a, b):
    """Zero-based first difference, including a missing trailing token."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def audit(rows, expected, requirements):
    failures = []
    if len(rows) != expected:
        failures.append(f"incomplete: {len(rows)} of {expected} requests")
    references = {}
    coverage = {}
    for row in rows:
        name = row['case']
        totals = coverage.setdefault(name, dict(drafts=0, speculative=0, rollbacks=0, suffix_windows=0))
        for key in totals:
            totals[key] += row.get(key, 0)
        key = row['prompt']
        ref = references.setdefault(key, row)
        if row.get('input_sha256') != ref.get('input_sha256'):
            failures.append(f"{name}/{key}: input IDs differ from reference")
        index = difference(ref['output_ids'], row['output_ids'])
        row['first_difference'] = index
        row['matches_reference'] = index is None and ref['finish'] == row['finish']
        if not row['output_ids'] or row['generated'] != len(row['output_ids']):
            failures.append(f"{name}/{key}: empty output or DONE count differs")
        if row['reused'] != 0:
            failures.append(f"{name}/{key}: reused prompt tokens")
        if not row['matches_reference']:
            failures.append(f"{name}/{key}: token difference {index}, finish {row['finish']}")
    for name, minimum in requirements.items():
        totals = coverage.get(name, {})
        for key, value in minimum.items():
            if totals.get(key, 0) < value:
                failures.append(f"{name}: {key}={totals.get(key, 0)} < required {value}")
    return dict(passed=not failures, failures=failures, coverage=coverage)


class Engine:
    def __init__(self, arm, log, switch, timeout):
        self.timeout = timeout
        self.log = log
        self.queue = queue.Queue()
        env = dict(os.environ, **arm.get('env', {}))
        if any('switch' in c for c in arm['cases']):
            env['STRATA_PIPELINE_SWITCH'] = str(switch.resolve())
            env['STRATA_PIPELINE_DEBUG'] = '1'
        self.stderr = log.open('w', encoding='utf-8')
        try:
            self.proc = subprocess.Popen([arm['exe'], *arm.get('prefix_args', []), '--serve', *arm['args']],
                                         cwd=arm.get('cwd'), env=env, stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=self.stderr,
                                         text=True, encoding='utf-8', bufsize=1)
        except Exception:
            self.stderr.close()
            raise
        def read():
            try:
                for line in self.proc.stdout:
                    self.queue.put(line.strip())
            finally:
                self.queue.put(None)
        threading.Thread(target=read, daemon=True).start()
        self.info = []

    def line(self):
        try:
            line = self.queue.get(timeout=self.timeout)
        except queue.Empty:
            raise TimeoutError(f"engine silent for {self.timeout}s; see {self.log}") from None
        if line is None:
            raise RuntimeError(f"engine exited ({self.proc.poll()}); see {self.log}")
        if line.startswith('ERR'):
            raise RuntimeError(line)
        return line

    def ready(self):
        while True:
            line = self.line()
            self.info.append(line)
            if line.startswith('READY'):
                return

    def generate(self, ids, max_new, keys):
        self.proc.stdin.write(f"GEN {max_new} {keys} {','.join(map(str, ids))}\n")
        self.proc.stdin.flush()
        output = []
        start = time.monotonic()
        while True:
            line = self.line()
            if line.startswith('T '):
                output.append(int(line.split()[1]))
            elif line.startswith('DONE '):
                f = line.split()
                if len(f) < 9:
                    raise RuntimeError('DONE lacks draft/reuse telemetry: ' + line)
                return dict(output_ids=output, done=line, generated=int(f[1]), finish=f[5],
                            drafts=int(f[7]), accepted=int(f[6]), reused=int(f[8]),
                            seconds=time.monotonic() - start)

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.proc.stdin.close()
        self.proc.stdout.close()
        self.stderr.close()


def validate(manifest):
    prompts = manifest['prompts']
    if not prompts or any(not p['ids'] or any(type(t) is not int or t < 0 for t in p['ids']) for p in prompts):
        raise ValueError('prompts need nonempty, nonnegative integer token IDs')
    if len({p['name'] for p in prompts}) != len(prompts):
        raise ValueError('prompt names must be unique')
    names = [c['name'] for a in manifest['arms'] for c in a['cases']]
    if not names or len(set(names)) != len(names):
        raise ValueError('case names must be nonempty and unique across arms')
    if (type(manifest.get('repeats', 2)) is not int or manifest.get('repeats', 2) < 2 or
            type(manifest.get('max_new', 128)) is not int or manifest.get('max_new', 128) < 1):
        raise ValueError('use at least two repeats and one output token')
    for arm in manifest['arms']:
        for case in arm['cases']:
            if any(k not in ('spec_min_p',) or not 0 <= v <= 1 for k, v in case.get('keys', {}).items()):
                raise ValueError('only spec_min_p in [0,1] is allowed; decoding stays greedy')
            for k, v in case.get('switch', {}).items():
                if k not in ('pw', 'theta', 'force_miss', 'short_read') or type(v) not in (int, float) or v < 0:
                    raise ValueError('invalid pipeline switch')
            if any(k not in ('drafts', 'speculative', 'rollbacks', 'suffix_windows') or type(v) is not int or v < 0
                   for k, v in case.get('require', {}).items()):
                raise ValueError('invalid coverage requirement')


def run(manifest, output, timeout=600):
    validate(manifest)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    repeats = manifest.get('repeats', 2)
    expected = len(manifest['prompts']) * repeats * sum(len(a['cases']) for a in manifest['arms'])
    requirements = {c['name']: c.get('require', {}) for a in manifest['arms'] for c in a['cases']}
    report = dict(complete=False, expected=expected, rows=[], arms=[], requirements=requirements)
    def save():
        temp = output / 'result.tmp'
        temp.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temp.replace(output / 'result.json')
    save()
    try:
        for ai, arm in enumerate(manifest['arms']):
            log = output / f'engine-{ai}.log'
            switch = output / f'switch-{ai}.txt'
            # A task-owned file; never overwrite a service's switch or configuration.
            switch.write_text('pw=0 theta=0 force_miss=0 short_read=0', encoding='utf-8')
            engine = Engine(arm, log, switch, timeout)
            try:
                engine.ready()
                report['arms'].append(dict(name=arm['name'], info=engine.info,
                                          exe_sha256=hashlib.sha256(Path(arm['exe']).read_bytes()).hexdigest()))
                for repeat in range(repeats):
                    # Reverse the case order after each round; retain first-round serial references.
                    cases = arm['cases'] if repeat % 2 == 0 else list(reversed(arm['cases']))
                    for prompt in manifest['prompts']:
                        for case in cases:
                            controls = dict(pw=0, theta=0, force_miss=0, short_read=0)
                            controls.update(case.get('switch', {}))
                            switch.write_text(' '.join(f'{k}={v}' for k, v in controls.items()), encoding='utf-8')
                            offset = log.stat().st_size
                            keys = ' '.join(f'{k}={v}' for k, v in case.get('keys', {}).items())
                            row = engine.generate(prompt['ids'], manifest.get('max_new', 128), keys)
                            # stderr is emitted before DONE; read only this request's log segment.
                            with log.open('rb') as stream:
                                stream.seek(offset)
                                trace = stream.read().decode('utf-8', errors='replace')
                            counts = PIPE.findall(trace)
                            row.update(speculative=sum(int(c[0]) for c in counts),
                                       rollbacks=sum(int(c[2]) for c in counts),
                                       suffix_windows=sum(map(int, SUFFIX.findall(trace))),
                                       pipeline_summaries=[list(map(int, c)) for c in counts],
                                       arm=arm['name'], case=case['name'], prompt=prompt['name'], repeat=repeat,
                                       input_sha256=hashlib.sha256(json.dumps(prompt['ids']).encode()).hexdigest())
                            report['rows'].append(row)
                            save()
                            if 'switch' in case:
                                confirmed = SWITCH.findall(trace)
                                wanted = [float(controls[k]) for k in ('pw', 'theta', 'force_miss', 'short_read')]
                                if not any(all(abs(float(v) - w) < 0.0006 for v, w in zip(values, wanted))
                                           for values in confirmed):
                                    raise RuntimeError(f"{case['name']}: engine did not confirm requested pipeline controls")
                            print(f"{case['name']} {prompt['name']} repeat={repeat} tokens={len(row['output_ids'])} "
                                  f"drafts={row['drafts']} speculative={row['speculative']} rollbacks={row['rollbacks']}", flush=True)
            finally:
                engine.close()
        report.update(audit(report['rows'], expected, requirements), complete=True)
    except Exception as exc:
        report.update(passed=False, error=repr(exc))
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--manifest', type=Path)
    mode.add_argument('--audit', type=Path, help='recheck a saved result without running an engine')
    parser.add_argument('--output', type=Path, help='new directory for full IDs and engine logs')
    parser.add_argument('--timeout', type=float, default=600, help='maximum silence per engine protocol line')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    if args.audit:
        saved = json.loads(args.audit.read_text(encoding='utf-8'))
        result = audit(saved['rows'], saved['expected'], saved['requirements'])
        result['complete'] = saved.get('complete', False)
        if not result['complete'] or saved.get('error'):
            result['passed'] = False
            result['failures'].append('saved run did not complete successfully')
    else:
        if args.output is None:
            parser.error('--output is required with --manifest')
        result = run(json.loads(args.manifest.read_text(encoding='utf-8')), args.output, args.timeout)
    print(json.dumps({k: result[k] for k in ('complete', 'passed', 'failures', 'error') if k in result}))
    return 0 if result.get('passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
