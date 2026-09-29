"""Controlled local Pi coding task with cache off/on; private server, dry-run by default.

Requires an installed Pi SDK (--pi-package). Never contacts an existing server.
Only --run creates files or loads a model. Retains generated code, private agent
events, exact inputs, timings, engine metrics and independent validator results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SPEC = '''Implement a Python 3 standard-library expense report.
report.py exports summarize(text: str) -> list[dict]. Input is CSV with headers
id,timestamp,team,amount. Strip surrounding whitespace in values. A valid row has
a nonempty id and team, an ISO timestamp WITH timezone (Z or explicit offset),
and a finite decimal amount with at most two fractional digits. Ignore invalid
rows. Among valid rows with the same id, the last valid row wins (a later invalid
row must not erase a valid row). Normalize timestamps to UTC and group by UTC
date and team. Return groups sorted by (day, team), each with exactly the keys
day (YYYY-MM-DD), team, count (int), total (decimal string with two places).
Negative amounts and zero are valid. Use decimal arithmetic, not binary floats.
Empty input and header-only input return []. Correctly parse quoted commas.
cli.py accepts exactly one argument, a CSV filename or - for stdin. Print the
JSON list and exit 0. An unreadable filename prints a concise message to stderr
and exits 2 without a traceback. Do not add third-party dependencies.
'''
VALIDATOR = r'''
import importlib.util, json, subprocess, sys
from pathlib import Path
root=Path(sys.argv[1]); spec=importlib.util.spec_from_file_location('report',root/'report.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
header='id,timestamp,team,amount\n'
cases=[('',[]),(header,[]),
 (header+'a,2026-01-02T00:30:00+02:00,"Blue, East",0.10\nb,2026-01-01T12:00:00Z,"Blue, East",0.20\n',
  [{'day':'2026-01-01','team':'Blue, East','count':2,'total':'0.30'}]),
 (header+'a,2026-01-01T00:00:00Z,X,1.00\na,2026-01-02T00:00:00Z,Y,2.00\na,bad,X,9.00\n',
  [{'day':'2026-01-02','team':'Y','count':1,'total':'2.00'}]),
 (header+' a ,2026-01-01T00:00:00Z, X ,-2.10\nb,2026-01-01T00:00:00Z,X,0\n',
  [{'day':'2026-01-01','team':'X','count':2,'total':'-2.10'}]),
 (header+'a,2026-01-01T00:00:00,X,1\nb,2026-01-01T00:00:00Z,X,NaN\nc,2026-01-01T00:00:00Z,X,1.001\n,2026-01-01T00:00:00Z,X,2\nd,2026-01-01T00:00:00Z,,2\n',[]),
 (header+'a,2026-01-02T00:00:00Z,B,1\nb,2026-01-01T00:00:00Z,Z,2\nc,2026-01-02T00:00:00Z,A,3\n',
  [{'day':'2026-01-01','team':'Z','count':1,'total':'2.00'},
   {'day':'2026-01-02','team':'A','count':1,'total':'3.00'},
   {'day':'2026-01-02','team':'B','count':1,'total':'1.00'}])]
checks=[]
for i,(text,want) in enumerate(cases):
    got=m.summarize(text); assert got==want,(i,got,want); checks.append('library-'+str(i))
text,want=cases[2]
p=subprocess.run([sys.executable,str(root/'cli.py'),'-'],input=text,text=True,capture_output=True,timeout=10)
assert p.returncode==0 and json.loads(p.stdout)==want,p; checks.append('stdin')
f=root/'validator-input.csv';f.write_text(text)
p=subprocess.run([sys.executable,str(root/'cli.py'),str(f)],text=True,capture_output=True,timeout=10)
assert p.returncode==0 and json.loads(p.stdout)==want,p;checks.append('file');f.unlink()
p=subprocess.run([sys.executable,str(root/'cli.py'),str(root/'does-not-exist.csv')],text=True,capture_output=True,timeout=10)
assert p.returncode==2 and p.stderr and 'Traceback' not in p.stderr,p;checks.append('missing-file')
print(json.dumps({'passed':checks}))
'''


def get_json(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--pi-package', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--port', type=int, default=11436)
    ap.add_argument('--order', default='off,on', choices=('off,on', 'on,off'))
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    if not args.run:
        print('Dry run: private server, identical Pi A/B/A coding tasks, cache off/on, independent validators.')
        return
    args.output = args.output.resolve()
    args.output.mkdir(parents=False, exist_ok=False)
    cfg = json.loads(args.config.read_text())
    cfg.update(exe=str(args.engine.resolve()), cwd=str(ROOT), host='127.0.0.1', port=args.port)
    work = args.output / 'workspace'
    agent = args.output / 'agent'
    agent.mkdir()
    model = cfg['model_name']
    if model != 'qwen3.8-flash-next-iq3_s':
        raise ValueError('This task is currently validated for qwen3.8-flash-next-iq3_s')
    (agent/'models.json').write_text(json.dumps({'providers': {'strata-benchmark': {
        'baseUrl': f'http://127.0.0.1:{args.port}/v1', 'api': 'openai-completions', 'apiKey': 'local',
        'models': [{'id': model, 'reasoning': False, 'input': ['text'], 'contextWindow': 131072,
                    'maxTokens': 2048, 'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0}}],
        'compat': {'supportsStore': False, 'supportsDeveloperRole': False, 'supportsReasoningEffort': False}
    }}}))
    corpus = 'id,timestamp,team,amount\n' + ''.join(
        f'r{i},2026-01-{i%28+1:02d}T12:00:00Z,Team{i%7},{i%100}.{i%97:02d}\n' for i in range(280))
    initial = {'SPEC.md': SPEC, 'sample.csv': corpus,
               'report.py': 'def summarize(text):\n    raise NotImplementedError\n', 'cli.py': ''}
    manifest = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in initial.items()}
    (args.output/'inputs.json').write_text(json.dumps(initial, indent=2))
    validator = args.output/'validate.py'
    validator.write_text(VALIDATOR)
    results = []
    for label in args.order.split(','):
        run = args.output/label
        run.mkdir()
        if work.exists():
            shutil.rmtree(work)  # only this harness's new, disposable task workspace
        work.mkdir()
        for name, text in initial.items():
            (work/name).write_text(text)
        run_cfg = dict(cfg, args=list(cfg['args']), log=str(run/'engine.log'))
        run_cfg['args'] += ['--conversation-cache-mib', '8192' if label=='on' else '0']
        config = run/'config.json'
        config.write_text(json.dumps(run_cfg, indent=2))
        config.with_name('config.shared-settings.json').write_text(json.dumps({
            'reasoning_effort': 'none', 'temperature': 0, 'max_tokens': 2048}))
        record = {'mode': label, 'inputs_sha256': manifest}
        with (run/'server.log').open('w') as log:
            server = subprocess.Popen([sys.executable, '-m', 'serve.server', '--engine', 'strata',
                '--config', str(config), '--host', '127.0.0.1', '--port', str(args.port)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                base = f'http://127.0.0.1:{args.port}'
                for _ in range(180):
                    if server.poll() is not None:
                        raise RuntimeError('Private server exited; inspect server.log')
                    try:
                        get_json(base+'/health')
                        break
                    except OSError:
                        time.sleep(1)
                else:
                    raise TimeoutError('Private server did not become ready')
                started = time.monotonic()
                with (run/'agent.log').open('w') as agent_log:
                    task = subprocess.run(['node', str(ROOT/'tools/agent_cache_task.mjs'),
                        str(args.pi_package.resolve()), str(work), str(agent), str(run/'task.json')],
                        cwd=work, stdout=agent_log, stderr=subprocess.STDOUT, timeout=750,
                        env={**os.environ, 'PI_OFFLINE': '1'})
                record.update(wall_s=time.monotonic()-started, agent_exit=task.returncode)
                check = subprocess.run([sys.executable, str(validator), str(work)],
                    text=True, capture_output=True, timeout=30)
                (run/'validation.txt').write_text(check.stdout+check.stderr)
                record.update(validator_exit=check.returncode, success=task.returncode==check.returncode==0)
                metrics = get_json(base+'/metrics?requests=all')
                (run/'metrics.json').write_text(json.dumps(metrics, indent=2))
                record.update(totals=metrics['totals'], engine=metrics['engine'])
                # Kernel high-water RSS for the engine, unlike periodic request-end samples.
                children = Path(f'/proc/{server.pid}/task/{server.pid}/children').read_text().split()
                record['child_memory_kib'] = {}
                for pid in children:
                    status = Path(f'/proc/{pid}/status').read_text().splitlines()
                    record['child_memory_kib'][pid] = {s.split(':')[0]: int(s.split()[1])
                        for s in status if s.startswith(('VmHWM:', 'VmRSS:'))}
                shutil.copytree(work, run/'artifacts')
            finally:
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGINT)
                    try:
                        server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid, signal.SIGTERM)
                        server.wait(timeout=10)
        results.append(record)
        (args.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
        print(json.dumps(record), flush=True)
    for key in ('expert_slots', 'context', 'kv', 'kv_resident', 'spec', 'mtp_max', 'lookup', 'pool_workers', 'pcie_frac'):
        if results[0]['engine'][key] != results[1]['engine'][key]:
            raise AssertionError(f'Unmatched engine configuration: {key}')
    if not all(r['success'] for r in results):
        raise SystemExit('Agent task or independent validation failed; no successful-task speedup claim')


if __name__ == '__main__':
    main()
