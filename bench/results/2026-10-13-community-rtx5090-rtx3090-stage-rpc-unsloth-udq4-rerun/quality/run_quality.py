#!/usr/bin/env python3
"""Run only Strata's existing needle and community coding fixtures.

Attaches to an existing native API. Never starts/stops/reconfigures an engine.
Outputs private artifacts; review and sanitize before publishing.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

FIXTURE_ROOT = 'bench/results/2026-10-10-community-rx-9060-xt/fixtures'
TASKS = {
    'merge-windows': {'allowed': ['windows.ts'], 'tests': 25},
    'refactor-orders': {'allowed': ['orders.ts', 'validation.ts', 'aggregation.ts'], 'tests': 24},
    'reservation-workflow': {'allowed': ['stock.ts', 'requests.ts', 'allocate.ts'], 'tests': 32},
}
PROMPT = ('Read AGENTS.md and README.md in this directory and perform the supplied '
          'coding task exactly. Follow the allowed-file scope; never change tests, '
          'instructions or configuration. Run node --test acceptance.test.ts. '
          'Permit at most two repair rounds after the initial test run. '
          'No network, installation, delegation or files outside this project. '
          'Finish with the actual pass/fail count and changed files.')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(root):
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def fetch(url, path):
    with urllib.request.urlopen(url.rstrip('/') + path, timeout=10) as response:
        return json.load(response)


def command(args, timeout=40):
    result = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    return {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


def facts(args, remote=False):
    service = 'strata-rpc-coordinator-5090' if remote else 'strata-rpc-worker-3090'
    config = str(args.source / 'strata-rpc.json') if remote else str(Path.home() / '.local/state/strata-stage-rpc/worker.json')
    commands = {
        'service': ['systemctl', '--user', 'show', service + '.service', '--property=MainPID,InvocationID,ActiveState,MemoryCurrent,MemoryPeak'],
        'config_sha256': ['sha256sum', config],
        'binary_sha256': ['sha256sum', str(args.source / 'build-rpc-kquant/strata')],
        'source': ['git', '-C', str(args.source), 'rev-parse', 'HEAD'],
        'dirty': ['git', '-C', str(args.source), 'status', '--short'],
        'source_diff_sha256': ['bash', '-c', 'git -C ' + shlex.quote(str(args.source)) + ' diff --binary | sha256sum'],
        'build_identity': [str(args.source / 'build-rpc-kquant/strata'), '--rpc-build-info'],
        'cpu': ['lscpu', '-J'],
        'ram': ['free', '-b'],
        'os': ['uname', '-srvm'],
        'storage': ['findmnt', '-T', '/mnt/ware/llm_stuff', '-o', 'SOURCE,FSTYPE,OPTIONS', '-n'],
        'gpu': ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,driver_version,power.limit', '--format=csv,noheader,nounits'],
    }
    return {key: command(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', args.remote, shlex.join(cmd)] if remote else cmd)
            for key, cmd in commands.items()}


def property_value(fact, name):
    return next((line.split('=', 1)[1] for line in fact['service']['stdout'].splitlines() if line.startswith(name + '=')), None)


def unchanged(before, after):
    result = {}
    for role in ('local', 'remote'):
        for key in ('config_sha256', 'binary_sha256', 'source', 'source_diff_sha256'):
            result[role + '_' + key] = before[role][key] == after[role][key] and before[role][key]['exit_code'] == 0
        for key in ('MainPID', 'InvocationID', 'ActiveState'):
            first, last = property_value(before[role], key), property_value(after[role], key)
            result[role + '_' + key] = first is not None and first == last
    return result


def ensure_idle(args, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        health, metrics = fetch(args.url, '/health'), fetch(args.url, '/metrics')
        if not health.get('loaded'):
            raise RuntimeError('Backend is not loaded; refusing restart or replay')
        if metrics['live']['state'] == 'idle' and not metrics['live'].get('queued'):
            return metrics
        time.sleep(2)
    raise RuntimeError('Backend did not become idle; no other request was cancelled')


class Sampler:
    def __init__(self, args):
        self.args, self.stop, self.phase = args, threading.Event(), 'setup'
        self.errors, self.samples = [], 0
        self.ready = threading.Event()
        self.guard_error = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        with (self.args.out / 'telemetry.jsonl').open('w') as log:
            while not self.stop.is_set():
                try:
                    metrics = fetch(self.args.url, '/metrics?requests=all')
                    api = {'requests': []}
                    if not getattr(self, 'api_monitor_disabled', False):
                        try:
                            api = fetch(self.args.url, '/api/requests')
                        except urllib.error.HTTPError as error:
                            if error.code != 404:
                                raise
                            self.api_monitor_disabled = True
                            error.close()
                    gpu = command(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'])
                    row = {'time': time.time(), 'phase': self.phase, 'live': metrics.get('live'),
                           'totals': metrics.get('totals'), 'requests': metrics.get('requests'),
                           'api_requests': api.get('requests', []), 'worker_process_vram': gpu,
                           'nodes': [{k: node.get(k) for k in ('id', 'role', 'layers', 'hardware_static', 'hardware', 'freshness', 'engine_cache')}
                                     for node in metrics.get('nodes', [])]}
                    allocations = []
                    for line in gpu.get('stdout', '').splitlines():
                        cells = [part.strip() for part in line.split(',')]
                        if len(cells) == 2 and cells[0].isdigit() and int(cells[0]) == getattr(self.args, 'worker_pid', None):
                            try:
                                allocations.append(float(cells[1]))
                            except ValueError:
                                pass
                    row['worker_process_vram_mib'] = sum(allocations) if allocations else None
                    if allocations and sum(allocations) > 20480:
                        self.guard_error = 'Worker inference allocation exceeded 20480 MiB'
                    log.write(json.dumps(row) + '\n')
                    log.flush()
                    self.samples += 1
                    self.ready.set()
                except Exception as error:
                    self.errors.append({'time': time.time(), 'phase': self.phase, 'error': str(error)})
                self.stop.wait(2)


def terminate_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def independent_test(args, project, output):
    # Read-only fixture, no host home, no credentials, no network namespace.
    cmd = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session',
           '--ro-bind', '/usr', '/usr', '--ro-bind', '/lib', '/lib', '--ro-bind', '/lib64', '/lib64',
           '--ro-bind', str(args.node), '/node', '--ro-bind', str(project), '/work',
           '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--chdir', '/work',
           '--clearenv', '--setenv', 'PATH', '/usr/bin:/bin', '--setenv', 'HOME', '/tmp',
           '/node', '--test', 'acceptance.test.ts']
    started = time.time()
    try:
        result = command(cmd, timeout=60)
    except subprocess.TimeoutExpired:
        result = {'exit_code': 124, 'stdout': '', 'stderr': 'Independent test timeout'}
    output.write_text(result['stdout'] + result['stderr'])
    counts = {key: int(match.group(1)) if (match := re.search(r'^# ' + key + r' (\d+)$', result['stdout'], re.M)) else None
              for key in ('tests', 'pass', 'fail', 'cancelled', 'skipped', 'todo')}
    return {**counts, 'exit_code': result['exit_code'], 'duration_s': time.time() - started,
            'command': ['node', '--test', 'acceptance.test.ts'],
            'isolation': 'read-only Bubblewrap mounts; fresh network namespace; cleared environment; 60 s timeout'}


def event_summary(path):
    events, invalid = [], 0
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            invalid += 1
    completed = [event.get('item', {}) for event in events if event.get('type') == 'item.completed']
    tools = [item for item in completed if item.get('type') in ('command_execution', 'file_change', 'mcp_tool_call')]
    commands = [item for item in tools if item.get('type') == 'command_execution']
    return {'events': len(events), 'invalid_jsonl_lines': invalid,
            'turn_completed': sum(event.get('type') == 'turn.completed' for event in events),
            'turn_failed': sum(event.get('type') == 'turn.failed' for event in events),
            'errors': [event for event in events if event.get('type') == 'error'],
            'usage': [event.get('usage') for event in events if event.get('type') == 'turn.completed'],
            'tool_calls': len(tools), 'command_calls': len(commands),
            'command_errors': sum(item.get('exit_code') not in (None, 0) or item.get('status') == 'failed' for item in commands),
            'file_change_calls': sum(item.get('type') == 'file_change' for item in tools),
            'test_command_calls': sum('node --test' in (item.get('command') or '') for item in commands)}


def run_trial(args, sampler, fixture, attempt, deadline):
    trial_dir = args.out / 'coding' / (fixture + '-' + str(attempt))
    trial_dir.mkdir(parents=True)
    project = trial_dir / 'project'
    if sampler.guard_error:
        raise RuntimeError(sampler.guard_error)
    shutil.copytree(args.fixture_source / FIXTURE_ROOT / fixture, project)
    original = manifest(project)
    write(trial_dir / 'original-manifest.json', original)
    before = ensure_idle(args)
    env = dict(os.environ)
    for key in list(env):
        if re.search(r'(API_KEY|ACCESS_TOKEN|AUTH_TOKEN|GITHUB_TOKEN|HF_TOKEN)$', key):
            env.pop(key)
    env.update(DUAL_CODEX_PROXY_URL=args.url, DUAL_CODEX_WORKDIR=str(project), DUAL_CODEX_SANDBOX='workspace-write')
    cmd = [str(args.launcher), '-c', 'model_reasoning_effort="medium"',
           'exec', '--ignore-user-config', '--json', '--ephemeral', '--skip-git-repo-check',
           '--color', 'never', '-o', str(trial_dir / 'final.txt'), PROMPT]
    started = time.time()
    sampler.phase = fixture + '-' + str(attempt)
    write(trial_dir / 'started.json', {'started_utc': utc(), 'started': started, 'command': cmd,
                                     'reasoning': 'medium', 'baseline_totals': before['totals']})
    print('CODING_STARTED ' + sampler.phase, flush=True)
    timed_out = False
    with (trial_dir / 'events.jsonl').open('w') as stdout, (trial_dir / 'stderr.log').open('w') as stderr:
        process = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        timeout = min(900, max(1, deadline - time.monotonic()))
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_group(process)
    finished = time.time()
    after = ensure_idle(args, timeout=90)
    current = manifest(project)
    changed = sorted(key for key in set(original) | set(current) if original.get(key) != current.get(key))
    permitted = TASKS[fixture]['allowed']
    violations = sorted(set(changed) - set(permitted))
    # Score a separate frozen copy so later filesystem activity cannot change the tested solution.
    frozen = trial_dir / 'verified-copy'
    shutil.copytree(project, frozen)
    verification = independent_test(args, frozen, trial_dir / 'acceptance.log')
    summary = event_summary(trial_dir / 'events.jsonl')
    passed = (not timed_out and process.returncode == 0 and not violations and
              verification['exit_code'] == 0 and verification['tests'] == TASKS[fixture]['tests'] and
              verification['pass'] == TASKS[fixture]['tests'] and verification['fail'] == 0 and
              verification['skipped'] == 0 and verification['cancelled'] == 0 and
              summary['turn_completed'] > 0 and not summary['turn_failed'] and not summary['invalid_jsonl_lines'])
    record = {'fixture': fixture, 'attempt': attempt, 'started': started, 'finished': finished,
              'wall_s': finished - started, 'exit_code': process.returncode, 'timed_out': timed_out,
              'passed': passed, 'changed_files': changed, 'policy_violations': violations,
              'original_manifest': original, 'solution_manifest': current,
              'verification': verification, 'codex': summary,
              'native_counter_deltas': {key: after['totals'][key] - value for key, value in before['totals'].items() if key != 'since'}}
    write(trial_dir / 'result.json', record)
    print('CODING_FINISHED ' + json.dumps({k: record[k] for k in ('fixture', 'attempt', 'passed', 'wall_s', 'exit_code')}) +
          ' verification=' + json.dumps(verification), flush=True)
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path.home() / 'AI-tools/Strata-rpc')
    ap.add_argument('--fixture-source', type=Path, required=True, help='Frozen upstream tools and corpus, separate from the live engine')
    ap.add_argument('--launcher', type=Path, default=Path.home() / 'AI-tools/llama-rpc/scripts/dual_codex_wsl.sh')
    ap.add_argument('--remote', default='user@coordinator')
    ap.add_argument('--url', default='http://127.0.0.1:8090')
    ap.add_argument('--node', type=Path, default=Path(shutil.which('node')).resolve())
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    deadline = time.monotonic() + 14400
    initial = ensure_idle(args)
    before = {'health': fetch(args.url, '/health'), 'props': fetch(args.url, '/props'),
              'local': facts(args), 'remote': facts(args, True), 'totals': initial['totals']}
    args.worker_pid = int(property_value(before['local'], 'MainPID'))
    write(args.out / 'before.json', before)
    write(args.out / 'run.json', {'started_utc': utc(), 'profile': 'dual-strata-unsloth-udq4-rpc',
                                 'managed_run': json.loads((Path.home() / '.local/state/llama-stack/active.json').read_text()).get('run_id'),
                                 'source': str(args.source), 'url': args.url, 'sample_interval_s': 2,
                                 'reasoning_coding': 'medium', 'temperature_coding': 'server default 1.0',
                                 'coding_prompt': PROMPT, 'attempts_per_fixture': 3, 'trial_timeout_s': 900,
                                 'overall_timeout_s': 14400, 'fixture_source': str(args.fixture_source),
                                 'fixture_manifest': manifest(args.fixture_source / FIXTURE_ROOT),
                                 'needle_tool_sha256': digest(args.fixture_source / 'tools/needle_bench.py'),
                                 'launcher_sha256': digest(args.launcher), 'harness_sha256': digest(Path(__file__)),
                                 'codex_version': command(['codex', '--version']), 'node_version': command([str(args.node), '--version'])})
    sampler = Sampler(args)
    sampler.thread.start()
    results, needle_exit, failure = [], None, None
    try:
        if not sampler.ready.wait(30):
            raise RuntimeError('No complete telemetry sample before the first request')
        if sampler.guard_error:
            raise RuntimeError(sampler.guard_error)
        sampler.phase = 'needles'
        remote_out = str(args.out / 'needles.json')
        remote_cmd = 'mkdir -p ' + shlex.quote(str(args.out)) + ' && cd ' + shlex.quote(str(args.fixture_source)) + ' && ' + shlex.join([
            'python3', '-u', 'tools/needle_bench.py', '--url', 'http://127.0.0.1:8090',
            '--lengths', '32k,128k', '--depths', '10,50,90', '--out', remote_out])
        print('NEEDLES_STARTED artifacts=' + str(args.out), flush=True)
        with (args.out / 'needles.log').open('w') as log:
            process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', args.remote, remote_cmd],
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                process.wait(timeout=min(10800, max(1, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                terminate_group(process)
                raise RuntimeError('Needle collection timed out; remote output may be partial')
        needle_exit = process.returncode
        copied = command(['rsync', '-a', args.remote + ':' + remote_out, str(args.out / 'needles.json')])
        if copied['exit_code']:
            raise RuntimeError('Needle artifact retrieval failed: ' + copied['stderr'])
        print('NEEDLES_FINISHED exit=' + str(needle_exit) + ' rows=' + str(len(json.loads((args.out / 'needles.json').read_text()))), flush=True)
        ensure_idle(args)
        for fixture in TASKS:
            for attempt in range(1, 4):
                if time.monotonic() >= deadline:
                    raise RuntimeError('Overall four-hour budget exhausted; remaining attempts not run')
                results.append(run_trial(args, sampler, fixture, attempt, deadline))
    except Exception as error:
        failure = {'type': type(error).__name__, 'message': str(error)}
        print('COLLECTION_ERROR ' + json.dumps(failure), flush=True)
    finally:
        sampler.phase = 'finalize'
        sampler.stop.set()
        sampler.thread.join(timeout=45)
    after = {'health': fetch(args.url, '/health'), 'local': facts(args), 'remote': facts(args, True),
             'totals': fetch(args.url, '/metrics')['totals']}
    write(args.out / 'after.json', after)
    needles = json.loads((args.out / 'needles.json').read_text()) if (args.out / 'needles.json').exists() else []
    all_samples = [json.loads(line) for line in (args.out / 'telemetry.jsonl').read_text().splitlines()]
    requests = {}
    peaks = {}
    for sample in all_samples:
        for request in sample['requests']:
            if request['time'] >= start:
                requests[request['time']] = request
        for node in sample['nodes']:
            peak = peaks.setdefault(node['id'], {})
            for key in ('gpu_mem_used', 'gpu_util', 'gpu_power', 'ram_used', 'net_rx_mib_s', 'net_tx_mib_s'):
                value = (node.get('hardware') or {}).get(key)
                if isinstance(value, (float, int)):
                    peak[key] = max(value, peak.get(key, value))
    # Include the final completion even if the sampler was between polls.
    for request in fetch(args.url, '/metrics?requests=all')['requests']:
        if request['time'] >= start:
            requests[request['time']] = request
    write(args.out / 'native-requests.json', sorted(requests.values(), key=lambda request: request['time']))
    changes = unchanged(before, after)
    needle_complete = len(needles) == 6 and {(r.get('length'), r.get('depth')) for r in needles} == {
        (length, depth) for length in ('32k', '128k') for depth in (10, 50, 90)}
    record = {'started': start, 'finished': time.time(), 'completed_utc': utc(),
              'needle_exit_code': needle_exit, 'needle_cases': len(needles),
              'needle_complete': needle_complete, 'needle_found': sum(r.get('found', False) for r in needles),
              'needle_errors': sum('error' in r for r in needles),
              'needle_passed': needle_complete and all(r.get('found') is True and 'error' not in r for r in needles),
              'coding_attempts': results, 'coding_passed': sum(r['passed'] for r in results),
              'coding_expected': 9, 'samples': sampler.samples, 'telemetry_errors': sampler.errors,
              'worker_vram_guard_error': sampler.guard_error,
              'failure': failure, 'unchanged': changes, 'observed_peaks': peaks,
              'native_counter_deltas': {key: after['totals'][key] - value for key, value in before['totals'].items() if key != 'since'},
              'native_requests_collected': len(requests)}
    write(args.out / 'summary.json', record)
    print('QUALITY_FINISHED ' + json.dumps({k: record[k] for k in ('needle_found', 'needle_cases', 'coding_passed', 'coding_expected', 'samples', 'failure', 'unchanged')}), flush=True)
    return int(failure is not None or not all(changes.values()) or len(results) != 9)


if __name__ == '__main__':
    raise SystemExit(main())
