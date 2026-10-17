"""Explicit opt-in runner for owned servers only. No production start/stop/config operations."""
from __future__ import annotations

import concurrent.futures
import csv
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from catalog import ROOT, WORKSPACE, PYTHON, PORT, URL, arg, digest, save, validate_config

PROTECTED = []  # standalone report: never rewrite or capture production configuration



def protected_hashes() -> dict:
    return {str(p): digest(p.read_bytes()) for p in PROTECTED if p.exists()}


def modules():
    import psutil
    import requests
    return psutil, requests


def production_busy(psutil) -> list[str]:
    """Read OS metadata, without connecting to the production API or sending TCP probes."""
    reasons = []
    for connection in psutil.net_connections(kind='tcp'):
        if connection.status == psutil.CONN_LISTEN and connection.laddr.port == 8080:
            reasons.append('production port 8080 is listening')
    for process in psutil.process_iter(['pid', 'name', 'exe']):
        name = (process.info['name'] or '').lower()
        if name == 'strata.exe':
            reasons.append(f'a Strata engine is already active (PID {process.info["pid"]})')
    return reasons


def gpu_snapshot() -> list[dict]:
    fields = ['index', 'name', 'driver_version', 'memory.total', 'memory.used', 'utilization.gpu',
              'temperature.gpu', 'power.draw']
    process = subprocess.run(['nvidia-smi', '--query-gpu=' + ','.join(fields),
                              '--format=csv,noheader,nounits'], capture_output=True,
                             text=True, encoding='utf-8', errors='replace', timeout=10, check=True,
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    result = []
    for row in csv.reader(io.StringIO(process.stdout.replace('\x00', ''))):
        item = dict(zip(fields, [v.strip() for v in row]))
        item['index'] = int(item['index'])
        for key in fields[3:]:
            try:
                item[key] = float(item[key])
            except ValueError:
                item[key] = None
        result.append(item)
    if len(result) < 2 or any(r['memory.used'] is None for r in result[:2]):
        raise RuntimeError('Cannot verify both GPU memory readings.')
    return result


def preflight(psutil) -> dict:
    reasons = production_busy(psutil)
    if reasons:
        raise RuntimeError('Experiment refused: ' + '; '.join(reasons) + '. No service was touched.')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', PORT))  # fail if the separate experimental port is owned by somebody else
    gpu = gpu_snapshot()
    if any(row['memory.used'] > 2048 for row in gpu[:2]):
        raise RuntimeError('A GPU is occupied by another workload (>2048 MiB). No model started.')
    available = psutil.virtual_memory().available / 2**30
    if available < 44:
        raise RuntimeError(f'Only {available:.1f} GiB RAM available; this suite requires 44 GiB before loading.')
    return dict(gpu=gpu, available_ram_gib=available, cpu_logical=psutil.cpu_count(),
                cpu_physical=psutil.cpu_count(logical=False))


class OwnedServer:
    def __init__(self, config: dict, folder: Path, psutil, requests):
        self.psutil, self.requests = psutil, requests
        self.folder = folder
        self.proc = None
        self.created = None
        self.handles = []
        self.cfg = config
        self.http = requests.Session()
        self.http.trust_env = False

    def start(self):
        validate_config(self.cfg)
        cfg_path = self.folder / 'config.json'
        save(cfg_path, self.cfg)
        env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_')}
        env.pop('CUDA_VISIBLE_DEVICES', None)
        for credential in ('STRATA_API_KEY', 'OPENAI_API_KEY', 'CODEX_API_KEY', 'GH_TOKEN', 'GITHUB_TOKEN'):
            env.pop(credential, None)
        env.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        env.update({str(k): str(v) for k, v in self.cfg.get('env', {}).items()})
        for name in ('server.stdout.log', 'server.stderr.log'):
            self.handles.append((self.folder / name).open('wb'))
        self.proc = subprocess.Popen([str(PYTHON), '-u', '-m', 'serve.server', '--engine', 'strata',
                                      '--config', str(cfg_path), '--host', '127.0.0.1', '--port', str(PORT)],
                                     cwd=self.cfg['server_root'], env=env, stdin=subprocess.DEVNULL,
                                     stdout=self.handles[0], stderr=self.handles[1],
                                     creationflags=(getattr(subprocess, 'CREATE_NO_WINDOW', 0) |
                                                    getattr(subprocess, 'NORMAL_PRIORITY_CLASS', 0)))
        self.created = self.psutil.Process(self.proc.pid).create_time()

    def assert_owned(self):
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError('The owned experimental server has exited.')
        owner = self.psutil.Process(self.proc.pid)
        if owner.create_time() != self.created:
            raise RuntimeError('Experimental process identity changed; no request sent.')
        allowed = {owner.pid, *(p.pid for p in owner.children(recursive=True))}
        listeners = [c for c in self.psutil.net_connections(kind='tcp')
                     if c.status == self.psutil.CONN_LISTEN and c.laddr.port == PORT]
        if not listeners:
            raise ConnectionError('The experimental server has not opened its port yet.')
        if any(c.pid not in allowed for c in listeners):
            raise RuntimeError('Experimental port is owned by another process; no request sent.')

    def get(self, endpoint):
        self.assert_owned()
        response = self.http.get(URL + endpoint, timeout=(5, 10))
        response.raise_for_status()
        return response.json()

    def ready(self, timeout=600):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f'Experimental server exited while loading: {self.proc.returncode}.')
            try:
                health = self.get('/health')
                if health.get('loaded'):
                    if health.get('model') != self.cfg['model_name']:
                        raise RuntimeError('Experimental model identity mismatch.')
                    status = self.get('/v1/status')
                    if status.get('engine') != self.cfg['expected_engine_version']:
                        raise RuntimeError('Engine version mismatch.')
                    return dict(health=health, status=status, metrics=self.get('/metrics'))
            except (ConnectionError, self.requests.RequestException):
                pass
            time.sleep(2)
        raise TimeoutError('Experimental model did not load within 600 seconds.')

    def stop(self):
        """No port-based kill and no /unload sent to an unknown server."""
        descendants = []
        same_parent = False
        if self.proc is not None:
            try:
                parent = self.psutil.Process(self.proc.pid)
                if parent.create_time() == self.created:
                    same_parent = True
                    descendants = [(p, p.create_time()) for p in parent.children(recursive=True)]
            except self.psutil.NoSuchProcess:
                pass
            if same_parent and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=10)
            for child, created in reversed(descendants):
                try:
                    if child.is_running() and child.create_time() == created:
                        child.terminate()
                        try:
                            child.wait(timeout=5)
                        except self.psutil.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=5)
                except self.psutil.NoSuchProcess:
                    pass
        for handle in self.handles:
            handle.close()
        self.http.close()


class Sampler:
    def __init__(self, server: OwnedServer, interval=2.0):
        self.server, self.interval = server, interval
        self.samples = []
        self.errors = []
        self.abort_reason = None
        self.low_ram_samples = 0
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        psutil = self.server.psutil
        while not self.done.is_set():
            try:
                if (ROOT / 'STOP_REQUESTED').exists():
                    self.abort_reason = 'Operator requested a scope change; stopping this owned experiment.'
                    return
                # Protect a newly started production listener too; never stop it.
                if any(c.status == psutil.CONN_LISTEN and c.laddr.port == 8080
                       for c in psutil.net_connections(kind='tcp')):
                    self.abort_reason = 'Production port 8080 became active; aborting the experiment only.'
                    return
                proc = psutil.Process(self.server.proc.pid)
                family = [proc, *proc.children(recursive=True)]
                rss = 0
                for member in family:
                    try:
                        rss += member.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                available = psutil.virtual_memory().available / 2**30
                self.low_ram_samples = self.low_ram_samples + 1 if available < 0.5 else 0
                swap = psutil.swap_memory()
                self.samples.append(dict(time=time.time(), gpu=gpu_snapshot(), process_rss_gib=rss / 2**30,
                                         ram_available_gib=available, swap_used_gib=swap.used / 2**30))
                if self.low_ram_samples >= 3:
                    self.abort_reason = 'Available RAM stayed below 0.5 GiB for three samples; experiment aborted.'
                    return
            except Exception as exc:
                self.errors.append(f'{type(exc).__name__}: {exc}')
            self.done.wait(self.interval)

    def start(self):
        self.thread.start()

    def stop(self):
        self.done.set()
        self.thread.join(timeout=15)


def parse_events(lines):
    for raw in lines:
        if not raw.startswith(b'data:'):
            continue
        data = raw[5:].strip()
        if data == b'[DONE]':
            return
        obj = json.loads(data)
        if obj.get('error'):
            raise RuntimeError(f'Stream error: {obj["error"]}')
        yield obj


def request(server, case, barrier, sampler, timeout=1800):
    server.assert_owned()
    body = dict(model='strata-experiment', messages=case['messages'], temperature=0, seed=42,
                reasoning_effort='none', max_tokens=case['max_tokens'], stream=True,
                stream_options=dict(include_usage=True))
    payload_hash = digest(json.dumps(body, ensure_ascii=False, sort_keys=True).encode('utf-8'))
    session = server.requests.Session()
    session.trust_env = False
    session.trust_env = False
    if barrier:
        barrier.wait(timeout=30)
    start, first, chunks, usage, finish = time.perf_counter(), None, [], None, None
    try:
        with session.post(URL + '/v1/chat/completions', json=body, stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            def timed_lines():
                for line in response.iter_lines(chunk_size=1, delimiter=b'\n'):
                    if sampler.abort_reason:
                        raise RuntimeError(sampler.abort_reason)
                    if time.perf_counter() - start > timeout:
                        raise TimeoutError(f'Request exceeded its {timeout}-second wall deadline.')
                    yield line  # check deadlines on keepalives too, not just generated tokens
            for chunk in parse_events(timed_lines()):
                if isinstance(chunk.get('usage'), dict):
                    usage = chunk['usage']
                for choice in chunk.get('choices') or []:
                    delta = choice.get('delta') or {}
                    text = (delta.get('content') or '') + (delta.get('reasoning_content') or '')
                    if text:
                        first = first if first is not None else time.perf_counter()
                        chunks.append(text)
                    finish = choice.get('finish_reason') or finish
        end = time.perf_counter()
        if not usage or finish is None or not usage.get('completion_tokens'):
            raise RuntimeError('Missing usage, finish reason or generated tokens.')
        text = ''.join(chunks)
        return dict(case=case['name'], payload_sha256=payload_hash, messages_sha256=case['messages_sha256'],
                    start=start, end=end, elapsed_s=end - start, ttft_s=first - start if first else None,
                    usage=usage, finish=finish, answer_sha256=digest(text.encode('utf-8')),
                    answer=text.replace('\x00', '\ufffd'))
    finally:
        session.close()


def native_records(before: dict, after: dict) -> list[dict]:
    identities = {json.dumps(x, sort_keys=True) for x in before.get('requests', [])}
    return [x for x in after.get('requests', []) if json.dumps(x, sort_keys=True) not in identities]


def attach_native(rows, native):
    remaining = list(native)
    for row in rows:
        matches = [n for n in remaining if n.get('prompt_tokens') == row['usage'].get('prompt_tokens') and
                   n.get('output_tokens') == row['usage'].get('completion_tokens')]
        row['native'] = matches[0] if len(matches) == 1 else None
        if row['native'] is not None:
            remaining.remove(row['native'])
        row['native_match'] = 'unique' if row['native'] is not None else 'ambiguous_or_missing'


def run_group(server, cases, sampler, folder, mode):
    before = server.get('/metrics')
    barrier = threading.Barrier(len(cases)) if len(cases) > 1 else None
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cases)) as pool:
        futures = [pool.submit(request, server, case, barrier, sampler) for case in cases]
        try:
            rows = [future.result(timeout=1900) for future in futures]
        except BaseException:
            server.stop()  # unblock all clients immediately after a failed or interrupted group
            for future in futures:
                future.cancel()
            raise
    # /metrics can become visible just after the last SSE frame.
    deadline = time.monotonic() + 5
    while True:
        after = server.get('/metrics')
        native = native_records(before, after)
        if len(native) >= len(cases) or time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    attach_native(rows, native)
    errors = []
    if len(native) != len(cases):
        errors.append('native request count mismatch')
    if mode == 'cold' and (len(native) != len(cases) or any(n.get('reused') != 0 for n in native)):
        errors.append('cold run has missing reuse counters or nonzero reused tokens')
    for index, (row, case) in enumerate(zip(rows, cases)):
        if case.get('required_output_tokens') and row['usage']['completion_tokens'] != case['required_output_tokens']:
            errors.append(f'Expected exactly{case["required_output_tokens"]} generated tokens, got{row["usage"]["completion_tokens"]}')
        if case.get('expected_prompt_tokens') is not None and row['usage']['prompt_tokens'] != case['expected_prompt_tokens']:
            errors.append(f'Expected frozen prompt count {case["expected_prompt_tokens"]}, got {row["usage"]["prompt_tokens"]}')
        answer_path = folder / f'{index}-{case["name"]}.txt'
        answer_path.write_text(row.pop('answer'), encoding='utf-8')
        row['answer_file'] = str(answer_path.relative_to(ROOT))
        if case['kind'] == 'code':
            try:
                check = subprocess.run([str(PYTHON), '-I', str(ROOT / 'experiments.py'),
                                        '--validate-code', str(answer_path)], capture_output=True,
                                       text=True, encoding='utf-8', timeout=30,
                                       env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1'},
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                row['validation'] = json.loads(check.stdout) if check.returncode == 0 else dict(
                    passed=False, error=check.stderr[-500:])
            except Exception as exc:
                row['validation'] = dict(passed=False, error=f'{type(exc).__name__}: {exc}')
        else:
            from validators import validate_text
            row['validation'] = validate_text(case, answer_path.read_text(encoding='utf-8'))
    wall = max(r['end'] for r in rows) - min(r['start'] for r in rows)
    return dict(case_names=[c['name'] for c in cases], concurrency=len(cases), wall_s=wall,
                end_to_end_output_tok_s=sum(r['usage']['completion_tokens'] for r in rows) / wall,
                ttft_max_s=max(r['ttft_s'] for r in rows if r['ttft_s'] is not None),
                comparable=not errors, errors=errors, rows=rows, native_records=native,
                metrics_before=before, metrics_after=after)


def activation(profile, cfg, status, text):
    requested_slots = cfg.get('parallel', 1)
    actual_slots = status.get('concurrency', {}).get('serving')
    result = dict(serving_expected=requested_slots, serving_actual=actual_slots,
                  valid=actual_slots == requested_slots, evidence=[])
    rules = []
    if '--kv-grow' in cfg['args']:
        rules.extend([('dual_kv_grow', '--kv-grow: independent KV and expert budgets on 2 GPU(s)'),
                      ('kv_gpu0', 'elastic K/V CUDA0:'), ('kv_gpu1', 'elastic K/V CUDA1:')])
    if '--pipeline-windows' in cfg['args']:
        number = cfg['args'][cfg['args'].index('--pipeline-windows') + 1]
        rules.append(('pipeline', f'--pipeline-windows {number}: two verifiers per stage'))
        if number == '2':
            rules.append(('pipeline_decode', 'decode too;'))
    if '--resident-experts' in cfg['args']:
        rules.append(('resident', 'resident RAM:'))
    if '--adapt-async' in cfg['args']:
        rules.append(('async', 'adaptive tier asynchronous (--adapt-async 1)'))
    if cfg.get('env', {}).get('STRATA_EXCHANGE_ROTATE') == '1':
        rules.append(('rotation', 'exchange rotation:'))
    if cfg.get('env', {}).get('STRATA_DMA_BOUNCE') == '1':
        rules.extend([('dma_gpu0_actual', 'strata DMA CUDA0: actual cudaMemcpyBatchAsync'),
                      ('dma_gpu1_actual', 'strata DMA CUDA1: actual cudaMemcpyBatchAsync')])
    if '--host-core' in cfg['args']:
        rules.append(('host_core', '(--host-core last)'))
    for kind, token in rules:
        present = token in text
        result['evidence'].append(dict(feature=kind, confirmed=present, token=token))
        result['valid'] = result['valid'] and present
    if not rules and profile not in ('v39-pp256', 'v40-pp256', 'v40-pp128'):
        result['activation_note'] = 'configured; no positive runtime activation marker required for this flag'
    if 'starting it again' in text:
        result['valid'] = False
        result['evidence'].append(dict(feature='engine_restart', confirmed=True))
    if 'cudaMemcpyBatchAsync failed' in text or 'expert upload failed' in text:
        result['evidence'].append(dict(feature='dma_fallback', confirmed=True))
    return result


def visit(item, cfg, cases, runroot, cache_mode, suite, c4):
    psutil, requests = modules()
    hardware = preflight(psutil)
    label = f'{item["pair"]}-r{item["round"]}-{item["arm"]}'
    folder = runroot / label
    folder.mkdir()  # no overwrite on resume
    config = json.loads(json.dumps(cfg))
    config['model_name'] = 'strata-experiment-' + uuid.uuid4().hex
    config['log'] = str(folder / 'engine.log')
    if cache_mode == 'cold':
        config['args'] = arg(config['args'], '--prompt-cache', '0')
    if suite == 'cachecheck':
        for flag, value in [('--prompt-cache', '6'), ('--conversation-cache-mib', '4096'),
                            ('--conversation-cache-slots', '4')]:
            config['args'] = arg(config['args'], flag, value)
    result = dict(**item, hardware_before=hardware, cache_mode=cache_mode, warmup_protocol=4 if suite.startswith('repeat10-') else 'transition' if suite == 'cachecheck' else 3 if suite in ('combo-dma', 'combo-dma-full') else 2, groups=[], errors=[],
                  engine_sha256=digest(Path(cfg['exe']).read_bytes()),
                  source_sha256=digest((Path(cfg['server_root']) / 'serve/server.py').read_bytes()))
    server = OwnedServer(config, folder, psutil, requests)
    sampler = Sampler(server)
    sampling_started = False
    t0 = time.perf_counter()
    try:
        server.start()
        result['server_pid'] = server.proc.pid
        result['server_priority'] = psutil.Process(server.proc.pid).nice()
        result['ready'] = server.ready()
        if result['ready']['health'].get('max_context') != 262144 or result['ready']['metrics']['engine'].get('kv') != 'int8':
            raise RuntimeError('The experimental engine did not retain complete 262144 / int8 capacity.')
        result['startup_s'] = time.perf_counter() - t0
        sampler.start()
        sampling_started = True
        warmup = dict(cases['code-8192'], name='warmup', max_tokens=128)
        warmup['messages'] = [{'role': 'user', 'content': 'Warmup only. List ten common programming concepts.'}]
        # Same warmup for every fresh process; never counted as measured work.
        if suite in ('combo-dma', 'combo-dma-full') or suite.startswith('repeat10-'):
            warmup = dict(cases['code-8192'], name='fixed-code-warmup', max_tokens=128)
            seeded = request(server, warmup, None, sampler)
            if seeded['usage']['prompt_tokens'] != 8192 or seeded['usage']['completion_tokens'] != 128:
                raise RuntimeError('Fixed warmup must have exactly8192 input and128 output tokens.')
            result['fixed_warmup'] = {key: seeded[key] for key in ('payload_sha256', 'usage', 'finish', 'elapsed_s')}
        else:
            request(server, warmup, None, sampler)
        names = ['short-zh', 'copy-edit'] if suite == 'quick' else ['short-zh', 'copy-edit', 'code8k', 'json8k', 'chat32k']
        if suite in ('capacity262', 'repeat262'):
            names = ['code8k', 'json8k', 'chat32k', 'code261k']
        elif suite == 'reserve262':
            names = ['code8k', 'json8k', 'copy-edit']
        elif suite == 'curve':
            names = [f'code-{n}' for n in (8192, 32768, 65536, 131072, 204800, 261000)]
        elif suite == 'longcheck':
            names = [f'code-{n}' for n in (32768, 131072, 261000)]
        elif suite == 'combo-dma':
            names = [f'code-{n}' for n in (32768, 131072, 261000)]
        elif suite == 'combo-dma-full':
            names = ['code-261000']
        elif suite == 'full':
            names = ['code-261000']
        elif suite == 'cachecheck':
            names = [f'code-{n}' for n in (32768, 8192, 32768, 131072, 8192, 32768)]
        elif suite.startswith('repeat10-'):
            names = [f'code-{n}' for n in (65536, 131072, 261000)]
        if suite == 'long':
            names.append('code253k')
        for sequence, name in enumerate(names):
            # Warm this workload's decode/attention shapes, not only one English prompt.
            # Cold mode still has --prompt-cache 0, audited with native reused counters.
            # Reuse mode deliberately seeds this exact prefix before measuring.
            print(f'    {item["profile"]}: {name} ({"first near-full capacity request" if name == "code261k" and suite == "capacity262" else "warm + measure"})', flush=True)
            if suite in ('combo-dma', 'combo-dma-full') or suite.startswith('repeat10-'):
                first_folder = folder / (name + '-first')
                first_folder.mkdir()
                first = run_group(server, [cases[name]], sampler, first_folder, cache_mode)
                first['label'] = name + '-first'
                first['shape_state'] = 'first_after_previous_context'
                result['groups'].append(first)
                save(folder / 'result.json', result)
                if suite.startswith('repeat10-') and (not first['comparable'] or not all(r['validation']['passed'] for r in first['rows'])):
                    raise RuntimeError('First workload output/count check failed; stopped without replacing the sample.')
                if suite == 'repeat10-first':
                    continue
            elif suite != 'cachecheck' and not (name == 'code261k' and suite == 'capacity262'):
                request(server, cases[name], None, sampler)
            label = f'{sequence:02d}-{name}' if suite == 'cachecheck' else name
            group_folder = folder / label
            group_folder.mkdir()
            group = run_group(server, [cases[name]], sampler, group_folder, cache_mode)
            group['label'] = label
            result['groups'].append(group)
            save(folder / 'result.json', result)
            if suite.startswith('repeat10-') and (not group['comparable'] or not all(r['validation']['passed'] for r in group['rows'])):
                raise RuntimeError('Measured workload output/count check failed; stopped without replacing the sample.')
        if suite in ('curve', 'longcheck', 'full', 'cachecheck', 'combo-dma', 'combo-dma-full') or suite.startswith('repeat10-'):
            if suite == 'cachecheck':
                restored = result['groups'][2]['rows'][0].get('native') or {}
                if not restored.get('reused', 0):
                    result['errors'].append('32K conversation did not reuse a parked prefix after 8K switch.')
            result['final_status'] = server.get('/v1/status')
            return result
        # Even quick mode needs two simultaneous requests to exercise the batch path.
        members = (['copy-edit', 'short-zh'] if suite == 'quick' else
                   ['code8k', 'json8k', 'chat8k', 'copy-edit'] if c4 else ['code8k', 'json8k'])
        group_folder = folder / f'concurrent{len(members)}'
        group_folder.mkdir()
        # Capture real batch-window shapes before the timed group as well.
        warm_barrier = threading.Barrier(len(members))
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(members)) as pool:
            warm_futures = [pool.submit(request, server, cases[n], warm_barrier, sampler) for n in members]
            try:
                for future in warm_futures:
                    future.result(timeout=1900)
            except BaseException:
                server.stop()
                raise
        group = run_group(server, [cases[n] for n in members], sampler, group_folder, cache_mode)
        group['label'] = f'concurrent{len(members)}'
        result['groups'].append(group)
        result['final_status'] = server.get('/v1/status')
    except (Exception, KeyboardInterrupt) as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
        if isinstance(exc, KeyboardInterrupt):
            result['interrupted'] = True
    finally:
        if sampling_started:
            sampler.stop()
        # Read and retain diagnostics before/after owned-process cleanup.
        save(folder / 'samples.json', dict(samples=sampler.samples, errors=sampler.errors))
        if sampler.abort_reason:
            result['errors'].append(sampler.abort_reason)
        try:
            result['exit_before_cleanup'] = server.proc.poll() if server.proc else None
            server.stop()
        except Exception as exc:
            result['errors'].append(f'Cleanup failed: {exc}')
        text = '\n'.join(p.read_text(encoding='utf-8', errors='replace').replace('\x00', '\ufffd')
                         for p in (folder / 'engine.log', folder / 'server.stdout.log', folder / 'server.stderr.log')
                         if p.exists())
        result['activation'] = activation(item['profile'], config,
                                          result.get('final_status', result.get('ready', {}).get('status', {})), text)
        result['resource_summary'] = resource_summary(sampler.samples)
        result['sample_errors'] = sampler.errors
        try:
            # Finish cleanup before another arm; GPU memory may be released asynchronously on WDDM.
            deadline = time.monotonic() + 60
            while True:
                result['gpu_after'] = gpu_snapshot()
                if all(r['memory.used'] <= 2048 for r in result['gpu_after'][:2]):
                    break
                if time.monotonic() >= deadline:
                    result['errors'].append('GPU memory was not released within 60 seconds.')
                    break
                time.sleep(2)
        except Exception as exc:
            result['errors'].append(f'Cannot verify GPU cleanup: {exc}')
        result['valid'] = (not result['errors'] and result['activation']['valid'] and
                           bool(result['groups']) and all(g['comparable'] for g in result['groups']))
        save(folder / 'result.json', result)
    return result


def resource_summary(samples):
    result = dict(sample_count=len(samples))
    if samples:
        result['ram_available_min_gib'] = min(s['ram_available_gib'] for s in samples)
        result['process_rss_peak_gib'] = max(s['process_rss_gib'] for s in samples)
        result['gpu_peak_used_mib'] = {str(index): max(row['memory.used'] for s in samples for row in s['gpu']
                                                    if row['index'] == index)
                                      for index in (0, 1)}
    return result
