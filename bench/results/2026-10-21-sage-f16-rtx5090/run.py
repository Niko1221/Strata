import argparse, fcntl, json, os, socket, subprocess, sys, time, urllib.request, threading
import psutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = None
MODEL = None
PORT = 8196
BASE = f'http://127.0.0.1:{PORT}'
sys.path.insert(0, str(ROOT / 'tools/qwen35'))
from q36_harness import Eng

def http(path, body=None, timeout=7200):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def gpu():
    return subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()

def make_prompts(context):
    def tokenize(s): return http('/tokenize', {'content': s, 'add_special': False, 'parse_special': True})['tokens']
    prompts = []
    sizes = [('short', 64), ('512', 512), ('4096', 4096)]
    if context >= 262144: sizes += [('32768',32768),('131072',131072),('261888',261888)]
    corpus = tokenize('\n'.join(f'Record {i:05d}: component=valve pressure={100 + i % 71} inspection=passed.' for i in range(20000 if context >= 262144 else 2000)))
    for name, size in sizes:
        prefix = tokenize('<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\nBenchmark ' + name + '.\n')
        suffix = tokenize('\nWrite a detailed Python function that validates these records and summarizes pressure by component. Explain the algorithm.\n<|im_end|>\n<|im_start|>assistant\n<think>\n')
        ids = prefix + corpus[:max(0, size - len(prefix) - len(suffix))] + suffix
        prompts.append({'name': name, 'ids': ids})
    prompt_path.write_text(json.dumps(prompts, indent=2) + '\n')
    return prompts

ap = argparse.ArgumentParser()
ap.add_argument('kind', choices=['llama', 'strata'])
ap.add_argument('--name', required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--model', type=Path, required=True)
ap.add_argument('--strata-bin', type=Path)
ap.add_argument('--llama-server', type=Path)
ap.add_argument('--port', type=int, default=8196)
ap.add_argument('--threads', type=int, default=16)
ap.add_argument('--ncmoe', type=int, default=26)
ap.add_argument('--chunk', type=int, default=512)
ap.add_argument('--context', type=int, default=262144)
ap.add_argument('--mtp', type=int, default=0)
ap.add_argument('--kv', choices=['fp32','f16'])
ap.add_argument('--reps', type=int, default=1)
ap.add_argument('--gen', type=int, default=256)
ap.add_argument('--cells', default='261888')
a = ap.parse_args()
OUT = a.out.resolve(); OUT.mkdir(parents=True, exist_ok=True)
MODEL = a.model.resolve()
PORT = a.port; BASE = f'http://127.0.0.1:{PORT}'
if a.kind == 'strata' and not a.strata_bin: ap.error('--strata-bin is required for strata')
if a.kind == 'llama' and not a.llama_server: ap.error('--llama-server is required for llama')
# Convert paths to JSON-safe metadata while retaining Path objects for command construction.
options = {k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()}
prompt_path = OUT / ('prompts.json' if a.context == 8192 else f'prompts-{a.context}.json')
assert MODEL.exists(), MODEL
lock = open('/tmp/strata-sage-benchmark.lock', 'w')
fcntl.flock(lock, fcntl.LOCK_EX)
meta = {'kind': a.kind, 'options': options, 'start': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'gpu_before': gpu(), 'runs': []}
log_path = OUT / (a.name + '.server.log')
path = OUT / (a.name + '.json')
process = None
engine = None
log = None
progress = (OUT / (a.name + '.progress.log')).open('w')
stop_monitor = threading.Event()
samples = []
def monitor():
    psutil.cpu_percent()
    while not stop_monitor.wait(5):
        child = process if process is not None else (engine.p if engine else None)
        sample = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'cpu_percent': psutil.cpu_percent(),
                  'available_ram_gib': psutil.virtual_memory().available / 2**30, 'gpu': gpu()}
        if child and child.poll() is None:
            try:
                proc = psutil.Process(child.pid)
                sample.update({'process_cpu_s': sum(proc.cpu_times()[:2]), 'process_rss_gib': proc.memory_info().rss / 2**30})
            except psutil.Error: pass
        samples.append(sample)
threading.Thread(target=monitor, daemon=True).start()
class LoggedEng(Eng):
    def line(self, timeout=600):
        line = super().line(timeout)
        if line and line.startswith(('PP ', 'READY', 'DONE', 'ERR')):
            progress.write(time.strftime('%Y-%m-%dT%H:%M:%S%z') + ' ' + line + '\n')
            progress.flush()
        return line
try:
    if a.kind == 'llama':
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', PORT))
        cmd = [str(a.llama_server.resolve()), '-m', str(MODEL), '--host', '127.0.0.1', '--port', str(PORT),
               '-ngl', '99', '-ncmoe', str(a.ncmoe), '-t', str(a.threads), '-tb', str(a.threads), '-c', str(a.context), '-np', '1',
               '-b', str(a.chunk), '-ub', str(a.chunk), '-fa', 'on', '--fit', 'off', '--no-webui', '--no-warmup', '-lv', '4']
        if a.ncmoe < 0:
            for flag in ['-ngl', '-ncmoe']:
                i = cmd.index(flag)
                del cmd[i:i + 2]
            cmd[cmd.index('--fit') + 1] = 'on'
        meta['command'] = cmd
        log = log_path.open('w')
        t0 = time.monotonic()
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        while time.monotonic() - t0 < 900:
            if process.poll() is not None: raise RuntimeError(f'llama-server exited {process.returncode}; see {log_path}')
            try:
                if http('/health', timeout=2).get('status') == 'ok': break
            except Exception: pass
            time.sleep(.5)
        else: raise TimeoutError('server loading timeout')
        meta['load_s'] = time.monotonic() - t0
        prompts = json.loads(prompt_path.read_text()) if prompt_path.exists() else make_prompts(a.context)
        def run(prompt, n):
            t = time.monotonic()
            r = http('/completion', {'prompt': prompt, 'n_predict': n, 'temperature': 0., 'seed': 271828, 'cache_prompt': False,
                                      'ignore_eos': True, 'return_tokens': True, 'n_probs': 0})
            return {'wall_s': time.monotonic() - t, 'tokens': r.get('tokens'), 'content': r.get('content'), 'timings': r.get('timings'),
                    'stop': r.get('stop_type'), 'cached': r.get('tokens_cached'), 'truncated': r.get('truncated'), 'tokens_evaluated': r.get('tokens_evaluated')}
    else:
        args = ['--native', str(MODEL), '--f16-weights', '--max-context', str(a.context), '--prefill', str(a.chunk), '--mtp', str(a.mtp),
                '--pool-workers', str(a.threads)]
        if a.kv: args += ['--kv', a.kv]
        meta['command'] = [str(a.strata_bin.resolve()), '--serve', *args]
        engine = LoggedEng(str(a.strata_bin.resolve()), args, log=str(log_path))
        meta['load_s'], meta['ready'] = engine.load_s, engine.ready
        if not prompt_path.exists(): raise SystemExit('Run the llama side first to write the exact token ids in --out.')
        prompts = json.loads(prompt_path.read_text())
        def run(prompt, n):
            toks, done, ttft, wall = engine.solo(prompt, n)
            f = done.split()
            np, nd, ppms, decms = int(f[2]), int(f[1]), float(f[3]), float(f[4])
            return {'wall_s': wall, 'ttft_s': ttft, 'tokens': toks, 'done': done,
                    'timings': {'prompt_n': np, 'prompt_ms': ppms, 'prompt_per_second': 1000 * np / max(.001, ppms),
                                'predicted_n': nd, 'predicted_ms': decms, 'predicted_per_second': 1000 * max(0, nd - 1) / max(.001, decms)}}
    print(a.name, 'ready in', round(meta['load_s'], 2), 'seconds', flush=True)
    meta['warmup'] = run(prompts[0]['ids'], 32)
    cells = set(a.cells.split(','))
    for rep in range(a.reps):
        for prompt in prompts:
            if prompt['name'] not in cells: continue
            result = run(prompt['ids'], a.gen)
            result.update({'name': prompt['name'], 'rep': rep, 'gpu_after': gpu()})
            meta['runs'].append(result)
            path.write_text(json.dumps(meta, indent=2) + '\n')
            print(a.name, prompt['name'], rep, result['timings'], flush=True)
except BaseException as e:
    meta['error'] = repr(e)
    print(a.name, 'FAILED', repr(e), flush=True)
    raise
finally:
    stop_monitor.set()
    meta['resource_samples'] = samples
    meta['gpu_end'] = gpu()
    progress.close()
    if engine: engine.close()
    if process:
        process.terminate()
        try: process.wait(timeout=30)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
    if log: log.close()
    path.write_text(json.dumps(meta, indent=2) + '\n')
