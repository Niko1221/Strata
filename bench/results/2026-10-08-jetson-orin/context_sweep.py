"""Compare preserved/rebased builds at context limits and near-limit prompt lengths.

Only one model server runs at a time. Results and system samples are retained
per configuration, including failures. Cold OS caches are not forced.
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import psutil
import requests

ROOT = Path(__file__).resolve().parents[3]
PYTHON = ROOT / '.venv/bin/python'
BASE = Path('/home/calvin/models/strata-orin-validation')


def save(path, obj):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2) + '\n')
    temporary.replace(path)


def sample(proc, output, stop):
    low = 0
    with output.open('w') as f:
        while not stop.is_set():
            memory = psutil.virtual_memory()
            rss = 0
            try:
                server = psutil.Process(proc.pid)
                for child in [server] + server.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
            except psutil.NoSuchProcess:
                pass
            f.write(json.dumps(dict(wall_time=time.time(), available_bytes=memory.available,
                                   system_used_bytes=memory.total-memory.available,
                                   process_tree_rss_bytes=rss, swap_used_bytes=psutil.swap_memory().used,
                                   disk=psutil.disk_io_counters()._asdict())) + '\n')
            f.flush()
            low = low + 1 if memory.available < 2 * 2**30 else 0
            if low >= 5 and proc.poll() is None:
                print('emergency stop: less than 2 GiB available for five samples', flush=True)
                os.killpg(proc.pid, signal.SIGTERM)
            stop.wait(1)


def request(url, model, repeat, round, generate, task, path):
    text = f'Experiment {round}, size {repeat}. Context: ' + 'red green blue yellow ' * repeat
    text += ('\nWrite at least 200 words explaining how a computer processes information. Start with the processor.'
             if task == 'throughput' else '\nAfter reading, reply with exactly OK.')
    body = dict(model=model, messages=[dict(role='user', content=text)], temperature=0,
                max_tokens=generate, chat_template_kwargs=dict(enable_thinking=False))
    started = time.time()
    r = requests.post(url + '/v1/chat/completions', json=body, timeout=21600)
    obj = dict(start_time=started, end_time=time.time(), pattern_repeats=repeat,
               round=round, task=task, output_cap=generate, http_status=r.status_code,
               response=r.json())
    save(path, obj)  # preserve failed/empty responses too
    r.raise_for_status()
    assert obj['response']['usage']['completion_tokens'] > 0, obj
    if task == 'recall':
        assert obj['response']['choices'][0]['message']['content'].strip() == 'OK', obj
    return obj


def run_cell(a, variant, context, near):
    label, source, exe = variant
    directory = Path(a.out) / label / str(context)
    directory.mkdir(parents=True, exist_ok=True)
    result = dict(label=label, context_limit=context, source_root=str(source), exe=str(exe),
                  source_commit=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                  binary_sha256=hashlib.file_digest(exe.open('rb'), 'sha256').hexdigest(), requests=[], status='starting')
    result_path = directory / 'result.json'
    save(result_path, result)
    cfg = json.loads((BASE / 'strata-orin-validation-mtp.json').read_text())
    cfg.update(exe=str(exe), cwd=str(source), model_name=f'Orin-{label}-ctx{context}', log=str(directory/'engine.txt'))
    cfg['args'][cfg['args'].index('--max-context')+1] = str(context)
    config = directory / 'config.json'; save(config, cfg)
    stop = threading.Event(); proc = None; thread = None
    url = 'http://127.0.0.1:18081'
    print('starting cell', label, context, flush=True)
    try:
        with (directory/'server.txt').open('w') as log:
            proc = subprocess.Popen([str(PYTHON), '-m', 'serve.server', '--engine', 'strata',
                                     '--config', str(config), '--host', '127.0.0.1', '--port', '18081'],
                                    cwd=source, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            thread = threading.Thread(target=sample, args=(proc,directory/'memory.jsonl',stop), daemon=True); thread.start()
            deadline = time.monotonic()+600
            while time.monotonic()<deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f'server exited before readiness: {proc.returncode}')
                try:
                    h = requests.get(url+'/health',timeout=2).json()
                    if h.get('loaded') and h.get('model')==cfg['model_name']:
                        assert h['max_context']==context
                        result['health']=h;break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('server readiness exceeded ten minutes')
            for round in range(a.rounds):
                for repeat in [32,128,256]:
                    if repeat*4+45+64>context:
                        continue
                    print('matched',label,context,round,repeat,flush=True)
                    row=request(url,cfg['model_name'],repeat,round,64,'throughput',directory/f'matched-{round}-{repeat}.json')
                    result['requests'].append(row);save(result_path,result)
            if near:
                repeat=(context-256)//4
                print('near-limit',label,context,repeat,flush=True)
                row=request(url,cfg['model_name'],repeat,99,64,'throughput',directory/'near-limit.json')
                result['requests'].append(row);save(result_path,result)
                row=request(url,cfg['model_name'],1,100,8,'recall',directory/'recovery.json')
                result['requests'].append(row);save(result_path,result)
            result['status']='passed'
    except Exception as error:
        result['status']='failed';result['error']=repr(error)
        print('cell failed',label,context,repr(error),flush=True)
    finally:
        if proc and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGINT)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        stop.set()
        if thread:thread.join(timeout=5)
        samples=[json.loads(line) for line in (directory/'memory.jsonl').read_text().splitlines()]
        if samples:
            result['memory']=dict(minimum_available_bytes=min(r['available_bytes'] for r in samples),
                                  maximum_system_used_bytes=max(r['system_used_bytes'] for r in samples),
                                  maximum_process_tree_rss_bytes=max(r['process_tree_rss_bytes'] for r in samples),
                                  maximum_swap_used_bytes=max(r['swap_used_bytes'] for r in samples),
                                  host_disk_read_bytes=samples[-1]['disk']['read_bytes']-samples[0]['disk']['read_bytes'])
            result['headroom_preserved']=result['memory']['minimum_available_bytes']>=6*2**30
        save(result_path,result)
    print('finished cell',label,context,result['status'],flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--contexts',type=int,nargs='+',default=[1024,2048,4096,8192,16384,32768,65536,131072,262144])
    p.add_argument('--near-limit',action='store_true');p.add_argument('--current-only',action='store_true')
    a=p.parse_args();a.out=str(Path(a.out).resolve());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    variants=[('baseline',Path('/tmp/strata-orin-baseline-src'),BASE/'baseline-engine-c32492c/strata'),
              ('rebased',ROOT,ROOT/'engine-cuda12/strata')]
    if a.current_only:variants=variants[1:]
    results=[]
    for index,context in enumerate(a.contexts):
        for variant in (variants if index%2==0 else list(reversed(variants))):
            results.append(run_cell(a,variant,context,a.near_limit and variant[0]=='rebased'))
            save(out/'matrix.json',dict(contexts=a.contexts,rounds=a.rounds,near_limit=a.near_limit,
                                      cache_policy='auto; fresh server per cell; OS cache retained',cells=results))


if __name__=='__main__':main()
