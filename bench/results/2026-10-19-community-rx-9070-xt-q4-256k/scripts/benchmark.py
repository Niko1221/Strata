"""Local-only streaming measurements; no third-party inference services."""
import argparse
import datetime
import hashlib
import json
import random
from pathlib import Path
import subprocess
import threading
import time
import urllib.error
import urllib.request

import psutil

BASE = 'http://127.0.0.1:8081'
OUT = Path('results')

def disk_snapshot():
    # Windows physical-disk counters: includes all processes, not just Strata.
    return {name: value._asdict() for name, value in psutil.disk_io_counters(perdisk=True).items()}

def request(path, data=None):
    req = urllib.request.Request(BASE + path, data=None if data is None else json.dumps(data).encode(),
                                 headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=1800)

def monitor(stop, rows):
    psutil.cpu_percent()
    while not stop.wait(1):
        mem = psutil.virtual_memory()
        rows.append({'time': time.time(), 'ram_used_bytes': mem.used,
                     'ram_available_bytes': mem.available, 'cpu_percent': psutil.cpu_percent(),
                     'swap_used_bytes': psutil.swap_memory().used,
                     'disk_counters': disk_snapshot()})

def gpu_snapshot():
    # Native WDDM counters. Engine loads are reported separately, never summed as GPU percent.
    code = ('$r=@{memory=@(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory | '
            'Select-Object Name,DedicatedUsage,SharedUsage); engines=@(Get-CimInstance '
            'Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine | Where-Object UtilizationPercentage -gt 0 | '
            'Select-Object Name,UtilizationPercentage)}; $r | ConvertTo-Json -Depth 5 -Compress')
    p = subprocess.run(['powershell.exe','-NoProfile','-Command',code], capture_output=True, text=True, timeout=20)
    return {'returncode': p.returncode, 'output': p.stdout, 'error': p.stderr}

def gpu_monitor(stop, rows):
    while not stop.is_set():
        try:
            rows.append({'time':time.time(), **gpu_snapshot()})
        except Exception as exc:
            rows.append({'time':time.time(),'error':repr(exc)})
        stop.wait(2)

def run(prompt, label, max_tokens=192, expected=None):
    data = {'model':'strata','messages':[{'role':'user','content':prompt}], 'temperature':0,
            'seed':42,'max_tokens':max_tokens,'reasoning_effort':'none','stream':True,
            'stream_options':{'include_usage':True}}
    rows, events, gpu_rows = [], [], []
    stop = threading.Event()
    thread = threading.Thread(target=monitor,args=(stop,rows),daemon=True)
    gpu_thread = threading.Thread(target=gpu_monitor,args=(stop,gpu_rows),daemon=True)
    gpu_before = gpu_snapshot()
    disk_before = disk_snapshot()
    disk_start = time.perf_counter()
    thread.start()
    gpu_thread.start()
    start = time.perf_counter()
    first = None
    content = ''
    result = {'label':label,'request':data,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
              'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        with request('/v1/chat/completions',data) as response:
            for raw in response:
                if not raw.startswith(b'data: '): continue
                payload=raw[6:].strip()
                if payload==b'[DONE]': break
                event=json.loads(payload)
                events.append(event)
                for choice in event.get('choices',[]):
                    delta=choice.get('delta',{})
                    text=delta.get('content') or delta.get('reasoning_content') or ''
                    if text and first is None: first=time.perf_counter()-start
                    content += text
        result['ok']=True
    except Exception as exc:
        result.update(ok=False,error=repr(exc))
    finally:
        result.update(wall_seconds=time.perf_counter()-start,ttft_seconds=first,response=content,events=events)
        stop.set(); thread.join(timeout=3); gpu_thread.join(timeout=25)
    result.update(samples=rows,gpu_samples=gpu_rows,gpu_before=gpu_before,gpu_after=gpu_snapshot(),
                  disk_before=disk_before,disk_after=disk_snapshot())
    result['disk_interval_seconds'] = time.perf_counter()-disk_start
    if expected is not None:
        result['recall_check']={'expected':expected,'all_present':all(value in content for value in expected)}
    for event in events:
        for key in ['usage','timings']:
            if event.get(key): result[key]=event[key]
    OUT.mkdir(exist_ok=True)
    path=OUT/(label+'.json')
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result.get(k) for k in ['label','ok','wall_seconds','ttft_seconds','usage','timings','error']}),flush=True)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--label',default='baseline')
    parser.add_argument('--runs',type=int,default=3)
    parser.add_argument('--context',type=int,default=0)
    parser.add_argument('--probe',choices=['repeated','varied'],default='repeated')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    with request('/health') as response:
        (OUT/(args.label+'-health.json')).write_bytes(response.read())
    for n in range(args.runs):
        expected=None
        prompt=(f'Benchmark run {n}. Write a TypeScript function that retries an async operation with '
                'exponential backoff, preserving the final error. Include a concise usage example.')
        if args.context:
            # Count the actual rendered prompt with the server before inference; reserve output space.
            target=args.context-512
            line='export function item(index: number): number { return index * 3 + 17; }\n'
            count=max(1,target//22)
            for _ in range(4):
                if args.probe=='varied':
                    rng=random.Random(703+n)
                    expected=[f'ALPHA{rng.randrange(100000,999999)}',f'BETA{rng.randrange(100000,999999)}',f'OMEGA{rng.randrange(100000,999999)}']
                    parts=[f'Unique run {args.label}-{n}. Inspect this synthetic TypeScript source bundle.\n',f'export const firstBuild = "{expected[0]}";\n']
                    for i in range(count):
                        if i==count//2:parts.append(f'export const middleBuild = "{expected[1]}";\n')
                        parts.append(f'// module-{i}.ts\nexport function item{i}(index: number): number {{ return index * {rng.randrange(1,997)} + {rng.randrange(1,9999)}; }}\n')
                    parts.append(f'export const lastBuild = "{expected[2]}";\nReturn only the three string values of firstBuild, middleBuild, lastBuild, in that order.')
                    prompt=''.join(parts)
                else:
                    expected=['ORBIT739']
                    prompt=f'Unique run {args.label}-{n}. Remember secret value ORBIT739 at the start.\n'+line*count+'\nWhat secret value was at the start? Answer with that value only.'
                with request('/v1/messages/count_tokens',{'model':'strata','max_tokens':192,'messages':[{'role':'user','content':prompt}]}) as response:
                    actual=json.load(response)['input_tokens']
                if abs(actual-target)<64: break
                count=max(1,int(count*target/actual))
            print(json.dumps({'target':target,'rendered_tokens':actual}),flush=True)
        result=run(prompt,f'{args.label}-{n}',expected=expected)
        if not result['ok']: break

if __name__=='__main__': main()
