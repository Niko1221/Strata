#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补充矩阵：长输出 decode/并发。上下文 1K、64K；并发 1/4/8/16。"""
import json, os, random, string, sys, threading, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '.')
import importlib.util as _il, pathlib
_spec = _il.spec_from_file_location("bench_strata", str(pathlib.Path(__file__).with_name("benchmark-matrix.py")))
B = _il.module_from_spec(_spec); _spec.loader.exec_module(B)

def make_prompt_long(rng, target_tokens, ratio):
    n_words = int(target_tokens / ratio)
    parts = []
    total = 0
    while total < n_words:
        chunk = B.unique_block(rng, 800)
        parts.append(chunk)
        total += 800
    tail = rng.randrange(10**12)
    header = (f"以下是编号 {tail} 的随机语料（词表中数字均为随机编号，无实际含义）。"
              f"请通读后写一篇 700 字左右的中文总结，介绍语料的构成与统计特征，"
              f"并列举其中的高频词。必须写满 600 字以上。\n\n")
    return header + '\n'.join(parts)

def one(prompt, out, max_tokens=1600):
    body = {"model": B.MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(B.BASE, data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json',
                 'Authorization': f'Bearer {B.KEY}'})
    t0 = time.perf_counter(); t1 = None; tlast = None; usage = {}; err = None
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            for line in r:
                line = line.decode('utf-8', 'replace').strip()
                if not line.startswith('data:'):
                    continue
                p = line[5:].strip()
                if p == '[DONE]':
                    break
                d = json.loads(p)
                if d.get('usage'):
                    usage = d['usage']
                for ch in d.get('choices', []):
                    dl = ch.get('delta', {})
                    if dl.get('content') or dl.get('reasoning_content'):
                        now = time.perf_counter()
                        if t1 is None: t1 = now
                        tlast = now
    except Exception as e:
        err = type(e).__name__
    out.append(dict(start=t0, first=t1, last=tlast, end=time.perf_counter(),
                    usage=usage, error=err))

def run_cell(ctx, conc, ratio, log):
    rng = random.Random(os.getpid() * 31 + ctx * 7 + conc + int(time.time()))
    prompts = [make_prompt_long(rng, ctx, ratio) for _ in range(conc)]
    outs = []
    barrier = threading.Barrier(conc)
    def w(p):
        barrier.wait()
        one(p, outs)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        list(ex.map(w, prompts))
    wall = time.perf_counter() - t0
    ok = [o for o in outs if not o['error'] and o['first']]
    rows = [dict(error=o['error']) for o in outs if o['error']]
    for o in ok:
        u = o['usage']
        pt = u.get('prompt_tokens') or 0
        ct = u.get('completion_tokens') or 0
        tt = o['first'] - o['start']
        dec = o['last'] - o['first']
        rows.append(dict(prompt_tokens=pt, completion_tokens=ct, ttft=tt,
                         decode_s=dec, prefill_tps=pt / tt if tt else None,
                         decode_tps=ct / dec if dec else None,
                         total=o['end'] - o['start']))
    cell = dict(ctx=ctx, conc=conc, ok=len(ok), fail=len(outs) - len(ok),
                wall=wall, requests=sorted(rows, key=lambda r: r.get('ttft') or 0),
                wall_throughput_tps=(sum(r.get('completion_tokens', 0) for r in rows) / wall) if rows else 0)
    for key, agg in [('ttft', 'ttft'), ('prefill_tps', 'prefill'), ('decode_tps', 'decode')]:
        vals = sorted(r[key] for r in rows if r.get(key))
        if vals:
            cell[f'{agg}_med'] = vals[len(vals)//2]
            cell[f'{agg}_min'] = vals[0]
            cell[f'{agg}_max'] = vals[-1]
    completions = sorted(r['completion_tokens'] for r in rows if 'completion_tokens' in r)
    if completions:
        cell['completion_med'] = completions[len(completions)//2]
    log.write(json.dumps(cell, ensure_ascii=False) + '\n')
    log.flush()
    rec = {k: v for k, v in cell.items() if k != 'requests'}
    print(json.dumps(rec, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    ratio = B.calibrate()
    with open('bench2_results.jsonl', 'a') as log:
        for ctx in (1000, 64000):
            for conc in (1, 4, 8, 16):
                print(f'=== decode cell ctx={ctx} conc={conc} ===', flush=True)
                run_cell(ctx, conc, ratio, log)
