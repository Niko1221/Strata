#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strata 端点压测：TTFT / prefill / decode × 上下文 × 并发（冷缓存）。"""
import json, os, random, string, sys, threading, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


CFG = None
BASE = os.environ.get('STRATA_BASE', 'http://127.0.0.1:8080/v1/chat/completions')
MODEL = os.environ.get('STRATA_MODEL', 'qwen3.8-flash-next-iq3_s')

def api_key():
    return os.environ.get('STRATA_API_KEY', '')

KEY = api_key()

# ---------- 唯一冷 prompt 生成 ----------
WORDS_CN = '记忆 系统 工程 检索 写入 更新 评估 架构 缓存 推理 任务 状态 上下文 向量 索引 策略 权限 版本 审计 生命周期 巩固 遗忘 抽取 组织 调度 一致性 冲突 优先级 召回 排序 过滤 压缩 归档 失效 删除 溯源 证据 范围 作用域 偏好 约束 轨迹 快照 摘要 嵌入 分块 粒度 阈值 延迟 吞吐 并发 队列 槽位 令牌 窗口 位置 编码 解码 采样 温度'.split()

def unique_block(rng, n_words):
    return ' '.join(rng.choice(WORDS_CN) + str(rng.randrange(10**6)) for _ in range(n_words))

def make_prompt(rng, target_tokens, ratio):
    """ratio = 实测 token/词；生成从第一个词就唯一的长 prompt。"""
    n_words = int(target_tokens / ratio)
    parts = []
    total = 0
    while total < n_words:
        chunk = unique_block(rng, 800)
        parts.append(chunk)
        total += 800
    header = (f"以下是编号 {rng.randrange(10**12)} 的随机语料，"
              f"种子 {''.join(rng.choices(string.ascii_letters, k=24))}。"
              f"请阅读后只回答：DONE\n\n")
    return header + '\n'.join(parts)

# ---------- 单请求 ----------
def one_request(prompt, max_tokens, t_barrier=None, out=None):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json',
                 'Authorization': f'Bearer {KEY}'})
    t0 = time.perf_counter()
    ttft = None          # 第一个数据块
    t_first_tok = None   # 第一个 token（content 或 reasoning_content）
    t_last = None
    tok_deltas = 0
    usage = {}
    err = None
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            for line in r:
                line = line.decode('utf-8', 'replace').strip()
                if not line.startswith('data:'):
                    continue
                payload = line[5:].strip()
                if payload == '[DONE]':
                    break
                d = json.loads(payload)
                if d.get('usage'):
                    usage = d['usage']
                for ch in d.get('choices', []):
                    dl = ch.get('delta', {})
                    piece = dl.get('content') or dl.get('reasoning_content')
                    if piece:
                        now = time.perf_counter()
                        if t_first_tok is None:
                            t_first_tok = now
                        t_last = now
                        tok_deltas += 1
                if ttft is None and d.get('choices'):
                    ttft = time.perf_counter()
    except Exception as e:
        err = type(e).__name__
    t_end = time.perf_counter()
    res = dict(start=t0, ttft_chunk=ttft, first_tok=t_first_tok,
               last_tok=t_last, end=t_end, tok_deltas=tok_deltas,
               usage=usage, error=err)
    if out is not None:
        out.append(res)
    return res

# ---------- 一个矩阵格 ----------
def run_cell(ctx_tokens, conc, max_tokens, ratio, log):
    rng = random.Random(os.getpid() * 1000 + ctx_tokens * 17 + conc)
    prompts = [make_prompt(rng, ctx_tokens, ratio) for _ in range(conc)]
    outs = []
    barrier = threading.Barrier(conc)
    def worker(p):
        barrier.wait()
        return one_request(p, max_tokens, out=outs)
    t_cell0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        list(ex.map(worker, prompts))
    t_cell1 = time.perf_counter()
    ok = [o for o in outs if not o['error']]
    fail = len(outs) - len(ok)
    start = min(o['start'] for o in ok) if ok else 0
    rows = []
    for o in ok:
        pt = o['usage'].get('prompt_tokens') or 0
        ct = o['usage'].get('completion_tokens') or o['tok_deltas'] or 0
        ttft = (o['first_tok'] - o['start']) if o['first_tok'] else None
        dec = (o['last_tok'] - o['first_tok']) if (o['last_tok'] and o['first_tok']) else None
        rows.append(dict(prompt_tokens=pt, completion_tokens=ct,
                         ttft=ttft, decode_s=dec,
                         prefill_tps=(pt / ttft) if ttft else None,
                         decode_tps=(ct / dec) if dec else None,
                         total=o['end'] - o['start']))
    cell = dict(ctx=ctx_tokens, conc=conc, max_tokens=max_tokens,
                ok=len(ok), fail=fail, wall=t_cell1 - t_cell0,
                wall_throughput_tps=(sum(r['completion_tokens'] for r in rows) / (t_cell1 - t_cell0)) if ok else 0,
                requests=sorted(rows, key=lambda r: r['ttft'] or 0))
    rec = {k: v for k, v in cell.items() if k != 'requests'}
    ttfts = [r['ttft'] for r in rows if r['ttft']]
    if ttfts:
        rec.update(ttft_min=min(ttfts), ttft_med=sorted(ttfts)[len(ttfts)//2],
                   ttft_max=max(ttfts))
    pf = [r['prefill_tps'] for r in rows if r['prefill_tps']]
    if pf:
        rec.update(prefill_med_tps=sorted(pf)[len(pf)//2])
    dps = [r['decode_tps'] for r in rows if r['decode_tps']]
    if dps:
        rec.update(decode_med_tps=sorted(dps)[len(dps)//2])
    log.write(json.dumps(cell, ensure_ascii=False) + '\n')
    log.flush()
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    return cell

def calibrate():
    """测 prompt 生成器的 token/词 比率（两轮迭代收敛）。"""
    ratio = 8.0
    for _ in range(2):
        rng = random.Random(42)
        p = make_prompt(rng, 4000, ratio)
        r = one_request(p, 8)
        pt = r['usage'].get('prompt_tokens', 0)
        words = p.count(' ') + 1
        new = pt / words
        print(f"校准: 词块数≈{words} → prompt_tokens={pt} → tokens/词={new:.3f}", flush=True)
        if abs(new - ratio) < 0.3:
            ratio = new
            break
        ratio = new
    return ratio

if __name__ == '__main__':
    ctxs = [64000, 128000, 250000]
    concs = [1, 4, 8, 16]
    max_tokens = 1024
    if len(sys.argv) > 1:  # 快速冒烟：只跑 64K×1
        ctxs, concs = [64000], [1]
    ratio = calibrate()
    with open('bench_results.jsonl', 'a') as log:
        for c in ctxs:
            for k in concs:
                print(f'=== cell ctx={c} conc={k} ===', flush=True)
                run_cell(c, k, max_tokens, ratio, log)
