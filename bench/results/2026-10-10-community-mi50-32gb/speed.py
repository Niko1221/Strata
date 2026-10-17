# -*- coding: utf-8 -*-
"""W770 speed bench, per docs/COMMUNITY_BENCHMARKS.md.

N prompt lengths x RUNS runs, max_tokens 256, temperature 0, one request at a time.

Every run puts a DIFFERENT salt at the head of the prompt, so the first token differs and nothing can
be reused: each run is genuinely new work (the spec asks new-vs-reused to be separated, and we report
both -- the engine prints them).

The numbers reported are the ENGINE's own, taken from its
    strata serve: prompt <N> tokens = <R> reused + <K> read in <ms> ms (<tok/s>), <G> generated in <ms> ms
                  (<tok/s>), drafts accepted <A> of <B>, <C> checkpoints
    strata serve: decode expert cache hit rate: <X>% (<hits> hits / <lookups> lookups)
lines -- never "generated / wall clock", which the spec explicitly forbids for decode.

Usage: python3 _w770_speed.py <base-url> <arm> <out.json> <lens csv> [runs] [engine log] [max_tokens]
"""
import hashlib, json, pathlib, random, sys, time, urllib.request

URL = sys.argv[1]
ARM = sys.argv[2]
OUT = sys.argv[3]
LENS = sys.argv[4]
RUNS = int(sys.argv[5]) if len(sys.argv) > 5 else 3
LOG = sys.argv[6] if len(sys.argv) > 6 else None
MAXTOK = int(sys.argv[7]) if len(sys.argv) > 7 else 256
CPT = 2.4                      # chars per token measured on this box (1.89 .. 3.00 by block)
ROOT = pathlib.Path('/src')

_cache = {}


def filler(n_chars):
    """A deterministic text blob built from the repository (same idea as tools/needle_bench.py)."""
    if 'txt' not in _cache:
        files = sorted((ROOT / 'docs').glob('*.md'))
        files += sorted((ROOT / 'src').rglob('*.cpp'))[:40]
        files += sorted((ROOT / 'include').rglob('*.hpp'))[:30]
        parts, total, need = [], 0, 900000
        while total < need and files:
            for f in files:
                try:
                    t = f.read_text(encoding='utf-8', errors='replace')
                except OSError:
                    continue
                parts.append(t)
                total += len(t)
                if total >= need:
                    break
        _cache['txt'] = ''.join(parts)
    return _cache['txt'][:n_chars]


def log_n(p):
    if not p:
        return 0, []
    try:
        lines = open(p, encoding='utf-8', errors='replace').read().splitlines()
        return len(lines), lines
    except OSError:
        return 0, []


def post(prompt, maxtok):
    body = {'model': 'strata', 'max_tokens': maxtok, 'temperature': 0,
            'chat_template_kwargs': {'enable_thinking': False},
            'messages': [{'role': 'user', 'content': prompt}]}
    req = urllib.request.Request(URL.rstrip('/') + '/v1/chat/completions',
                                 data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        o = json.loads(r.read())
    return o, time.time() - t0


res = []
for L in LENS.split(','):
    ntok = int(L)
    for i in range(RUNS):
        salt = 'bench %s len %d run %d nonce %08x' % (ARM, ntok, i, random.getrandbits(32))
        prompt = salt + '\n' + filler(int(ntok * CPT))
        n0, _ = log_n(LOG)
        try:
            o, wall = post(prompt, MAXTOK)
        except Exception as e:                                  # noqa: BLE001
            print('  %-5s len=%-7d run=%d FAILED %s' % (ARM, ntok, i, e), flush=True)
            res.append({'arm': ARM, 'len_target': ntok, 'run': i, 'salt': salt, 'error': str(e)})
            continue
        u = o.get('usage') or {}
        txt = (o['choices'][0]['message'].get('content') or '')
        time.sleep(1.5)                     # let the engine finish printing its per-request lines
        _, lines = log_n(LOG)
        new = lines[n0:]
        pl = next((l for l in new if 'strata serve: prompt ' in l), '')
        hl = next((l for l in new if 'hit rate' in l), '')
        res.append({'arm': ARM, 'len_target': ntok, 'run': i, 'salt': salt,
                    'prompt_tokens': u.get('prompt_tokens'),
                    'completion_tokens': u.get('completion_tokens'),
                    'wall_s': round(wall, 2),
                    'text_sha16': hashlib.sha256(txt.encode()).hexdigest()[:16],
                    'engine_prompt_line': pl, 'engine_hit_line': hl})
        print('  %-5s len=%-7d run=%d prompt=%-7s gen=%-4s wall=%.1fs | %s'
              % (ARM, ntok, i, u.get('prompt_tokens'), u.get('completion_tokens'), wall,
                 (pl.split('strata serve: ')[-1] if pl else '(no engine line)')[:118]), flush=True)

json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('SPEED_DONE %s -> %s (%d records)' % (ARM, OUT, len(res)))
