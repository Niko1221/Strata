"""DeepSeek-V4.1 engine: logit_bias over real HTTP chat requests, the emitted token IDs captured at the engine boundary.

The baseline engine (without the change) and the candidate start one after another from the same run config. The
checks: requests without a bias come out the same (greedy and sampled); a ban in both forms; a positive bias; 400s for
a bad id or value; a cleared and an empty bias; a 103,215-token ban; streaming and sampled bans; a ban of a token the
unbiased answer writes late (greedy and sampled, 48 tokens, DSpark drafting: verification and the drafter's picks);
banning every token with Han or kana text; the same with the DSpark drafter off; and two conversations at once
(--parallel 2), overlapping in time, each with its own list of tokens its answer would write late. Every request turns
thinking off.
"""
import argparse, json, sys, threading, time, urllib.error, urllib.request
from pathlib import Path
ap = argparse.ArgumentParser(description=__doc__)
for name in ['source', 'config', 'baseline', 'candidate', 'output']:
    ap.add_argument('--' + name, type=Path, required=True)
a = ap.parse_args()
source = a.source.resolve()
sys.path[:0] = [str(source), str(source / 'tools')]
from serve.server import StrataEngine, Service, engine_args, child_env, serve
from serve.frontend import ChatTemplate
from serve import dsml
import strata_tokenizer as ST

cfg = json.loads(a.config.read_text())
assert cfg.get('arch') == 'deepseek41', 'expects a DeepSeek-V4.1 run config ("arch": "deepseek41")'
cfg = {k: v for k, v in cfg.items()
       if k in ['args', 'user_args', 'tokenizer', 'gpu', 'gpus_asked', 'lib_dirs', 'env', 'arch', 'layer_split', 'drafting']}
tp = Path(cfg['tokenizer'])
vocab = json.loads((tp / 'vocab.json').read_text(encoding='utf-8'))
tokens = [None] * len(vocab)
for t, i in vocab.items():
    tokens[i] = t
tcfg = json.loads((tp / 'tokenizer.json').read_text(encoding='utf-8'))
tok = ST.Tokenizer(tokens, (tp / 'merges.txt').read_text(encoding='utf-8').split('\n'),
                   json.loads((tp / 'token_type.json').read_text()), tcfg.get('pre') or 'qwen35', tcfg.get('special_ids') or None)
template = ChatTemplate(dsml.pick_template(tp / 'chat_template.jinja')[0])
HAN = [(0x2E80, 0x2FDF), (0x3005, 0x3007), (0x3021, 0x3029), (0x3040, 0x30FF), (0x31C0, 0x31FF), (0x3400, 0x4DBF),
       (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0xFF66, 0xFF9F), (0x20000, 0x323AF)]
def han(text):
    return any(lo <= ord(c) <= hi for c in text for lo, hi in HAN)
han_ids = [i for i in range(len(tokens)) if han(tok.decode([i]))]
print('tokens with Han or kana text:', len(han_ids), flush=True)
out = a.output
out.mkdir(parents=True, exist_ok=True)
rows = []

def bias_key(value):
    return json.dumps(value, sort_keys=True)

class AuditEngine(StrataEngine):
    """Keeps each call's emitted IDs under its request's logit_bias (two requests at once carry different lists)."""
    def generate(self, ids, max_new, sampling, *args, **kwargs):
        emitted = []
        self.calls.append((bias_key((sampling or {}).get('logit_bias')), emitted))
        for token in super().generate(ids, max_new, sampling, *args, **kwargs):
            if token is not None:
                emitted.append(token)
            yield token

def start(mode, exe, args, parallel):
    # (each engine with the libraries built next to it: the baseline and the candidate are two build directories)
    local = dict(cfg, args=args, exe=str(exe.resolve()), cwd=str(source), log=str(out / (mode + '.log')), parallel=parallel,
                 lib_dirs=[str(exe.resolve().parent)])
    engine = AuditEngine(local['exe'], engine_args(local), cwd=local['cwd'], log=local['log'], env=child_env(local))
    engine.calls = []
    service = Service(engine, tok, template, model_name='bias-validation')
    service.model_arch = 'deepseek41'
    service.set_family('deepseek41')
    service.requested_slots = parallel
    return engine, service, serve(service, port=0)

def call(mode, engine, server, label, prompt='Write the Chinese word for hello. Only the word.', expect=200, **kw):
    req = {'model': 'bias-validation', 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 8,
           'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}, **kw}
    data = json.dumps(req).encode()
    started, wall = time.perf_counter(), time.time()
    http = urllib.request.Request(f'http://127.0.0.1:{server.server_address[1]}/v1/chat/completions', data=data,
                                  headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(http, timeout=600) as response:
            status = response.status
            raw = response.read().decode()
    except urllib.error.HTTPError as e:
        with e:
            status = e.code
            raw = e.read().decode()
    ids = []
    if status == 200:
        mine = [e for k, e in engine.calls if k == bias_key(kw.get('logit_bias'))]
        assert mine, (label, 'no engine call carried this request\'s logit_bias')
        ids = list(mine[-1])
    timings = json.loads(raw).get('timings', {}) if status == 200 and not req.get('stream') else {}
    row = {'mode': mode, 'label': label, 'status': status, 'seconds': round(time.perf_counter() - started, 3),
           'start': wall, 'end': time.time(), 'draft_n': timings.get('draft_n'), 'draft_n_accepted': timings.get('draft_n_accepted'),
           'request_bytes': len(data), 'token_ids': ids, 'text': tok.decode(ids) if ids else '', 'response': raw}
    rows.append(row)
    (out / 'rows.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(mode, label, status, ids, repr(row['text']), 'drafts', row['draft_n'], row['draft_n_accepted'], flush=True)
    assert status == expect, (label, status, raw[:300])
    return row

def args_with(flag, value):
    args = list(cfg['args'])
    if flag in args:
        args[args.index(flag) + 1] = value
    else:
        args += [flag, value]
    return args

LONG = dict(prompt='Write three sentences about Python.', max_tokens=48)

def late(ids, after=5):
    """A token the answer writes first at position `after` or later (a ban that the first pick alone cannot show)."""
    return next(t for i, t in enumerate(ids) if i >= after and t not in ids[:i])

first = sampled = None
for mode in ['baseline', 'candidate', 'no-dspark', 'parallel']:
    exe = a.baseline if mode == 'baseline' else a.candidate
    args = args_with('--dspark', 'off') if mode == 'no-dspark' else list(cfg['args'])
    engine, service, server = start(mode, exe, args, 2 if mode == 'parallel' else 1)
    c = lambda *x, **k: call(mode, engine, server, *x, **k)
    try:
        if mode == 'baseline':
            first = c('unchanged-greedy')
            sampled = c('unchanged-sampled', prompt='Write one sentence about Python.', temperature=0.7, top_k=20, seed=42)
            continue
        assert engine.info.get('logit_bias') == 1, 'the candidate does not advertise logit_bias=1'
        banned = first['token_ids'][0]
        if mode == 'candidate':
            assert c('unchanged-greedy')['token_ids'] == first['token_ids'], 'default greedy output changed'
            s = c('unchanged-sampled', prompt='Write one sentence about Python.', temperature=0.7, top_k=20, seed=42)
            assert s['token_ids'] == sampled['token_ids'], 'default sampled output changed'
            one = c('ban-object', logit_bias={str(banned): -100})
            two = c('ban-pairs', logit_bias=[[banned, False]])
            assert banned not in one['token_ids'] and one['token_ids'] == two['token_ids']
            boosted = tok.encode(' banana', parse_special=False)[0]
            assert c('positive-bias', logit_bias={str(boosted): 100}, max_tokens=3)['token_ids'][0] == boosted
            c('invalid-id', expect=400, logit_bias={str(len(tokens)): 1}, stream=True)
            c('invalid-value', expect=400, logit_bias={'0': 101}, stream=True)
            assert c('cleared-bias')['token_ids'] == first['token_ids']
            assert c('empty-bias', logit_bias={})['token_ids'] == first['token_ids']
            large = set(range(103214)) | {banned}
            if len(large) < 103215:
                large.add(103214)
            assert not set(c('103215-bans', logit_bias=[[i, False] for i in sorted(large)])['token_ids']) & large
            assert banned not in c('stream-ban', logit_bias={str(banned): -100}, stream=True)['token_ids']
            assert banned not in c('sampled-ban', logit_bias={str(banned): -100}, temperature=0.8, seed=123)['token_ids']
            h = c('han-ban', logit_bias={str(i): -100 for i in han_ids}, max_tokens=24)
            assert not set(h['token_ids']) & set(han_ids)
            assert c('cleared-after-han')['token_ids'] == first['token_ids']
            long = c('long-unbiased', **LONG)
            assert long['draft_n'], 'DSpark drafted nothing: the verify path was not exercised'
            word = late(long['token_ids'])
            lb = c('long-late-ban', logit_bias={str(word): -100}, **LONG)
            assert word not in lb['token_ids'] and lb['token_ids'] != long['token_ids'] and lb['draft_n']
            ls = c('long-sampled', temperature=0.7, top_k=20, seed=7, **LONG)
            sword = late(ls['token_ids'])
            lsb = c('long-sampled-late-ban', logit_bias={str(sword): -100}, temperature=0.7, top_k=20, seed=7, **LONG)
            assert sword not in lsb['token_ids'] and lsb['draft_n']
        elif mode == 'no-dspark':
            plain = c('unbiased')
            assert banned not in c('ban-object', logit_bias={str(banned): -100})['token_ids']
            h = c('han-ban', logit_bias={str(i): -100 for i in han_ids}, max_tokens=24)
            assert not set(h['token_ids']) & set(han_ids)
            assert c('cleared-bias')['token_ids'] == plain['token_ids']
        else:   # two conversations at once, long enough to overlap, each banned (or not) where its answer is late
            alone = c('unbiased', **LONG)
            zh = dict(prompt='用中文写三句话介绍Python。', max_tokens=48)
            zalone = c('zh-unbiased', **zh)
            word, zword = late(alone['token_ids']), late(zalone['token_ids'])
            results = {}
            def run(label, **kw):
                results[label] = c(label, **kw)
            for pair in ([('ban-a', dict(LONG, logit_bias={str(word): -100})), ('free-b', LONG)],
                         [('zh-ban-a', dict(zh, logit_bias={str(zword): -100})), ('free-b2', LONG)],
                         [('han-a', dict(zh, logit_bias={str(i): -100 for i in han_ids})),
                          ('ban-b', dict(LONG, logit_bias={str(word): -100}))]):
                threads = [threading.Thread(target=run, args=(label,), kwargs=kw) for label, kw in pair]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                x, y = (results[label] for label, _ in pair)
                assert x['start'] < y['end'] and y['start'] < x['end'], (pair[0][0], pair[1][0], 'did not overlap')
            assert word not in results['ban-a']['token_ids'] and results['ban-a']['token_ids'] != alone['token_ids']
            for free in ('free-b', 'free-b2'):   # unbiased beside a biased one (free-b would write ban-a's token late)
                assert results[free]['token_ids'] == alone['token_ids'], (free, 'took the other conversation\'s list')
            assert zword not in results['zh-ban-a']['token_ids']
            assert not set(results['han-a']['token_ids']) & set(han_ids)
            assert word not in results['ban-b']['token_ids']
    finally:
        server.shutdown()
        server.server_close()
        engine.close()

(out / 'complete.json').write_text(json.dumps({'status': 'passed', 'requests': len(rows), 'han_tokens': len(han_ids),
    'baseline': str(a.baseline), 'candidate': str(a.candidate), 'config': str(a.config)}, indent=2, ensure_ascii=False))
print('passed', len(rows), 'requests', flush=True)
