#!/usr/bin/env python3
"""Exercise the real HTTP server, including solo promotion and return to solo.

Compare raw engine token IDs and streamed content/reasoning bytes with solo
references; check that pipeline groups do not claim a retained slot cache.
"""

def main(argv=None):
    import argparse
    import json
    import os
    from pathlib import Path
    import shlex
    import sys
    import threading
    import urllib.request

    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--exe')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--max-new', type=int, default=256)
    ap.add_argument('--later-max-new', type=int, default=64)
    ap.add_argument('--stagger-after', default='8,16,24')
    ap.add_argument('--long-tokens', type=int, default=0)
    ap.add_argument('--keys', default='')
    ap.add_argument('--mt-min', default='1')
    ap.add_argument('--expect-return-solo', choices=('auto','yes','no'), default='auto')
    ap.add_argument('--extra', default='')
    a = ap.parse_args(argv)
    if not 2 <= a.n <= min(a.batch, 8) or not 0 < a.later_max_new < a.max_new:
        ap.error('require at least two prompts and 0 < --later-max-new < --max-new')
    stagger_after = [int(x) for x in a.stagger_after.split(',')]
    if len(stagger_after) != a.n-1 or any(x <= 0 or x >= a.max_new for x in stagger_after):
        ap.error('--stagger-after must give one valid anchor token count per later prompt')
    repo = Path(a.repo).resolve()
    sys.path[:0] = [str(repo), str(repo / 'tools')]
    from batch_test import QUESTIONS, tokenizer
    from serve.frontend import ChatTemplate
    from serve.server import Service, StrataEngine, engine_args, serve

    cfg = json.loads(Path(a.config).read_text())
    if a.exe:
        cfg['exe'] = a.exe
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = ':'.join(cfg['lib_dirs'] + [env.get('LD_LIBRARY_PATH', '')])
    if a.mt_min:
        env['STRATA_IQ_MT_MIN'] = a.mt_min
    else:
        env.pop('STRATA_IQ_MT_MIN', None)
    engine = StrataEngine(cfg['exe'], engine_args(cfg) + ['--batch',str(a.batch)] + shlex.split(a.extra), cwd=cfg['cwd'],
                          log=a.out + '.engine.log', env=env)
    commands = []
    token_records = {'solo':{},'staggered':{}}
    phase = 'solo'
    generate = engine.generate
    def record_generate(ids, max_new, sampling, cancel, *args, **kwargs):
        current = phase
        token_records[current].setdefault(str(max_new), [])
        record_tokens = {'prompt':list(ids),'tokens':[]}
        token_records[current][str(max_new)].append(record_tokens)
        for t in generate(ids,max_new,sampling,cancel,*args,**kwargs):
            if t is not None:
                record_tokens['tokens'].append(int(t))
            yield t
    engine.generate = record_generate
    send = engine._send
    def record(line):
        f = line.split()
        commands.append({'command': f[0], 'slot': int(f[1]) if f[0].startswith(('BGEN','BSTOP')) else None,
                         'prompt_tokens':len(f[-1].split(',')) if f[0].startswith(('GEN','BGEN')) else None,
                         'anchor_emitted':len(token_records['staggered'].get(str(a.max_new),[{'tokens':[]}])[0]['tokens'])})
        return send(line)
    engine._send = record
    tpl = Path(cfg['tokenizer']) / 'chat_template.jinja'
    svc = Service(engine, tokenizer(cfg['tokenizer']), ChatTemplate(tpl if tpl.exists() else repo/'serve/chat_template.jinja'),
                  model_name=cfg['model_name'])
    httpd = serve(svc, port=0)
    url = f'http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions'
    limits = [a.max_new] + [a.later_max_new] * (a.n-1)
    questions = QUESTIONS[:a.n]
    if a.long_tokens:
        filler = 'The committee reviewed the budget, schedule and risks and asked the staff to report back. '
        questions[-1] = filler * (a.long_tokens // max(1,len(tokenizer(cfg['tokenizer']).encode(filler))) + 1) + questions[-1]
    sampling = {'temperature':0}
    for entry in shlex.split(a.keys):
        k,v = entry.split('=',1)
        sampling[k] = json.loads(v)
    parts = [{"content": '', "reasoning_content": ''} for _ in limits]
    finished = [False] * len(limits)
    errors = []
    cv = threading.Condition()
    chunks = 0
    workers = []
    references = []
    events = []

    def request(i, stream):
        body = {'model':cfg['model_name'], **sampling, 'max_tokens':limits[i], 'stream':stream,
                'messages':[{'role':'user','content':questions[i]}],
                'chat_template_kwargs':{'enable_thinking':False}}
        return urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})

    def worker(i):
        nonlocal chunks
        try:
            with urllib.request.urlopen(request(i, True), timeout=120) as r:
                for raw in r:
                    if not raw.startswith(b'data: '):
                        continue
                    if raw.strip() == b'data: [DONE]':
                        break
                    packet = json.loads(raw[6:])
                    if 'error' in packet:
                        raise RuntimeError(packet['error'])
                    delta = packet['choices'][0]['delta']
                    with cv:
                        for k in parts[i]:
                            parts[i][k] += delta.get(k) or ''
                        if i == 0 and any(delta.get(k) for k in parts[i]):
                            chunks += 1
                        cv.notify_all()
        except Exception as exc:
            with cv:
                errors.append({'request':i,'error':repr(exc)})
        finally:
            with cv:
                finished[i] = True
                cv.notify_all()

    try:
        for i in range(len(limits)):
            with urllib.request.urlopen(request(i, False), timeout=120) as r:
                msg = json.load(r)['choices'][0]['message']
            references.append({k:msg.get(k) or '' for k in parts[i]})
        commands.clear()
        phase = 'staggered'
        for i in range(len(limits)):
            if i:
                with cv:
                    ready = cv.wait_for(lambda: len(token_records['staggered'].get(str(a.max_new),[{'tokens':[]}])[0]['tokens']) >= stagger_after[i-1] or finished[0] or errors, timeout=60)
                    if not ready or finished[0] or errors:
                        raise RuntimeError('anchor finished or failed before a staggered HTTP request')
                    events.append({'request':i,'anchor_chunks':chunks,'anchor_tokens':len(token_records['staggered'][str(a.max_new)][0]['tokens']),'anchor_active':not finished[0]})
            t = threading.Thread(target=worker, args=(i,))
            workers.append(t)
            t.start()
        for t in workers:
            t.join(timeout=180)
        if any(t.is_alive() for t in workers):
            raise RuntimeError('HTTP worker did not finish within the deadline')
        same = [x == y for x,y in zip(parts,references)]
        opcodes = [x['command'] for x in commands]
        first_batch = next((i for i,x in enumerate(opcodes) if x.startswith('BGEN')), len(opcodes))
        promoted = 'STOP' in opcodes[:first_batch] and first_batch < len(opcodes)
        returned = 'GEN' in opcodes[first_batch:] and 'BSTOP' in opcodes[first_batch:]
        token_matches = []
        for limit,refs in token_records['solo'].items():
            for ref in refs:
                other = next((r for r in token_records['staggered'].get(limit,[]) if r['prompt']==ref['prompt']),None)
                token_matches.append(other is not None and ref['tokens']==other['tokens'])
        expected_return = bool(engine.info.get('slot_cache')) and os.environ.get('STRATA_PARALLEL_SOLO') != '0'
        if a.expect_return_solo != 'auto':
            expected_return = a.expect_return_solo == 'yes'
        pipeline_groups = max(engine.slot_group, default=0) + 1
        pipelined = pipeline_groups > 1 and len(cfg.get('gpu') or []) > 1
        cache_capability_consistent = not pipelined or not engine.info.get('slot_cache')
        result = {'matched':same,'token_matches':token_matches,'slot_cache':engine.info.get('slot_cache'),
                  'pipeline_groups':pipeline_groups,'cache_capability_consistent':cache_capability_consistent,
                  'expected_return_to_solo':expected_return,'references':references,'staggered':parts,'admissions':events,
                  'commands':commands,'promoted':promoted,'returned_to_solo':returned,'errors':errors,
                  'mt_min':a.mt_min,'token_records':token_records}
        Path(a.out if a.out.endswith('.json') else a.out+'.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('references','staggered','token_records')},indent=2),flush=True)
        if (errors or not all(same) or len(token_matches)!=a.n or not all(token_matches) or not promoted or
                returned != expected_return or not cache_capability_consistent):
            return 2
    finally:
        httpd.shutdown()
        httpd.server_close()
        engine.close()


if __name__ == "__main__":
    import sys
    sys.exit(main())
