"""G5 native-model qualification. Owns one engine on an idle authorized GPU.

No scripted model output or proposal injection. Run each configuration separately;
compare raw numerical diagnostics independently of the exact fixed-logit tests.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from target_only_probe import sha256, tokenizer
from serve.frontend import ChatTemplate
from serve.grammar import GrammarConstraint
from serve.server import StrataEngine, child_env


def audit(trace):
    cursors, specs, grammars = [], [], []
    for line in trace.splitlines():
        if 'strata trace: CURSOR ' in line:
            c = {k: int(v) for k, v in re.findall(r'(\w+)=(-?\d+)', line)}
            assert c['consumed'] == c['prompt'] + c['produced'] - 1, c
            assert c['selection_position'] == c['consumed'] - 1, c
            cursors.append(c)
        elif 'strata trace: SPEC ' in line:
            s = {k: v for k, v in re.findall(r'(\w+)=([^ ]+)', line)}
            s = {k: v if k == 'source' else float(v) if k.endswith('_ms') else int(v) for k, v in s.items()}
            assert 1 <= s['kept'] <= s['reachable'] <= s['proposed'] <= 8 and s['fallback'] == 0, s
            specs.append(s)
        elif 'strata trace: GRAMMAR ' in line:
            g = {k: int(v) for k, v in re.findall(r'(\w+)=(-?\d+)', line)}
            assert cursors and g['tokens'] == cursors[-1]['produced'], (g, cursors[-1:])
            assert not g['terminal'] or g['accepting']
            grammars.append(g)
    assert len(specs) == len(grammars)
    return dict(cursor_windows=len(cursors), grammar_windows=len(grammars),
                sources={s: sum(r['source'] == s for r in specs) for s in ('target', 'mtp', 'suffix')},
                blocked_drafts=sum(r['blocked_draft'] >= 0 for r in specs),
                end_drafts=sum(r['end_draft'] for r in specs),
                reachable_rows=sorted(set(r['reachable'] for r in specs)),
                retained_counts=sorted(set(r['kept'] for r in specs)),
                coupled_windows=sum(r['coupled'] for r in specs), fallback_count=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--mode', choices=('target','mtp','coupled','suffix','suffix-coupled'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--diagnostics', action='store_true', help='short numerical run; not a timing benchmark')
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text(encoding='utf-8'))
    a.out.mkdir(parents=True, exist_ok=False)
    report = dict(result='running', mode=a.mode, scripted_model_output=False, diagnostics=a.diagnostics,
                  executable_sha256=sha256(cfg['exe']), args=cfg['args'], cases=[])
    engine = None
    tok = tokenizer(cfg['tokenizer'])
    template = ChatTemplate(Path(cfg['tokenizer']) / 'chat_template.jinja')
    env = child_env(cfg)
    env['STRATA_TRACE'] = '1'; env['STRATA_DECODE_TIMING'] = '1'
    env.pop('STRATA_SPEC_COUPLED', None)
    env.pop('STRATA_GRAMMAR_LOGITS', None)
    if a.diagnostics: env['STRATA_GRAMMAR_LOGITS'] = str(a.out / 'head-rows.bin')
    log = a.out / 'engine.txt'
    def save():
        (a.out / 'result.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    def prompt(text):
        return tok.encode(template.render([dict(role='user', content=text)], enable_thinking=False), parse_special=True)
    def raw(tokens):
        return b''.join(tok.token_bytes(t) for t in tokens if t not in (248044,248046))
    def gen(name, source, ids, cap=64, sampling=None):
        started=time.perf_counter()
        output=[t for t in engine.generate(ids, cap, sampling or {}, threading.Event(),
                                          constraint=GrammarConstraint(source) if source else None) if t is not None]
        last=dict(engine.last)
        assert len(output)==last['generated'] and len(output)<=cap
        report['cases'].append(dict(name=name, grammar=source, prompt=ids, tokens=output,
            raw_bytes_hex=raw(output).hex(), last=last, seconds=time.perf_counter()-started))
        save(); print(name, last['finish'], last['generated'], 'drafts',last['drafts_accepted'], '/',last['drafts_offered'], flush=True)
        return output,last
    try:
        engine=StrataEngine(cfg['exe'],cfg['args'],cwd=cfg['cwd'],log=str(log),env=env)
        report['info']=dict(engine.info)
        assert engine.info['grammar']=='gbnf-v2'
        if a.mode=='target': assert engine.info['decode_mode']=='target' and engine.info['mtp_loaded']==0
        else: assert engine.info['decode_mode']=='mtp' and engine.info['mtp_loaded']==1
        assert bool(engine.info['lookup'])==a.mode.startswith('suffix')
        coupled=a.mode.endswith('coupled')
        assert ('--coupled-draft' in cfg['args'])==coupled
        ids=prompt('Write the digits 0 through 9 separated by spaces. No explanation.')
        sequence='0 1 2 3 4 5 6 7 8 9'
        source='root ::= '+json.dumps(sequence)
        sampled=dict(temperature=0.8,top_p=0.85,top_k=20,min_p=0.04,seed=434,
                     penalty_last_n=64,repetition_penalty=1.1,frequency_penalty=0.1,presence_penalty=0.05)
        if a.diagnostics:
            sequence='0 1 2 3 4 5'
            source='root ::= '+json.dumps(sequence)
            for name,params in [('greedy',{'temperature':0,'seed':434}),('sampled',sampled)]:
                tokens,last=gen(name,source,ids,16,params)
                assert raw(tokens).decode()==sequence and last['finish']=='stop'
            # Third request permits real choice, retaining exact prefix metadata
            # up to the first numerical divergence across native graph shapes.
            gen('sampled alternatives','root ::= [0-9] (" " [0-9])*',ids,8,sampled)
        else:
            plain,_=gen('ordinary before constraint',None,ids,24)
            tokens,last=gen('greedy sequence',source,ids)
            assert raw(tokens).decode()==sequence and last['finish']=='stop'
            gen('sampled penalized sequence',source,ids,sampling=sampled)
            for cap in (1,2,3,4,5,6,7,8,9,10,11,12):
                part,last=gen('budget '+str(cap),source,ids,cap)
                assert sequence.encode().startswith(raw(part)) and last['finish']=='length'
            literal='begin: café 🐈; end'
            unicode='root ::= '+json.dumps(literal,ensure_ascii=False)
            for params in ({'temperature':0},sampled):
                whole,last=gen('Unicode full',unicode,ids,sampling=params)
                assert raw(whole).decode()==literal and last['finish']=='stop'
            for cap in range(1,min(len(whole),10)):
                part,last=gen('Unicode budget '+str(cap),unicode,ids,cap)
                assert literal.encode().startswith(raw(part)) and last['finish']=='length'
            empty,last=gen('epsilon','root ::= ""',ids,1)
            assert raw(empty)==b'' and last['finish']=='stop'
            recursive,last=gen('recursive','root ::= "(" root ")" | "x"',prompt('Write ((x)).'),64)
            answer=raw(recursive).decode()
            assert last['finish']=='stop' and re.fullmatch(r'\(*x\)*',answer) and answer.count('(')==answer.count(')')
            # Long repeats naturally exercise the existing suffix policy, which
            # still chooses its source itself. No forced lookup hook is used.
            phrase='alpha beta gamma delta '
            repeat_ids=prompt('Repeat this line exactly, eight times, without extra text:\n'+phrase.strip())
            repeating='root ::= '+json.dumps(phrase*8)
            for i in range(2):
                rep,last=gen('suffix opportunity '+str(i),repeating,repeat_ids,96)
                assert raw(rep).decode()==phrase*8 and last['finish']=='stop'
            for prefix in (1,3):
                generator=engine.generate(ids,256,{},threading.Event(),constraint=GrammarConstraint('root ::= '+json.dumps(phrase*30)))
                visible=[]
                try:
                    for t in generator:
                        if t is not None:
                            visible.append(t)
                            if len(visible)==prefix: break
                finally: generator.close()
                assert engine.last['finish']=='cancel'
                report['cases'].append(dict(name='STOP drain '+str(prefix),tokens=visible,last=dict(engine.last)))
                clean,last=gen('clean after STOP '+str(prefix),'root ::= "OK"',ids)
                assert raw(clean)==b'OK' and last['finish']=='stop'
            partial,_=gen('live context source',source,ids,5)
            continued_ids=ids+partial+tok.encode('\nNow say OK.')
            live,last=gen('live context continuation','root ::= "OK"',continued_ids)
            assert raw(live)==b'OK' and last['reused']>=len(ids)+len(partial)-1
            plain_after,_=gen('ordinary after constraints',None,ids,24)
            report['ordinary_repeat_identical']=plain_after==plain
        engine.close();engine=None
        report['trace_audit']=audit(log.read_text(encoding='utf-8'))
        summary=report['trace_audit']
        if a.mode!='target': assert summary['sources']['mtp'] and max(summary['reachable_rows'])>1
        if a.mode.startswith('suffix') and not a.diagnostics: assert summary['sources']['suffix']>0
        if coupled: assert summary['coupled_windows']>0
        report['result']='pass'
    except Exception:
        report['result']='fail';report['failure']=traceback.format_exc();raise
    finally:
        if engine is not None: engine.close()
        save()


if __name__=='__main__': main()
