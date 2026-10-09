"""Isolated native migration probe. Existing model, new output directory only.

python tools/bench_rope_migration.py --config /path/config.json --output /new/path
The output includes every failure and measured quality difference. No automatic
quality acceptance threshold is applied; correctness assertions remain separate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import struct
import numpy as np
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from serve.server import StrataEngine, child_env
from serve.frontend import ChatTemplate
import strata_tokenizer as ST


def saved_prefix(path, max_tokens):
    """Read this probe's freshly saved v1 token receipt; native RESTORE validates the file fully."""
    with path.open('rb') as f:
        header=f.read(64)
        assert len(header)==64 and header[:8]==b'STRSESS\x01'
        assert struct.unpack_from('<II',header,8)==(1,64)
        f.seek(64+18*8+3*8)
        count=struct.unpack('<Q',f.read(8))[0]
        assert 0<count<=max_tokens
        data=f.read(count*4)
        assert len(data)==count*4
        return list(struct.unpack('<'+str(count)+'i',data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--tokens', type=int, default=4096)
    ap.add_argument('--source-context', type=int, default=8192)
    ap.add_argument('--target-context', type=int, default=16384)
    ap.add_argument('--kv-resident', type=int, default=0)
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--mtp', help='MTP runtime directory; convert and validate its FP16 draft state too')
    ap.add_argument('--mtp-window', type=int, default=32768)
    ap.add_argument('--extend-tokens', type=int, nargs='*', default=[],
                    help='After migration, extend the same canonical chat to these total token lengths')
    ap.add_argument('--ordinary-only', action='store_const', const='none', dest='fresh_rope',
                    help='Fresh ordinary-RoPE retrieval only; no session writes or YaRN runs')
    ap.add_argument('--fresh-rope', choices=('none','yarn'),
                    help='One fresh retrieval probe under ordinary RoPE or YaRN 4x; no migration')
    args = ap.parse_args()
    if args.fresh_rope and args.compare:
        ap.error('fresh-only and --compare are separate experiments')
    if args.extend_tokens and (not args.compare or args.fresh_rope or
            any(n <= args.tokens or n + 128 > args.target_context for n in args.extend_tokens)):
        ap.error('extension requires --compare and larger lengths fitting target context plus 128 tokens')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.config).read_text())
    rows = []
    def record(**row):
        rows.append(row)
        (output / 'results.json').write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)
    tp = Path(config['tokenizer'])
    vocab = json.loads((tp/'vocab.json').read_text())
    vocab_tokens = [None]*len(vocab)
    for text, index in vocab.items(): vocab_tokens[index] = text
    tok = ST.Tokenizer(vocab_tokens, (tp/'merges.txt').read_text().split('\n'),
                       json.loads((tp/'token_type.json').read_text()))
    template = ChatTemplate(tp/'chat_template.jinja')
    def prompt_text(padding):
        return ('The first code is CEDAR-731.\n' + ' apple'*(padding//2) +
                '\nThe middle code is MARBLE-482.\n' + ' orange'*(padding-padding//2) +
                '\nThe last code is QUARTZ-956.\nReturn all three codes in order separated by |. Nothing else.')
    def prompt(padding):
        text = prompt_text(padding)
        return tok.encode(template.render([{'role':'user','content':text}], tools=None,
                                         enable_thinking=False), parse_special=True)
    lo, hi = 0, args.tokens
    while lo < hi:
        mid = (lo+hi+1)//2
        if len(prompt(mid)) <= args.tokens: lo = mid
        else: hi = mid-1
    ids = prompt(lo)
    assert len(ids) == args.tokens, len(ids)
    (output/'tokens.json').write_text(json.dumps(ids))
    base_args = []
    raw = config['args']; i = 0
    remove = {'--max-context','--kv','--kv-resident','--mtp','--spec','--conversation-cache-mib',
              '--rope-scaling',
              '--rope-scale','--prefill','--suffix-draft','--mtp-window',
              '--conversation-cache-spill-dir','--conversation-cache-disk-mib','--conversation-cache-dir'}
    while i < len(raw):
        if raw[i] in remove: i += 2
        elif raw[i] in ('--experimental-rope-yarn4-cache','--conversation-cache-disk-only'): i += 1
        else: base_args.append(raw[i]); i += 1
    base_args += ['--kv','fp16','--spec','2','--suffix-draft','0','--conversation-cache-mib','0',
                  '--experimental-rope-yarn4-cache','--prefill','8192']
    if args.mtp:
        base_args[base_args.index('--spec')+1]='4'
        base_args += ['--mtp',args.mtp,'--mtp-window',str(args.mtp_window)]
    if args.kv_resident: base_args += ['--kv-resident',str(args.kv_resident)]
    def start(label, yarn, context):
        argv = base_args+['--max-context',str(context),'--rope-scaling','yarn' if yarn else 'none']
        if yarn: argv += ['--rope-scale','4','--yarn-orig-ctx','262144']
        cfg = dict(config, args=argv)
        cfg['exe'] = str(ROOT/'build/strata')
        cfg['cwd'] = str(ROOT)
        env = child_env(cfg)
        env['STRATA_PREFILL_CPU_SHARE']='0'
        env['STRATA_ROPE_TABLE']='1'
        env['STRATA_LOGPOS'] = str(output/(label+'.logpos.tsv'))
        env['STRATA_MIGRATION_LOGITS'] = str(output/(label+'.logits.bin'))
        env['STRATA_MIGRATION_LOGITS_FROM'] = str(args.tokens-1)
        started = time.perf_counter()
        engine = StrataEngine(cfg['exe'], argv, cwd=str(ROOT), log=str(output/(label+'.engine.log')), env=env)
        record(kind='start', label=label, seconds=time.perf_counter()-started, argv=argv)
        engine.probe_label=label
        return engine
    def generate(engine, label, tokens, limit):
        started = time.perf_counter(); first = None; result = []
        for token in engine.generate(tokens, limit, {'temperature':0}, threading.Event()):
            if token is not None:
                first = first or time.perf_counter(); result.append(token)
        elapsed = time.perf_counter()-started
        text = tok.decode(result)
        record(kind='generate',label=label,ttft_s=None if first is None else first-started,
               total_s=elapsed,output_tokens=len(result),text=text,reused=engine.reused,
               drafts_accepted=engine.last.get('drafts_accepted'),drafts_offered=engine.last.get('drafts_offered'),
               decode_tps=(len(result)-1)/max(.000001,elapsed-(first-started)) if first and len(result)>1 else None,
               retrieval_task=limit>1,
               correct=all(code in text for code in ('CEDAR-731','MARBLE-482','QUARTZ-956')) if limit>1 else None,
               input_tokens=len(tokens),input_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest())
        return result
    forced=tok.encode('CEDAR-731|MARBLE-482|QUARTZ-956. These are the three codes recorded in the document.',parse_special=False)
    distributions={}
    eval_lengths={}
    def forced_eval(engine,label,prompt_ids=None):
        prompt_ids=ids if prompt_ids is None else prompt_ids
        eval_lengths[label]=len(prompt_ids)
        path=output/(engine.probe_label+'.logits.bin')
        offset=path.stat().st_size if path.exists() else 0
        generate(engine,label+'-forced',prompt_ids+forced,1)
        data=path.read_bytes()[offset:] if path.exists() else b''
        parsed={}
        at=0
        while at<len(data):
            pos,target,count=struct.unpack_from('<qii',data,at);at+=16
            row=np.frombuffer(data,dtype='<f4',count=count,offset=at).astype(np.float64);at+=count*4
            if not len(prompt_ids)-1<=pos<len(prompt_ids)+len(forced)-1:continue
            assert np.isfinite(row).all(),(label,pos)
            row-=np.max(row);row-=np.log(np.exp(row).sum())
            parsed[pos]=(target,row)
        distributions[label]=parsed
        record(kind='forced-rows',label=label,positions=sorted(parsed),rows=len(parsed))
    def compare(reference,candidate):
        a,b=distributions[reference],distributions[candidate]
        common=sorted(a.keys() & b.keys())
        assert eval_lengths[reference]==eval_lengths[candidate]
        length=eval_lengths[reference]
        expected=list(range(length-1,length+len(forced)-1))
        assert sorted(a)==sorted(b)==expected,('incomplete full-vocabulary diagnostic rows',reference,candidate,sorted(a),sorted(b),expected)
        metrics=[]
        for pos in common:
            ta,la=a[pos];tb,lb=b[pos];assert ta==tb
            metrics.append({'position':pos,'target':ta,'kl':float(np.sum(np.exp(la)*(la-lb))),
                'top_agrees':bool(la.argmax()==lb.argmax()),'reference_nll':float(-la[ta]),'candidate_nll':float(-lb[ta]),
                'nll_delta':float(la[ta]-lb[ta])})
        mean_kl=float(np.mean([x['kl'] for x in metrics]));top=float(np.mean([x['top_agrees'] for x in metrics]));delta=float(np.mean([x['nll_delta'] for x in metrics]))
        record(kind='quality',reference=reference,candidate=candidate,rows=len(metrics),mean_kl=mean_kl,
            max_kl=max(x['kl'] for x in metrics),top_agreement=top,mean_nll_delta=delta,perplexity_ratio=float(np.exp(delta)),individual=metrics)
    source = output/'ordinary.sess'
    engine = None
    try:
        if args.fresh_rope:
            yarn=args.fresh_rope=='yarn'
            engine = start('yarn' if yarn else 'ordinary',yarn,args.target_context)
            generate(engine,'A-fresh-yarn' if yarn else 'ordinary-task',ids,64)
            record(kind='complete',measurement_scope='fresh three-code retrieval; no quality verdict',
                   rope_profile='yarn4' if yarn else 'ordinary',
                   actual_prompt_tokens=len(ids),allocation_limit=args.target_context,
                   full_1m_sequence_verified=len(ids)>=1048576)
            return
        engine = start('ordinary',False,args.source_context)
        generate(engine,'ordinary-prefix',ids[:-1],1)
        record(kind='save-source', **engine.session_file('save',str(source)))
        saved = saved_prefix(source,args.source_context)
        assert saved == ids[:len(saved)] and len(saved)<len(ids), (len(saved),len(ids))
        record(kind='boundary',cached=len(saved),replay_before_sampling=len(ids)-len(saved))
        if args.compare: forced_eval(engine,'ordinary')
        generate(engine,'ordinary-task',ids,64)
        engine.close(); engine=None
        engine = start('yarn',True,args.target_context)
        if args.compare:
            generate(engine,'A-prefix',ids[:-1],1)
            forced_eval(engine,'A')
        generate(engine,'A-fresh-yarn',ids,64)
        # Exact replay C: a disjoint prompt clears live state; parked RAM/disk caches disabled.
        generate(engine,'reset',[42,43,44,45],1)
        if args.compare:
            generate(engine,'C-prefix',ids[:-1],1)
            forced_eval(engine,'C')
        generate(engine,'C-exact-yarn-replay',ids,64)
        started=time.perf_counter()
        migrated=engine.session_file('migrate_yarn4',f'{args.source_context} {source}')
        record(kind='B-migration',wall_s=time.perf_counter()-started,**migrated)
        if args.compare: forced_eval(engine,'B')
        generate(engine,'B-migrated-yarn',ids,64)
        # A fresh continuation must not lose the approximate-history marker on SAVE.
        roundtrip=output/'migrated-roundtrip.sess' if args.tokens<=32768 else Path(str(source)+'.yarn4')
        if args.tokens<=32768:record(kind='save-migrated',**engine.session_file('save',str(roundtrip)))
        generate(engine,'reset-after-save',[46,47,48,49],1)
        restored=engine.session_file('restore',str(roundtrip))
        record(kind='restore-migrated',**restored)
        assert restored.get('approximate_migrated_history'), restored
        reloaded = generate(engine,'B-reloaded-yarn',ids,64)
        assert all(code in tok.decode(reloaded) for code in ('CEDAR-731','MARBLE-482','QUARTZ-956'))
        for target in args.extend_tokens:
            def extended(padding):
                messages=[{'role':'user','content':prompt_text(lo)},
                          {'role':'assistant','content':'CEDAR-731|MARBLE-482|QUARTZ-956'},
                          {'role':'user','content':'Additional reference material follows.\n'+' blue'*padding+
                           '\nRecall the first, middle and last codes from the earlier document. '
                           'Return all three in order separated by |. Nothing else.'}]
                return tok.encode(template.render(messages,tools=None,enable_thinking=False),parse_special=True)
            low,high=0,target
            while low<high:
                mid=(low+high+1)//2
                if len(extended(mid))<=target:low=mid
                else:high=mid-1
            extended_ids=extended(low)
            assert len(extended_ids)==target,(len(extended_ids),target)
            assert extended_ids[:len(saved)]==saved,'extension must preserve the canonical source prefix exactly'
            record(kind='extension-input',tokens=target,source_cached_tokens=len(saved),
                   sha256=hashlib.sha256(json.dumps(extended_ids).encode()).hexdigest())
            for arm in ('A','C','B'):
                label=f'{arm}-extend-{target}'
                if arm=='B':
                    record(kind='extension-restore',label=label,
                           **engine.session_file('restore',str(source)+'.yarn4'))
                else:
                    generate(engine,label+'-reset',[42,43,44,45],1)
                generate(engine,label+'-prefix',extended_ids[:-1],1)
                if arm=='B':
                    assert engine.reused>=len(saved)-8,('migration unexpectedly replayed old prefix',engine.reused,len(saved))
                forced_eval(engine,label,extended_ids)
                generate(engine,label+'-retrieval',extended_ids,64)
            compare(f'A-extend-{target}',f'C-extend-{target}')
            compare(f'A-extend-{target}',f'B-extend-{target}')
        engine.close(); engine=None
        if args.compare:
            compare('A','C');compare('A','B');compare('ordinary','A')
            engine=start('ordinary-large-capacity',False,args.target_context)
            generate(engine,'D-prefix',ids[:-1],1)
            forced_eval(engine,'D')
            generate(engine,'D-ordinary-large-capacity',ids,64)
            engine.close();engine=None
            compare('ordinary','D');compare('D','A')
        record(kind='complete',measurement_scope=(
            'per-case logit and retrieval measurements; no automatic quality verdict'
            if args.compare else 'retrieval/lifecycle measurements only'),
            actual_prompt_tokens=len(ids), allocation_limit=args.target_context,
            extended_contexts=args.extend_tokens,
            full_1m_sequence_verified=1048576 in args.extend_tokens)
    except BaseException as exc:
        record(kind='failure',error=repr(exc))
        raise
    finally:
        if engine: engine.close()


if __name__ == '__main__': main()
