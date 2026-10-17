"""Native-engine cold-prefix benchmark. Does not launch a permanent service."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT=Path(os.environ.get('BENCH_OUTPUT', '.')).resolve()
SOURCE=Path(os.environ['STRATA_SRC']).resolve()
sys.path[:0]=[str(SOURCE),str(SOURCE/'tools')]
from serve.server import StrataEngine, child_env
from serve.frontend import ChatTemplate
import strata_tokenizer as ST

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--label',required=True)
    ap.add_argument('--exe',required=True)
    ap.add_argument('--remote')
    ap.add_argument('--split',type=int)
    ap.add_argument('--prefill',type=int,default=8192)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--sizes',default='1024,8192,32768')
    ap.add_argument('--output-tokens',type=int,default=128)
    ap.add_argument('--spec',type=int,default=4)
    ap.add_argument('--max-context',type=int,default=65536)
    ap.add_argument('--mmap',action='store_true')
    ap.add_argument('--no-mtp',action='store_true')
    args=ap.parse_args()
    output=ROOT/args.label
    output.mkdir(exist_ok=False)
    config=json.loads(args.config.read_text())
    raw=config['args']
    argv=[]
    for flag in ['--pack','--native','--ple-gguf','--expert-profile']:
        argv += [flag,raw[raw.index(flag)+1]]
    argv+=['--expert-cache','auto','--prefill',str(args.prefill),'--max-context',str(args.max_context),
           '--kv','fp16','--stats','--vram-reserve-mib','1536','--prompt-cache','0',
           '--conversation-cache-mib','0','--spec',str(args.spec),'--suffix-draft','0']
    if args.spec and not args.no_mtp:argv+=['--mtp',str(Path.home()/'Strata-data/mtp/rt')]
    if args.mmap:argv+=['--mmap-experts']
    argv+=config.get('benchmark_extra_args',[])
    if args.remote:argv+=['--layer-split',str(args.split),'--split-device','0','--remote-stage',args.remote]
    tp=Path(config['tokenizer']);vocab=json.loads((tp/'vocab.json').read_text());vocab_tokens=[None]*len(vocab)
    for text,index in vocab.items():vocab_tokens[index]=text
    tok=ST.Tokenizer(vocab_tokens,(tp/'merges.txt').read_text().split('\n'),json.loads((tp/'token_type.json').read_text()))
    template=ChatTemplate(tp/'chat_template.jinja')
    def encode(text):return tok.encode(template.render([{'role':'user','content':text}],tools=None,enable_thinking=False),parse_special=True)
    def prompt(n):
        def ids(pad):return encode('The first project code is CEDAR-731.\n'+' apple'*(pad//2)+'\nThe middle code is MARBLE-482.\n'+' orange'*(pad-pad//2)+'\nThe last code is QUARTZ-956.\nReturn the three codes in order separated by |, then write a Python function that validates them, with a docstring and example calls.')
        lo,hi=0,n
        while lo<hi:
            mid=(lo+hi+1)//2
            if len(ids(mid))<=n:lo=mid
            else:hi=mid-1
        result=ids(lo);assert len(result)==n,(len(result),n)
        return result
    prompts={n:prompt(n) for n in map(int,args.sizes.split(','))}
    (output/'prompts.json').write_text(json.dumps(prompts))
    rows=[]
    def record(row):
        rows.append(row);(output/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
    env=child_env(config)
    for k in list(env):
        if k.startswith(('STRATA_LOGPOS','STRATA_MIGRATION','STRATA_ROPE_TABLE')):env.pop(k,None)
    env['STRATA_PREFILL_CPU_SHARE']='0'
    env['STRATA_REMOTE_TIMING']='1'
    env['STRATA_STAGE_TOKEN']=os.environ.get('STRATA_STAGE_TOKEN', '')
    env['STRATA_REMOTE_TIMEOUT_S']='60'
    engine=None
    try:
        start=time.perf_counter()
        engine=StrataEngine(args.exe,argv,cwd=str(SOURCE),log=str(output/'engine.log'),env=env)
        record({'kind':'start','seconds':time.perf_counter()-start,'argv':argv,'exe':args.exe})
        for repeat in range(args.repeats):
            for n,ids in prompts.items():
                (output/'status.json').write_text(json.dumps({'state':'running','pid':os.getpid(),'repeat':repeat,'tokens':n}))
                start=time.perf_counter();first=None;out=[]
                for t in engine.generate(ids,args.output_tokens,{'temperature':0},threading.Event()):
                    if t is not None:
                        if first is None:first=time.perf_counter()
                        out.append(t)
                total=time.perf_counter()-start;ttft=first-start if first else None
                text=tok.decode(out)
                record({'kind':'request','repeat':repeat,'tokens':n,'ttft_s':ttft,'total_s':total,
                        'decode_tps':(len(out)-1)/(total-ttft) if len(out)>1 and total>ttft else None,
                        'prefill_progress_ms':engine.progress_ms,'prefill_tps':engine.prefill_tok_s_mean,
                        'reused_tokens':engine.reused,'output_tokens':len(out),'output_ids':out,'text':text,
                        'all_markers':all(x in text for x in ['CEDAR-731','MARBLE-482','QUARTZ-956']),
                        'input_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),'stats':engine.last})
        record({'kind':'complete'})
        (output/'status.json').write_text(json.dumps({'state':'passed'}))
    except Exception as e:
        record({'kind':'failure','error':repr(e)})
        (output/'status.json').write_text(json.dumps({'state':'failed','error':repr(e)}))
        raise
    finally:
        if engine:engine.close()

if __name__=='__main__':main()
