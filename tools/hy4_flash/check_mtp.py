#!/usr/bin/env python3
"""Run the same token prompts through hy4-flash serve with and without MTP; retain raw evidence."""
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import shutil
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "terminal"))
from strata_control.hy4_flash import environment
from tools.strata_tokenizer import Tokenizer
from serve.frontend import ChatTemplate


def run(args, mtp, prompts):
    env = {**os.environ, **environment(), 'CUDA_VISIBLE_DEVICES': str(args.gpu), 'HY4_FLASH_DIRECT_CONTEXT': str(args.context)}
    env.update(dict(value.split('=',1) for value in args.env))
    mode = 'mtp' if mtp else 'baseline'
    command = [str(args.engine.resolve()), '--serve', '--native', str(args.model.resolve()),
               '--pack', str(args.pack.resolve()), '--context', str(args.context),
               '--expert-vram', str(args.expert_vram)]
    if args.resident_experts:
        command.append('--resident-experts')
    if mtp:
        command += ['--mtp', '--draft-tokens', str(args.draft_tokens)]
    results = []
    with (args.out / f'{mode}.stderr.log').open('w') as stderr, (args.out / f'{mode}.stdout.log').open('w') as stdout:
        process = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=stderr, text=True, bufsize=1, cwd=ROOT)
        lines = queue.Queue()
        def pump():
            for line in process.stdout:
                stdout.write(line); stdout.flush(); lines.put(line.strip())
            lines.put(None)
        threading.Thread(target=pump, daemon=True).start()
        def read():
            line = lines.get(timeout=300)
            if line is None:
                raise RuntimeError(f'{mode} process exited ({process.wait(timeout=10)}); see stderr')
            if line.startswith('ERR '):
                raise RuntimeError(line)
            return line
        try:
            while not read().startswith('READY '):
                pass
            print(mode, 'READY', flush=True)
            for label, tokens in prompts:
                start = time.monotonic()
                process.stdin.write(f'GEN {args.tokens} temperature={args.temperature} top_p=1 seed=42 '+','.join(map(str,tokens))+'\n')
                process.stdin.flush()
                output=[];first=None
                while True:
                    line=read()
                    if line.startswith('T '):
                        output.append(int(line.split()[1]))
                        if first is None:first=time.monotonic()-start
                    elif line.startswith('DONE '):
                        result={'label':label,'input':tokens,'output':output,'done':line,
                                'seconds':time.monotonic()-start,'first_seconds':first}
                        results.append(result)
                        print(mode,label,line,'wall',round(result['seconds'],3),flush=True)
                        break
        finally:
            if process.poll() is None:
                try:
                    process.stdin.write('QUIT\n');process.stdin.flush();process.wait(timeout=30)
                except (OSError,subprocess.TimeoutExpired):
                    process.terminate();process.wait(timeout=30)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--pack',type=Path,required=True)
    p.add_argument('--engine',type=Path,default=ROOT/'build/strata')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--context',type=int,default=8192)
    p.add_argument('--expert-vram',type=float,default=48)
    p.add_argument('--resident-experts',action='store_true')
    p.add_argument('--tokens',type=int,default=128)
    p.add_argument('--draft-tokens',type=int,default=3)
    p.add_argument('--mtp-only',action='store_true')
    p.add_argument('--temperature',type=float,default=0)
    p.add_argument('--long',action='store_true')
    p.add_argument('--env',action='append',default=[],help='override an engine environment setting: KEY=VALUE')
    p.add_argument('--repeat',type=int,default=1,help='repeat the prompt set in the same engine process')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    # The shared CUDA library is mmap'ed during inference. Snapshot executable
    # and library so another build cannot truncate live code under the test.
    runtime=a.out.resolve()/'runtime';runtime.mkdir(exist_ok=True)
    engine=a.engine.resolve()
    shutil.copy2(engine,runtime/'strata')
    for library in engine.parent.glob('libstrata*.so'):
        shutil.copy2(library,runtime/library.name)
    a.engine=runtime/'strata'
    tokenizer=Tokenizer.from_gguf(a.model)
    template=ChatTemplate(a.pack/'tokenizer/chat_template.jinja')
    prompts=[]
    for i,text in enumerate(['你好，请用一句话介绍自己。','请写一个Python函数，用二分查找查找有序数组中的整数。','从1开始列出前30个正整数，以逗号分隔。']):
        rendered=template.render([{'role':'user','content':text}],reasoning_effort='no_think')
        prompts.append((f'prompt{i}',tokenizer.encode(rendered,parse_special=True)))
    if a.long:
        text=('这是一份用于测试长上下文缓存和回滚的记录。编号、顺序与内容都需要保持一致。\n'*160)+ '请总结上述记录，并解释为什么需要保持顺序。'
        rendered=template.render([{'role':'user','content':text}],reasoning_effort='no_think')
        prompts.append(('long',tokenizer.encode(rendered,parse_special=True)))
        for n in range(4):
            prompts.append((f'rewind{n}',prompts[-1][1][:-n-1]))
    prompts=[(f'{label}-r{repeat+1}',ids) for repeat in range(a.repeat) for label,ids in prompts]
    result={}
    if not a.mtp_only:
        result['baseline']=run(a,False,prompts)
        (a.out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    result['mtp']=run(a,True,prompts)
    if 'baseline' in result:
        result['equal']=[b['output']==m['output'] for b,m in zip(result['baseline'],result['mtp'])]
    (a.out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('RESULT',result.get('equal'),flush=True)
    if 'equal' in result and not all(result['equal']):
        raise SystemExit(1)


if __name__=='__main__':main()
