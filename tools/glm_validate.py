"""Run real GLM model regressions and reproducible timing cases, one engine at a time.

Example: python tools/glm_validate.py --exe build/strata-glm.exe --model MODEL --out bench/results/RUN --full
The full suite needs the prompt1536.txt generated for the benchmark (or --prompt FILE).
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time
import numpy as np

PROMPT = [154822,154824,785,6722,315,9621,374]
CONTINUATION = [12089,11,264,3283,3881,369,1181,9077,3840,11,19812,17621]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exe',required=True); ap.add_argument('--model',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--prompt'); ap.add_argument('--full',action='store_true'); ap.add_argument('--cpu-only',action='store_true')
    ap.add_argument('--expert-ram-gb',default='28'); ap.add_argument('--wait-baseline',action='store_true')
    a=ap.parse_args(); out=Path(a.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    if a.wait_baseline:
        print('waiting for sequential baseline runs',flush=True)
        while True:
            try:
                b=json.loads((out/'baseline.json').read_text())
                if len(b)==4: break
            except (OSError,ValueError): pass
            time.sleep(5)
    base=[str(Path(a.exe).resolve()),'--model',str(Path(a.model).resolve()),'--expert-ram-gb',a.expert_ram_gb,'--threads','16']
    cases=[('cpu-f32-tf',[]),('cpu-bf16-tf',['--kv','bf16']),('cpu-fp8-tf',['--kv','fp8'])]
    if not a.cpu_only: cases += [('gpu-f32-tf',['--gpu','0'])]
    cases=[(n,opts+['--tokens',','.join(map(str,PROMPT+CONTINUATION)),'--tf','--logits-out',str(out/(n+'.f32'))]) for n,opts in cases]
    if a.full:
        prompt=a.prompt or str(out/'prompt1536.txt')
        cases += [('cpu-layer1536',['--tokens-file',prompt,'--logits-out',str(out/'cpu-layer1536.f32')])]
        if not a.cpu_only:
            cases += [('gpu-layer1536',['--gpu','0','--tokens-file',prompt,'--logits-out',str(out/'gpu-layer1536.f32')]),
                      ('gpu-chunk1536',['--gpu','0','--prefill','chunk','--prefill-chunk','256','--tokens-file',prompt,'--logits-out',str(out/'gpu-chunk1536.f32')])]
        for name,opts in [('cpu-decode',[]), *([] if a.cpu_only else [('gpu-decode',['--gpu','0']),('gpu-prefetch',['--gpu','0','--prefetch'])])]:
            cases += [(name,opts+['--tokens',','.join(map(str,PROMPT)),'--gen','100','--ignore-eos','--repeat','2'])]
    results=[]
    for name,args in cases:
        print('starting',name,flush=True); start=time.monotonic()
        with (out/(name+'.stdout.txt')).open('w') as so,(out/(name+'.stderr.txt')).open('w') as se:
            p=subprocess.run(base+args,stdout=so,stderr=se)
        result={'name':name,'returncode':p.returncode,'seconds':time.monotonic()-start,'command':base+args}
        if name.endswith('-tf') and p.returncode==0:
            ids=[int(x) for x in re.findall(r'argmax (\d+)',(out/(name+'.stdout.txt')).read_text())]
            result['reference_agreement']=sum(x==y for x,y in zip(ids[6:18],CONTINUATION))
        results.append(result); (out/'validation.json').write_text(json.dumps(results,indent=2))
        print(result['name'],result['returncode'],round(result['seconds'],1),result.get('reference_agreement'),flush=True)
        if p.returncode: return p.returncode
    for left,right,tol in [('cpu-f32-tf','gpu-f32-tf',1e-3),('cpu-layer1536','gpu-layer1536',1e-3),('gpu-layer1536','gpu-chunk1536',1e-3)]:
        if not (out/(left+'.f32')).exists() or not (out/(right+'.f32')).exists(): continue
        x,y=(np.fromfile(out/(n+'.f32'),dtype=np.float32) for n in (left,right))
        error=float(np.max(np.abs(x-y))/max(float(np.max(np.abs(x))),1e-6))
        agree=int(np.sum(x.reshape(-1,154880).argmax(1)==y.reshape(-1,154880).argmax(1)))
        result={'comparison':[left,right],'normalized_max_error':error,'top1_agreement':agree,'rows':len(x)//154880,'pass':error<tol}
        results.append(result); print(result,flush=True)
    (out/'validation.json').write_text(json.dumps(results,indent=2))
    return int(any(r.get('pass') is False for r in results) or any(r.get('reference_agreement',12)!=12 for r in results if 'fp8' not in r.get('name','')))

if __name__=='__main__': raise SystemExit(main())
