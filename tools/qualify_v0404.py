"""Bounded post-performance correctness checks; never use their TPS as fast-mode results."""
from pathlib import Path
import json
import os
import subprocess
import sys

root=Path(__file__).resolve().parents[1]; out=root/'build-validation'
env=dict(os.environ)
env.update(STRATA_QFUSE='0',STRATA_NO_IQ512='1',STRATA_NO_IQ256='1',STRATA_NO_IQ4NL='1')
extra='--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow --expert-cache 6000 --prefill 256'
results=[]
for arm in ['baseline','candidate']:
    cfg=json.loads((out/(arm+'.json')).read_text())
    env['STRATA_BATCH_DENSE_PARALLEL']='1' if arm=='candidate' else '0'
    for kind in ['tokens','lifecycle']:
        name=f'{kind}-{arm}'; env['BATCH_TEST_LOG']=str(out/(name+'.stderr.log'))
        script='batch_test.py' if kind=='tokens' else 'batch_interleave_test.py'
        args=[sys.executable,str(root/'tools'/script),'--config',str(out/(arm+'.json')),
              '--exe',cfg['exe'],'--max-new','128','--extra',extra]
        if kind=='tokens': args+=['--batch','4','--n','4','--dump',str(out/(name+'.json'))]
        else: args+=['--long','1536']
        print('START',name,flush=True)
        with (out/(name+'.log')).open('w') as log:
            p=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT)
            try: rc=p.wait(timeout=600)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],stdout=log,stderr=subprocess.STDOUT)
                raise
        results.append(dict(name=name,exit_code=rc))
        print('DONE',name,rc,flush=True)
        (out/'qualification.json').write_text(json.dumps(results,indent=2))
        if rc: raise RuntimeError(f'{name} failed; inspect saved evidence')
b=json.loads((out/'tokens-baseline.json').read_text())
c=json.loads((out/'tokens-candidate.json').read_text())
assert b==c, 'Cross-build token mismatch; inspect dumps'
print('CROSS BUILD EXACT',sum(map(len,c['batch'])),'batch tokens and solo references',flush=True)
