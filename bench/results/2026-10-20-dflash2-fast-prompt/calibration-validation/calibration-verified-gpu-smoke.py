import json, sys, time, hashlib
from pathlib import Path
root=Path.cwd(); sys.path.insert(0,str(root/'tools')); sys.path.insert(0,str(root))
import calibrate as C
import calibrate_dflash2 as D
from serve.server import StrataEngine, child_env
cfg=json.loads((root/'strata-dflash2-fast-prompt.json').read_text())
args=C.with_arg(C.engine_args(cfg), "--expert-cache", "2700")
fixtures=json.loads((root/'bench/results/2026-10-19-dflash2-pr-performance/prompts.json').read_text())
out=root.parent/'Strata-data/dflash2-long-next'
starts=[]
def launch(a):
 index=len(starts); starts.append(a)
 return StrataEngine(cfg['exe'],a,cwd=str(root),log=str(out/f'calibration-verified-gpu-{index}.log'),env=child_env(cfg))
# A bounded hardware smoke, not a full production calibration/performance claim.
C.SWEEP_ROUNDS=1; C.MAX_NEW=32; D.PREFILLS=(); D.FLOORS=(.7,.85); D.LONG_TOKENS=(768,)
import strata_tokenizer as ST
folder=Path(cfg['tokenizer']); vocab=json.loads((folder/'vocab.json').read_text()); tokens=[None]*len(vocab)
for token,index in vocab.items(): tokens[index]=token
tok=ST.Tokenizer(tokens,(folder/'merges.txt').read_text().split('\n'),json.loads((folder/'token_type.json').read_text()))
make=C.doc_prompt(tok)
source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'tools/calibrate.py',root/'tools/calibrate_dflash2.py']}
t0=time.monotonic()
settings,report=D.measure_profiles(args,{'--pcie-frac':'0','--spec-min-p':'.7','--dflash-block':'7',
 '--adapt-every':'0','--adapt-swaps':'0'},[fixtures['code']],make,launch, baseline_args=args,
 say=lambda *x:print(*x,flush=True))
result={'settings':settings,'report':report,'starts':starts,'seconds':time.monotonic()-t0,'source_hashes':source_hashes,
 'limits':'smoke only: 32 output tokens, 768-token document, one confirmation round; full defaults not run'}
(out/'calibration-verified-gpu-smoke.json').write_text(json.dumps(result,indent=2))
print('SMOKE COMPLETE',len(starts),settings,report.get('failed'),flush=True)
assert report['profiles'] and report['confirmation']['baseline'] and report['confirmation']['candidate']
assert not report['failed'], report['failed']
assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==v for p,v in source_hashes.items())
