from pathlib import Path
import json,sys,os
root=Path.cwd();out=(root/'../Strata-data/dflash2-context-audit-20261022').resolve();sys.path.insert(0,str(root/'tools'));from dflash_sampling import Engine,replace
cfg=json.loads((out/'full/dflash2-il-command.json').read_text());args=cfg['command'][1:-1];replace(args,'--dflash',str((root/'../Strata-data/dflash2-official/dflash2-bf16.gguf').resolve()))
env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')};env.update(cfg['env']);env.update(STRATA_DF_PARITY=str(out/'stage'),STRATA_DF_PARITY_CYCLE='0');(out/'stage').mkdir(exist_ok=False)
m=json.loads((out/'code_long.json').read_text());e=Engine(root/'build-dflash2/strata',args,root,env,out/'stage-capture.log')
try:
 r=e.request(m['prompt'],'temperature=0 seed=1234 spec_min_p=.7 dflash_block=7',8);assert r['tokens']==m['output'][:8],r
 (out/'stage-capture.json').write_text(json.dumps(r));print('PASS: long stage capture target tokens',flush=True)
finally:e.close()
