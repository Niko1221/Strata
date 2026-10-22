from pathlib import Path
import sys,json,hashlib
root=Path.cwd();out=(root/'../Strata-data/dflash2-diagnosis-20261022').resolve();sys.path.insert(0,str(root/'tools'));import strata_tokenizer as ST
p=root/'../Strata-data/models/Maximilian228--Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata/pack/tokenizer';v=json.loads((p/'vocab.json').read_text());ts=[None]*len(v)
for t,i in v.items():ts[i]=t
tok=ST.Tokenizer(ts,(p/'merges.txt').read_text().split('\n'),json.loads((p/'token_type.json').read_text()));old=json.loads((root/'bench/results/2026-10-19-dflash2-pr-performance/prompts.json').read_text())
prefix='<|im_start|>user\n';suffix='<|im_end|>\n<|im_start|>assistant\n<think>\n'
code=tok.decode(old['code'])[len(prefix):-len(suffix)];long=tok.decode(old['long'])[len(prefix):-len(suffix)];notes,question=long.rsplit('\n\n',1)
texts={'code_short':code,'code_long':notes+'\n\n'+code,'cache_short':notes.splitlines()[0]+'\n\n'+question,'cache_long':long}
res={k:tok.encode(prefix+t+suffix,parse_special=True) for k,t in texts.items()};assert res['code_short']==old['code'] and res['cache_long']==old['long']
(out/'prompts.json').write_text(json.dumps(res));(out/'prompts-text.json').write_text(json.dumps(texts,indent=2));(out/'prompt-provenance.json').write_text(json.dumps({'lengths':{k:len(x) for k,x in res.items()},'source_sha256':hashlib.sha256((root/'bench/results/2026-10-19-dflash2-pr-performance/prompts.json').read_bytes()).hexdigest()},indent=2));print({k:len(x) for k,x in res.items()})
