from pathlib import Path
import sys,json,os,time,statistics
root=Path.cwd();out=(root/'../Strata-data/dflash2-context-audit-20261022').resolve();sys.path.insert(0,str(root/'tools'));from dflash_sampling import Engine,validate_residency
import strata_tokenizer as ST
p=root/'../Strata-data/models/Maximilian228--Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata/pack/tokenizer';v=json.loads((p/'vocab.json').read_text());ts=[None]*len(v)
for t,i in v.items():ts[i]=t
tok=ST.Tokenizer(ts,(p/'merges.txt').read_text().split('\n'),json.loads((p/'token_type.json').read_text()))
texts=json.loads((root/'bench/results/2026-10-22-dflash2-length-diagnosis/prompts-text.json').read_text());notes=texts['cache_long'].rsplit('\n\n',1)[0]
text='<|im_start|>user\n'+(notes+'\n')*3+'\n'+texts['code_short']+'<|im_end|>\n<|im_start|>assistant\n<think>\n';ids=tok.encode(text,parse_special=True);assert 7000<len(ids)<8000
(out/'long-prompt.json').write_text(json.dumps(dict(text=text,tokens=ids)));cfg=json.loads((out/'full/dflash2-il-command.json').read_text());args=cfg['command'][1:-1];result={}
for arm in ['full','tail']:
 env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')};env.update(cfg['env'])
 if arm=='tail':env['STRATA_DFLASH2_PROMPT_TAIL']='1'
 e=Engine(out/'strata-prompt-tail',args,root,env,out/(arm+'-long.log'));rows=[]
 try:
  validate_residency(e.info,2900)
  for rep in [-1,0,1,2]:
   r=e.request(ids,'temperature=0 seed=1234 spec_min_p=.7 dflash_block=7',32);assert 'error' not in r,r;r['rep']=rep;r['prompt_ms']=float(r['done'].split()[3]);rows.append(r);print(arm,rep,r['prompt_ms'],r['accepted'],r['offered'],flush=True)
  control=e.request(ids,'temperature=1 top_k=1 top_p=1 min_p=0 seed=1234',32);assert control['offered']==0 and all(r['tokens']==control['tokens'] for r in rows)
  result[arm]=dict(rows=rows,control=control,info=e.info,median_prompt_ms=statistics.median(r['prompt_ms'] for r in rows if r['rep']>=0));(out/'long-tail.json').write_text(json.dumps(result,indent=2))
 finally:e.close()
for x,y in zip(result['full']['rows'],result['tail']['rows']):
 for k in ['tokens','accepted','offered']:assert x[k]==y[k],k
print('PASS',len(ids),'prompt change pct',100*(result['tail']['median_prompt_ms']/result['full']['median_prompt_ms']-1),flush=True)
