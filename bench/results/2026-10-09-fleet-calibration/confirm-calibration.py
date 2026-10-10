from pathlib import Path
import json,sys,threading,statistics,time,os
# Put a matching Strata checkout at ./source and configs under ./<run-name>.
root=Path(__file__).resolve().parent
sys.path[:0]=[str(root/'source'),str(root/'source/tools')]
import calibrate as cal
import strata_tokenizer as st
from serve.server import StrataEngine,child_env
name=sys.argv[1];p=root/name
base=json.loads((p/'config.json').read_text());tuned=json.loads((p/'calibrated-config.json').read_text())
tp=Path(base['tokenizer']);v=json.loads((tp/'vocab.json').read_text());toks=[None]*len(v)
for t,i in v.items():toks[i]=t
tok=st.Tokenizer(toks,(tp/'merges.txt').read_text().split('\n'),json.loads((tp/'token_type.json').read_text()))
ids=[cal.chat_ids(tok,x) for x in cal.PROMPTS]
results=[]
for round,arm in enumerate(['default','tuned','tuned','default']):
 cfg=dict(base if arm=='default' else tuned)
 cfg['args']=cal.apply(base['args'],{}) if arm=='default' else tuned['args']
 eng=StrataEngine(cfg['exe'],cal.engine_args(cfg),cwd=cfg.get('cwd'),env=child_env(cfg),log=str(p/f'confirm-{round}-{arm}.log'))
 try:
  session=cal.Session(eng,ids);session.warm_up(1)
  samples=[]
  for prompt in ids:
   tokens=[x for x in eng.generate(prompt,512,{'temperature':0},threading.Event()) if x is not None]
   last=dict(eng.last or {})
   samples.append({'tokens':tokens,'last':last,'tok_s':len(tokens)*1000/last['decode_ms']})
  row={'round':round,'arm':arm,'samples':samples,'median_tok_s':statistics.median(x['tok_s'] for x in samples),'time':time.time()}
  results.append(row)
  (p/'confirmation.json').write_text(json.dumps({'complete':False,'rounds':results},indent=2))
  print(arm,row['median_tok_s'],flush=True)
 finally:cal.close(eng)
med={a:statistics.median(r['median_tok_s'] for r in results if r['arm']==a) for a in ['default','tuned']}
(p/'confirmation.json').write_text(json.dumps({'complete':True,'rounds':results,'medians':med,'gain_pct':100*(med['tuned']/med['default']-1)},indent=2))
print('COMPLETE',json.dumps(med),flush=True)
