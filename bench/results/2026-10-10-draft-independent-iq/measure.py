import json,os,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parent;os.chdir(r)
exe=str(r/'build/iq_avx2_parity') if (r/'build/iq_avx2_parity').exists() else str(r/'fixed-iq-parity')
results=[]
for label,extra in [('default',{}),('avx2',{'STRATA_NO_IQ512':'1'}),('ggml',{'STRATA_NO_IQ512':'1','STRATA_NO_IQ256':'1','STRATA_NO_IQ4NL':'1'}),('iq3s_mt1',{'STRATA_IQ3S_MT1':'1'}),('legacy',{'STRATA_IQ_MT_MIN':'2'})]:
 env=os.environ.copy();env.pop('STRATA_IQ_MT_MIN',None);env.update(extra)
 with (r/('test-'+label+'.log')).open('w') as f: p=subprocess.run([exe],env=env,stdout=f,stderr=subprocess.STDOUT)
 results.append(dict(label=label,code=p.returncode,expected=1 if label=='legacy' else 0))
for rep in range(3):
 for label,extra in ([('default',{}),('legacy',{'STRATA_IQ_MT_MIN':'2'})] if rep%2==0 else [('legacy',{'STRATA_IQ_MT_MIN':'2'}),('default',{})]):
  env=os.environ.copy();env.pop('STRATA_IQ_MT_MIN',None);env.update(extra)
  with (r/(f'bench-{rep}-{label}.log')).open('w') as f:p=subprocess.run([exe,'--bench','--dispatch','--cpu','2','--mb','128','--reps','3','--nt','1,2,4','--pairs','iq2_xxs/iq4_nl,iq2_xs/iq4_nl,iq2_s/iq4_nl,iq3_xxs/iq4_nl,iq3_s/iq4_nl,iq4_xs/iq4_nl'],env=env,stdout=f,stderr=subprocess.STDOUT)
  results.append(dict(label=f'bench-{rep}-{label}',code=p.returncode,expected=0))
(r/'measurement-status.json').write_text(json.dumps(results,indent=2))
