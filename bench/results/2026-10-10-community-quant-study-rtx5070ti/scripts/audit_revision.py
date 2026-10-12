#!/usr/bin/env python3
"""Independent numerical/content audit for the review revision."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib
import json
import math
import statistics
import subprocess
import re
import gzip
import numpy as np
from PIL import Image,ImageDraw

BASE=Path(__file__).resolve().parent.parent
D=json.loads((BASE/'data/ALL-METRICS.json').read_text());M=D['models']
HTML=(BASE/'community-updated.html').read_text()
CHECKS=[]
def check(label,test):
    CHECKS.append({'claim':label,'passed':bool(test)})
    print(('PASS' if test else 'FAIL'),label)
    if not test:raise AssertionError(label)

class Tables(HTMLParser):
    def __init__(self):super().__init__();self.tables=[];self.t=None;self.row=None;self.cell=None
    def handle_starttag(self,tag,attrs):
        if tag=='table':self.t=[]
        elif tag=='tr' and self.t is not None:self.row=[]
        elif tag in ('td','th') and self.row is not None:self.cell=''
    def handle_data(self,s):
        if self.cell is not None:self.cell+=s
    def handle_endtag(self,tag):
        if tag in ('td','th') and self.cell is not None:self.row.append(self.cell.strip());self.cell=None
        elif tag=='tr' and self.row is not None:self.t.append(self.row);self.row=None
        elif tag=='table':self.tables.append(self.t);self.t=None
p=Tables();p.feed(HTML)
p.tables=[[r for r in t if not re.match(r"^T\d{2} ·",r[0])] for t in p.tables]
check('10 files, 20 effort arms, 18 tool arms',len(M)==10 and sum(len(m['heplus']) for m in M.values())==20 and sum(len(m.get('trials')or{}) for m in M.values())==18)
he_tables=[t for t in p.tables if t[0][:2]==['Arm / lab','First %']]
he_rows=[r for t in he_tables for r in t[1:]]
check('All 20 coding arms printed exactly once in coding tables',len(he_rows)==20 and len({r[0] for r in he_rows})==20)
for k,m in M.items():
 for e,s in m['heplus'].items():
    hp=BASE/'data/heplus'/f'{k}-{e}.json'
    raw=json.loads(hp.read_text()) if hp.exists() else json.loads(gzip.decompress(hp.with_suffix('.json.gz').read_bytes()))
    ts=raw['tasks'];r=next(r for r in he_rows if r[0]==f'{m["short"]} {m["lab"]} {e}')
    check(f'{k}/{e} final/count/first/repairs/wall',r[1]==f'{raw["first_attempt_score"]:.1f}' and r[2].split()[0].strip('*')==f'{raw["score"]:.1f}' and r[3]==f'{raw["passed"]}/{raw["scored_total"]}' and r[4]==f'{raw["repair_successes"]}/{raw["repair_attempted"]}' and r[5].split()[0]==f'{int(round(raw["total_time"]))//60}m{int(round(raw["total_time"]))%60:02d}s')
    check(f'{k}/{e} raw score denominator',round(sum(t['passed'] is True for t in ts)/raw['scored_total']*100,1)==raw['score'])
    check(f'{k}/{e} blank-final count',int(r[7])==sum(not (t.get('response')or'').strip() for t in ts))
tr_tables=[t for t in p.tables if t[0][:2]==['Arm / lab','Mean +/- SD']]
tr_rows=[r for t in tr_tables for r in t[1:]]
check('All 18 tool arms printed exactly once',len(tr_rows)==18 and len({r[0] for r in tr_rows})==18)
for k,m in M.items():
 for e,t in (m.get('trials')or{}).items():
    arr=[v['points'] for v in t['per_scenario'].values()]
    finals=[round(sum(x[i] for x in arr)/184*100) for i in range(5)]
    check(f'{k}/{e} tool mean, SD and 460 evaluations',len(arr)==92 and all(len(x)==5 for x in arr) and statistics.mean(finals)==t['mean'] and round(statistics.stdev(finals),1)==t['std'])
    check(f'{k}/{e} independently counted floor/ceiling',round(sum(all(x==2 for x in a) for a in arr)/92*100,1)==t['pass_hat_5'] and round(sum(any(x==2 for x in a) for a in arr)/92*100,1)==t['pass_at_5'])
    r=next(r for r in tr_rows if r[0]==f'{m["short"]} {m["lab"]} {e}')
    check(f'{k}/{e} printed tool headline and warnings',r[1].startswith(f'{t["mean"]:.1f} +/- {t["std"]:.1f}') and r[3]==f'{t["pass_at_5"]:.1f}' and r[4].split()[0]==f'{t["pass_hat_5"]:.1f}' and r[5]==f'{t["gap"]:.1f}' and int(r[7])==len(t['warnings']))
kn_tables=[t for t in p.tables if t[0][:2]==['File / lab','GSM8K /100']]
kn_rows=[r for t in kn_tables for r in t[1:]]
check('All 10 knowledge/debug/needle rows printed once',len(kn_rows)==10)
for k,m in M.items():
    r=next(r for r in kn_rows if r[0]==f'{m["short"]} {m["lab"]}')
    check(f'{k} printed knowledge/debug/needle',r[1]==f'{m["knowledge"]["gsm8k"]["accuracy"]:.0f}' and r[2]==f'{m["knowledge"]["mmlu"]["accuracy"]:.0f}' and r[3]==f'{m["knowledge"]["ifeval"]["prompt_accuracy"]:.1f} / {m["knowledge"]["ifeval"]["instruction_accuracy"]:.1f}' and r[4]==m['fivebugs']['vis'] and r[5].split()[0]==m['fivebugs']['hid'] and r[6]==f'{m["needle"]["retrieval_accuracy"]:.0f} @ {int(m["needle"]["effective_context"]):,}')
perf_tables=[t for t in p.tables if t[0][:2]==['File / lab','Engine'] and 'Context tokens' in t[0][2]]
check('Both effort profiles share all 10 model rows',len(perf_tables)==2 and all(len(t)==11 for t in perf_tables))
for effort,t in zip(['low','xhigh'],perf_tables):
 for k,m in M.items():
    r=next(r for r in t[1:] if r[0]==f'{m["short"]} {m["lab"]}')
    pp=(m.get('perf')or{}).get(effort)
    if not pp:
        check(f'{k}/{effort} unavailable malformed profile labeled n/a',all(x=='n/a' for x in r[2:]));continue
    ps=sorted(pp.values(),key=lambda x:x['context_tokens'])
    check(f'{k}/{effort} all decode, first-token rates, depths and deep TTFT',r[2]==' / '.join(f"{x['context_tokens']:,.0f}" for x in ps) and r[3]==' / '.join(f"{x['tg_mean']:.1f}" for x in ps) and r[4]==' / '.join(f"{x['prefill_e2e_tps']:,.0f}" for x in ps) and r[5]==f"{ps[-1]['e2e_ttft_ms']/1000:.2f}")
check('No standalone Gyro-M chapter',not re.search(r'<h2[^>]*>[^<]*Gyro',HTML))
check('Coding, tool and knowledge tables integrate model rows',len(he_tables)==len(tr_tables)==len(kn_tables)==1)
registry=json.loads((BASE/'claims.json').read_text())['recommendations']
check('One recommendation registry with six aligned roles',len(registry)==6 and {x['role'] for x in registry}=={'all_rounder','reliability','fast_coding','coding_ceiling','knowledge_debug','experimental_not_production'})
# Verify composite independently; forks/partial must have neither a rank nor composite.
sb=json.loads((BASE/'data/FINAL-SCOREBOARD.json').read_text());axes=['coding','floor','hardmode','mmlu','gsm8k','debug','decode']
el=[r for r in sb if r['rank'] is not None]
check('Composite excludes all fork and partial arms',len(el)==14 and all(r['cohort']=='stock' and r['complete_scoring'] for r in el) and all(r['composite'] is None and r['rank'] is None for r in sb if r not in el))
for r in el:
    score=statistics.mean((r[a]-min(x[a] for x in el))/(max(x[a] for x in el)-min(x[a] for x in el))*100 for a in axes)
    check(f'Composite {r["key"]}/{r["arm"]}',math.isclose(score,r['composite'],abs_tol=1e-9))
check('Registry ranks match the scoreboard',all(x['rank']==next(r['rank'] for r in sb if r['key']==x['key'] and r['arm']==x['effort']) for x in registry))
check('First pick is the same all-rounder as the scoreboard',registry[0]['key']==el[0]['key'] and registry[0]['effort']==el[0]['arm'])
check('Composite rows ranked exactly',all(el[i]['composite']>=el[i+1]['composite'] for i in range(len(el)-1)) and [r['rank'] for r in el]==list(range(1,15)))
check('Outcome and critical caveats retained',all(x in HTML for x in ['completed_with_invalid_slices','HumanEval/39','HumanEval/99','205 MiB','78 C','ple_parity','157/163','153/162','155/163','2.7.1.dev16+gf7c34a130','129,024']))
check('Inflated prompt throughput not recommended', 'Prefill is huge' not in HTML and '239–430K' not in HTML and 'NOT an engine-kernel prefill' in HTML)
check('No fabricated placeholder values','see raw run ledger' not in HTML)
check('No emoji, stars or forced page breaks',not any(ord(c)>=0x1f300 or c=='★' for c in HTML) and 'page-break-before:always' not in HTML.replace(' ',''))
images=re.findall(r'<img src="([^"]+)"',HTML)
check('All 10 figures exist and are used',len(images)==10 and len(set(images))==10 and all((BASE/f).is_file() for f in images))
pdf=BASE/'strata-community.pdf';txt=subprocess.run(['pdftotext',str(pdf),'-'],capture_output=True,text=True,check=True).stdout
check('PDF contains new model, status, scoring denominators and provenance',all(x in txt for x in ['Gyro-M','completed_with_invalid_slices','156/164','157/164','153/162','205','129,024',D['gyro_m_campaign']['manifest']['engine_sha256']]))
info=subprocess.run(['pdfinfo',str(pdf)],capture_output=True,text=True,check=True).stdout
pages=int(re.search(r'Pages:\s+(\d+)',info).group(1));check('PDF has expected complete page inventory',pages==len(list((BASE/'qa').glob('page-*.png'))))
fill=[]
for path in sorted((BASE/'qa').glob('page-*.png')):
    im=np.array(Image.open(path).convert('L'));ys=np.flatnonzero((im<245).any(axis=1));fill.append(round(ys[-1]/im.shape[0],3))
check('No near-empty non-final pages',all(x>=.55 for x in fill[:-1]))
pdf_pages=txt.split('\f')
table_inventory=json.loads((BASE/'data/TABLE-INVENTORY.json').read_text())
table_locations={}
for tab in table_inventory:
    locations=[i+1 for i,page in enumerate(pdf_pages) if tab['id'] in page]
    table_locations[tab['id']]=locations
    check(f"Whole table {tab['id']} stays on one PDF page",len(locations)==1 and pdf_pages[locations[0]-1].count(tab['id'])==2)
patterns=[r'ghp_[A-Za-z0-9]{15,}',r'sk-[A-Za-z0-9_-]{20,}',r'Bearer\s+[A-Za-z0-9_-]{20,}']
check('No secret patterns in delivered PDF text/HTML',not any(re.search(x,txt+HTML) for x in patterns))
report={'checks':CHECKS,'passed':len(CHECKS),'failed':0,'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'pdf_bytes':pdf.stat().st_size,'pages':pages,'page_fill':fill,'figures':len(images),'table_pages':table_locations}
(BASE/'qa/AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
print('AUDIT_SUMMARY',json.dumps({k:report[k] for k in ['passed','failed','pages','page_fill','figures','pdf_sha256']}))
# Make 2-page final-QA panels, each page preserved at native rendered size.
paths=sorted((BASE/'qa').glob('page-*.png'))
for i in range(0,len(paths),2):
    ims=[Image.open(x).convert('RGB') for x in paths[i:i+2]];can=Image.new('RGB',(sum(x.width for x in ims)+12,max(x.height for x in ims)+20),'#dfe4e9');draw=ImageDraw.Draw(can);x=0
    for path,im in zip(paths[i:i+2],ims):draw.text((x+4,3),path.stem,fill='#17202a');can.paste(im,(x,20));x+=im.width+12
    can.save(BASE/'qa'/f'final-panel-{i//2+1}.png')
