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
D=json.loads((BASE/'data/ALL-METRICS.json').read_text());M=D['models'];REF=D['hosted_api_reference'];HR=REF['human_eval_plus'];TR=REF['tool_eval_bench'];HFB=REF['five_bugs'];HKS=REF['sampled_knowledge_and_recall'];HPERF=REF['performance']
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
check('10 local files, 20 local effort arms, 18 local tool arms',len(M)==10 and sum(len(m['heplus']) for m in M.values())==20 and sum(len(m.get('trials')or{}) for m in M.values())==18)
check('Hosted GPT-6-luna role remains external-reference-only',REF['role']=='external_baseline_reference_only' and REF['recommendation'] is None and REF['eligible_for_local_composite'] is False and 'gpt-6-luna' not in M)
he_tables=[t for t in p.tables if t[0][:2]==['Arm / lab','First %']]
he_rows=[r for t in he_tables for r in t[1:]]
local_he_rows=[r for r in he_rows if not r[0].startswith('GPT-6-luna MAX')]
check('All 20 local coding arms printed exactly once',len(local_he_rows)==20 and len({r[0] for r in local_he_rows})==20)
he_host_rows=[r for r in he_rows if r[0].startswith('GPT-6-luna MAX')]
check('Hosted HE+ reference row matches source without placeholders',len(he_host_rows)==1 and len(he_host_rows[0])==10 and he_host_rows[0][1]==f'{HR["first_percent"]:.1f}' and he_host_rows[0][2]==f'{HR["final_percent"]:.1f}' and he_host_rows[0][3]==f'{HR["final_passes"]}/{len(HR["tasks"])}' and he_host_rows[0][4]==f'{HR["repair_successes"]}/{HR["repair_attempted"]}' and he_host_rows[0][5]==f'{int(round(HR["wall_seconds"]))//60}m{int(round(HR["wall_seconds"]))%60:02d}s' and he_host_rows[0][6]==f'{HR["provider_reported_tokens_per_second"]:.1f}†' and he_host_rows[0][7]==str(HR['blank_final_count']) and he_host_rows[0][8]==str(HR['harness_errors']) and 'OpenAI' in he_host_rows[0][9] and not any(re.search(r'(?i)\b(?:n/?a|missing|not available)\b',cell) for cell in he_host_rows[0]))
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
local_tr_rows=[r for r in tr_rows if not r[0].startswith('GPT-6-luna MAX')]
check('All 18 local tool arms printed exactly once',len(local_tr_rows)==18 and len({r[0] for r in local_tr_rows})==18)
tr_host_rows=[r for r in tr_rows if r[0].startswith('GPT-6-luna MAX')]
check('Hosted TEB row has all supported metrics and no placeholders',len(tr_host_rows)==1 and len(tr_host_rows[0])==7 and tr_host_rows[0][1]==f'{TR["mean"]:.1f} +/- {TR["stddev"]:.1f}' and tr_host_rows[0][2]==f'{TR["pass_at_5_percent"]:.1f}' and tr_host_rows[0][3]==f'{TR["pass_hat_5_percent"]:.1f}' and tr_host_rows[0][4]==f'{TR["reliability_gap_percent"]:.1f}' and tr_host_rows[0][5]==str(TR['final_trial_safety_warnings']) and 'OpenAI' in tr_host_rows[0][6] and not any(re.search(r'(?i)\b(?:n/?a|missing|not available)\b',cell) for cell in tr_host_rows[0]))
number=lambda s:float(re.search(r'-?\d+(?:\.\d+)?',s).group())
teb_order=[(number(r[3]),number(r[1])) for r in tr_rows]
check('TEB table places hosted reference inline by Pass^5, then mean',teb_order==sorted(teb_order,key=lambda x:(-x[0],-x[1])))
for k,m in M.items():
 for e,t in (m.get('trials')or{}).items():
    arr=[v['points'] for v in t['per_scenario'].values()]
    finals=[round(sum(x[i] for x in arr)/184*100) for i in range(5)]
    check(f'{k}/{e} tool mean, SD and 460 evaluations',len(arr)==92 and all(len(x)==5 for x in arr) and statistics.mean(finals)==t['mean'] and round(statistics.stdev(finals),1)==t['std'])
    check(f'{k}/{e} independently counted floor/ceiling',round(sum(all(x==2 for x in a) for a in arr)/92*100,1)==t['pass_hat_5'] and round(sum(any(x==2 for x in a) for a in arr)/92*100,1)==t['pass_at_5'])
    r=next(r for r in tr_rows if r[0]==f'{m["short"]} {m["lab"]} {e}')
    check(f'{k}/{e} printed tool headline and warnings',r[1].startswith(f'{t["mean"]:.1f} +/- {t["std"]:.1f}') and r[2]==f'{t["pass_at_5"]:.1f}' and r[3].split()[0]==f'{t["pass_hat_5"]:.1f}' and r[4]==f'{t["gap"]:.1f}' and int(r[5])==len(t['warnings']))
kn_tables=[t for t in p.tables if t[0][:2]==['File / lab','GSM8K /100']]
kn_rows=[r for t in kn_tables for r in t[1:]]
local_kn_rows=[r for r in kn_rows if not r[0].startswith('GPT-6-luna MAX')]
check('All 10 local knowledge/debug/needle rows printed once',len(local_kn_rows)==10)
kn_host_rows=[r for r in kn_rows if r[0].startswith('GPT-6-luna MAX')]
check('Hosted sampled-suite reference row matches sanitized source',len(kn_host_rows)==1 and kn_host_rows[0][1]==f'{HKS["gsm8k"]["accuracy_percent"]:.0f}' and kn_host_rows[0][2]==f'{HKS["mmlu"]["accuracy_percent"]:.0f}' and kn_host_rows[0][3]==f'{HKS["ifeval"]["prompt_accuracy_percent"]:.1f} / {HKS["ifeval"]["instruction_accuracy_percent"]:.1f}' and kn_host_rows[0][4]==f'{HFB["visible_passes"]}/{HFB["visible_total"]}' and kn_host_rows[0][5]==f'{HFB["hidden_passes"]}/{HFB["hidden_total"]}' and kn_host_rows[0][6]==f'{HKS["needle"]["accuracy_percent"]:.0f} @ {HKS["needle"]["effective_context_tokens"]:,}' and 'OpenAI' in kn_host_rows[0][7])
kn_order=[number(re.search(r'\d+',r[5]).group()) for r in kn_rows]
check('Sampled-suite table places hosted reference inline by hidden-debug score',kn_order==sorted(kn_order,reverse=True))
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
check('Reference baseline visible, provider identified, and non-recommendation',all(x in HTML for x in ['GPT-6-luna MAX','OpenAI','not a recommendation','external reference']))
check('Hosted sampling caveat and unavailable matched performance are explicit',REF['sampling_caveat'] in HTML and 'all hosted rates are withheld' in HTML and 'not visible-text throughput' in HTML)
hard_tables=[t for t in p.tables if t[0][:2]==['Arm / lab','Hard Mode mean']]
hard_rows=[r for t in hard_tables for r in t[1:]]
hard_host_rows=[r for r in hard_rows if r[0].startswith('GPT-6-luna MAX')]
check('Hard Mode table places hosted reference inline by mean score',len(hard_host_rows)==1 and 'OpenAI' in hard_host_rows[0][-1] and [number(r[1]) for r in hard_rows]==sorted([number(r[1]) for r in hard_rows],reverse=True))
hosted_html_rows=re.findall(r'<tr class="hosted-reference">(.*?)</tr>',HTML,re.S)
hosted_table_rows=he_host_rows+tr_host_rows+kn_host_rows+hard_host_rows
check('Hosted rows have no missing values, N/A placeholders, or medal markup',len(hosted_html_rows)>=4 and all(all(cell.strip() for cell in row) and not any(re.search(r'(?i)\b(?:n/?a|missing|not available)\b',cell) for cell in row) for row in hosted_table_rows) and all('class="med' not in row for row in hosted_html_rows))
he_order=[(int(r[3].split('/')[-1])==164 and int(r[8])==0,number(r[2])) for r in he_rows]
check('HumanEval+ table places hosted reference within score-ranked coverage tier',he_order==sorted(he_order,key=lambda x:(not x[0],-x[1])))
check('No emoji, stars or forced page breaks',not any(ord(c)>=0x1f300 or c=='★' for c in HTML) and 'page-break-before:always' not in HTML.replace(' ',''))
images=re.findall(r'<img src="([^"]+)"',HTML)
check('All 10 figures exist and are used',len(images)==10 and len(set(images))==10 and all((BASE/f).is_file() for f in images))
pdf=BASE/'strata-community.pdf';txt=subprocess.run(['pdftotext',str(pdf),'-'],capture_output=True,text=True,check=True).stdout
check('PDF contains local provenance plus hosted baseline scores and scope',all(x in txt for x in ['Gyro-M','completed_with_invalid_slices','156/164','157/164','153/162','205','129,024','GPT-6-luna MAX','OpenAI','95.7','92.4','75.0','20/25','not a recommendation','unavailable for matched decode comparison',D['gyro_m_campaign']['manifest']['engine_sha256']]))
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
