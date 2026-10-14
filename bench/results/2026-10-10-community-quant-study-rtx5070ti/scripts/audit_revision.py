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
D=json.loads((BASE/'data/ALL-METRICS.json').read_text());M=D['models'];REF=D['hosted_api_reference'];SOL_REF=D['hosted_api_reference_sol'];HR=REF['human_eval_plus'];TR=REF['tool_eval_bench'];HFB=REF['five_bugs'];HKS=REF['sampled_knowledge_and_recall'];HPERF=REF['performance'];SHR=SOL_REF['human_eval_plus'];STR=SOL_REF['tool_eval_bench'];SFB=SOL_REF['five_bugs'];SHKS=SOL_REF['sampled_knowledge_and_recall'];SPERF=SOL_REF['performance']
HTML=(BASE/'community-updated.html').read_text();README=(BASE/'README.md').read_text()
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
check('10 local files, 20 local effort arms, 20 local tool arms',len(M)==10 and sum(len(m['heplus']) for m in M.values())==20 and sum(len(m.get('trials')or{}) for m in M.values())==20)
check('Both hosted identities remain external-reference-only',REF['role']=='external_baseline_reference_only' and REF['recommendation'] is None and REF['eligible_for_local_composite'] is False and SOL_REF['role']=='external_baseline_reference_only' and SOL_REF['recommendation'] is None and SOL_REF['eligible_for_local_composite'] is False and 'gpt-6-luna' not in M and 'gpt-6.1-sol' not in M)
check('First-upload GPT-6.1-SOL source and Five Bugs selection are pinned',SOL_REF['source_commit']=='a48292f5766f66efcdb1e23396569b0907c44c04' and SOL_REF['source_score_sha256']=='bb040b5e27d0602730638dc3f511a02f1f9e9abd0b46969c543becffd7b1d7d8' and SFB['visible_passes']==25 and SFB['visible_total']==25 and SFB['hidden_passes']==20 and SFB['hidden_total']==25 and SFB['post_feedback_validation_included'] is False and SPERF['valid'] is False)
he_tables=[t for t in p.tables if t[0][:2]==['Arm / lab','First %']]
he_rows=[r for t in he_tables for r in t[1:]]
HOSTED_NAMES=['GPT-6-luna MAX','GPT-6.1-SOL ultra']
def hosted_row(r):return any(r[0].startswith(n) for n in HOSTED_NAMES)
local_he_rows=[r for r in he_rows if not hosted_row(r)]
check('All 20 local coding arms printed exactly once',len(local_he_rows)==20 and len({r[0] for r in local_he_rows})==20)
he_host_rows=[r for r in he_rows if hosted_row(r)]
check('Both hosted HE+ rows match their source, omit incomparable fields and contain no placeholders',len(he_host_rows)==2 and all(len(r)==7 and not any(re.search(r'(?i)\b(?:n/?a|missing|not available)\b',cell) for cell in r) for r in he_host_rows))
for prefix,metric in [('GPT-6-luna MAX',HR),('GPT-6.1-SOL ultra',SHR)]:
 r=next((r for r in he_host_rows if r[0].startswith(prefix)),[]);total=metric.get('task_total',len(metric.get('tasks',[])));excluded=metric.get('infra_excluded',metric.get('harness_errors',0))
 check(f'{prefix} HE+ fields and engine provenance',len(r)==7 and r[1]==f'{metric["first_percent"]:.1f}' and r[2]==f'{metric["final_percent"]:.1f}' and r[3]==f'{metric["final_passes"]}/{total}' and r[4]==f'{metric["repair_successes"]}/{metric["repair_attempted"]}' and r[5]==str(excluded) and 'OpenAI' in r[6])
for k,m in M.items():
 for e,s in m['heplus'].items():
    hp=BASE/'data/heplus'/f'{k}-{e}.json'
    raw=json.loads(hp.read_text()) if hp.exists() else json.loads(gzip.decompress(hp.with_suffix('.json.gz').read_bytes()))
    ts=raw['tasks'];r=next(r for r in he_rows if r[0]==f'{m["short"]} {m["lab"]} {e}')
    check(f'{k}/{e} final/count/first/repairs/infra',r[1]==f'{raw["first_attempt_score"]:.1f}' and r[2].split()[0].strip('*')==f'{raw["score"]:.1f}' and r[3]==f'{raw["passed"]}/{raw["scored_total"]}' and r[4]==f'{raw["repair_successes"]}/{raw["repair_attempted"]}' and r[5]==str(raw.get('harness_errors',0)))
    check(f'{k}/{e} raw score denominator',round(sum(t['passed'] is True for t in ts)/raw['scored_total']*100,1)==raw['score'])
    blank_tasks=[t for t in ts if not (t.get('response')or'').strip()]
    harness_failures=sum(bool(t.get('harness_error')) for t in ts)
    check(f'{k}/{e} blank-final failures stay scored and denominator matches harness exclusions',all(t.get('passed') is not True for t in blank_tasks) and raw['scored_total']==len(ts)-harness_failures and raw.get('harness_errors',0)==harness_failures)
tr_tables=[t for t in p.tables if t[0][:2]==['Arm / lab','Mean +/- SD']]
tr_rows=[r for t in tr_tables for r in t[1:]]
local_tr_rows=[r for r in tr_rows if not hosted_row(r)]
check('All 20 local tool arms printed exactly once',len(local_tr_rows)==20 and len({r[0] for r in local_tr_rows})==20)
tr_host_rows=[r for r in tr_rows if hosted_row(r)]
check('Both hosted TEB rows have every common metric and no placeholders',len(tr_host_rows)==2 and all(len(r)==7 and not any(re.search(r'(?i)\b(?:n/?a|missing|not available)\b',cell) for cell in r) for r in tr_host_rows))
def hosted_floor(metric):return metric.get('pass_hat_5_percent',metric.get('pass_all5_percent'))
for prefix,metric in [('GPT-6-luna MAX',TR),('GPT-6.1-SOL ultra',STR)]:
 r=next((r for r in tr_host_rows if r[0].startswith(prefix)),[])
 check(f'{prefix} TEB common values and engine label',len(r)==7 and r[1]==f'{metric["mean"]:.1f} +/- {metric["stddev"]:.1f}' and r[2]==f'{metric["pass_at_5_percent"]:.1f}' and r[3]==f'{hosted_floor(metric):.1f}' and r[4]==f'{metric["reliability_gap_percent"]:.1f}' and r[5]==str(metric['final_trial_safety_warnings']) and 'OpenAI' in r[6])
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
local_kn_rows=[r for r in kn_rows if not hosted_row(r)]
check('All 10 local knowledge/debug/needle rows printed once',len(local_kn_rows)==10)
kn_host_rows=[r for r in kn_rows if hosted_row(r)]
check('Both hosted sampled-suite rows match their source and keep Needle token bases explicit',len(kn_host_rows)==2 and all(len(r)==8 for r in kn_host_rows))
for prefix,ref in [('GPT-6-luna MAX',REF),('GPT-6.1-SOL ultra',SOL_REF)]:
 r=next((r for r in kn_host_rows if r[0].startswith(prefix)),[]);hks=ref['sampled_knowledge_and_recall'];hfb=ref['five_bugs'];needle=hks['needle']
 needle_text=f'{needle["accuracy_percent"]:.0f} @ {needle["actual_max_input_tokens"]:,} actual' if prefix.startswith('GPT-6.1') else f'{needle["accuracy_percent"]:.0f} @ {needle["effective_context_tokens"]:,} effective'
 check(f'{prefix} sampled-suite values and engine provenance',len(r)==8 and r[1]==f'{hks["gsm8k"]["accuracy_percent"]:.0f}' and r[2]==f'{hks["mmlu"]["accuracy_percent"]:.0f}' and r[3]==f'{hks["ifeval"]["prompt_accuracy_percent"]:.1f} / {hks["ifeval"]["instruction_accuracy_percent"]:.1f}' and r[4]==f'{hfb["visible_passes"]}/{hfb["visible_total"]}' and r[5]==f'{hfb["hidden_passes"]}/{hfb["hidden_total"]}' and r[6]==needle_text and 'OpenAI' in r[7])
kn_order=[number(re.search(r'\d+',r[5]).group()) for r in kn_rows]
check('Sampled-suite table places both hosted references inline by hidden-debug score',kn_order==sorted(kn_order,reverse=True))
kn_top=max(kn_order);kn_top_rows=[r for r in kn_rows if number(re.search(r'\d+',r[5]).group())==kn_top]
check('Knowledge/debugging top is the disclosed 20/25 three-way tie, with local-first deterministic order',kn_top==20 and len(kn_top_rows)==3 and kn_top_rows[0][0].startswith('S-IQ3_S') and kn_top_rows[1][0].startswith('GPT-6-luna MAX') and kn_top_rows[2][0].startswith('GPT-6.1-SOL ultra') and 'Top hidden Five Bugs score: 20/25, tied among' in HTML and 'not a unique win' in HTML and 'top score is tied at 20/25' in README and all(name in HTML and name in README for name in ['S-IQ3_S','GPT-6-luna MAX','GPT-6.1-SOL ultra']))
for k,m in M.items():
    r=next(r for r in kn_rows if r[0]==f'{m["short"]} {m["lab"]}')
    check(f'{k} printed knowledge/debug/needle',r[1]==f'{m["knowledge"]["gsm8k"]["accuracy"]:.0f}' and r[2]==('n/a' if m['knowledge']['mmlu']['accuracy'] is None else f'{m["knowledge"]["mmlu"]["accuracy"]:.0f}') and r[3]==f'{m["knowledge"]["ifeval"]["prompt_accuracy"]:.1f} / {m["knowledge"]["ifeval"]["instruction_accuracy"]:.1f}' and r[4]==m['fivebugs']['vis'] and r[5].split()[0]==m['fivebugs']['hid'] and r[6]==f'{m["needle"]["retrieval_accuracy"]:.0f} @ {int(m["needle"]["effective_context"]):,} effective')
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
check('One recommendation registry with seven aligned roles',len(registry)==7 and {x['role'] for x in registry}=={'all_rounder','fastest_total_wall','reliability','fast_coding','coding_ceiling','knowledge_debug','experimental_not_production'})
# Comprehensive decision matrix: validate raw evidence, no arbitrary aggregate.
sb=json.loads((BASE/'data/FINAL-SCOREBOARD.json').read_text());policy=json.loads((BASE/'data/DAILY-DRIVER-POLICY.json').read_text())
check('All 20 unique local effort rows appear without score or numeric rank',len(sb)==20 and len({(r['key'],r['arm']) for r in sb})==20 and all(not any(k in r for k in ['rank','composite','overall','score']) for r in sb))
check('Decision policy has no weights and identifies stock default with broad coverage',policy['weights'] is None and policy['default']=={'key':'swift-iq3_s','effort':'low'} and M['swift-iq3_s']['cohort']=='stock')
for r in sb:
 m=M[r['key']];he=m['heplus'][r['arm']];t=m['trials'][r['arm']]
 check(f'Matrix measured coding/tool evidence {r["key"]}/{r["arm"]}',r['coding_first']==he['first_attempt_score'] and r['coding_final']==he['score'] and r['coding_scored']==164 and r['coding_wall_s']==he['total_time'] and r['coding_output_tokens']==he['total_output_tokens'] and r['tool_mean']==t['mean'] and r['tool_floor']==t['pass_hat_5'] and r['tool_ceiling']==t['pass_at_5'] and r['hardmode']==t['cats']['P'] and r['warning_count_last_trial']==t['n_warnings'])
 sample=r['sampled'] or r['legacy_sampled']
 check(f'Matrix sampled domain evidence {r["key"]}/{r["arm"]}',sample['gsm8k']==m['knowledge']['gsm8k']['accuracy'] and sample['algebra']==m['knowledge']['mmlu']['accuracy'] and sample['ifeval_prompt']==m['knowledge']['ifeval']['prompt_accuracy'] and sample['ifeval_instruction']==m['knowledge']['ifeval']['instruction_accuracy'] and sample['debug_visible']==m['fivebugs']['vis'] and sample['debug_hidden']==m['fivebugs']['hid'] and sample['needle_accuracy']==m['needle']['retrieval_accuracy'] and sample['needle_context_tokens']==m['needle']['effective_context'])
cap_table=next(t for t in p.tables if t[0][:2]==['Setup / effort','Code first/final %'])
deploy_table=next(t for t in p.tables if t[0][:2]==['Setup / effort','Code wall'])
for r in sb:
 label=f'{M[r["key"]]["short"]} {M[r["key"]]["lab"]} {r["arm"]}'
 row=next(x for x in cap_table[1:] if x[0]==label);sample=r['sampled'] or r['legacy_sampled'];mark='†' if r['legacy_sampled'] else ''
 expected=[f'{r["coding_first"]:.1f} / {r["coding_final"]:.1f}',f'{r["tool_mean"]:.1f}±{r["tool_sd"]:.1f} / {r["tool_floor"]:.1f}',f'{r["hardmode"]:.1f}',f'{sample["gsm8k"]:.0f} / '+('n/a' if sample['algebra'] is None else f'{sample["algebra"]:.0f}')+mark,f'{sample["ifeval_prompt"]:.1f} / {sample["ifeval_instruction"]:.1f}'+mark,sample['debug_visible']+' / '+sample['debug_hidden']+mark,f'{sample["needle_accuracy"]:.0f} @ {int(sample["needle_context_tokens"]):,}'+mark,f'{round(r["total_wall_s"])//60}m{round(r["total_wall_s"])%60:02d}s']
 check(f'Printed all-test capability row {r["key"]}/{r["arm"]}',row[1:]==expected)
 tw=round(r['total_wall_s']);dw=round(r['sampled']['debug_wall_s']);cw=round(r['coding_wall_s']);tow=round(r['tool_wall_s'])
 check(f'Speed total wall = code+debug+tool {r["key"]}/{r["arm"]}',abs(tw-(cw+dw+tow))<=2 and row[8]==f'{tw//60}m{tw%60:02d}s')
 wall_row=next(x for x in deploy_table[1:] if x[0]==label);v=round(r['coding_wall_s'])
 check(f'Printed measured coding wall {r["key"]}/{r["arm"]}',wall_row[1]==f'{v//60}m{v%60:02d}s' and wall_row[4]==f'{r["coding_output_tokens"]:,.0f}')
check('Gyro-S valid rc2 auxiliary evidence imported; algebra scored offline, rc1 preserved',all(r['sampled'] is not None and r['sampled']['gsm8k']==96.0 and r['sampled']['ifeval_prompt']==76.0 and r['sampled']['algebra']==79.0 and r['legacy_sampled'] is None for r in sb if r['key']=='gyro-s') and M['gyro-s']['legacy_rc1_auxiliary']['knowledge']['mmlu']['accuracy']==76.0)
check('One default, experimental forks not recommended as daily defaults',[(r['key'],r['arm']) for r in sb if r['verdict']=='Recommended default']==[('swift-iq3_s','low')] and all(r['verdict']=='Experimental; not default' for r in sb if r['cohort']!='stock'))
check('Recommendation registry contains no overall ranks',all('rank' not in r for r in registry) and registry[0]['key']=='swift-iq3_s' and registry[0]['effort']=='low')
check('Capability and deployment matrices include all 20 setups',any(t[0][:2]==['Setup / effort','Code first/final %'] and len(t)==21 for t in p.tables) and any(t[0][:2]==['Setup / effort','Code wall'] and len(t)==21 for t in p.tables))
check('Outcome and critical caveats retained',all(x in HTML for x in ['completed_with_invalid_slices','HumanEval/39','HumanEval/99','205 MiB','78 C','ple_parity','157/163','153/162','155/163','2.7.1.dev16+gf7c34a130','129,024']))
check('Inflated prompt throughput not recommended', 'Prefill is huge' not in HTML and '239–430K' not in HTML and 'not kernel prefill' in HTML and 'Gyro-M depth-zero decode is descriptive only' not in HTML)
check('No fabricated placeholder values','see raw run ledger' not in HTML)
check('Both hosted references visible, provider-identified, and outside local ranks',all(x in HTML for x in ['GPT-6-luna MAX','GPT-6.1-SOL ultra','OpenAI','External quality anchors, not picks','excluded from local counts, ranks, recommendations and the OVERALL composite']))
check('Both hosted sampling caveats, score-matrix limitation and unavailable matched performance are explicit',REF['sampling_caveat'] in HTML and SOL_REF['sampling_caveat'] in HTML and 'matched hosted rates remain withheld' in HTML and 'not visible-text throughput' in HTML and 'No SOL decode-performance measures exist' in HTML and 'TC-56 timeout was recovered' in HTML and 'no per-task outcome matrix' in HTML and 'a48292f' in HTML)
hard_tables=[t for t in p.tables if t[0][:2]==['Arm / lab','Hard Mode mean']]
hard_rows=[r for t in hard_tables for r in t[1:]]
hard_host_rows=[r for r in hard_rows if hosted_row(r)]
check('Hard Mode table places both hosted references inline by mean score',len(hard_host_rows)==2 and all('OpenAI' in r[-1] for r in hard_host_rows) and [number(r[1]) for r in hard_rows]==sorted([number(r[1]) for r in hard_rows],reverse=True))
hosted_html_rows=re.findall(r'<tr class="hosted-reference">(.*?)</tr>',HTML,re.S)
hosted_table_rows=he_host_rows+tr_host_rows+kn_host_rows+hard_host_rows
check('Hosted rows have no missing values, N/A placeholders, or medal markup',len(hosted_html_rows)>=8 and all(all(cell.strip() for cell in row) and not any(re.search(r'(?i)\b(?:n/?a|missing|not available)\b',cell) for cell in row) for row in hosted_table_rows) and all('class="med' not in row for row in hosted_html_rows))
he_order=[(int(r[3].split('/')[-1])==164 and int(r[5])==0,number(r[2])) for r in he_rows]
check('HumanEval+ table places both hosted references within score-ranked coverage tier',he_order==sorted(he_order,key=lambda x:(not x[0],-x[1])))
check('No emoji, stars or forced page breaks',not any(ord(c)>=0x1f300 or c=='★' for c in HTML) and 'page-break-before:always' not in HTML.replace(' ',''))
images=re.findall(r'<img src="([^"]+)"',HTML)
check('All 10 figures exist and are used',len(images)==10 and len(set(images))==10 and all((BASE/f).is_file() for f in images))
pdf=BASE/'strata-community.pdf';txt=subprocess.run(['pdftotext',str(pdf),'-'],capture_output=True,text=True,check=True).stdout
check('PDF contains local provenance plus both hosted scores, first-upload provenance and scope',all(x in ' '.join(txt.split()) for x in ['Gyro-M','completed_with_invalid_slices','156/164','157/164','153/162','205','129,024','GPT-6-luna MAX','GPT-6.1-SOL ultra','OpenAI','95.7','92.4','75.0','95.1','97.6','90.2','83.7','20/25','25/25','a48292f','neither hosted result supplies a matched decode measurement','not recommendations or',D['gyro_m_campaign']['manifest']['engine_sha256']]))
check('PDF explicitly explains the sampled-suite hidden-score tie',all(x in ' '.join(txt.split()) for x in ['Top hidden Five Bugs score: 20/25, tied among','not a unique win']))
info=subprocess.run(['pdfinfo',str(pdf)],capture_output=True,text=True,check=True).stdout
pages=int(re.search(r'Pages:\s+(\d+)',info).group(1));check('PDF has expected complete page inventory',pages==len(list((BASE/'qa').glob('page-*.png'))))
fill=[]
for path in sorted((BASE/'qa').glob('page-*.png')):
    im=np.array(Image.open(path).convert('L'));ys=np.flatnonzero((im<245).any(axis=1));fill.append(round(ys[-1]/im.shape[0],3))
# Full-width tables/scoreboards can end around mid-page without being sparse.
check('No near-empty non-final pages',all(x>=.50 for x in fill[:-1]))
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
(BASE/'data/independent-audit.json').write_text(json.dumps({'method':'Current audit_revision.py: independently counted raw coding denominators and 92x5 tool points, validated every matrix domain against canonical measurements, source identity, coverage and public PDF tables; no master score.','result':'passed','audited_coding_arms':20,'audited_tool_arms':20,'partial_humaneval_arms':D['inventory']['partial_he_arms'],'daily_driver_policy':policy,'checks':CHECKS,'pdf_sha256':report['pdf_sha256']},indent=2)+'\n')
print('AUDIT_SUMMARY',json.dumps({k:report[k] for k in ['passed','failed','pages','page_fill','figures','pdf_sha256']}))
# Make 2-page final-QA panels, each page preserved at native rendered size.
paths=sorted((BASE/'qa').glob('page-*.png'))
for i in range(0,len(paths),2):
    ims=[Image.open(x).convert('RGB') for x in paths[i:i+2]];can=Image.new('RGB',(sum(x.width for x in ims)+12,max(x.height for x in ims)+20),'#dfe4e9');draw=ImageDraw.Draw(can);x=0
    for path,im in zip(paths[i:i+2],ims):draw.text((x+4,3),path.stem,fill='#17202a');can.paste(im,(x,20));x+=im.width+12
    can.save(BASE/'qa'/f'final-panel-{i//2+1}.png')
