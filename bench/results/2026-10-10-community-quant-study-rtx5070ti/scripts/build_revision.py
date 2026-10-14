#!/usr/bin/env python3
"""Portable review compiler: all metrics from data/ALL-METRICS.json; no rig access."""
from pathlib import Path
import datetime
import html
import json
import math
import re
import subprocess
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE/'scripts'))
from report_eligibility import decode_axis_status,gyro_s_scoring_gate
D=json.loads((BASE/'data/ALL-METRICS.json').read_text())
M=D['models']; C=D['gyro_m_campaign']; G=M['gyro-m']; CH=BASE/'charts'; CH.mkdir(exist_ok=True)
S_RUN=D['gyro_s_campaign']; S_ID=M['gyro-s']['setup_identity']; M_ID=C['manifest']; S_GATE=gyro_s_scoring_gate(S_RUN)
assert S_GATE['eligible'], '; '.join(S_GATE['reasons'])
assert M['gyro-s']['cohort']=='rc2' and M['gyro-s']['auxiliary_cohort']=='rc2'
assert S_ID['engine_sha256']==S_RUN['manifest']['engine_sha256']==M_ID['engine_sha256']
assert S_ID['engine_commit']==S_RUN['manifest']['engine_commit']==M_ID['engine_commit']
assert S_ID['model_sha256']==S_RUN['manifest']['model_sha256'] and S_ID['model_revision']==S_RUN['manifest']['model_revision']
assert S_ID['context']==M_ID['context']==131072 and S_ID['kv']==M_ID['kv']=='int8'
assert M['gyro-m']['cohort']=='rc2' and M['gyro-m']['size_gb']==91965774272/1e9
for effort in ['low','xhigh']:
    assert M['gyro-s']['heplus'][effort]['audit']['complete_scoring']
    assert len(M['gyro-s']['trials'][effort]['per_scenario'])==92
    assert len(M['gyro-s']['perf'][effort])==3
REF=D['hosted_api_reference']; REF_DETAIL=json.loads((BASE/REF['artifact']).read_text())
SOL_REF=D['hosted_api_reference_sol']; SOL_DETAIL=json.loads((BASE/SOL_REF['artifact']).read_text())
HR=REF['human_eval_plus']; TR=REF['tool_eval_bench']; HFB=REF['five_bugs']; HKS=REF['sampled_knowledge_and_recall']; HPERF=REF['performance']
SHR=SOL_REF['human_eval_plus']; STR=SOL_REF['tool_eval_bench']; SFB=SOL_REF['five_bugs']; SHKS=SOL_REF['sampled_knowledge_and_recall']; SPERF=SOL_REF['performance']
for hosted_ref in [REF,SOL_REF]:
    assert hosted_ref['role']=='external_baseline_reference_only' and hosted_ref['eligible_for_local_composite'] is False and hosted_ref['recommendation'] is None
HOSTED_REFERENCES=[
    {'key':'hosted-reference-luna','label':'GPT-6-luna MAX (OpenAI) · hosted API reference','name':'GPT-6-luna MAX','engine':'OpenAI Codex Responses API · max','color':'#7B2CBF','ref':REF,'detail':REF_DETAIL},
    {'key':'hosted-reference-sol','label':'GPT-6.1-SOL ultra (OpenAI) · hosted Codex reference','name':'GPT-6.1-SOL ultra','engine':'OpenAI Codex CLI 0.162.1 · ultra','color':'#9B59B6','ref':SOL_REF,'detail':SOL_DETAIL},
]
HOSTED_BY_KEY={x['key']:x for x in HOSTED_REFERENCES}
HOSTED_NAMES=[x['name'] for x in HOSTED_REFERENCES]

ORDER=['base-iq3_xxs','q2_0','iq3_s','ista-iq2_xs','swift-iq3_xxs','swift-iq2_xs','swift-iq3_s','swift-q2_0','gyro-s','gyro-m']
STOCK=[k for k in ORDER if M[k]['cohort']=='stock']
COLOR={'ISTA-DASLab':'#0072B2','UkisAI':'#D55E00','AgentionAI':'#009E73'}
SRC={'ISTA-DASLab':'https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF','UkisAI':'https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF','AgentionAI':'https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF'}
plt.rcParams.update({'font.size':9,'figure.dpi':160,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
from decision_matrix import build_matrix, POLICY, RECOMMENDATIONS

def save(name,fig):
    fig.savefig(CH/name,bbox_inches='tight',facecolor='white'); plt.close(fig)

def legend(ax,fig,handles=None,ncol=3,bottom=.24):
    ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.16),ncol=ncol,frameon=False,fontsize=8)
    fig.subplots_adjust(bottom=bottom)

def label(k):return M[k]['short']
def hlabel(k):return f"{M[k]['short']} ({M[k]['lab']})"
def clr(k):return COLOR[M[k]['lab']]
def f(v,n=1):return f'{v:,.{n}f}' if isinstance(v,(int,float)) else 'n/a'
def wall(s):return f'{int(round(s))//60}m{int(round(s))%60:02d}s'
def escape(x):return html.escape(str(x))
def name(k):return f'<span class="dot" style="background:{clr(k)}"></span><b>{label(k)}</b> <span class="lab" style="color:{clr(k)}">{M[k]["lab"]}</span>'
def medal(v,values,lower=False):
    levels=sorted(set(x for x in values if isinstance(x,(int,float))),reverse=not lower)[:3]
    if v not in levels:return ''
    r=levels.index(v);return f'<span class="med {"gsb"[r]}">{r+1}</span>'
def rating(t):
    r=t['rating']; stars=r.count('★');return f'{stars}/5 '+('safety-capped' if 'safety-capped' in r else ('Good' if 'Good' in r else 'Excellent'))

# Comprehensive daily-driver evidence matrix: no synthetic overall score.
SB=build_matrix(D)
(BASE/'data/FINAL-SCOREBOARD.json').write_text(json.dumps(SB,indent=2)+'\n')
(BASE/'data/DAILY-DRIVER-POLICY.json').write_text(json.dumps(POLICY,indent=2)+'\n')
FULL=[(k,e,r) for k,m in M.items() for e,r in m['heplus'].items() if r['audit']['complete_scoring']]
FAST=min(FULL,key=lambda x:x[2]['total_time'])
MAXHE=max(r['score'] for _,_,r in FULL);BESTHE=[(k,e) for k,e,r in FULL if r['score']==MAXHE]
localtools=[(k,e,t) for k in ORDER for e,t in (M[k].get('trials')or{}).items()]
stocktools=[(k,e,t) for k in STOCK for e,t in M[k]['trials'].items()]
BESTMEAN=max(stocktools,key=lambda x:x[2]['mean']);BESTFLOOR=max(stocktools,key=lambda x:x[2]['pass_hat_5'])
hewallratio=G['heplus']['xhigh']['total_time']/G['heplus']['low']['total_time']
hetokenratio=G['heplus']['xhigh']['total_output_tokens']/G['heplus']['low']['total_output_tokens']
claims={'inventory':D['inventory'],'daily_driver_policy':POLICY,'recommendations':RECOMMENDATIONS,'analysis_only_axes':{'gyro-s':['legacy_rc1_auxiliary'],'reason':'Gyro-S valid rc2 math, instructions, debugging, retrieval and algebra are imported. Legacy rc1 artifacts are archived, not substituted.'},'fastest_full_he':{'key':FAST[0],'effort':FAST[1],'wall_s':FAST[2]['total_time']},'best_complete_he':{'score':MAXHE,'arms':BESTHE},'stock_best_mean':{'key':BESTMEAN[0],'effort':BESTMEAN[1],'mean':BESTMEAN[2]['mean']},'stock_best_floor':{'key':BESTFLOOR[0],'effort':BESTFLOOR[1],'floor':BESTFLOOR[2]['pass_hat_5']},'gyro_s_scoring_gate':S_GATE,'partial_he_arms':D['inventory']['partial_he_arms']}
claims['gyro_m_speed_attribution']=C['performance_attribution_audit']
claims['hosted_api_reference']={'display_name':REF['display_name'],'run_id':REF['run_id'],'role':REF['role'],'recommendation':None,'eligible_for_local_composite':False,'human_eval_plus':{'first_percent':HR['first_percent'],'final_percent':HR['final_percent'],'solved':f"{HR['final_passes']}/{len(HR['tasks'])}",'repairs':f"{HR['repair_successes']}/{HR['repair_attempted']}",'provider_reported_tokens_per_second':HR['provider_reported_tokens_per_second'],'token_rate_caveat':HR['token_rate_caveat']},'tool_eval_bench':{'mean':TR['mean'],'stddev':TR['stddev'],'pass_at_5_percent':TR['pass_at_5_percent'],'pass_hat_5_percent':TR['pass_hat_5_percent'],'hard_mode_mean_percent':TR['hard_mode_mean_percent']},'performance_status':HPERF['status']}
claims['hosted_api_references']=[{'display_name':x['ref']['display_name'],'run_id':x['ref']['run_id'],'role':x['ref']['role'],'recommendation':None,'eligible_for_local_composite':False,'source_commit':x['ref'].get('source_commit')} for x in HOSTED_REFERENCES]
(BASE/'claims.json').write_text(json.dumps(claims,indent=2)+'\n')

# F1 coding results: one ranked row per effort arm plus the hosted references.
coding_entries=[(k,e,M[k]['heplus'][e],None) for k in ORDER for e in ['low','xhigh']]+[(h['key'],'hosted',h['ref']['human_eval_plus'],h) for h in HOSTED_REFERENCES]
def _coding_sort_key(x):
 k,e,r,hosted=x
 complete=(r.get('infra_excluded',r.get('harness_errors',0))==0 and r.get('task_total',len(r.get('tasks',[])))==164) if hosted else r['audit']['complete_scoring']
 return (not complete,-(r['final_percent'] if hosted else r['score']))
coding_entries.sort(key=_coding_sort_key)
fig,ax=plt.subplots(figsize=(10.6,9.0))
coding_labels=[]
for i,(k,e,r,hosted) in enumerate(coding_entries):
 if hosted:
  score=r['final_percent'];c=hosted['color'];label_text=hosted['label'];complete=True;hatch=None;alpha=.96
 else:
  score=r['score'];c=clr(k);complete=r['audit']['complete_scoring'];hatch='//' if e=='xhigh' else None;alpha=.92 if e=='low' else .58
  cohort_tag=' [rc2]' if M[k]['cohort']=='rc2' else ''
  label_text=f'{label(k)} {e} · {M[k]["lab"]}{cohort_tag}'+('*' if not complete else '')
 ax.barh(i,score,height=.68,color=c if complete else 'white',edgecolor=c,alpha=alpha,hatch=hatch)
 ax.text(score+.25,i,f'{score:.1f}',va='center',fontsize=9.5,fontweight='bold')
 coding_labels.append(label_text)
ax.set_yticks(range(len(coding_entries)));ax.set_yticklabels(coding_labels,fontsize=9.5);ax.invert_yaxis();ax.set_xlim(0,100)
ax.set_xlabel('Final reported pass rate (%)',fontsize=11);ax.set_title('HumanEval+: complete coverage first, then sorted by final score',fontsize=12)
legend(ax,fig,[Patch(color='#526778',label='low effort'),Patch(color='#526778',alpha=.58,hatch='//',label='xhigh effort')]+[Patch(color=h['color'],label=h['label']) for h in HOSTED_REFERENCES],ncol=2,bottom=.27)
ax.tick_params(axis='x',labelsize=10);ax.grid(axis='x',alpha=.2);save('F1-coding.png',fig)

# F2 low time/quality with one explicit offset per point.
fig,ax=plt.subplots(figsize=(9,4.3))
off={'base-iq3_xxs':(-45,23),'q2_0':(-25,11),'iq3_s':(-14,-19),'ista-iq2_xs':(-22,12),'swift-iq3_xxs':(-42,-22),'swift-iq2_xs':(8,8),'swift-iq3_s':(-24,12),'swift-q2_0':(-28,-20),'gyro-s':(8,8),'gyro-m':(8,-15)}
for k in ORDER:
 r=M[k]['heplus']['low'];ok=r['audit']['complete_scoring'];xx=r['total_time']/60
 ax.scatter(xx,r['score'],s=M[k]['size_gb']*4.3,color=clr(k) if ok else 'white',edgecolors=clr(k),alpha=.8,zorder=3)
 ax.annotate(label(k)+(' *' if not ok else ''),(xx,r['score']),xytext=off[k],textcoords='offset points',fontsize=8,bbox=dict(fc='white',ec='none',alpha=.82,pad=.5),arrowprops=dict(arrowstyle='-',color=clr(k),lw=.6))
ax.axvspan(8,20,color='#009e73',alpha=.05);ax.set_xlim(8,57);ax.set_ylim(93.6,97)
ax.set_xlabel('Real wall time for all 164 attempted tasks at low effort (minutes)');ax.set_ylabel('Reported final pass rate (%)');ax.set_title('Time-to-quality: all local coding arms scored on 164 tasks')
legend(ax,fig,[Patch(color=v,label=k) for k,v in COLOR.items()]);ax.grid(alpha=.2);save('F2-wall.png',fig)

# F3 low-effort reliability: all local files and both hosted references, score-positioned.
rr=[(k,'low',M[k]['trials']['low']) for k in ORDER if M[k].get('trials')]
rr += [(h['key'],'hosted',h['ref']['tool_eval_bench']) for h in HOSTED_REFERENCES]
rr.sort(key=lambda x:(-(x[2].get('pass_hat_5_percent',x[2].get('pass_all5_percent')) if x[0] in HOSTED_BY_KEY else x[2]['pass_hat_5']),-(x[2]['mean'])))
fig,ax=plt.subplots(figsize=(8.4,4.8))
for i,(k,e,t) in enumerate(rr):
 y=len(rr)-i-1
 if k in HOSTED_BY_KEY:
  floor=t.get('pass_hat_5_percent',t.get('pass_all5_percent'));mean=t['mean'];c=HOSTED_BY_KEY[k]['color']
 else:floor=t['pass_hat_5'];mean=t['mean'];c=clr(k)
 ax.plot([floor,mean],[y,y],c=c,alpha=.65,lw=2)
 ax.scatter(mean,y,color=c,s=52);ax.scatter(floor,y,facecolors='white',edgecolors=c,s=52)
 ax.text(floor-.45,y+.17,f(floor),ha='right',fontsize=8);ax.text(mean+.5,y+.17,f(mean),fontsize=8)
ax.set_yticks(range(len(rr)))
ax.set_yticklabels([HOSTED_BY_KEY[k]['label'] if k in HOSTED_BY_KEY else f'{label(k)} {e}'+(' [rc2]' if M[k]['cohort']=='rc2' else '') for k,e,t in reversed(rr)],fontsize=7)
ax.set_xlim(65,96);ax.set_xlabel('Floor and average score (%)');ax.set_title('Five-run reliability, low effort — local and hosted runs')
legend(ax,fig,[plt.Line2D([],[],marker='o',ls='',c='#46606d',label='average of five runs'),plt.Line2D([],[],marker='o',ls='',markerfacecolor='white',c='#46606d',label='full marks in all five runs')],2);ax.grid(axis='x',alpha=.2);save('F3-reliability.png',fig)

# F4 Hard Mode: rank every local arm and both hosted references by five-run mean.
hard_entries=localtools+[(h['key'],'hosted',h['ref']['tool_eval_bench']) for h in HOSTED_REFERENCES]
hard_entries.sort(key=lambda x:-(x[2]['hard_mode_mean_percent'] if x[0] in HOSTED_BY_KEY else x[2]['cats']['P']))
fig,ax=plt.subplots(figsize=(8.8,6.4));hard_labels=[]
for i,(k,e,t) in enumerate(hard_entries):
 if k in HOSTED_BY_KEY:
  hosted=HOSTED_BY_KEY[k];score=t['hard_mode_mean_percent'];c=hosted['color'];hard_labels.append(hosted['label'])
 else:score=t['cats']['P'];c=clr(k);hard_labels.append(f'{label(k)} {e} · {M[k]["lab"]}'+(' [rc2]' if M[k]['cohort']=='rc2' else ''))
 ax.barh(i,score,height=.63,color=c,alpha=.86)
 ax.text(score+.6,i,f(score),va='center',fontsize=8)
ax.set_yticks(range(len(hard_entries)));ax.set_yticklabels(hard_labels,fontsize=7);ax.invert_yaxis()
ax.set_xlim(0,100);ax.set_xlabel('Hard Mode (P), mean of five runs (%)');ax.set_title('Hard Mode: local arms and hosted references, sorted by mean')
legend(ax,fig,[Patch(color=COLOR['ISTA-DASLab'],label='ISTA-DASLab'),Patch(color=COLOR['UkisAI'],label='UkisAI'),Patch(color=COLOR['AgentionAI'],label='AgentionAI')]+[Patch(color=h['color'],label=h['label']) for h in HOSTED_REFERENCES],ncol=3,bottom=.27);ax.grid(axis='x',alpha=.2);save('F4-hardmode.png',fig)

# F5 debugging, ranked by hidden checks across local and hosted rows.
rr=STOCK+['gyro-s','gyro-m']+[h['key'] for h in HOSTED_REFERENCES]
rr.sort(key=lambda k:-(HOSTED_BY_KEY[k]['ref']['five_bugs']['hidden_passes'] if k in HOSTED_BY_KEY else int(M[k]['fivebugs']['hid'].split('/')[0])))
fig,ax=plt.subplots(figsize=(8.5,4.6));y=np.arange(len(rr))
labels=[]
for i,k in enumerate(rr):
 if k in HOSTED_BY_KEY:
  bugs=HOSTED_BY_KEY[k]['ref']['five_bugs'];vis=bugs['visible_passes'];hid=bugs['hidden_passes'];c=HOSTED_BY_KEY[k]['color'];labels.append(HOSTED_BY_KEY[k]['label'])
 else:
  b=M[k]['fivebugs'];vis=int(b['vis'].split('/')[0]);hid=int(b['hid'].split('/')[0]);c=clr(k);labels.append(label(k)+(' [rc2]' if M[k]['cohort']=='rc2' else ''))
 ax.barh(i,vis,height=.7,color=c,alpha=.2)
 ax.barh(i,hid,height=.4,color=c,alpha=.92)
 ax.text(hid+.2,i,f'{hid}/25',va='center',fontsize=8)
ax.set_yticks(y);ax.set_yticklabels(labels,fontsize=8);ax.invert_yaxis();ax.set_xlim(0,27);ax.set_xlabel('Successful sessions / 25');ax.set_title('Five Bugs: rows ordered by hidden-check score')
legend(ax,fig,[Patch(color='#566e79',alpha=.2,label='visible checks'),Patch(color='#566e79',label='hidden checks')]+[Patch(color=h['color'],label=h['label']) for h in HOSTED_REFERENCES],ncol=2,bottom=.27);save('F5-debug.png',fig)

# F6 output token totals (no conflation of all-task average with tokens per passing task).
x=np.arange(len(ORDER));bw=.36;fig,ax=plt.subplots(figsize=(9,3.8))
for ei,e in enumerate(['low','xhigh']):
 vals=[M[k]['heplus'][e]['total_output_tokens']/1000 for k in ORDER]
 ax.bar(x+(ei-.5)*bw,vals,bw,color=[clr(k) for k in ORDER],alpha=.95 if e=='low' else .45,hatch=None if e=='low' else '//',label=e)
ax.set_xticks(x);ax.set_xticklabels([label(k)+(' [rc2]' if M[k]['cohort']=='rc2' else '') for k in ORDER],rotation=25,ha='right',fontsize=8);ax.set_ylabel('Completion tokens across the run (thousands)');ax.set_title('Reasoning economics: token totals include repairs and failures')
legend(ax,fig,ncol=2);ax.grid(axis='y',alpha=.2);save('F6-tokens.png',fig)

# F7 only profiles whose zero-context decode cell is attributable to the pinned setup.
perf_keys=[k for k in ORDER if M[k].get('perf') and all(decode_axis_status(M[k]['perf'][e])['eligible'] for e in ['low','xhigh'])]
fig,ax=plt.subplots(figsize=(9,3.8));xs=np.arange(len(perf_keys));alphas=[.95,.62,.34]
for di in range(3):
 for i,k in enumerate(perf_keys):
    pp=sorted(M[k]['perf']['low'].values(),key=lambda r:r['context_tokens']);r=pp[di]
    ax.bar(i+(di-1)*.25,r['tg_mean'],.23,color=clr(k),alpha=alphas[di])
ax.set_xticks(xs);ax.set_xticklabels([label(k)+(' [rc2]' if M[k]['cohort']=='rc2' else '') for k in perf_keys],rotation=25,ha='right',fontsize=8);ax.set_ylabel('Decode tokens / second, low effort');ax.set_title('Attribution-eligible decode profiles — actual depth in tables')
legend(ax,fig,[Patch(color='#566e79',alpha=a,label=z) for a,z in zip(alphas,['0 background tokens','32,768 background tokens','deepest tested context'])]);ax.grid(axis='y',alpha=.2);save('F7-decode-all.png',fig)

# F8 both efforts for all measured files, rather than a Gyro-only chart.
fig,ax=plt.subplots(figsize=(8.2,4.5))
for i,k in enumerate(perf_keys):
 lo=M[k]['perf']['low']['0']['tg_mean'];hi=M[k]['perf']['xhigh']['0']['tg_mean'];c=clr(k)
 ax.plot([lo,hi],[i,i],c=c,lw=2,alpha=.5)
 ax.scatter(lo,i,s=55,facecolors='white',edgecolors=c,label='low' if i==0 else None)
 ax.scatter(hi,i,s=55,color=c,label='xhigh' if i==0 else None)
ax.set_yticks(range(len(perf_keys)));ax.set_yticklabels([label(k)+(' [rc2]' if M[k]['cohort']=='rc2' else '') for k in perf_keys],fontsize=8)
ax.set_xlabel('Decode tokens / second at zero background context');ax.set_title('Effort changes decode — all measurable files')
legend(ax,fig,ncol=2);ax.grid(axis='x',alpha=.2);save('F8-decode-effort.png',fig)

# F9 every tool arm and hosted reference: identical reliability decomposition.
rr=[(k,e,t) for k in ORDER for e,t in (M[k].get('trials')or{}).items()]
def tool_decomposition(detail,scenarios):
 points=[row['points'] for row in scenarios]
 result={'solid':sum(all(p==2 for p in v) for v in points),'flaky':sum(any(p==2 for p in v) and not all(p==2 for p in v) for v in points),'partial_only':sum(max(v)<2 and max(v)>0 for v in points),'zero_only':sum(max(v)==0 for v in points)}
 assert sum(result.values())==detail['metrics']['tool_eval_bench']['scenarios']
 return result
for hosted in HOSTED_REFERENCES:
 detail=hosted['detail'];scenario_rows=detail['metrics']['tool_eval_bench']['scenarios_by_trial']
 rr.append((hosted['key'],'hosted',{'decomposition':tool_decomposition(detail,scenario_rows)}))
rr.sort(key=lambda x:-x[2]['decomposition']['solid'])
fig,ax=plt.subplots(figsize=(8.6,6.6));labs=['Full marks in all five','Passed some runs','Partial only','Zero in all runs'];cols=['#25845b','#e2a729','#9ba6af','#c0392b']
for i,(k,e,t) in enumerate(rr):
 dd=t['decomposition'];left=0
 for j,key in enumerate(['solid','flaky','partial_only','zero_only']):
    v=dd[key];ax.barh(i,v,left=left,color=cols[j],height=.65)
    if j==0:ax.text(left+v/2,i,str(v),ha='center',va='center',fontsize=8,color='white');left+=v
    else:left+=v
labels=[f'{label(k)} {e}'+(' [rc2]' if M[k]['cohort']=='rc2' else '') if k not in HOSTED_BY_KEY else HOSTED_BY_KEY[k]['label'] for k,e,t in rr]
ax.set_yticks(range(len(rr)));ax.set_yticklabels(labels,fontsize=8);ax.invert_yaxis();ax.set_xlim(0,92);ax.set_xlabel('Number of agentic scenarios / 92');ax.set_title('Reliability-floor decomposition — sorted by Pass^5 floor')
ax.legend(handles=[Patch(color=c,label=l) for c,l in zip(cols,labs)],loc='upper center',bbox_to_anchor=(.5,-.14),ncol=2,frameon=False,fontsize=8)
fig.subplots_adjust(bottom=.21)
save('F9-decomposition.png',fig)

# F10: pronounced within-column contrast, raw labels, no master score.
from scorecard_colors import PALETTE, MISSING, cell_style
fig,ax=plt.subplots(figsize=(13.2,9.4));domains=['Coding','Tool floor','Hard Mode','Math','Algebra','Instructions','Hidden bugs','Speed']
def wstr(x):return f"{int(round(x))//60}m{int(round(x))%60:02d}"
values_by_row=[]
for r in SB:
 s=r['sampled']
 values_by_row.append([r['coding_final'],r['tool_floor'],r['hardmode']]+([s['gsm8k'],s['algebra'],s['ifeval_prompt'],100*int(s['debug_hidden'].split('/')[0])/25] if s else [None]*4)+[r['total_wall_s']])
bounds=[(min(v[j] for v in values_by_row if v[j] is not None),max(v[j] for v in values_by_row if v[j] is not None)) for j in range(len(domains))]
(BASE/'data/SCORECARD-COLOR-SCALE.json').write_text(json.dumps({'purpose':'presentation only; no score, weights or recommendation changes','method':'Five contrasting bins across the observed range of each column; missing values gray; constant columns neutral. Speed is total wall time (code+debug+5-run tool), colored inverted so faster is stronger. Colors magnify within-test differences, not absolute pass/fail or statistical significance.','palette':list(PALETTE),'missing':MISSING,'column_bounds':{name:{'minimum':lo,'maximum':hi} for name,(lo,hi) in zip(domains,bounds)}},indent=2)+'\n')
for i,values in enumerate(values_by_row):
 for j,v in enumerate(values):
  bg,fg=cell_style(v,*bounds[j],higher_is_better=(j!=len(domains)-1))
  ax.add_patch(Rectangle((j,i),1,1,facecolor=bg,edgecolor='white',linewidth=.8))
  lab=wstr(v) if (j==len(domains)-1 and v is not None) else (f(v) if v is not None else 'n/a')
  ax.text(j+.5,i+.5,lab,ha='center',va='center',fontsize=9,color=fg,fontweight='normal')
ax.set_yticks(np.arange(len(SB))+.5);ax.set_yticklabels([r['label']+' '+r['arm'] for r in SB],fontsize=9.5)
ax.set_xticks(np.arange(len(domains))+.5);ax.set_xticklabels([(f'Speed\n{wstr(lo)}–{wstr(hi)}' if j==len(domains)-1 else f'{name}\n{lo:.1f}–{hi:.1f}%') for j,(name,(lo,hi)) in enumerate(zip(domains,bounds))],fontsize=9.5);ax.xaxis.tick_top();ax.set_xlim(0,len(domains));ax.set_ylim(len(SB),0)
ax.set_title('Daily-driver evidence — compare colors WITHIN each test, not across tests',fontsize=13,pad=34)
ax.legend(handles=[Patch(color=PALETTE[0],label='Weaker within test'),Patch(color=PALETTE[2],label='Middle of range'),Patch(color=PALETTE[4],label='Stronger within test'),Patch(color=MISSING,label='Not available')],loc='upper center',bbox_to_anchor=(.5,-.035),ncol=4,frameon=False,fontsize=9.5)
fig.subplots_adjust(bottom=.10)
ax.tick_params(length=0)
for sp in ax.spines.values():sp.set_visible(False)
save('F10-scoreboard.png',fig)

exec(compile((BASE/'scripts/report_body.py').read_text(),str(BASE/'scripts/report_body.py'),'exec'))
