#!/usr/bin/env python3
"""Portable review compiler: all metrics from data/ALL-METRICS.json; no rig access."""
from pathlib import Path
import datetime
import html
import json
import math
import re
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

BASE=Path(__file__).resolve().parent.parent
D=json.loads((BASE/'data/ALL-METRICS.json').read_text())
M=D['models']; C=D['gyro_m_campaign']; G=M['gyro-m']; CH=BASE/'charts'; CH.mkdir(exist_ok=True)
ORDER=['base-iq3_xxs','q2_0','iq3_s','ista-iq2_xs','swift-iq3_xxs','swift-iq2_xs','swift-iq3_s','swift-q2_0','gyro-s','gyro-m']
STOCK=[k for k in ORDER if M[k]['cohort']=='stock']
COLOR={'ISTA-DASLab':'#0072B2','UkisAI':'#D55E00','AgentionAI':'#009E73'}
SRC={'ISTA-DASLab':'https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF','UkisAI':'https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF','AgentionAI':'https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF'}
plt.rcParams.update({'font.size':9,'figure.dpi':160,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
AXES=['coding','floor','hardmode','mmlu','gsm8k','debug','decode']

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

# Composite is confined to eligible stock-engine arms. Partial and fork rows stay visible but unranked.
SB=[]
for k in ORDER:
 for e in ['low','xhigh']:
    m=M[k];r=m['heplus'][e];t=(m.get('trials')or{}).get(e);p=(m.get('perf')or{}).get(e)
    row={'key':k,'arm':e,'label':hlabel(k),'cohort':m['cohort'],'coding':r['score'],'floor':t['pass_hat_5'] if t else None,'hardmode':t['cats']['P'] if t else None,'mmlu':m['knowledge']['mmlu']['accuracy'],'gsm8k':m['knowledge']['gsm8k']['accuracy'],'debug':int(m['fivebugs']['hid'].split('/')[0]),'decode':p['0']['tg_mean'] if p else None,'capped':'safety-capped' in t['rating'] if t else None,'complete_scoring':r['audit']['complete_scoring'],'composite':None,'rank':None}
    row['eligible']=m['cohort']=='stock' and row['complete_scoring'] and all(row[a] is not None for a in AXES)
    row['reason']='different engine cohort' if m['cohort']!='stock' else ('partial HumanEval+ denominator' if not row['complete_scoring'] else '')
    SB.append(row)
el=[r for r in SB if r['eligible']]
norm={a:(min(r[a] for r in el),max(r[a] for r in el)) for a in AXES}
for r in el:r['composite']=sum((r[a]-norm[a][0])/(norm[a][1]-norm[a][0])*100 if norm[a][1]>norm[a][0] else 50 for a in AXES)/len(AXES)
el.sort(key=lambda r:-r['composite'])
for i,r in enumerate(el,1):r['rank']=i
SB=el+[r for r in SB if not r['eligible']]
(BASE/'data/FINAL-SCOREBOARD.json').write_text(json.dumps(SB,indent=2)+'\n')
TOP=el[0];rankmap={(r['key'],r['arm']):r['rank'] for r in SB}
FULL=[(k,e,r) for k,m in M.items() for e,r in m['heplus'].items() if r['audit']['complete_scoring']]
FAST=min(FULL,key=lambda x:x[2]['total_time'])
MAXHE=max(r['score'] for _,_,r in FULL);BESTHE=[(k,e) for k,e,r in FULL if r['score']==MAXHE]
stocktools=[(k,e,t) for k in STOCK for e,t in M[k]['trials'].items()]
BESTMEAN=max(stocktools,key=lambda x:x[2]['mean']);BESTFLOOR=max(stocktools,key=lambda x:x[2]['pass_hat_5'])
hewallratio=G['heplus']['xhigh']['total_time']/G['heplus']['low']['total_time']
hetokenratio=G['heplus']['xhigh']['total_output_tokens']/G['heplus']['low']['total_output_tokens']
claims={'inventory':D['inventory'],'fastest_full_he':{'key':FAST[0],'effort':FAST[1],'wall_s':FAST[2]['total_time']},'best_complete_he':{'score':MAXHE,'arms':BESTHE},'stock_best_mean':{'key':BESTMEAN[0],'effort':BESTMEAN[1],'mean':BESTMEAN[2]['mean']},'stock_best_floor':{'key':BESTFLOOR[0],'effort':BESTFLOOR[1],'floor':BESTFLOOR[2]['pass_hat_5']},'stock_composite_leader':TOP,'gyro_m_he_wall_ratio':hewallratio,'gyro_m_he_token_ratio':hetokenratio,'gyro_m_resources':{'min_free_vram_mib':C['min_free_vram_mib'],'max_temperature_c':C['max_temperature_c']},'partial_he_arms':D['inventory']['partial_he_arms'],'tool_method':{'version':C['manifest']['teb_version'],'scenarios':G['trials']['low']['n_scenarios'],'trials':5}}
(BASE/'claims.json').write_text(json.dumps(claims,indent=2)+'\n')

# F1 coding overview. Partial-denominator bars are hollow, not hidden or ranked.
fig,ax=plt.subplots(figsize=(9.1,3.8));x=np.arange(len(ORDER));bw=.36
for ei,e in enumerate(['low','xhigh']):
 for i,k in enumerate(ORDER):
    r=M[k]['heplus'][e];ok=r['audit']['complete_scoring']
    ax.bar(i+(ei-.5)*bw,r['score'],bw,color=clr(k) if ok else 'white',edgecolor=clr(k),alpha=.92 if e=='low' else .55,hatch=None if e=='low' else '//')
    if not ok:ax.text(i+(ei-.5)*bw,r['score']+.07,'*',ha='center',fontsize=12)
ax.set_xticks(x);ax.set_xticklabels([label(k) for k in ORDER],rotation=25,ha='right',fontsize=8)
ax.set_ylim(91.5,97.5);ax.set_ylabel('Reported final pass rate (%)');ax.set_title('HumanEval+: similar ceilings, different effort costs')
legend(ax,fig,[Patch(color='#526778',label='low'),Patch(color='#526778',alpha=.5,hatch='//',label='xhigh'),Patch(facecolor='white',edgecolor='#526778',label='* partial scoring denominator')])
ax.grid(axis='y',alpha=.2);save('F1-coding.png',fig)

# F2 low time/quality with one explicit offset per point.
fig,ax=plt.subplots(figsize=(9,4.3))
off={'base-iq3_xxs':(-45,23),'q2_0':(-25,11),'iq3_s':(-14,-19),'ista-iq2_xs':(-22,12),'swift-iq3_xxs':(-42,-22),'swift-iq2_xs':(8,8),'swift-iq3_s':(-24,12),'swift-q2_0':(-28,-20),'gyro-s':(8,8),'gyro-m':(8,-15)}
for k in ORDER:
 r=M[k]['heplus']['low'];ok=r['audit']['complete_scoring'];xx=r['total_time']/60
 ax.scatter(xx,r['score'],s=M[k]['size_gb']*4.3,color=clr(k) if ok else 'white',edgecolors=clr(k),alpha=.8,zorder=3)
 ax.annotate(label(k)+(' *' if not ok else ''),(xx,r['score']),xytext=off[k],textcoords='offset points',fontsize=8,bbox=dict(fc='white',ec='none',alpha=.82,pad=.5),arrowprops=dict(arrowstyle='-',color=clr(k),lw=.6))
ax.axvspan(8,20,color='#009e73',alpha=.05);ax.set_xlim(8,57);ax.set_ylim(93.6,97)
ax.set_xlabel('Real wall time for all 164 attempted tasks at low effort (minutes)');ax.set_ylabel('Reported final pass rate (%)');ax.set_title('Time-to-quality: bubble area = file size; hollow = partial scoring')
legend(ax,fig,[Patch(color=v,label=k) for k,v in COLOR.items()]);ax.grid(alpha=.2);save('F2-wall.png',fig)

# F3 low-effort reliability: all tool-capable files in one view.
rr=sorted([(k,'low',M[k]['trials']['low']) for k in ORDER if M[k].get('trials')],key=lambda x:(M[x[0]]['cohort']!='stock',-x[2]['pass_hat_5']))
fig,ax=plt.subplots(figsize=(8.4,4.5))
for i,(k,e,t) in enumerate(rr):
 y=len(rr)-i-1;c=clr(k);ax.plot([t['pass_hat_5'],t['mean']],[y,y],c=c,alpha=.6,lw=2)
 ax.scatter(t['mean'],y,color=c,s=52);ax.scatter(t['pass_hat_5'],y,facecolors='white',edgecolors=c,s=52)
 ax.text(t['pass_hat_5']-.45,y+.17,f(t['pass_hat_5']),ha='right',fontsize=8);ax.text(t['mean']+.5,y+.17,f(t['mean']),fontsize=8)
ax.set_yticks(range(len(rr)));ax.set_yticklabels([f'{label(k)} low'+(' [rc2]' if k=='gyro-m' else '') for k,e,t in reversed(rr)],fontsize=8)
ax.set_xlim(65,94);ax.set_xlabel('Floor and average score (%)');ax.set_title('Five-run reliability, low effort — all measured files')
legend(ax,fig,[plt.Line2D([],[],marker='o',ls='',c='#46606d',label='average of five runs'),plt.Line2D([],[],marker='o',ls='',markerfacecolor='white',c='#46606d',label='full marks in all five runs')],2);ax.grid(axis='x',alpha=.2);save('F3-reliability.png',fig)

# F4 Hard Mode means sorted within each cohort, not from last-trial score.
rr=sorted(stocktools,key=lambda x:x[2]['cats']['P'])+[("gyro-m",e,G['trials'][e]) for e in ['low','xhigh']]
fig,ax=plt.subplots(figsize=(8.6,6.3))
for i,(k,e,t) in enumerate(rr):
 ax.barh(i,t['cats']['P'],height=.63,color=clr(k),alpha=.82)
 ax.text(t['cats']['P']+.6,i,f(t['cats']['P']),va='center',fontsize=8)
ax.set_yticks(range(len(rr)));ax.set_yticklabels([f'{label(k)} {e}'+(' [rc2]' if k=='gyro-m' else '') for k,e,t in rr],fontsize=8)
ax.set_xlim(0,100);ax.set_xlabel('Hard Mode (P), mean of five runs (%)');ax.set_title('Long multi-step chains: extra effort often helps this axis')
ax.grid(axis='x',alpha=.2);save('F4-hardmode.png',fig)

# F5 debugging, ranked within cohorts. Unlike the old plot, no overlapping equal points.
rr=sorted(STOCK,key=lambda k:-int(M[k]['fivebugs']['hid'].split('/')[0]))+['gyro-s','gyro-m']
fig,ax=plt.subplots(figsize=(8.5,4.3));y=np.arange(len(rr))
for i,k in enumerate(rr):
 b=M[k]['fivebugs'];ax.barh(i,int(b['vis'].split('/')[0]),height=.7,color=clr(k),alpha=.2)
 ax.barh(i,int(b['hid'].split('/')[0]),height=.4,color=clr(k),alpha=.92)
 ax.text(int(b['hid'].split('/')[0])+.2,i,b['hid'],va='center',fontsize=8)
ax.set_yticks(y);ax.set_yticklabels([label(k) for k in rr],fontsize=8);ax.invert_yaxis();ax.set_xlim(0,27);ax.set_xlabel('Successful sessions / 25');ax.set_title('Five Bugs: hidden edge cases separate superficial fixes')
legend(ax,fig,[Patch(color='#566e79',alpha=.2,label='visible checks'),Patch(color='#566e79',label='hidden checks')],2);save('F5-debug.png',fig)

# F6 output token totals (no conflation of all-task average with tokens per passing task).
fig,ax=plt.subplots(figsize=(9,3.8))
for ei,e in enumerate(['low','xhigh']):
 vals=[M[k]['heplus'][e]['total_output_tokens']/1000 for k in ORDER]
 ax.bar(x+(ei-.5)*bw,vals,bw,color=[clr(k) for k in ORDER],alpha=.95 if e=='low' else .45,hatch=None if e=='low' else '//',label=e)
ax.set_xticks(x);ax.set_xticklabels([label(k) for k in ORDER],rotation=25,ha='right',fontsize=8);ax.set_ylabel('Completion tokens across the run (thousands)');ax.set_title('Reasoning economics: token totals include repairs and failures')
legend(ax,fig,ncol=2);ax.grid(axis='y',alpha=.2);save('F6-tokens.png',fig)

# F7 all finite low-effort profiles, with true deep-context scope in tables.
perf_keys=[k for k in ORDER if M[k].get('perf')]
fig,ax=plt.subplots(figsize=(9,3.8));xs=np.arange(len(perf_keys));alphas=[.95,.62,.34]
for di in range(3):
 for i,k in enumerate(perf_keys):
    pp=sorted(M[k]['perf']['low'].values(),key=lambda r:r['context_tokens']);r=pp[di]
    ax.bar(i+(di-1)*.25,r['tg_mean'],.23,color=clr(k),alpha=alphas[di])
ax.set_xticks(xs);ax.set_xticklabels([label(k)+(' [rc2]' if k=='gyro-m' else '') for k in perf_keys],rotation=25,ha='right',fontsize=8);ax.set_ylabel('Decode tokens / second, low effort');ax.set_title('Measured decode profiles — all finite files, actual depth in tables')
legend(ax,fig,[Patch(color='#566e79',alpha=a,label=z) for a,z in zip(alphas,['0 background tokens','32,768 background tokens','deepest tested context'])]);ax.grid(axis='y',alpha=.2);save('F7-decode-all.png',fig)

# F8 both efforts for all measured files, rather than a Gyro-only chart.
fig,ax=plt.subplots(figsize=(8.2,4.5))
for i,k in enumerate(perf_keys):
 lo=M[k]['perf']['low']['0']['tg_mean'];hi=M[k]['perf']['xhigh']['0']['tg_mean'];c=clr(k)
 ax.plot([lo,hi],[i,i],c=c,lw=2,alpha=.5)
 ax.scatter(lo,i,s=55,facecolors='white',edgecolors=c,label='low' if i==0 else None)
 ax.scatter(hi,i,s=55,color=c,label='xhigh' if i==0 else None)
ax.set_yticks(range(len(perf_keys)));ax.set_yticklabels([label(k)+(' [rc2]' if k=='gyro-m' else '') for k in perf_keys],fontsize=8)
ax.set_xlabel('Decode tokens / second at zero background context');ax.set_title('Effort changes decode — all measurable files')
legend(ax,fig,ncol=2);ax.grid(axis='x',alpha=.2);save('F8-decode-effort.png',fig)

# F9 every tool arm: identical reliability decomposition across the report.
rr=sorted([(k,e,t) for k in ORDER for e,t in (M[k].get('trials')or{}).items()],key=lambda x:(M[x[0]]['cohort']!='stock',-x[2]['pass_hat_5']))
fig,ax=plt.subplots(figsize=(8.6,6.2));labs=['Full marks in all five','Passed some runs','Partial only','Zero in all runs'];cols=['#25845b','#e2a729','#9ba6af','#c0392b']
for i,(k,e,t) in enumerate(rr):
 dd=t['decomposition'];left=0
 for j,key in enumerate(['solid','flaky','partial_only','zero_only']):
    v=dd[key];ax.barh(i,v,left=left,color=cols[j],height=.65)
    if j==0:ax.text(left+v/2,i,str(v),ha='center',va='center',fontsize=8,color='white');left+=v
    else:left+=v
ax.set_yticks(range(len(rr)));ax.set_yticklabels([f'{label(k)} {e}'+(' [rc2]' if k=='gyro-m' else '') for k,e,t in rr],fontsize=8);ax.invert_yaxis();ax.set_xlim(0,92);ax.set_xlabel('Number of agentic scenarios / 92');ax.set_title('What the reliability floor is made of — all measured arms')
ax.legend(handles=[Patch(color=c,label=l) for c,l in zip(cols,labs)],loc='upper center',bbox_to_anchor=(.5,-.14),ncol=2,frameon=False,fontsize=8)
fig.subplots_adjust(bottom=.21)
save('F9-decomposition.png',fig)

# F10 board: partial scores qualified; unranked fork rows are separated.
cm=LinearSegmentedColormap.from_list('strata',['#fff','#7fd1c1','#239987','#0b3d5c'])
fig,ax=plt.subplots(figsize=(10,6.6));columns=AXES+['capped']
for i,r in enumerate(SB):
 for j,a in enumerate(columns):
    v=r[a]
    if v is None:n=None;txt='n/a'
    elif a=='capped':n=.08 if v else .75;txt='capped' if v else 'not capped'
    else:
     lo,hi=norm[a];n=(v-lo)/(hi-lo) if hi>lo else .5;n=max(0,min(1,n));txt=f(v,1 if a in ['coding','floor','hardmode'] else 0)
     if a=='coding' and not r['complete_scoring']:txt+='*'
    bg=cm(float(n)) if n is not None else '#e9edf1';ax.add_patch(Rectangle((j,i),1,1,facecolor=bg,edgecolor='white'))
    ax.text(j+.5,i+.5,txt,ha='center',va='center',fontsize=7,color='white' if n is not None and n>.65 else '#23313c')
 if r['rank']:ax.text(-.25,i+.5,f'{r["rank"]}.  {r["composite"]:.0f}',ha='right',va='center',fontsize=8,fontweight='bold')
 else:ax.text(-.25,i+.5,'unranked',ha='right',va='center',fontsize=7,color='#7b8d98')
ax.text(-.25,-.55,'Rank /\nOVERALL',ha='right',va='center',fontsize=8,fontweight='bold')
ax.axhline(len(el),color='#627b89',ls='--');ax.set_yticks(np.arange(len(SB))+.5);ax.set_yticklabels([f'{label(r["key"])} {r["arm"]}' for r in SB],fontsize=8)
for tick,r in zip(ax.get_yticklabels(),SB):tick.set_color(clr(r['key']))
ax.set_xticks(np.arange(len(columns))+.5);ax.set_xticklabels(['Coding\nHE+','Agents\nPass^5','Hard\nMode','MMLU\nsample','GSM8K\nsample','Debug\n/25','Decode\ntok/s','Rating\ngate'],fontsize=8)
ax.xaxis.tick_top();ax.set_xlim(-1.5,len(columns));ax.set_ylim(len(SB),-1.2);ax.set_title('Decision board: stock-engine composite only; forks and partial arms unranked',pad=27)
ax.tick_params(length=0)
for sp in ax.spines.values():sp.set_visible(False)
save('F10-scoreboard.png',fig)

exec(compile((BASE/'scripts/report_body.py').read_text(),str(BASE/'scripts/report_body.py'),'exec'))
