"""Render equal-scenario or equal-category views from the pinned category counts."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--unit', choices=['scenario', 'category'], default='category')
unit = parser.parse_args().unit
ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT/'source-data.json').read_text())
ORDER = list('ABCDEFGHIJKLMNO')
NAMES = ['Tool selection', 'Arguments', 'Multi-step chains', 'Restraint / refusal',
         'Error recovery', 'Localization', 'Structured reasoning', 'Instruction following',
         'Context & state', 'Code patterns', 'Safety / boundaries', 'Toolset scale',
         'Planning', 'Composition', 'Structured output']
COLORS = ['#176a85', '#e4ad3a', '#8da4b6', '#586c92', '#17a39b', '#d294b1',
          '#a77750', '#90ad52', '#7865ba', '#adc6bd', '#b85d66', '#689bc3',
          '#dc795a', '#6b9682', '#b299cb']
INK, MUTED, BG = '#183149', '#607386', '#f4f7fb'
plt.rcParams.update({'font.family':'DejaVu Sans', 'svg.fonttype':'none',
                     'svg.hashsalt':'equal-weight-20261008', 'font.size':11})
capacities = {k:max(c['max'] for m in data['models'] for c in m['category_scores'] if c['category']==k) for k in ORDER}
assert sum(capacities.values()) == 138
rows = []
for model in data['models']:
    cats = {c['category']:c for c in model['category_scores']}
    assert set(cats) == set(ORDER)
    for c in cats.values():
        assert c['max'] == 2*(c['pass_count']+c['partial_count']+c['fail_count'])
        assert c['earned'] == 2*c['pass_count']+c['partial_count']
    earned=sum(c['earned'] for c in cats.values())
    available=sum(c['max'] for c in cats.values())
    assert (earned,available)==(model['standard_points'],model['standard_max'])
    terms = {k:Fraction(100*cats[k]['earned'],available) if unit=='scenario'
             else Fraction(100*cats[k]['earned'],15*cats[k]['max']) for k in ORDER}
    score=sum(terms.values(),Fraction(0))
    rows.append(dict(id=model['id'],label=model['label'],reference=model['reference'],
                     score=float(score),exact_score=str(score),points=earned,max_points=available,
                     scenarios=available//2,category_scores=cats,
                     contributions={k:float(v) for k,v in terms.items()},source_url=model['source_url']))
ranked=sorted((r for r in rows if not r['reference']),key=lambda r:(-r['score'],r['label']))
reference=next(r for r in rows if r['reference'])
stem='equal-'+unit
(ROOT/(stem+'-results.json')).write_text(json.dumps(dict(unit=unit,
    category_weights={k:'1/15' for k in ORDER} if unit=='category' else None,
    ranked_variants=ranked,reference=reference),indent=2)+'\n')
table='| Variant | Score /100 | Original benchmark points | Scored scenarios |\n|---|---:|---:|---:|\n'
for r in ranked+[reference]:
    table+=f"| {r['label']} | {r['score']:.2f} | {r['points']}/{r['max_points']} | {r['scenarios']} |\n"
(ROOT/(stem.upper()+'-RESULTS.md')).write_text(table)

fig=plt.figure(figsize=(18,13),facecolor=BG)
title='Every test counts equally.' if unit=='scenario' else 'Every category counts equally.'
subtitle=('Pass = 2 points  /  partial = 1  /  fail = 0  /  every category included' if unit=='scenario'
          else '15 categories x 6.67%  /  all categories included  /  no preference multipliers')
fig.text(.045,.955,title,fontsize=28,weight='bold',color=INK)
fig.text(.045,.918,subtitle,fontsize=14,color=MUTED)
fig.legend(handles=[Patch(color=color,label=f'{k}: {name}') for k,name,color in zip(ORDER,NAMES,COLORS)],
           loc='upper left',bbox_to_anchor=(.04,.885),ncol=5,frameon=False,fontsize=10)
ax=fig.add_axes([.225,.19,.71,.56],facecolor=BG)
yref=len(ranked)+.7
extent=69 if unit=='scenario' else 100
for y,r in list(enumerate(ranked))+[(yref,reference)]:
    left=0
    for k,color in zip(ORDER,COLORS):
        c=r['category_scores'][k]
        cap=capacities[k]/2 if unit=='scenario' else 100/15
        fill=c['earned']/2 if unit=='scenario' else 100/15*c['earned']/c['max']
        ax.barh(y,cap,left=left,height=.57,color='white',edgecolor='#b6c4cf',linewidth=.7)
        ax.barh(y,fill,left=left,height=.57,color=color,edgecolor=BG,linewidth=.5)
        if unit=='scenario' and c['max']<capacities[k]:
            missing=(capacities[k]-c['max'])/2
            ax.barh(y,missing,left=left+cap-missing,height=.57,color='#dce3ea',
                    edgecolor='#6b7985',hatch='////',linewidth=.7)
        left+=cap
    ax.text(extent*1.018,y,f"{r['score']:.2f}",va='center',weight='bold',color=INK,fontsize=12)
left=0
for k in ORDER:
    cap=capacities[k]/2 if unit=='scenario' else 100/15
    ax.axvline(left,color='#c5d0db',linewidth=.6,zorder=0)
    ax.text(left+cap/2,-.8,k,ha='center',va='center',weight='bold',color=INK,fontsize=10)
    left+=cap
ax.text(extent*1.018,-.8,'Score',va='center',weight='bold',color=INK,fontsize=10)
ax.axhline(len(ranked)-.1,color='#c5d0db',linewidth=1)
ax.set_yticks(list(range(len(ranked)))+[yref],[r['label'] for r in ranked]+['Reference'],fontsize=12,color=INK)
ax.set_ylim(yref+.8,-1.3);ax.set_xlim(0,extent*1.10)
ax.set_xticks([0,10,20,30,40,50,60,69] if unit=='scenario' else [0,20,40,60,80,100])
ax.set_xlabel('Scenario-equivalent credit (points / 2); score at right normalizes by scored tests' if unit=='scenario'
              else 'Category budgets total 100 points; score sums the colored portions',color=MUTED,labelpad=10)
ax.tick_params(length=0,pad=9,colors=MUTED)
for spine in ax.spines.values():spine.set_visible(False)
fig.text(.045,.122,'Color = earned credit. Empty = missed credit. Section boundaries stay fixed for every model.',color=INK,weight='bold',fontsize=12)
fig.text(.045,.088,('Gyro: hatched slot = TC-45 excluded by the benchmark, not a failure. Scores use 68 tests; others use 69.'
                  if unit=='scenario' else 'Gyro instruction-following uses 4 scored tests; others use 5. Its category still receives 6.67%.'),color=MUTED,fontsize=10)
fig.text(.045,.061,'Single runs, different runtime cohorts. Structured-output API limitations affect scores. Reference is the llm-60 Q8 retest.',color=MUTED,fontsize=10)
fig.text(.045,.03,'Tool-Eval-Bench by SeraphimSerapis  |  github.com/SeraphimSerapis/tool-eval-bench',color=INK,fontsize=10)
for ext in ['png','svg']:
    p=ROOT/'figures'/(stem+'-ranking.'+ext)
    fig.savefig(p,dpi=160,facecolor=BG,metadata={'Date':None} if ext=='svg' else None)
    if ext=='svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
plt.close(fig)
print(table)
