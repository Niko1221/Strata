"""PR figure: measurements, estimated policy, and context-capacity limits."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

HERE=Path(__file__).resolve().parent
DATA=HERE/'data'
model=json.loads((DATA/'decision-model.json').read_text())
results=json.loads((DATA/'results.json').read_text())
rows=model['rows']
cases={(c['input_target'],c['mtp']):c for c in results['cases']}
INK,MUTED,GRID='#142638','#586b7d','#e4eaf0'
OFF,ON='#475569','#087e8b'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':INK,
                     'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,
                     'axes.edgecolor':GRID,'svg.fonttype':'none'})
fig=plt.figure(figsize=(14.5,12),facecolor='#f8fafc')
fig.text(.075,.965,'Short control decisions: MTP off wins',fontsize=25,weight='bold',va='top')
fig.text(.075,.919,'Measured 48K–128K fresh inputs; estimates for longer answers',fontsize=16,va='top')
fig.text(.075,.884,'RTX 3070 8 GB  /  Flash-Next GSQ-RCO IQ3_S  /  INT8 KV  /  32K GPU-resident KV',
         fontsize=12,color=MUTED,va='top')
fig.legend(handles=[Patch(facecolor=OFF,label='MTP off'),Patch(facecolor=ON,label='MTP on')],
           bbox_to_anchor=(.965,.922),loc='upper right',ncol=2,frameon=False,fontsize=12)
left=fig.add_axes([.075,.54,.43,.26])
right=fig.add_axes([.59,.54,.365,.26])
for ax in [left,right]:
    ax.set_facecolor('white');ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=0,pad=8);ax.set_axisbelow(True)
    ax.set_ylim(-.65,3.6)
    ax.set_yticks([3,2,1,0],['48K','64K','96K','128K'])
left.set_title('Measured: 1,000 output tokens',loc='left',fontsize=17,weight='bold',pad=18)
left.set_xlim(0,455);left.set_xticks([0,100,200,300,400]);left.grid(axis='x',color=GRID,lw=.7)
left.set_xlabel('Request completion time (seconds)',labelpad=12)
left.text(.02,.99,'Dark: prefill · light: decode',transform=left.transAxes,fontsize=11,color=MUTED,va='top')
right.set_title('Projected break-even output',loc='left',fontsize=17,weight='bold',pad=18)
right.set_xlim(0,30500);right.set_xticks([0,8192,16384,24576],['0','8,192','16,384','24,576'])
right.grid(axis='x',color=GRID,lw=.7)
right.set_xlabel('Generated output tokens (estimate)',labelpad=12)
for i,s in enumerate(rows):
    y=3-i
    for mode,offset,color,light in [(False,.15,OFF,'#a7b2bf'),(True,-.15,ON,'#80b7be')]:
        c=cases[(s['input_tokens'],mode)]
        pre=c['timings']['prompt_ms']/1000;dec=c['timings']['decode_ms']/1000
        left.barh(y+offset,pre,height=.23,color=color,zorder=2)
        left.barh(y+offset,dec,left=pre,height=.23,color=light,zorder=2)
        left.text(c['wall_seconds']+5,y+offset,f'{c["wall_seconds"]:.1f} s',va='center',fontsize=12,color=color)
    boundary=s['projected_break_even_tokens']
    color='#9a4c12' if boundary>s['tested_output_capacity'] else OFF
    right.plot([0,boundary],[y,y],color=color,lw=2,ls=(0,(4,3)))
    right.scatter([boundary],[y],s=50,facecolors='white',edgecolors=color,zorder=3)
    right.text(boundary+900,y,f'≈{boundary:,.0f}',va='center',fontsize=13,color=color,weight='bold')
    if boundary>s['tested_output_capacity']:
        right.plot([s['tested_output_capacity']]*2,[y-.13,y+.13],color=INK,lw=2)
        right.text(1000,y-.35,'128K: only 2,048 output slots remain',fontsize=10.5,color='#9a4c12')

fig.text(.075,.448,'Known output length: which mode finishes sooner?',fontsize=17,weight='bold')
fig.text(.075,.418,'Signed values are seconds saved by enabling MTP; negative means MTP on is slower.',fontsize=11,color=MUTED)
table_ax=fig.add_axes([.10,.228,.855,.165]);table_ax.axis('off')
labels=['10 tokens','100 tokens','1,000 tokens','10,000 · model','100,000 · model']
texts=[];colors=[]
for s in rows:
    tr=[];cr=[]
    for q in s['decisions']:
        gain=q['mtp_on_savings_seconds']
        value=f'{gain:+.1f} s'.replace('-','−')
        if not q['fits_tested_context']:
            tr.append('Larger context\n'+value+'*');cr.append('#eceff3')
        elif gain>0:
            tr.append('ON candidate\n'+value+'*');cr.append('#d8eeee')
        else:
            tr.append('OFF\n'+value+('*' if q['output_tokens']>1000 else ''));cr.append('#e6ebf1')
    texts.append(tr);colors.append(cr)
table=table_ax.table(cellText=texts,colLabels=labels,rowLabels=['48K','64K','96K','128K'],
                     cellColours=colors,cellLoc='center',rowLoc='center',bbox=[0,0,1,1])
table.auto_set_font_size(False);table.set_fontsize(11.5)
for (r,c),cell in table.get_celld().items():
    cell.set_edgecolor('#f8fafc');cell.set_linewidth(3)
    if r==0:cell.set_facecolor('#f8fafc');cell.set_text_props(weight='bold',color=INK)
    if c==-1:cell.set_facecolor('#f8fafc');cell.set_text_props(weight='bold',color=INK)
fig.text(.075,.199,'10/100: recorded prefixes of 1K runs. * Longer-output arithmetic assumes constant rates; gray cells exceed tested capacity.',
         fontsize=10.5,color=MUTED)
fig.text(.075,.154,'Rule of thumb: projected crossing ≈ fresh input tokens ÷ 6',fontsize=16,weight='bold')
fig.text(.075,.12,'Use OFF for 10 / 100 / 1K control outputs. Consider ON well above the projected boundary, after checking capacity.',
         fontsize=11.5,color=MUTED)
fig.text(.075,.086,'Ten or 100 separate uncached 1K decisions still favor OFF: every decision pays prefill again.',
         fontsize=11.5,color=MUTED)
fig.text(.075,.048,'One pair per input length · warm engines · no prompt reuse · outputs differ · max context 133,120 · source 61b3fb5d',
         fontsize=10.5,color=MUTED)
fig.savefig(HERE/'control-decisions.png',dpi=180,facecolor=fig.get_facecolor())
fig.savefig(HERE/'control-decisions.svg',facecolor=fig.get_facecolor())
svg=HERE/'control-decisions.svg'
svg.write_text('\n'.join(l.rstrip() for l in svg.read_text().splitlines())+'\n',encoding='utf-8',newline='\n')
plt.close(fig)
print('Rendered control-decisions.png and control-decisions.svg')
