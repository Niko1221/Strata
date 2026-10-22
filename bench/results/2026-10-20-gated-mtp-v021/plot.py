"""Render saved per-position measurements; performs no inference."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
rows=json.loads((R/'singleton.json').read_text())['rows']
x=[r['index']+1 for r in rows]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,(ax,side)=plt.subplots(1,2,figsize=(11,4),gridspec_kw={'width_ratios':[2.2,1]})
fig.patch.set_facecolor('#f7f9fc')
ax.plot(x,[100*r['U-trace']['p_tv'] for r in rows],color='#cf6b2b',marker='o',lw=2,label='Original 0.2.1 MTP',zorder=3)
ax.plot(x,[100*r['A-trace']['p_tv'] for r in rows],color='#167e83',marker='s',lw=2,label='Patched MTP',zorder=4)
ax.set(xlabel='Generated token position',ylabel='Total variation from singleton target (%)',xticks=x,ylim=(-.25,4.3))
ax.grid(axis='y',alpha=.18);ax.legend(loc='upper left',frameon=False)
ax.annotate('3.613%',(12,3.613477),xytext=(10.2,3.95),fontsize=11,color='#994a18')
side.axis('off')
side.text(0,.94,'Raw-logit agreement',fontsize=14,fontweight='bold')
side.text(0,.78,'Against incremental singleton decoding',fontsize=9,color='#546174')
side.text(0,.56,'12 / 12',fontsize=30,fontweight='bold',color='#167e83');side.text(0,.44,'Patched MTP positions',fontsize=11)
side.text(0,.23,'1 / 12',fontsize=24,fontweight='bold',color='#cf6b2b');side.text(0,.12,'Original MTP positions',fontsize=11)
fig.suptitle('The patch preserves the singleton target across MTP windows',fontsize=15,fontweight='bold',x=.06,ha='left')
fig.text(.06,.015,'One fixed-prefix fixture • Coder IQ1_M • RX 7900 XTX • 12 positions • No throughput or model-quality claim',fontsize=9,color='#546174')
fig.tight_layout(rect=[0,.06,1,.92]);fig.savefig(R/'target-agreement.png',dpi=180,facecolor=fig.get_facecolor());plt.close(fig)
