"""Render PR figures. Requires matplotlib; inference is never run.

python render.py --analysis /path/to/analyze.py-output/analysis.json
Subsequent renders can use the saved plot-data.json without --analysis.
"""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
INK, MUTED, BLUE, TEAL, ORANGE = '#172B46', '#526579', '#3569B3', '#087F83', '#B76232'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': INK,
                     'savefig.facecolor': 'white', 'figure.facecolor': 'white'})

def box(ax, x, y, w, h, text, fill='#F0F4F8', edge='#D9E2EB', size=11):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.012,rounding_size=0.018',
                              facecolor=fill,edgecolor=edge,linewidth=1.2))
    ax.text(x,y,text,ha='center',va='center',fontsize=size,linespacing=1.65)

def arrow(ax,a,b,color=MUTED):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,linewidth=1.4,color=color))

def flow():
    fig, axes=plt.subplots(1,2,figsize=(14,6.6))
    fig.subplots_adjust(left=.025,right=.975,top=.80,bottom=.12,wspace=.10)
    fig.text(.04,.945,'Account for the probability removed by the gate',fontsize=21,weight='bold')
    fig.text(.04,.895,'One MTP proposal row  •  p: target distribution  •  q: draft distribution  •  τ: confidence threshold',fontsize=11,color=MUTED)
    for ax,patched in zip(axes,[False,True]):
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
        color=TEAL if patched else ORANGE
        ax.text(.5,1.01,'PATCHED: explicit NULL proposal' if patched else 'UPSTREAM: failed draw discarded',
                ha='center',fontsize=13,weight='bold',color=color)
        box(ax,.5,.87,.81,.13,'Draw D from q\nGate: q(D) ≥ τ',size=12)
        arrow(ax,(.38,.79),(.25,.66));arrow(ax,(.62,.79),(.75,.66))
        ax.text(.25,.725,'PASS',ha='center',fontsize=10,weight='bold',color=MUTED)
        ax.text(.75,.725,'FAIL',ha='center',fontsize=10,weight='bold',color=MUTED)
        if patched:
            box(ax,.25,.48,.42,.31,'Offer D with mass μ(D)\nAccept: min(1, p(D)/μ(D))\nReject: sample (p − μ)₊',fill='#E8F5F3',size=10.5)
            box(ax,.75,.48,.42,.31,'Keep a NULL proposal\n\nSample (p − μ)₊',fill='#E8F5F3',size=11)
            box(ax,.5,.13,.87,.17,'μ retains q on admitted tokens\nNULL mass = 1 − Σ μ',fill='#F4F8FA',size=12)
        else:
            box(ax,.25,.48,.42,.31,'Verify D against full q\nAccept: min(1, p(D)/q(D))\nReject: sample (p − q)₊',fill='#FBF1E9',size=10.5)
            box(ax,.75,.48,.42,.31,'Discard the failed draw\n\nSample p',fill='#FBF1E9',size=11)
            box(ax,.5,.13,.87,.17,'Selection depends on the drawn token\nThe resulting mixture is biased',fill='#F4F8FA',size=12)
        arrow(ax,(.25,.31),(.38,.23),color);arrow(ax,(.75,.31),(.62,.23),color)
    fig.text(.04,.045,'Residual distributions are normalized. This diagram shows the MTP-only row; the suffix-root union is described in the PR.',fontsize=10,color=MUTED)
    fig.savefig(HERE/'proposal-flow.png',dpi=180)
    plt.close(fig)

def throughput(data):
    fig,ax=plt.subplots(figsize=(12.5,6.8))
    fig.subplots_adjust(left=.25,right=.73,top=.77,bottom=.25)
    fig.text(.055,.94,'Decode throughput relative to upstream',fontsize=21,weight='bold')
    fig.text(.055,.885,'100 paired requests  •  20 paired engine blocks  •  RX 7900 XTX / Coder IQ1_M',fontsize=11.5,color=MUTED)
    ax.axvline(0,color='#9AA9B6',linewidth=1.1,linestyle=(0,(4,4)),zorder=0)
    for i,r in enumerate(data['rows']):
        y=4-i;color=INK if i==4 else BLUE if i<2 else TEAL
        if i==4:ax.axhspan(-.35,.35,color='#EDF2F7',zorder=0)
        ax.errorbar(r['estimate'],y,xerr=[[r['estimate']-r['low']],[r['high']-r['estimate']]],
                    fmt='D' if i==4 else 'o',color=color,markersize=7,elinewidth=2.5,capsize=5)
        ax.text(1.04,y,f"{r['estimate']:+.2f}%",transform=ax.get_yaxis_transform(),va='center',weight='bold',fontsize=12)
        ax.text(1.24,y,f"[{r['low']:+.2f}, {r['high']:+.2f}]",transform=ax.get_yaxis_transform(),va='center',fontsize=11,color=MUTED)
    ax.set_yticks(range(4,-1,-1),[r['label'] for r in data['rows']]);ax.tick_params(axis='y',length=0,pad=14)
    ax.set(xlim=(-4.5,6),ylim=(-.6,4.6),xlabel='Throughput change (%)   •   positive = faster')
    for side in ('left','right','top'):ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#CCD6DF');ax.xaxis.grid(True,color='#E8EDF2',linewidth=.8);ax.set_axisbelow(True)
    ax.text(1.04,1.07,'Estimate',transform=ax.transAxes,fontsize=10,color=MUTED)
    ax.text(1.24,1.07,'95% interval',transform=ax.transAxes,fontsize=10,color=MUTED)
    fig.text(.055,.145,'Points: paired geometric means. Intervals: 20,000 bootstrap draws over whole engine blocks.',fontsize=10.5,color=MUTED)
    fig.text(.055,.105,'Each workload/temperature cell has 25 pairs. Every interval includes zero.',fontsize=10.5,color=MUTED)
    fig.text(.055,.055,'Measured revisions: upstream 296b59d2 → patch 96c3060d. The combined row weights the four tested cells equally.',fontsize=10,color=MUTED)
    fig.savefig(HERE/'throughput.png',dpi=180)
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--analysis',type=Path);args=parser.parse_args()
    if args.analysis:
        j=json.loads(args.analysis.read_text());assert j['audit']['matrix_complete']
        labels=[('english@0.6','English · T=0.6'),('english@1.0','English · T=1.0'),
                ('source@0.6','Repeated context · T=0.6'),('source@1.0','Repeated context · T=1.0')]
        rows=[]
        for key,label in labels:
            c=j['cells'][key];assert c['primary']['n']==25
            ci=c['bootstrap95']['geomean'];rows.append(dict(label=label,estimate=100*(c['primary']['metrics']['geomean']-1),low=ci['lower_change_percent'],high=ci['upper_change_percent']))
        o=j['descriptive_overall'];ci=o['bootstrap95']['geomean']
        rows.append(dict(label='Four-cell combined',estimate=100*(o['geomean']-1),low=ci['lower_change_percent'],high=ci['upper_change_percent']))
        data={'measurements_sha256':hashlib.sha256((HERE.parent/'measurements.json').read_bytes()).hexdigest(),'rows':rows}
        (HERE/'plot-data.json').write_text(json.dumps(data,indent=2)+'\n')
    else:data=json.loads((HERE/'plot-data.json').read_text())
    assert data['measurements_sha256']==hashlib.sha256((HERE.parent/'measurements.json').read_bytes()).hexdigest()
    flow();throughput(data)

if __name__=='__main__':main()
