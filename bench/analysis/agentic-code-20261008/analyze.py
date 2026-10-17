"""Reproduce the preference-weighted analysis and figures; no inference or rescoring."""
from pathlib import Path
from fractions import Fraction
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
data = json.loads((ROOT / "source-data.json").read_text())
profile = json.loads((ROOT / "weights.json").read_text())
weights = profile["weights_percent"]
assert set(weights) == set("ABCDEFGHIJKLMNO")
assert weights == {"A":25,"B":10,"C":5,"D":0,"E":25,"F":0,"G":0,"H":0,"I":20,"J":5,"K":0,"L":0,"M":10,"N":0,"O":0}, "Update the explanatory figures when changing this profile"
assert all(isinstance(w, int) and w >= 0 for w in weights.values()) and sum(weights.values()) == 100
ORDER = [k for k in ["A", "E", "I", "B", "M", "C", "J", "D", "F", "G", "H", "K", "L", "N", "O"] if weights[k]]
NAMES = {"A":"Tool selection", "E":"Error recovery", "I":"Context & state", "B":"Arguments", "M":"Planning", "C":"Multi-step chains", "J":"Code patterns"}
COLORS = {"A":"#176a85", "E":"#17a39b", "I":"#7865ba", "B":"#e4ad3a", "M":"#dc795a", "C":"#8da4b6", "J":"#adc6bd"}
INK, MUTED, BG = "#183149", "#607386", "#f4f7fb"
plt.rcParams.update({"font.family":"DejaVu Sans", "svg.fonttype":"none", "svg.hashsalt":"agentic-code-20261008", "font.size":11})
rows = []
for model in data["models"]:
    cats = {c["category"]:c for c in model["category_scores"]}
    terms = {}
    for key in ORDER:
        c = cats[key]
        assert 0 <= c["earned"] <= c["max"] and c["max"] > 0
        terms[key] = Fraction(weights[key] * c["earned"], c["max"])
    exact = sum(terms.values(), Fraction(0))
    direct = math.fsum(weights[k] * cats[k]["earned"] / cats[k]["max"] for k in ORDER)
    assert math.isclose(float(exact), direct, abs_tol=1e-10)
    rows.append({"id":model["id"], "label":model["label"], "family":model["family"],
                 "reference":model["reference"], "score":float(exact), "exact_score":str(exact),
                 "contributions":{k:float(v) for k,v in terms.items()}, "file_gb":model["file_bytes"]/1e9,
                 "selected_points_available":sum(cats[k]["max"] for k in ORDER), "source_url":model["source_url"]})
assert len(rows) == 12 and all(r["selected_points_available"] == 58 for r in rows)
ranked = sorted((r for r in rows if not r["reference"]), key=lambda r:(-r["score"],r["label"]))
reference = next(r for r in rows if r["reference"])
(ROOT / "weighted-results.json").write_text(json.dumps({"weights":weights,"ranked_variants":ranked,"reference":reference},indent=2)+"\n")
table = "| Variant | Weighted score /100 | Difference from reference | GGUF GB |\n|---|---:|---:|---:|\n"
for r in ranked + [reference]:table += f"| {r['label']} | {r['score']:.2f} | {r['score']-reference['score']:+.2f} | {r['file_gb']:.2f} |\n"
(ROOT / "RESULTS.md").write_text(table)

def save(fig, name):
    for ext in ["png","svg"]:
        path = FIG / (name+"."+ext)
        fig.savefig(path,dpi=160,facecolor=BG,metadata={"Date":None} if ext == "svg" else None)
        if ext == "svg":path.write_text("\n".join(x.rstrip() for x in path.read_text().splitlines())+"\n")
    plt.close(fig)

def footer(fig, extra=""):
    fig.text(.045,.031,"Tool-Eval-Bench by SeraphimSerapis  |  github.com/SeraphimSerapis/tool-eval-bench",fontsize=9,color=INK)
    if extra:fig.text(.045,.057,extra,fontsize=9,color=MUTED)

fig=plt.figure(figsize=(16,12),facecolor=BG)
fig.text(.045,.948,"Pick the right tool. Recover when it fails.",fontsize=27,weight="bold",color=INK)
fig.text(.045,.912,"A preference-weighted view of the completed Flash Next tool-use runs",fontsize=14,color=MUTED)
ax=fig.add_axes([.235,.19,.69,.61],facecolor=BG)
yref=len(ranked)+.75
for i,r in list(enumerate(ranked))+[(yref,reference)]:
    left=0
    for k in ORDER:
        # Each category owns its full weight budget. Missed points stay empty;
        # later categories never shift left to hide a deficit.
        capacity=weights[k]
        earned=r["contributions"][k]
        assert 0 <= earned <= capacity
        ax.barh(i,capacity,left=left,height=.57,color="white",edgecolor="#bac8d3",linewidth=.7)
        ax.barh(i,earned,left=left,height=.57,color=COLORS[k],edgecolor=BG,linewidth=.5)
        left+=capacity
    assert left == 100
    ax.text(102,i,f"{r['score']:.2f}",va="center",weight="bold",color=INK,fontsize=12)
ax.axhline(len(ranked)-.1,color="#c5d0db",linewidth=1)
left=0
for k in ORDER:
    ax.axvline(left,color="#c5d0db",linewidth=.6,zorder=0)
    ax.text(left+weights[k]/2,-.75,k,ha="center",va="center",fontsize=10,weight="bold",color=INK)
    left+=weights[k]
ax.text(102,-.75,"Score",va="center",weight="bold",fontsize=10,color=INK)
ax.set_yticks(list(range(len(ranked)))+[yref],[r["label"] for r in ranked]+["Reference"],fontsize=12,color=INK)
ax.set_ylim(yref+.8,-1.25);ax.set_xlim(0,110);ax.set_xticks([0,25,50,70,80,90,95,100])
ax.tick_params(length=0,pad=10,colors=MUTED);ax.set_axisbelow(True);ax.grid(axis="x",color="#e0e7ed",linewidth=.7)
for sp in ax.spines.values():sp.set_visible(False)
fig.legend(handles=[Patch(color=COLORS[k],label=f"{k}: {NAMES[k]} {weights[k]}%") for k in ORDER]+[Patch(facecolor="white",edgecolor="#bac8d3",label="Empty = unearned points")],loc="upper left",bbox_to_anchor=(.045,.873),ncol=4,frameon=False,fontsize=10)
fig.text(.045,.122,"ISTA IQ3_XXS and Swift IQ3_XXS tie; Gyro-S is 0.67 points behind.",fontsize=13,weight="bold",color=INK)
fig.text(.045,.087,"Fixed category budgets: fill = earned points; empty = missed points. Safety and structured output contribute zero.",fontsize=10,color=MUTED)
footer(fig,"One trial per variant. Reference: same RTX PRO 6000 + mainline, mmap loading. Gyro still uses patched rc1.")
save(fig,"weighted-ranking")

fig=plt.figure(figsize=(16,12),facecolor=BG)
fig.text(.045,.95,"What matters in this scoring profile",fontsize=27,weight="bold",color=INK)
fig.text(.045,.912,"50% for tool selection + error recovery  /  5% for multi-step chains",fontsize=14,color=MUTED)
explanations={
 "A":("Choose the operation that fits the request.","Example: get_weather instead of web_search."),
 "E":("Handle errors without fabricating a result.","Example: surface a failed stock-price lookup."),
 "I":("Retain corrections and prior tool results.","Example: keep the revised meeting time."),
 "B":("Pass the intended arguments accurately.","Example: use the requested temperature units."),
 "M":("Break a goal into useful operations.","Example: resolve contacts, create event, notify."),
 "C":("Respect dependencies, with a smaller weight.","Example: read a found file before emailing it."),
 "J":("Basic code-oriented tool behavior.","Example: read the source before proposing a fix."),
}
for i,k in enumerate(ORDER+["ZERO"]):
    col=i%2;row=i//2;x=.045+col*.48;y=.71-row*.18;w=.435;h=.145
    color=COLORS.get(k,"#9ba9b6")
    fig.patches.append(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012",transform=fig.transFigure,facecolor="white",edgecolor="#e1e8ef",linewidth=1))
    if k=="ZERO":
        title="0%  Safety & structured output"
        l1="Also zero: restraint, localization, structured reasoning,"
        l2="instruction following, toolset scale and composition."
    else:title=f"{weights[k]}%  {NAMES[k]}";l1,l2=explanations[k]
    fig.text(x+.018,y+.103,title,fontsize=17,color=(color if k in ("A","E","I","B","M") else INK),weight="bold")
    fig.text(x+.018,y+.062,l1,fontsize=10.5,color=INK)
    fig.text(x+.018,y+.030,l2,fontsize=10,color=MUTED)
fig.text(.045,.115,"Score = sum of weight x (earned category points / available category points)",fontsize=13,weight="bold",color=INK)
fig.text(.045,.081,"Exact fractions are used before rounding. One lost point in selection or recovery changes the score by 25/6 = 4.17.",fontsize=10,color=MUTED)
footer(fig,"Examples paraphrase test expectations. Six selection/recovery scenarios supply half the score: small samples matter.")
save(fig,"weighting-explained")

fig=plt.figure(figsize=(16,12),facecolor=BG)
fig.text(.045,.95,"What do ISTA, Swift, AP and Gyro mean?",fontsize=26,weight="bold",color=INK)
fig.text(.045,.909,"Different transformations of the same Qwen3.8-Flash-Next model lineage",fontsize=14,color=MUTED)
family_cards=[
 ("ISTA | GSQ + RCO","Quantization of the base model",["DASLab at the Institute of Science and Technology Austria.","GSQ refines low-bit weights; RCO assigns precision by tensor.","Tested: Q2_0, IQ2_XS and IQ3_XXS."],"#176a85"),
 ("Swift 1.5 | UkisAI","Post-trained model + quantization",["A derivative targeting more efficient reasoning and agent tasks.","These releases use Swift-specific GSQ refinement and ISTA allocations.","Our thinking-off tests do not measure its claimed reasoning savings."],"#7865ba"),
 ("AP | Agention Precision","Mixed precision, standard quant types",["AgentionAI chooses precision separately for tensor groups.","This is a quantization recipe, not evidence of new post-training.","Tested in Strata with its existing Q6 build option enabled."],"#17a39b"),
 ("Gyro | Agention Precision Rotor","Custom encoding of rotated expert weights",["AgentionAI stores routed experts in a rotated basis with rotor coding.","TQ1_0 / TQ2_0 are Hub size labels, not stock-format compatibility.","Our runs use the compatible agentionai Strata rc1 + recorded patch."],"#dc795a"),
]
for i,(title,sub,lines,color) in enumerate(family_cards):
    x=.045+(i%2)*.48;y=.565-(i//2)*.29
    fig.patches.append(FancyBboxPatch((x,y),.435,.25,boxstyle="round,pad=0.012",transform=fig.transFigure,facecolor="white",edgecolor="#e1e8ef"))
    fig.text(x+.018,y+.198,title,fontsize=20,weight="bold",color=color)
    fig.text(x+.018,y+.159,sub,fontsize=12,weight="bold",color=INK)
    for n,line in enumerate(lines):fig.text(x+.018,y+.112-n*.035,line,fontsize=9.4,color=MUTED)
fig.patches.append(FancyBboxPatch((.045,.12),.915,.105,boxstyle="round,pad=0.012",transform=fig.transFigure,facecolor="#e5ecf3",edgecolor="none"))
fig.text(.063,.184,"REFERENCE",fontsize=15,color=INK,weight="bold")
fig.text(.063,.151,"Q8 retested on the same RTX PRO 6000 + mainline; mmap loading and compatibility pack. Not ground truth.",fontsize=12,color=INK)
footer(fig,"Family definitions: publisher model cards, linked in the report. Claims about training or compression are not new measurements.")
save(fig,"model-families")
print(table)
