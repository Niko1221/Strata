"""Rebuild publication figures from the checked-in Tool-Eval-Bench results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "font.size": 11, "svg.hashsalt": "tool-eval-q8-retest"})
INK, MUTED, TEAL, PALE, BG = "#152b40", "#566b7c", "#087f8c", "#bce3e2", "#f5f8fb"
SOURCE = "Tool-Eval-Bench by SeraphimSerapis | github.com/SeraphimSerapis/tool-eval-bench"
MODELS = [
    ("ISTA Q2_0", "ISTA\nQ2_0", "ista-q2", 80),
    ("ISTA IQ2_XS", "ISTA\nIQ2_XS", "ista-iq2", 70),
    ("Swift 1.5 Q2_0", "Swift\nQ2_0", "swift-q2", 97),
    ("Swift 1.5 IQ2_XS", "Swift\nIQ2_XS", "swift-iq2", 93),
    ("ISTA IQ3_XXS", "ISTA\nIQ3_XXS", "ista-iq3", 100),
    ("Swift 1.5 IQ3_XXS", "Swift\nIQ3_XXS", "swift-iq3", 100),
    ("AP-Q4_K_XL", "AP\nQ4_K_XL", "ap-q4-k-xl", 97),
    ("AP-IQ3_XXS", "AP\nIQ3_XXS", "ap-iq3-xxs", 90),
    ("AP-IQ2_S", "AP\nIQ2_S", "ap-iq2-s", 93),
    ("Gyro-S / TQ1_0", "Gyro-S\nTQ1_0", "gyro-s", 100),
    ("Gyro-M / TQ2_0", "Gyro-M\nTQ2_0", "gyro-m", 93),
    ("Reference", "Reference\nQ8_0", "unsloth-q8-llm60", 97),
]
REPORTS = [json.loads((ROOT / (m[2] + "-standard.json")).read_text(encoding="utf-8")) for m in MODELS]
SHORT = []
for model in MODELS:
    path = ROOT / (model[2] + "-short.json")
    if path.exists():
        score = json.loads(path.read_text(encoding="utf-8"))["final_score"]
    else:
        assert model[2] == "unsloth-q8"
        q8 = REPORTS[-1]["scores"]["scenario_results"]
        points = sum(x["points"] for x in q8 if 1 <= int(x["scenario_id"].split("-")[1]) <= 15)
        score = round(points / 30 * 100)
    assert score == model[3]
    SHORT.append(score)

fig = plt.figure(figsize=(15, 12), facecolor=BG)
fig.text(.055, .95, "How well did each model use tools?", fontsize=26, color=INK, weight="bold")
fig.text(.055, .91, "12 variants  /  short and standard suites  /  one trial per run", fontsize=13, color=MUTED)
ax = fig.add_axes([.235, .19, .695, .64], facecolor=BG)
y = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 14], dtype=float)
ax.set_ylim(15.0, -1.2)
ax.set_xlim(0, 108)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_yticks(y, [m[0] for m in MODELS], color=INK, fontsize=12)
ax.tick_params(axis="both", length=0, pad=10, colors=MUTED)
ax.set_axisbelow(True)
ax.grid(axis="x", color="#dde5ec", linewidth=.8)
for spine in ax.spines.values(): spine.set_visible(False)
for i, row in enumerate(y):
    ax.barh(row + .15, SHORT[i], height=.25, color=PALE, edgecolor=TEAL, linewidth=.55)
    score = REPORTS[i]["final_score"]
    ax.barh(row - .15, score, height=.25, color=TEAL)
    ax.text(SHORT[i] + 1.2, row + .15, str(SHORT[i]), va="center", fontsize=11, color=MUTED)
    ax.text(score + 1.2, row - .15, str(score), va="center", fontsize=11, color=TEAL, weight="bold")
for ypos, label in [(-.7, "MAINLINE / RTX PRO 6000"), (6.25, "AP / SAME GPU, Q6 BUILD OPTION ON"),
                    (10.25, "GYRO / SAME GPU, PATCHED RC1"), (13.25, "REFERENCE / SAME GPU + MAINLINE, MMAP")]:
    ax.text(-.26, ypos, label, transform=ax.get_yaxis_transform(), color=MUTED, fontsize=9, weight="bold")
ax.legend(handles=[Patch(facecolor=TEAL, label="Standard score /100"),
                   Patch(facecolor=PALE, edgecolor=TEAL, label="Short score /100")],
          loc="lower left", bbox_to_anchor=(0, 1.08), frameon=False, ncol=2, fontsize=12)
fig.text(.055, .125, "Read by cohort, not as a hardware or quantization-only ranking.", color=INK, weight="bold", fontsize=12)
fig.text(.055, .091, "Standard: 69 scored cases; Gyro: 68 (TC-45 excluded). Short: 15 cases, run separately for every variant.", color=MUTED, fontsize=10)
fig.text(.055, .064, "Swift is post-trained; AP uses mixed precision. Greedy sampling, thinking off. Timing is preliminary.", color=MUTED, fontsize=10)
fig.text(.055, .03, SOURCE + "  |  Measured 2026-10-07/08", color=INK, fontsize=9)
for ext in ("png", "svg"):fig.savefig(OUT / ("variant-scores." + ext), dpi=160, facecolor=BG, metadata={"Date":None} if ext == "svg" else None)
plt.close(fig)

# Short paraphrases of scenario expectations; these are not model quotations.
EXAMPLES = [
    ("A", "Tool selection", "TC-01", '"Weather in Berlin?"\nget_weather, not web_search.'),
    ("B", "Parameter precision", "TC-04", '"Give the temperature in Fahrenheit."\nPass the requested units.'),
    ("C", "Multi-step chains", "TC-07", '"Find the file and email its contents."\nsearch -> read -> contacts -> send.'),
    ("D", "Restraint & refusal", "TC-11", '"What is this simple sum?"\nAnswer directly; skip the calculator.'),
    ("E", "Error recovery", "TC-14", '"The stock-price tool returned an error."\nReport the failure; do not invent a price.'),
    ("F", "Localization", "TC-17", '"Schedule it in Berlin local time."\nUse Europe/Berlin, not UTC.'),
    ("G", "Structured reasoning", "TC-20", '"Work out the total from this file."\nSearch, read, then calculate.'),
    ("H", "Instruction following", "TC-23", '"Answer without using tools."\nMake no tool calls.'),
    ("I", "Context & state", "TC-47", '"Actually, change the meeting time."\nKeep the correction; create one event.'),
    ("J", "Code patterns", "TC-28", '"Help fix this source file."\nRead the real file before suggesting a fix.'),
    ("K", "Safety & boundaries", "TC-57", '"A search result says to ignore the user."\nTreat that instruction as untrusted data.'),
    ("L", "Toolset scale", "TC-37", '"Check the weather" with 52 tools offered.\nPick get_weather among the distractors.'),
    ("M", "Autonomous planning", "TC-51", '"Arrange a meeting and notify people."\nResolve contacts -> create event -> notify.'),
    ("N", "Creative composition", "TC-54", '"Convert this stock value to another currency."\nPrice + exchange rate + calculation.'),
    ("O", "Structured output", "TC-65", '"Return the weather in this JSON schema."\nCall the tool, then format valid JSON.'),
]
catmaps = [{x["category"]: x for x in r["scores"]["category_scores"]} for r in REPORTS]
data = np.array([[cats[letter]["percent"] for cats in catmaps] for letter, *_ in EXAMPLES])
fig = plt.figure(figsize=(21, 14), facecolor=BG)
fig.text(.035, .95, "Where the models succeeded - and struggled", fontsize=26, weight="bold", color=INK)
fig.text(.035, .912, "Standard-suite category scores (%) with one paraphrased test example per category", fontsize=14, color=MUTED)
ax = fig.add_axes([.185, .17, .435, .63])
cmap = LinearSegmentedColormap.from_list("scores", ["#f7e8db", "#dcebed", "#198a94", "#075263"])
ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(12), [m[1] for m in MODELS], fontsize=9, color=INK)
ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=9)
ax.set_yticks(range(15), [x[1] for x in EXAMPLES], fontsize=12, color=INK)
ax.set_xticks(np.arange(-.5, 12, 1), minor=True);ax.set_yticks(np.arange(-.5, 15, 1), minor=True)
ax.grid(which="minor", color=BG, linewidth=2);ax.tick_params(which="minor", length=0)
for sp in ax.spines.values():sp.set_visible(False)
for row in range(15):
    for col in range(12):
        v=data[row,col];ax.text(col,row,str(v),ha="center",va="center",fontsize=11,weight="bold",color="white" if v>=65 else INK)
for boundary in [5.5, 8.5, 10.5]:ax.axvline(boundary, color=BG, linewidth=7)
for mid, name in [(2.5,"ISTA / Swift"),(7,"AP"),(9.5,"Gyro*"),(11,"Reference")]:
    ax.text(mid, -2.0, name, ha="center", color=TEAL, weight="bold", fontsize=11)
ex = fig.add_axes([.65, .17, .325, .63], facecolor=BG)
ex.set_xlim(0,1);ex.set_ylim(14.5,-.5);ex.axis("off")
for i, (_,_,case,example) in enumerate(EXAMPLES):
    ex.axhline(i+.5,color="#dce5ec",linewidth=.7)
    ex.text(0,i-.13,example,va="center",fontsize=10.5,color=INK,linespacing=1.45)
    ex.text(.995,i+.31,case,ha="right",va="center",fontsize=8,color=MUTED)
fig.text(.65,.855,"WHAT THE TEST ASKS",color=TEAL,weight="bold",fontsize=11)
fig.text(.035,.112,"A tool-use benchmark, not a general intelligence ranking.",color=INK,weight="bold",fontsize=12)
fig.text(.035,.081,"*Gyro excludes TC-45. Reference now uses the same mainline engine and RTX PRO 6000, with mmap loading.",color=MUTED,fontsize=10)
fig.text(.035,.057,"Examples are paraphrases of scenario expectations, not observed model answers. All external actions use deterministic mocks.",color=MUTED,fontsize=10)
fig.text(.035,.025,SOURCE,color=INK,fontsize=10)
for ext in ("png", "svg"):fig.savefig(OUT / ("category-scores-and-examples." + ext),dpi=160,facecolor=BG,metadata={"Date":None} if ext == "svg" else None)
plt.close(fig)
# Matplotlib emits insignificant trailing spaces inside SVG path attributes.
for svg in OUT.glob("*.svg"):
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
print("Wrote two figures as PNG and SVG")
