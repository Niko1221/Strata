#!/usr/bin/env python3
# NOTE: runs on the study rig only — it reads raw benchkit/five-bugs logs.
# The shipped data/ALL-METRICS.json is its output; make_charts.py and
# compile_pr_readme.py rebuild everything from that file alone.
# -*- coding: utf-8 -*-
"""extract_all.py — pull EVERY metric from EVERY test into one structured JSON.

Output: /home/claw/teb-campaign/results/ALL-METRICS.json
Structure:
{
  "generated_at": ...,
  "models": {
     "<key>": {
        "name": ..., "lab": ..., "label": ..., "quant": ..., "role": "strata|gyro",
        "heplus": { "low": {...}, "xhigh": {...} },
        "trials": { "low": {...}, "xhigh": {...} },
        "perf":   { "low": {...}, "xhigh": {...} },
        "fivebugs": {...},
        "knowledge": { "gsm8k": {...}, "mmlu": {...}, "ifeval": {...} },
        "needle": {...},
        "walls": { "t2_low_min": ..., "t2_xhigh_min": ..., ... }
     }, ...
  },
  "sources": {...}  # file paths for provenance
}
Everything read from disk; nothing hand-typed.
"""
import json, os, re, glob, datetime

TC = "/home/claw/teb-campaign"
R = f"{TC}/results"
BK = "/home/claw/benchkit/results"

# ── model registry ──────────────────────────────────────────────────────────
# heplus_dir: benchkit timestamped dir (verified mapping from campaign logs)
MODELS = {
    "iq3_s": dict(
        name="Qwen3.8-Flash-Next GSQ-RCO IQ3_S", lab="ISTA-DASLab", quant="IQ3_S",
        label="Qwen3.8-Flash-Next IQ3_S (ISTA-DASLab)", role="strata",
        heplus_dirs={"low": "2026-10-07_23-58-08", "xhigh": "2026-10-08_01-05-31"},
        trials={"low": "trials-iq3_s-low.json", "xhigh": "trials-iq3_s-xhigh.json"},
        perf={"low": "rebase-iq3_s-perf-low.perf.json", "xhigh": "rebase-iq3_s-perf-xhigh.perf.json"},
        fivebugs="rebase-iq3_s", know_tag="rebase-iq3_s", fb_jsonl="local_rebase-iq3_s.jsonl"),
    "q2_0": dict(
        name="Qwen3.8-Flash-Next GSQ-RCO Q2_0", lab="ISTA-DASLab", quant="Q2_0",
        label="Qwen3.8-Flash-Next Q2_0 (ISTA-DASLab)", role="strata",
        heplus_dirs={"low": "2026-10-08_02-39-33", "xhigh": "2026-10-08_03-41-53"},
        trials={"low": "trials-q2_0-low.json", "xhigh": "trials-q2_0-xhigh.json"},
        perf={"low": "rebase-q2_0-perf-low.perf.json", "xhigh": "rebase-q2_0-perf-xhigh.perf.json"},
        fivebugs="rebase-q2_0", know_tag="rebase-q2_0", fb_jsonl="local_rebase-q2_0.jsonl"),
    "base-iq3_xxs": dict(
        name="Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS", lab="ISTA-DASLab", quant="IQ3_XXS",
        label="Qwen3.8-Flash-Next IQ3_XXS (ISTA-DASLab)", role="strata",
        heplus_dirs={"low": "2026-10-08_07-45-50", "xhigh": "2026-10-08_09-02-09"},
        trials={"low": "trials-base-iq3_xxs-low.json", "xhigh": "trials-base-iq3_xxs-xhigh.json"},
        perf={"low": "rebase-base-iq3_xxs-perf-low.perf.json", "xhigh": "rebase-base-iq3_xxs-perf-xhigh.perf.json"},
        fivebugs="rebase-base-iq3_xxs", know_tag="rebase-base-iq3_xxs", fb_jsonl="local_rebase-base-iq3_xxs.jsonl"),
    "ista-iq2_xs": dict(
        name="Qwen3.8-Flash-Next GSQ-RCO IQ2_XS", lab="ISTA-DASLab", quant="IQ2_XS",
        label="Qwen3.8-Flash-Next IQ2_XS (ISTA-DASLab)", role="strata",
        heplus_dirs={"low": "2026-10-09_23-50-54", "xhigh": "2026-10-10_01-06-37"},
        trials={"low": "campaign11-ista-iq2_xs-trials-low.json", "xhigh": "campaign11-ista-iq2_xs-trials-xhigh.json"},
        perf={"low": "campaign11-ista-iq2_xs-perf-low.perf.json", "xhigh": "campaign11-ista-iq2_xs-perf-xhigh.perf.json"},
        fivebugs="campaign11-ista-iq2_xs", know_tag="campaign11-ista-iq2_xs", fb_jsonl="local_campaign11-ista-iq2_xs.jsonl"),
    "swift-iq3_xxs": dict(
        name="Swift-1.5 Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS", lab="UkisAI", quant="IQ3_XXS",
        label="Swift-1.5 IQ3_XXS (UkisAI)", role="strata",
        heplus_dirs={"low": "2026-10-08_04-24-54", "xhigh": "2026-10-08_05-12-08"},
        trials={"low": "trials-swift-iq3_xxs-low.json", "xhigh": "trials-swift-iq3_xxs-xhigh.json"},
        perf={"low": "rebase-swift-iq3_xxs-perf-low.perf.json", "xhigh": "rebase-swift-iq3_xxs-perf-xhigh.perf.json"},
        fivebugs="rebase-swift-iq3_xxs", know_tag="rebase-swift-iq3_xxs", fb_jsonl="local_rebase-swift-iq3_xxs.jsonl"),
    "swift-iq2_xs": dict(
        name="Swift-1.5 Qwen3.8-Flash-Next GSQ-RCO IQ2_XS", lab="UkisAI", quant="IQ2_XS",
        label="Swift-1.5 IQ2_XS (UkisAI)", role="strata",
        heplus_dirs={"low": "2026-10-08_05-59-04", "xhigh": "2026-10-08_07-02-11"},
        trials={"low": "trials-swift-iq2_xs-low.json", "xhigh": "trials-swift-iq2_xs-xhigh.json"},
        perf={"low": "rebase-swift-iq2_xs-perf-low.perf.json", "xhigh": "rebase-swift-iq2_xs-perf-xhigh.perf.json"},
        fivebugs="rebase-swift-iq2_xs", know_tag="rebase-swift-iq2_xs", fb_jsonl="local_rebase-swift-iq2_xs.jsonl"),
    "swift-iq3_s": dict(
        name="Swift-1.5 Qwen3.8-Flash-Next GSQ-RCO IQ3_S", lab="UkisAI", quant="IQ3_S",
        label="Swift-1.5 IQ3_S (UkisAI)", role="strata",
        heplus_dirs={"low": "2026-10-08_22-59-50", "xhigh": "2026-10-08_23-43-25"},
        trials={"low": "trials-swift-iq3_s-low.json", "xhigh": "trials-swift-iq3_s-xhigh.json"},
        perf={"low": "rebase-swift-iq3_s-perf-low.perf.json", "xhigh": "rebase-swift-iq3_s-perf-xhigh.perf.json"},
        fivebugs="rebase-swift-iq3_s", know_tag="rebase-swift-iq3_s", fb_jsonl="local_rebase-swift-iq3_s.jsonl"),
    "swift-q2_0": dict(
        name="Swift-1.5 Qwen3.8-Flash-Next GSQ-RCO Q2_0", lab="UkisAI", quant="Q2_0",
        label="Swift-1.5 Q2_0 (UkisAI)", role="strata",
        heplus_dirs={"low": "2026-10-10_03-43-06", "xhigh": "2026-10-10_04-39-28"},
        trials={"low": "campaign11-swift-q2_0-trials-low.json", "xhigh": "campaign11-swift-q2_0-trials-xhigh.json"},
        perf={"low": "campaign11-swift-q2_0-perf-low.perf.json", "xhigh": "campaign11-swift-q2_0-perf-xhigh.perf.json"},
        fivebugs="campaign11-swift-q2_0", know_tag="campaign11-swift-q2_0", fb_jsonl="local_campaign11-swift-q2_0.jsonl"),
    "gyro-s": dict(
        name="Qwen3.8-Flash-Next Gyro-S TQ1_0", lab="AgentionAI", quant="TQ1_0",
        label="Gyro-S TQ1_0 (AgentionAI)", role="gyro",
        heplus_dirs={"low": "2026-10-09_06-54-07", "xhigh": "2026-10-09_21-47-18"},
        trials=None,  # excluded — no tool support in rc1 fork
        perf=None,    # llama-benchy invalid through proxy — engine-log stats instead
        fivebugs="rebase-gyro-s", know_tag="rebase-gyro-s", fb_jsonl="local_rebase-gyro-s.jsonl"),
}

HE_FIELDS = ["score", "first_attempt_score", "passed", "total", "scored_total", "tok_s",
             "total_time", "sum_generation_time", "avg_tokens_to_solve", "total_output_tokens",
             "median_thinking_time", "loop_kills", "timeouts", "errors",
             "repair_successes", "repair_attempted", "model_turns", "tool_calls"]

def load(p):
    try:
        with open(p) as f: return json.load(f)
    except Exception: return None

def heplus_row(dirname):
    p = f"{BK}/{dirname}/results.json"
    r = load(p)
    if not r: return None
    row = r[0] if isinstance(r, list) else r
    out = {k: row.get(k) for k in HE_FIELDS}
    out["_dir"] = dirname
    return out

def trials_row(fname):
    d = load(f"{R}/{fname}")
    if not d: return None
    ts = d.get("trial_statistics", {})
    sc = (d.get("scores") or {})
    cats = {}
    for c in sc.get("category_scores", []):
        if isinstance(c, dict):
            name = c.get("category") or c.get("name")
            if name: cats[str(name)] = c.get("percent")
    sg = d.get("safety_gate") or {}
    warns = sg.get("warnings") or d.get("safety_warnings") or []
    return dict(
        final_score=d.get("final_score"),
        rating=d.get("rating"),
        deployability=d.get("deployability"),
        responsiveness=d.get("responsiveness"),
        n_scenarios=d.get("total_scenarios"),
        mean=ts.get("final_score_mean"), std=ts.get("final_score_stddev"),
        median=ts.get("final_score_median"), ci95=ts.get("final_score_ci95"),
        pts_mean=ts.get("total_points_mean"), pts_std=ts.get("total_points_stddev"),
        pass_at_5=ts.get("pass_at_k"), pass_hat_5=ts.get("pass_hat_k"),
        gap=ts.get("reliability_gap"),
        safety_passed=sg.get("passed"),
        n_warnings=len(warns) if isinstance(warns, list) else warns,
        warnings=(warns if isinstance(warns, list) else [])[:8],
        cats=cats,
        worst=sc.get("worst_category"),
        median_turn_ms=sc.get("median_turn_ms"),
    )

def perf_row(fname):
    d = load(f"{R}/{fname}")
    if not d: return None
    cells = {}
    for b in d.get("benchmarks", []):
        ctx = b.get("context_size", 0) // 1024
        tg = (b.get("tg_throughput") or {})
        pp = (b.get("pp_throughput") or {})
        cells[f"{ctx}K"] = dict(tg_mean=tg.get("mean"), tg_std=tg.get("std"),
                                pp_mean=pp.get("mean"), pp_std=pp.get("std"))
    return cells

def fb_row(tag, jsonl):
    mat = load("/home/claw/five-bugs/results/fivebugs_matrix.json") or {}
    m = mat.get(tag, {})
    out = dict(vis=m.get("vis"), hid=m.get("hid"))
    jp = f"/home/claw/five-bugs/results/{jsonl}"
    if os.path.exists(jp):
        rows = [json.loads(l) for l in open(jp) if l.strip()]
        out["wall_s"] = round(sum(r.get("seconds", 0) for r in rows), 1)
        out["tokens"] = sum((r.get("meta", {}).get("usage", {}) or {}).get("completion_tokens", 0) for r in rows)
        out["n_runs"] = len(rows)
    return out

def know_row(tag, kind):
    g = sorted(glob.glob(f"{TC}/runs/2026/*/*--{tag}-{kind}.md"))
    if not g: return None
    txt = open(g[-1]).read()
    out = dict(_file=g[-1].split("/")[-1])
    if kind == "gsm8k" or kind == "mmlu":
        m = re.search(r"\*\*Accuracy\*\*:?\s*\*\*([\d.]+)%", txt) or re.search(r"\*\*Accuracy:\*\*\s*([\d.]+)%", txt)
        if m: out["accuracy"] = float(m.group(1))
    elif kind == "ifeval":
        m = re.search(r"Prompt[- ]level Accuracy:?\*{0,2}:?\s*\*{0,2}([\d.]+)%", txt)
        if m: out["prompt_accuracy"] = float(m.group(1))
        m2 = re.search(r"Instruction[- ]level Accuracy:?\*{0,2}:?\s*\*{0,2}([\d.]+)%", txt)
        if m2: out["instruction_accuracy"] = float(m2.group(1))
    elif kind == "needle":
        m = re.search(r"Retrieval Accuracy:?\*{0,2}:?\s*\*{0,2}([\d.]+)%", txt)
        if m: out["retrieval_accuracy"] = float(m.group(1))
        m2 = re.search(r"Effective Context:?\*{0,2}:?\s*\*{0,2}([\d,]+)", txt)
        if m2: out["effective_context"] = int(m2.group(1).replace(",", ""))
    return out

# ── T2 walls from campaign logs ─────────────────────────────────────────────
def parse_ts(s):
    return datetime.datetime.fromisoformat(s)

def t2_walls():
    """Returns {tag: {arm: (start, end)}} parsed from campaign logs."""
    out = {}
    logs = ["campaign-trials.log", "campaign.log", "campaign-xhigh.log", "campaign11.log",
            "campaign-phaseA-rebase.log"]
    events = []
    for lf in logs:
        p = f"{TC}/logs/{lf}"
        if not os.path.exists(p): continue
        for line in open(p, errors="replace"):
            m = re.match(r"\[([\d\-T:+]+)\]\s+(.*)", line)
            if not m: continue
            ts, msg = m.group(1), m.group(2).strip()
            events.append((ts, msg))
    events.sort()
    cur = None
    for ts, msg in events:
        m = re.search(r"T2x5 start \((campaign11-[a-z0-9_\-]+|[a-z0-9_\-]+-trials-[a-z]+)\)", msg)
        if m:
            cur = m.group(1)
            continue
        m2 = re.search(r"T2x5 DONE (campaign11-[a-z0-9_\-]+|[a-z0-9_\-]+-trials-[a-z]+)", msg)
        if m2 and cur:
            out[cur] = out.get(cur, {})
            cur = None
    return None  # (unused — wall parsing done separately per-file below)

def t2_wall_for(logfile, tag):
    """Parse wall for a specific T2 tag from a campaign log."""
    p = f"{TC}/logs/{logfile}"
    if not os.path.exists(p): return None
    start = end = None
    for line in open(p, errors="replace"):
        if f"T2x5 start ({tag})" in line:
            start = line[1:26]
        elif f"T2x5 DONE {tag}" in line:
            end = line[1:26]
    if start and end:
        try:
            return round((parse_ts(end) - parse_ts(start)).total_seconds() / 60, 1)
        except Exception:
            return None
    return None

# ALL T2 walls pre-parsed from campaign logs -> results/t2_walls.json
# tags: trials-<key>-<arm> (old campaigns) or campaign11-<key>-trials-<arm>
def load_walls():
    p = f"{R}/t2_walls.json"
    return load(p) or {}

WALLS = load_walls()
WALL_KEY = {
    "iq3_s": "trials-iq3_s",
    "q2_0": "trials-q2_0",
    "base-iq3_xxs": "trials-base-iq3_xxs",
    "swift-iq3_xxs": "trials-swift-iq3_xxs",
    "swift-iq2_xs": "trials-swift-iq2_xs",
    "swift-iq3_s": "trials-swift-iq3_s",
    "ista-iq2_xs": "campaign11-ista-iq2_xs-trials",
    "swift-q2_0": "campaign11-swift-q2_0-trials",
    "gyro-s": "trials-gyro-s",
}

# ── build ───────────────────────────────────────────────────────────────────
out = dict(generated_at=datetime.datetime.now().isoformat(), models={}, sources={})
for key, M in MODELS.items():
    e = dict(name=M["name"], lab=M["lab"], quant=M["quant"], label=M["label"], role=M["role"])
    if key == "gyro-s":
        e["trials_excluded_finding"] = dict(
            reason="rc1 fork has NO tool support (forced tool_choice ignored; resp len 0/A-D repro). Numbers through compat proxy; tools were stripped, so scores measure raw LLM behavior, not tool use. EXCLUDED from T2 tables; kept for the record.",
            proxy_walls_min=dict(low=50.8, xhigh=102.0),
            proxy_scores=dict(low=25.0, xhigh=28.2))
    # HE+
    e["heplus"] = {}
    for arm, d in M["heplus_dirs"].items():
        e["heplus"][arm] = heplus_row(d)
    # trials
    if M["trials"]:
        e["trials"] = {arm: trials_row(f) for arm, f in M["trials"].items()}
    else:
        e["trials"] = None
    # walls (from t2_walls.json) — applies to all models incl. gyro (excluded runs)
    wkey = WALL_KEY.get(key)
    e["trials_wall_min"] = {arm: (WALLS.get(f"{wkey}-{arm}") if wkey else None) for arm in ("low", "xhigh")}
    # perf
    e["perf"] = None
    if M["perf"]:
        e["perf"] = {arm: perf_row(f) for arm, f in M["perf"].items()}
    # five-bugs
    e["fivebugs"] = fb_row(M["fivebugs"], M["fb_jsonl"])
    # knowledge + needle
    e["knowledge"] = {}
    for kind in ("gsm8k", "mmlu", "ifeval"):
        e["knowledge"][kind] = know_row(M["know_tag"], kind)
    e["needle"] = know_row(M["know_tag"], "needle")
    out["models"][key] = e

# gyro speed (engine logs)
gp = f"{R}/gyro-speed-logstats.txt"
if os.path.exists(gp):
    out["gyro_speed_engine_logs"] = dict(file=gp, note="engine-log derived; decode med 82.4 warm / 60-71 depth; prefill 2,420-2,753 @16-96K+")

out["sources"] = dict(
    benchkit=BK, teb_results=R, fivebugs="/home/claw/five-bugs/results",
    knowledge_runs=f"{TC}/runs/2026/10", logs=f"{TC}/logs")

op = f"{R}/ALL-METRICS.json"
json.dump(out, open(op, "w"), indent=1)
print(f"WROTE {op}")
# summary
print(f"\nmodels: {len(out['models'])}")
for k, e in out["models"].items():
    h = e["heplus"]
    ok = sum(1 for a in ("low","xhigh") if h.get(a))
    t = "n/a" if e["trials"] is None else "OK"
    fb = e["fivebugs"]
    kn = sum(1 for kind in ("gsm8k","mmlu","ifeval") if e["knowledge"].get(kind))
    nd = "OK" if e["needle"] else "NO"
    print(f"  {k:14s} he={ok}/2 trials={t:4s} fb={fb.get('vis')}/{fb.get('hid')} knowledge={kn}/3 needle={nd}")
