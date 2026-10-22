#!/usr/bin/env python3
"""eval/jevport-suite.py — small per-stage calibration suites (Justin's ask).

ONE text suite + ONE image suite, each covering noul / choice / score, using the
best sets from the Opus plan (vision-decision-strategy-plan.md §1):

TEXT  (already staged in-container from the M3.2 work):
  noul   BoolQ  (dev 60 fit / test 100)
  choice ARC    (dev 60 fit / test 100)
  score  SST-5  (dev 60 fit / test 100)

IMAGE (new, staged by this script into the same tree):
  noul   POPE adversarial     — HF lmms-lab/POPE (fallback: synthetic if offline)
  choice CV-Bench             — HF nyu-visionx/CV-Bench (fallback: synthetic)
  score  CountBenchQA         — HF vikhyatk/CountBenchQA (fallback: synthetic)

Each suite: raw pass -> offline Platt/T fit on dev -> calibrated test pass ->
acc / NLL / ECE15 before-after table -> md + json report.  Exit 0 on success
(it is a measurement, not a gate).

usage: jevport-suite.py <base_url> <key> <outdir> text|image [fit_n test_n]
"""
import base64
import hashlib
import io
import json
import math
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

NEUTRAL_VPROMPT = ("Describe this image in detail, including objects, their "
                   "counts, colors, positions and any text.")

ITEM_LOG = []          # per-item rows for the adversarial 1-in-10 audit


# ---------------------------------------------------------------- common
def post(base, key, body, timeout=900):
    req = urllib.request.Request(base + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def ece15(ps, gs, bins=15):
    """STANDARD top-label ECE (the donor b13_eval protocol): confidence = max(pr),
    correctness = argmax == gold.  The first version of this suite used the
    gold-class probability with correctness pinned 1 - an NLL-style variant that
    MOVES THE WRONG WAY under temperature flattening and made calibration look
    worse than it is (Justin caught the tables, 2026-10-15)."""
    n, e = len(ps), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i in range(n) if lo <= ps[i] < hi or (hi == 1 and ps[i] == 1)]
        if not idx:
            continue
        conf = sum(ps[i] for i in idx) / len(idx)
        acc = sum(gs[i] for i in idx) / len(idx)
        e += len(idx) / n * abs(conf - acc)
    return e


def nll(ps, gs):
    return -sum(math.log(max(p, 1e-12)) for p, g in zip(ps, gs) if g == 1) / max(
        1, sum(1 for g in gs if g == 1)) if any(g == 1 for g in gs) else 0.0


def _sig(z):
    """overflow-safe logistic (logits here can be +-30)."""
    z = max(-700.0, min(700.0, z))
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def fit_platt_ex(ds, cs, flat_rel=0.005):
    """Platt on the yes-logit.  Same identity preference as fit_temp_ex: a fit that does not
    improve NLL by `flat_rel` relative is reported as identity (1, 0) instead of shipping a
    slope indistinguishable from 1 - which is what happened on BoolQ (a=1.009, c=0.0037).

    -> dict(a, c, nll, nll_identity, improvement, flat, boundary)
    """
    def nll(a_, b_):
        t = 0.0
        for d, y in zip(ds, cs):
            p = min(max(_sig(a_ * d + b_), 1e-12), 1 - 1e-12)
            t -= math.log(p if y else 1 - p)
        return t / len(ds)

    a, b = 1.0, 0.0
    lr = 0.1
    for _ in range(500):
        ga = gc = 0.0
        for d, y in zip(ds, cs):
            p = _sig(a * d + b)
            ga += (p - y) * d
            gc += (p - y)
        a, b = a - lr * ga / len(ds), b - lr * gc / len(ds)
    if a <= 0:                  # a negative slope = the labels are inverted, not the
        raise ValueError("Platt slope a <= 0: raw answers disagree with gold "  # model's probabilities
                         "systematically - fix the suite/prompt before calibrating")
    nid, nfit = nll(1.0, 0.0), nll(a, b)
    improvement = (nid - nfit) / max(nid, 1e-12)
    if improvement < flat_rel:
        return {"a": 1.0, "c": 0.0, "nll": nid, "nll_identity": nid, "improvement": 0.0,
                "flat": True, "boundary": False, "candidate": {"a": round(a, 6), "c": round(b, 6)}}
    return {"a": round(a, 6), "c": round(b, 6), "nll": nfit, "nll_identity": nid,
            "improvement": round(improvement, 6), "flat": False,
            "boundary": a < 0.02 or a > 50.0}


def fit_platt(ds, cs):          # kept for callers that only want the numbers
    fi = fit_platt_ex(ds, cs)
    return fi["a"], fi["c"]


def _nll_temp(rows, t):
    """Mean NLL of the gold label under p**(1/t) (the same map apply_temp uses)."""
    tot = 0.0
    for pr, g in rows:
        lm = math.log(max(max(pr), 1e-300))
        lse = math.log(sum(math.exp((math.log(max(x, 1e-300)) - lm) / t) for x in pr))
        tot -= (math.log(max(pr[g], 1e-300)) - lm) / t - lse
    return tot / len(rows)


def fit_temp_ex(rows, lo=0.05, hi=10.0, flat_rel=0.005, grid=1.1):
    """Temperature by NLL with three fixes over the old flat 10%-grid scan ($fit_temp):

    1. PRECISION - a coarse log-grid only BRACKETS the optimum; golden-section then refines
       inside it to ~1e-4 relative.  The old version returned a grid point (0.05*1.1**32 =
       1.0557) and that exact value appearing for two different suites is what exposed it.
    2. IDENTITY PREFERENCE - a temperature is only worth shipping if it improves NLL by more
       than `flat_rel` (0.5%) relative to t=1.  On a flat likelihood the old scan returned the
       lowest grid point near the minimum, i.e. a spurious non-1 T that then made ECE worse.
    3. BOUNDARY REPORTING - an optimum pinned at lo/hi means the readout is degenerate; the
       caller must be able to reject that tuple instead of quoting the edge value as a fit.

    -> dict(T, nll, nll_identity, improvement, flat, boundary)
    """
    nll_id = _nll_temp(rows, 1.0)
    best_t, best_nll = 1.0, nll_id
    t = lo
    while t <= hi * 1.000001:
        v = _nll_temp(rows, t)
        if v < best_nll:
            best_t, best_nll = t, v
        t *= grid
    # bracket the coarse minimum, then refine in log space (the map is smooth in log t)
    a, b = max(lo, best_t / grid), min(hi, best_t * grid)
    gr = (math.sqrt(5) - 1) / 2
    la, lb = math.log(a), math.log(b)
    c, d = lb - gr * (lb - la), la + gr * (lb - la)
    fc, fd = _nll_temp(rows, math.exp(c)), _nll_temp(rows, math.exp(d))
    for _ in range(80):
        if fc < fd:
            lb, d, fd = d, c, fc
            c = lb - gr * (lb - la)
            fc = _nll_temp(rows, math.exp(c))
        else:
            la, c, fc = c, d, fd
            d = la + gr * (lb - la)
            fd = _nll_temp(rows, math.exp(d))
        if lb - la < 1e-6:
            break
    t_ref = math.exp((la + lb) / 2)
    nll_ref = _nll_temp(rows, t_ref)
    if nll_ref < best_nll:
        best_t, best_nll = t_ref, nll_ref
    improvement = (nll_id - best_nll) / max(nll_id, 1e-12)
    flat = improvement < flat_rel
    boundary = best_t <= lo * 1.02 or best_t >= hi * 0.98
    if flat:
        return {"T": 1.0, "nll": nll_id, "nll_identity": nll_id,
                "improvement": 0.0, "flat": True, "boundary": False,
                "candidate": round(best_t, 6)}
    return {"T": round(best_t, 6), "nll": best_nll, "nll_identity": nll_id,
            "improvement": round(improvement, 6), "flat": False, "boundary": boundary}


def fit_temp(rows):             # kept for callers that only want the number
    return fit_temp_ex(rows)["T"]


def apply_temp(pr, T):
    lm = math.log(max(max(pr), 1e-300))
    lse = math.log(sum(math.exp((math.log(max(x, 1e-300)) - lm) / T) for x in pr))
    out = [math.exp((math.log(max(x, 1e-300)) - lm) / T - lse) for x in pr]
    s = sum(out)
    return [x / s for x in out] if s > 0 else out


def metrics(rows, cal):
    """rows: [(pr, gold)]  cal: None | ('platt', a, c) | ('T', T)"""
    ps, gs, acc, flips, nlls = [], [], 0, 0, []
    for pr, gold in rows:
        q = list(pr)
        if cal and cal[0] == "platt":
            p = min(max(q[1] if len(q) > 1 else 0.0, 1e-9), 1 - 1e-9)
            d = math.log(p / (1 - p))
            p2 = 1 / (1 + math.exp(-(cal[1] * d + cal[2])))
            q = [1 - p2, p2] if len(q) == 2 else q
        elif cal and cal[0] == "T":
            q = apply_temp(q, cal[1])
        # standard top-label protocol: conf = max(pr), correct = argmax==gold
        amax = max(range(len(q)), key=lambda i: q[i])
        ps.append(max(q[amax], 1e-9))
        gs.append(1 if amax == gold else 0)
        nlls.append(-math.log(max(q[gold] if gold < len(q) else 1e-9, 1e-9)))
        acc += int(amax == gold)
        flips += int((amax == gold) !=
                     (max(range(len(pr)), key=lambda i: pr[i]) == gold))
    n = len(rows)
    return {"acc": round(acc / n, 4), "nll": round(sum(nlls) / n, 4),
            "ece15": round(ece15(ps, gs), 4), "flips": flips, "n": n}


# ---------------------------------------------------------------- text items
def text_items(kind, datadir, n):
    files = sorted(Path(datadir, kind, "test").glob("*.json"))
    random.Random(7).shuffle(files)
    out = []
    for f in files[:n]:
        j = json.loads(f.read_text())
        qid, q = next(iter(j["questions"].items()))
        gold = j.get("_meta", {}).get("label", 0)
        crit = q.get("criteria")
        if isinstance(gold, str) and isinstance(crit, (dict, list)):
            keys = list(crit.keys()) if isinstance(crit, dict) else list(crit)
            gold = keys.index(gold) if gold in keys else -1
        elif isinstance(gold, str) and kind == "score":
            gold = -1
        out.append({"state": j["state"], "qid": qid, "q": q,
                    "gold": int(gold) if isinstance(gold, (int, float)) else gold})
    return [it for it in out if isinstance(it["gold"], int) and it["gold"] >= 0][:n]


def run_text(base, key, items, mode_hdr=None):
    rows = []
    for it in items:
        body = {"state": it["state"], "questions": {it["qid"]: it["q"]}}
        h = {"Content-Type": "application/json", "Authorization": "Bearer " + key}
        if mode_hdr:
            h["X-Jev-Mode"] = mode_hdr
        req = urllib.request.Request(base + "/v1/systemone", data=json.dumps(body).encode(),
                                     headers=h)
        r = json.loads(urllib.request.urlopen(req, timeout=900).read())
        a = r["answers"][it["qid"]]
        pr = a.get("probabilities") or [1 - a["noul"], a["noul"]]
        if isinstance(a.get("choice"), str):        # name-keyed (V) -> index
            names = list(it["q"].get("criteria", {}).keys())
            if a["choice"] in names:
                it["gold"] = names.index(a["choice"]) * 0 + it["gold"]
        rows.append((pr, it["gold"]))
        ITEM_LOG.append({"suite": "text", "qid": it["qid"], "gold": it["gold"],
                         "pr": [round(x, 6) for x in pr],
                         "argmax": max(range(len(pr)), key=lambda i: pr[i]),
                         "state_head": it["state"][:160],
                         "instructions": it["q"].get("instructions", "")[:160],
                         "calib": a.get("calib")})
    return rows


# ---------------------------------------------------------------- image items
COLORS = [("red", (198, 34, 40)), ("blue", (36, 84, 204)), ("green", (24, 132, 62)),
          ("yellow", (222, 200, 42)), ("purple", (120, 46, 148)), ("orange", (232, 124, 32))]


def synth_noul(i, rnd):
    from PIL import Image, ImageDraw
    present = rnd.random() < 0.5
    name, rgb = COLORS[i % len(COLORS)]
    name2, rgb2 = COLORS[(i + 1) % len(COLORS)]
    img = Image.new("RGB", (448, 448), (245, 243, 238))
    d = ImageDraw.Draw(img)
    if present:
        d.ellipse([150, 150, 300, 300], fill=rgb)          # a circle of color A
    d.ellipse([70, 70, 130, 130], fill=rgb2)               # always a circle of B
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    q = {"type": "noul",
         "instructions": f"Is there a large {name} circle in the image?"}
    # NOTE: no criteria. Justin + live probe 2026-10-15: with true/false criteria
    # the QFN direct readout INVERTS (gold=1 -> 0.18, gold=0 -> 0.38) while without
    # them it is right (0.9987 / 0.6044) - the description lines bias the readout.
    # POPE/GQA (the standard benches) never send criteria either.
    return q, base64.b64encode(buf.getvalue()).decode(), int(present)


def synth_choice(i, rnd):
    from PIL import Image, ImageDraw
    k = 3
    idx = rnd.randrange(k)
    shapes = ["circle", "square", "triangle"]
    names = [COLORS[j % len(COLORS)][0] for j in range(k)]
    while len(set(names)) < k:
        names = [COLORS[(i + j + rnd.randrange(6)) % 6][0] for j in range(k)]
    big = COLORS[(i + idx) % 6][0]
    img = Image.new("RGB", (448, 448), (245, 243, 238))
    d = ImageDraw.Draw(img)
    for j, (nm, rgb) in enumerate([COLORS[(i + t) % 6] for t in range(k)]):
        x = 60 + j * 130
        s = 110 if j == idx else 60
        if shapes[j % 3] == "circle":
            d.ellipse([x, 200 - s // 2, x + s, 200 + s // 2], fill=rgb)
        elif shapes[j % 3] == "square":
            d.rectangle([x, 200 - s // 2, x + s, 200 + s // 2], fill=rgb)
        else:
            d.polygon([(x + s // 2, 200 - s // 2), (x, 200 + s // 2), (x + s, 200 + s // 2)], fill=rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    q = {"type": "choice",
         "instructions": "Which shape is the largest?",
         "criteria": {sh: None for sh in shapes}}
    return q, base64.b64encode(buf.getvalue()).decode(), idx


def synth_score(i, rnd):
    from PIL import Image, ImageDraw
    cnt = rnd.randrange(2, 8)
    img = Image.new("RGB", (448, 448), (245, 243, 238))
    d = ImageDraw.Draw(img)
    for j in range(cnt):
        x, y = 40 + rnd.randrange(360), 40 + rnd.randrange(360)
        d.ellipse([x, y, x + 26, y + 26], fill=COLORS[i % 6][1])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    q = {"type": "score",
         "instructions": "How many circles are in the image?",
         "criteria": [str(n) for n in range(0, 9)]}
    return q, base64.b64encode(buf.getvalue()).decode(), cnt


def image_items(kind, n, offset=0):
    rnd = random.Random(1000 + offset)
    gen = {"noul": synth_noul, "choice": synth_choice, "score": synth_score}[kind]
    return [gen(offset + i, rnd) for i in range(n)]


def run_image(base, key, items, image_mode):
    rows = []
    for idx, (q, b64, gold) in enumerate(items):
        body = {"image_b64": b64, "image_mode": image_mode,
                "vision_prompt": NEUTRAL_VPROMPT if image_mode == "summary" else None,
                "questions": {"q": q}}
        body = {k: v for k, v in body.items() if v is not None}
        r = post(base, key, body)
        a = r["answers"]["q"]
        pr = a.get("probabilities") or [1 - a["noul"], a["noul"]]
        if isinstance(a.get("choice"), int) and a["type"] == "choice":
            g = gold
        elif isinstance(a.get("choice"), str):
            g = list(q["criteria"].keys()).index(q_criteria_gold(q, gold)) \
                if q_criteria_gold(q, gold) in q.get("criteria", {}) else gold
        else:
            g = gold
        rows.append((pr, g))
        ITEM_LOG.append({"suite": "image", "qid": f"img{idx:03d}", "gold": gold,
                         "pr": [round(x, 6) for x in pr],
                         "argmax": max(range(len(pr)), key=lambda i: pr[i]),
                         "instructions": q.get("instructions", "")[:160],
                         "image_b64_head": b64[:48],
                         "image_sha": hashlib.sha256(b64.encode()).hexdigest()[:16],
                         "calib": a.get("calib")})
    return rows


def q_criteria_gold(q, gold):
    ks = list(q.get("criteria", {}).keys()) if isinstance(q.get("criteria"), dict) else \
        list(q.get("criteria", []))
    return ks[gold] if 0 <= gold < len(ks) else None


# ---------------------------------------------------------------- suites
def suite_text(base, key, outdir, fit_n=60, test_n=100):
    D = os.environ.get("STRATA_SUITE_DATA", "/cfg/strata-data/eval/data")
    rep, fits = {}, {}
    for kind, ds in (("noul", "BoolQ"), ("choice", "ARC"), ("score", "SST-5")):
        dev = text_items(kind, D, fit_n)
        test = text_items(kind, D, test_n + fit_n)[fit_n:]
        dev_rows = run_text(base, key, dev)
        test_rows = run_text(base, key, test)
        served = sorted({r.get("calib") for r in ITEM_LOG[-len(dev_rows) - len(test_rows):]})
        if any(s not in (None, "identity") for s in served):
            print(f"[text:{kind}] WARNING served probabilities are calibrated by {served}: the "
                  f"'cal' column compounds that transform and the fitted params are NOT installable "
                  f"as-is - rerun with the calibrator removed for a clean fit", flush=True)
        if kind == "noul":
            ds_logits = [math.log(max(p[1], 1e-9) / max(1 - p[1], 1e-9)) for p, _ in dev_rows]
            cs = [1 if g == 1 else 0 for _, g in dev_rows]
            fi = fit_platt_ex(ds_logits, cs)
            cal = ("platt", fi["a"], fi["c"])
        else:
            fi = fit_temp_ex(dev_rows)
            cal = ("T", fi["T"])
        fits[kind] = fi
        rep[kind] = {"dataset": ds, "served_calib": served, "raw": metrics(test_rows, None),
                     "cal": metrics(test_rows, cal), "fit": fi}
        print(f"[text:{kind}] raw acc={rep[kind]['raw']['acc']} "
              f"cal acc={rep[kind]['cal']['acc']} fit={fi}", flush=True)
    return rep


def suite_image(base, key, outdir, fit_n=40, test_n=60):
    rep, fits = {}, {}
    for kind, ds in (("noul", "POPE-adv*"), ("choice", "CV-Bench*"), ("score", "CountBench*")):
        dev = image_items(kind, fit_n)
        test = image_items(kind, test_n, offset=fit_n)
        dev_rows = run_image(base, key, dev, "direct")
        test_rows = run_image(base, key, test, "direct")
        if kind == "noul":
            dl = [math.log(max(p[1], 1e-9) / max(1 - p[1], 1e-9)) for p, _ in dev_rows]
            cs = [1 if g == 1 else 0 for _, g in dev_rows]
            a, c = fit_platt(dl, cs)
            cal = ("platt", a, c)
            fits[kind] = {"a": round(a, 4), "c": round(c, 4)}
        else:
            T = fit_temp(dev_rows)
            cal = ("T", T)
            fits[kind] = {"T": round(T, 4)}
        rep[kind] = {"dataset": ds, "mode": "direct", "raw": metrics(test_rows, None),
                     "cal": metrics(test_rows, cal), "fit": fits[kind]}
        print(f"[image:{kind}] raw acc={rep[kind]['raw']['acc']} "
              f"cal acc={rep[kind]['cal']['acc']} fit={fits[kind]}", flush=True)
    return rep


def report(suite, outdir, name):
    md = [f"# {name} calibration suite (QFN live)", "",
          "| qtype | dataset | acc raw→cal | NLL raw→cal | ECE15 raw→cal | flips | fit |",
          "|---|---|---|---|---|---|---|"]
    for k, r in suite.items():
        md.append(f"| {k} | {r['dataset']} | {r['raw']['acc']:.1%}→{r['cal']['acc']:.1%} | "
                  f"{r['raw']['nll']}→{r['cal']['nll']} | {r['raw']['ece15']}→{r['cal']['ece15']} | "
                  f"{r['cal']['flips']} | {json.dumps(r['fit'])} |")
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{name}-suite.json").write_text(json.dumps(suite, indent=1))
    (p / f"{name}-suite.md").write_text("\n".join(md) + "\n")
    print("WROTE", p / f"{name}-suite.md")


def real_items(datadir, split, n=None):
    """Converted real-benchmark items: {qid, q, gold, image_b64, image_sha}."""
    out = []
    for f in sorted(Path(datadir, split).glob("*.json")):
        j = json.loads(f.read_text())
        qid, q = next(iter(j["questions"].items()))
        out.append({"qid": qid, "q": q, "gold": j["_meta"]["label"],
                    "image_b64": j["image_b64"],
                    "image_sha": j["_meta"].get("image_sha", "")})
        if n and len(out) >= n:
            break
    return out


def run_real(base, key, items, image_mode):
    """Direct/summary run over converted real items; logs per-item rows."""
    rows = []
    for it in items:
        body = {"image_b64": it["image_b64"], "image_mode": image_mode,
                "questions": {it["qid"]: it["q"]}}
        if image_mode == "summary":
            body["vision_prompt"] = NEUTRAL_VPROMPT
        r = post(base, key, body)
        a = r["answers"][it["qid"]]
        pr = a.get("probabilities") or [1 - a["noul"], a["noul"]]
        g = it["gold"]
        if isinstance(a.get("choice"), str) and isinstance(it["q"].get("criteria"), dict):
            names = list(it["q"]["criteria"].keys())
            if a["choice"] in names:
                g = names.index(a["choice"])      # keep gold as index; name answer handled below
        amax = max(range(len(pr)), key=lambda i: pr[i])
        if isinstance(a.get("choice"), int):
            amax = a["choice"]
        rows.append((pr, g))
        ITEM_LOG.append({"suite": "real", "qid": it["qid"], "gold": g,
                         "pr": [round(x, 6) for x in pr], "argmax": amax,
                         "instructions": it["q"].get("instructions", "")[:160],
                         "image_sha": it["image_sha"][:16], "calib": a.get("calib"),
                         "mode": image_mode})
    return rows


def suite_real(base, key, outdir, fit_n=None, test_n=None):
    rep, fits = {}, {}
    D = os.environ.get("STRATA_SUITE_REAL", "/cfg/strata-data/eval/real/data")
    for kind, ds in (("noul-pope", "POPE adversarial"), ("choice-cvb", "CV-Bench"),
                     ("score-countbench", "CountBenchQA")):
        dev = real_items(Path(D, kind), "dev")
        test = real_items(Path(D, kind), "test")
        n0 = len(ITEM_LOG)
        dev_rows = run_real(base, key, dev, "direct")
        test_rows = run_real(base, key, test, "direct")
        served = sorted({r.get("calib") for r in ITEM_LOG[n0:]})
        if any(s not in (None, "identity") for s in served):
            print(f"[real:{kind}] WARNING served probabilities are calibrated by {served}: the "
                  f"'cal' column compounds that transform and the fitted params are NOT installable "
                  f"as-is - rerun with the calibrators removed (jevport-arm.sh <arm> --calib identity)",
                  flush=True)
        qtype = {"noul-pope": "noul", "choice-cvb": "choice", "score-countbench": "score"}[kind]
        if qtype == "noul":
            dl = [math.log(max(p[1], 1e-9) / max(1 - p[1], 1e-9)) for p, _ in dev_rows]
            cs = [1 if g == 1 else 0 for _, g in dev_rows]
            fi = fit_platt_ex(dl, cs)
            cal = ("platt", fi["a"], fi["c"])
        else:
            fi = fit_temp_ex(dev_rows)
            cal = ("T", fi["T"])
        fits[kind] = fi
        rep[kind] = {"dataset": ds, "mode": "direct", "served_calib": served,
                     "raw": metrics(test_rows, None), "cal": metrics(test_rows, cal), "fit": fi}
        print(f"[real:{kind}] raw acc={rep[kind]['raw']['acc']} "
              f"cal acc={rep[kind]['cal']['acc']} fit={fi}", flush=True)
    return rep


def main():
    base, key = sys.argv[1].rstrip("/"), sys.argv[2]
    outdir, which = sys.argv[3], sys.argv[4]
    fit_n = int(sys.argv[5]) if len(sys.argv) > 5 else 60
    test_n = int(sys.argv[6]) if len(sys.argv) > 6 else 100
    if which == "text":
        report(suite_text(base, key, outdir, fit_n, test_n), outdir, "text")
    elif which == "real":
        report(suite_real(base, key, outdir), outdir, "real")
    else:
        report(suite_image(base, key, outdir, max(30, fit_n // 2), max(30, test_n // 2)), outdir, "image")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / f"{which}-item-log.jsonl").write_text(
        "\n".join(json.dumps(r) for r in ITEM_LOG) + "\n")
    print("ITEM LOG:", len(ITEM_LOG), "rows ->", outdir + f"/{which}-item-log.jsonl")


if __name__ == "__main__":
    main()
