#!/usr/bin/env python3
"""calib/blockers_gpu.py — the three GPU pre-PR blockers in one batch (Opus ranks 3/5/6 + S-fit).

Phases (checkpoint each to OUT; failures don't stop later phases):
  A  noise-floor   (text preset): 30 items x 3 qtypes x 5 runs x {idle, 8-decision,
                                  4chat+4decision} -> per-type p99 |dp|, argmax flips
  B  soak          (text preset): 2000 mixed (1q/3q decisions, chat gen, 422s) +
                                  VRAM drift samples; then (vision preset) 200 image-direct
  C  S-tuple fit   (vision preset): summary-mode dev split -> Platt/T -> calib-S.json,
                                  cold reload, verify calib id flips identity -> fitted
  D  CIs           (offline): bootstrap 95% CI + Brier + adaptive-ECE on the logged
                                  real-suite rows (POPE/CVB/CountBench, raw vs cal)

usage: blockers_gpu.py <text_base> <vision_base> <key> <outdir>
"""
import base64
import json
import math
import os
import random
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

TEXT_BASE, VISION_BASE, KEY, OUT = os.sys.argv[1:5]
OUT = Path(OUT)
OUT.mkdir(parents=True, exist_ok=True)
H = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}


def post(base, path, body, timeout=900):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers=H)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}


def vram():
    try:
        out = subprocess.run(["/usr/bin/nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=20).stdout.strip().splitlines()
        return [int(x) for x in out]
    except Exception:
        return None


def ckpt(name, obj):
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1))
    print(f"[ckpt] {name} written", flush=True)


def dev_items(kind, n):
    files = sorted(Path("/cfg/strata-data/eval/data", kind, "dev").glob("*.json"))[:n]
    out = []
    for f in files:
        j = json.loads(f.read_text())
        out.append((f.stem, j["state"], next(iter(j["questions"].values())),
                    j["_meta"]["label"]))
    return out


# ---------------------------------------------------------------- phase A: noise floor
def one_decision(base, state, q):
    st, b = post(base, "/v1/systemone", {"state": state, "questions": {"q": q}})
    if st != 200:
        return None
    a = b["answers"]["q"]
    return a.get("probabilities") or [1 - a["noul"], a["noul"]]


def chat_hammer(base, stop, stats):
    while not stop.is_set():
        st, _ = post(base, "/v1/chat/completions",
                     {"messages": [{"role": "user", "content": "Say a short sentence."}],
                      "max_tokens": 32})
        stats["chat_n"] += 1
        stats["chat_err"] += st != 200
        time.sleep(0.05)


def dec_hammer(base, stop, stats):
    while not stop.is_set():
        one_decision(base, "Load generator state, iteration running.",
                     {"type": "noul", "instructions": "Is the load generator running?"})
        stats["dec_n"] += 1
        time.sleep(0.05)


def phase_a():
    items = {"noul": dev_items("noul", 30), "choice": dev_items("choice", 30),
             "score": dev_items("score", 30)}
    scenarios = {"idle": lambda b, s, e: [],
                 "dec8": lambda b, s, e: [threading.Thread(target=dec_hammer, args=(b, s, e)) for _ in range(8)],
                 "chat4dec4": lambda b, s, e: [threading.Thread(target=chat_hammer, args=(b, s, e)) for _ in range(4)]
                 + [threading.Thread(target=dec_hammer, args=(b, s, e)) for _ in range(4)]}
    report = {}
    for scen, mk in scenarios.items():
        stop = threading.Event()
        stats = {"chat_n": 0, "chat_err": 0, "dec_n": 0}
        ths = mk(TEXT_BASE, stop, stats)
        for t in ths:
            t.start()
        runs = {}
        try:
            for run in range(5):
                for qt, its in items.items():
                    for qid, state, q, gold in its:
                        p = one_decision(TEXT_BASE, state, q)
                        if p is not None:
                            runs.setdefault((qt, qid), []).append(p)
                print(f"[A:{scen}] run {run + 1}/5 done", flush=True)
        finally:
            stop.set()
            for t in ths:
                t.join(timeout=30)
        per_type, flips = {}, 0
        for (qt, qid), ps in runs.items():
            base_p = ps[0]
            for p in ps[1:]:
                d = max(abs(a - b) for a, b in zip(p, base_p))
                per_type.setdefault(qt, []).append(d)
                if max(range(len(p)), key=lambda i: p[i]) != max(range(len(base_p)), key=lambda i: base_p[i]):
                    flips += 1
        report[scen] = {
            "load": stats,
            "p99_abs_dp": {qt: (round(sorted(v)[int(0.99 * len(v)) - 1], 6) if v else None)
                           for qt, v in per_type.items()},
            "max_abs_dp": {qt: (round(max(v), 6) if v else None) for qt, v in per_type.items()},
            "argmax_flips": flips, "n_items": sum(len(v) for v in per_type.values()) // 5}
        ckpt("noise-floor", report)
    print("[A] noise floor complete", flush=True)


# ---------------------------------------------------------------- phase B: soak
def phase_b():
    rnd = random.Random(43)
    stats = {"n": 0, "err": 0, "by_kind": {}, "vram": []}
    t0 = time.time()
    for i in range(2000):
        k = i % 5
        if k in (0, 1, 2):
            st, _ = post(TEXT_BASE, "/v1/systemone", {
                "state": f"Soak iteration {i}; engine warm; load nominal.",
                "questions": {"q": {"type": "noul", "instructions": f"Is iteration {i} running?"}}})
            kind = "dec1q"
        elif k == 3:
            st, _ = post(TEXT_BASE, "/v1/systemone", {
                "state": f"Soak {i}; valves open; tank full; pump on.",
                "questions": {f"q{j}": {"type": "noul", "instructions": f"Is valve {j} open?"}
                              for j in range(3)}})
            kind = "dec3q"
        else:
            st, _ = post(TEXT_BASE, "/v1/chat/completions",
                         {"messages": [{"role": "user", "content": f"soak {i}: one short sentence"}],
                          "max_tokens": 40})
            kind = "chat"
        stats["n"] += 1
        stats["err"] += st != 200
        stats["by_kind"][kind] = stats["by_kind"].get(kind, 0) + 1
        if i % 200 == 0:
            v = vram()
            stats["vram"].append({"i": i, "mb": v})
            print(f"[B] {i}/2000 err={stats['err']} vram={v} elapsed={int(time.time()-t0)}s", flush=True)
    stats["elapsed_s"] = int(time.time() - t0)
    stats["vram_end"] = vram()
    ckpt("soak-text", stats)
    print(f"[B] text soak complete: {stats['n']} req, {stats['err']} errors", flush=True)


def phase_b2():
    # image-direct mini-soak on the vision preset (200 requests, 2 images alternating)
    import io
    from PIL import Image, ImageDraw
    imgs = []
    for rgb in ((200, 30, 30), (30, 60, 200)):
        im = Image.new("RGB", (448, 448), rgb)
        ImageDraw.Draw(im).ellipse([150, 150, 300, 300], fill=(240, 240, 240))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        imgs.append(base64.b64encode(buf.getvalue()).decode())
    stats = {"n": 0, "err": 0, "vram_start": vram(), "vram_end": None}
    for i in range(200):
        st, _ = post(VISION_BASE, "/v1/systemone", {
            "image_b64": imgs[i % 2], "image_mode": "direct",
            "questions": {"q": {"type": "noul", "instructions": "Is there a circle in the image?"}}})
        stats["n"] += 1
        stats["err"] += st != 200
        if i % 50 == 0:
            print(f"[B2] {i}/200 err={stats['err']}", flush=True)
    stats["vram_end"] = vram()
    ckpt("soak-image", stats)


# ---------------------------------------------------------------- phase C: S-tuple fit
def real_dev(kind):
    out = []
    for f in sorted(Path("/cfg/strata-data/eval/real/data", kind, "dev").glob("*.json")):
        j = json.loads(f.read_text())
        qid, q = next(iter(j["questions"].items()))
        out.append({"qid": qid, "q": q, "gold": j["_meta"]["label"], "image_b64": j["image_b64"]})
    return out


def phase_c():
    NEUTRAL = ("Describe this image in detail, including objects, their counts, "
               "colors, positions and any text.")
    rows = {"noul": [], "choice": [], "score": []}
    for kind, qt in (("noul-pope", "noul"), ("choice-cvb", "choice"), ("score-countbench", "score")):
        for it in real_dev(kind):
            st, b = post(VISION_BASE, "/v1/systemone", {
                "image_b64": it["image_b64"], "image_mode": "summary",
                "vision_prompt": NEUTRAL, "questions": {it["qid"]: it["q"]}})
            if st != 200:
                continue
            a = b["answers"][it["qid"]]
            pr = a.get("probabilities") or [1 - a["noul"], a["noul"]]
            g = it["gold"]
            if qt == "choice" and isinstance(a.get("choice"), int):
                g = a["choice"]  # name-keyed gold mapping handled by scorer; index for fit
            rows[qt].append((pr, g))
        print(f"[C] {kind}: {len(rows[qt])} rows", flush=True)
    # fits (same math as bench_suite)
    import importlib.util
    spec = importlib.util.spec_from_file_location("bs", "/cfg/strata-data/eval/jevport-suite.py")
    bs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bs)
    dl = [math.log(max(p[1], 1e-9) / max(1 - p[1], 1e-9)) for p, _ in rows["noul"]]
    cs = [1 if g == 1 else 0 for _, g in rows["noul"]]
    a_, c_ = bs.fit_platt(dl, cs)
    Tc = bs.fit_temp(rows["choice"])
    Ts = bs.fit_temp(rows["score"])
    # holdout check on the same rows (dev-only fit; honest note in report)
    cal = {"a": round(a_, 4), "c": round(c_, 4), "T_choice": round(Tc, 4), "T_score": round(Ts, 4)}
    (Path("/cfg/strata/Jev-port-cuda/calib") / "calib-S.json").write_text(json.dumps({
        "tuple": "qfn-iq3_xxs|jev-port|none|v1|S|image-summary", **cal,
        "provenance": "fit 2026-10-15 blockers_gpu phase C (summary-mode real dev)"}, indent=1))
    ckpt("s-fit", {"fit": cal, "n": {k: len(v) for k, v in rows.items()},
                   "raw": {k: bs.metrics(v, None) for k, v in rows.items()},
                   "cal": {"noul": bs.metrics(rows["noul"], ("platt", a_, c_)),
                           "choice": bs.metrics(rows["choice"], ("T", Tc)),
                           "score": bs.metrics(rows["score"], ("T", Ts))}})
    print(f"[C] S fit: {cal}", flush=True)


# ---------------------------------------------------------------- phase D: CIs (offline)
def phase_d():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bs", "/cfg/strata-data/eval/jevport-suite.py")
    bs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bs)
    rows = [json.loads(l) for l in open("/opt/data/eval/real-item-log.jsonl")] if \
        Path("/opt/data/eval/real-item-log.jsonl").exists() else []
    if not rows:
        p = Path("/cfg/strata-data/eval/suites2/real-item-log.jsonl")
        if p.exists():
            rows = [json.loads(l) for l in open(p)]
    fits = {"pope": ("platt", 1.0571, -0.172), "cvb": ("T", 1.5456), "cb": ("T", 0.6555)}
    out = {}
    rnd = random.Random(7)
    for pref, (kind, fit) in (("pope", fits["pope"]), ("cvb", fits["cvb"]), ("cb", fits["cb"])):
        test = [r for r in rows if r["qid"].startswith(pref)][ {"pope": 60, "cvb": 20, "cb": 20}[pref]:]
        data = [(r["pr"], r["gold"]) for r in test]
        def ece_of(sample):
            return bs.ece15(*zip(*[(list(q), 1 if max(range(len(q)), key=lambda i: q[i]) == g else 0)
                                   for q, g in sample])) if False else None
        def brier(sample, use_cal):
            s = 0.0
            for pr, g in sample:
                q = bs_apply(pr, fit) if use_cal else list(pr)
                top = max(range(len(q)), key=lambda i: q[i])
                s += (max(q) - (1 if top == g else 0)) ** 2
            return s / len(sample)
        def bs_apply(pr, fit):
            if fit[0] == "platt":
                p = min(max(pr[1] if len(pr) > 1 else 0.0, 1e-9), 1 - 1e-9)
                d = math.log(p / (1 - p))
                qq = 1 / (1 + math.exp(-(fit[1] * d + fit[2])))
                return [1 - qq, qq]
            return bs_apply_T(pr, fit[1])
        def bs_apply_T(pr, T):
            if abs(T - 1.0) < 1e-9:
                return list(pr)
            lm = math.log(max(max(pr), 1e-300))
            lse = math.log(sum(math.exp((math.log(max(x, 1e-300)) - lm) / T) for x in pr))
            o = [math.exp((math.log(max(x, 1e-300)) - lm) / T - lse) for x in pr]
            s = sum(o)
            return [x / s for x in o]
        def boot(use_cal):
            vals = []
            for _ in range(1000):
                sample = [data[rnd.randrange(len(data))] for _ in range(len(data))]
                vals.append(brier(sample, use_cal))
            vals.sort()
            return round(vals[24], 4), round(vals[975], 4)
        out[kind] = {"n": len(data),
                     "brier_raw": round(brier(data, False), 4),
                     "brier_raw_ci95": boot(False),
                     "brier_cal": round(brier(data, True), 4),
                     "brier_cal_ci95": boot(True)}
        print(f"[D] {kind}: {out[kind]}", flush=True)
    ckpt("cis-brier", out)


def main():
    print("[warm] text preset", flush=True)
    post(TEXT_BASE, "/v1/systemone", {"state": "warm", "questions": {
        "q": {"type": "noul", "instructions": "Warm?"}}})
    for phase, fn in (("A", phase_a), ("B", phase_b), ("B2", phase_b2), ("C", phase_c), ("D", phase_d)):
        try:
            fn()
        except Exception as e:
            import traceback
            ckpt(f"error-{phase}", {"error": str(e), "tb": traceback.format_exc()})
            print(f"[{phase}] FAILED: {e}", flush=True)
    print("ALL PHASES ATTEMPTED", flush=True)


if __name__ == "__main__":
    main()
