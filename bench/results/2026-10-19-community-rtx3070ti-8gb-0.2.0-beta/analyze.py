"""Per request: the monitor samples inside its window (other processes' CPU, slowing GPU throttle reasons); disturbed =
other processes above the median request's + 1 core, or SW power cap / HW slowdown / thermal / power brake.  Prints
per-block medians (all runs and clean runs) and the pairwise comparisons the report uses."""
import csv
import json
import pathlib
import statistics

W = pathlib.Path(__file__).resolve().parent
RUNS = W / "runs"
BAD = 0x4 | 0x8 | 0x20 | 0x40 | 0x80


def monitor(d):
    rows = []
    p = d / "monitor.csv"
    if p.exists():
        for r in csv.DictReader(open(p, encoding="utf-8")):
            try:
                rows.append({"t": float(r["t"]), "other": float(r["other_cpu_s"] or 0), "top": r["other_top"],
                             "thr": int(r["throttle"], 16) if r["throttle"] else 0, "temp": float(r["temp_c"] or 0)})
            except (ValueError, KeyError):
                pass
    return rows


def reqs(d):
    out = []
    for sub in sorted(d.glob("targets-*")):
        p = sub / "results.json"
        if p.exists():
            for r in json.loads(p.read_text(encoding="utf-8")):
                if r["label"] == "warmup":
                    continue
                e = r["engine"]
                out.append({"block": d.name, "label": r["label"], "prompt": e["prompt_tokens"],
                            "reused": e.get("reused") or 0, "t0": e["time"], "t1": e["time"] + e["duration_s"],
                            "prefill": (e["prompt_tokens"] - (e.get("reused") or 0)) / (e["prompt_ms"] / 1000),
                            "decode": e["engine_generated"] / (e["decode_ms"] / 1000), "gen": e["engine_generated"],
                            "finish": r.get("finish_reason"), "ttft": r["client_ttft_s"], "total": r["client_elapsed_s"],
                            "acc": (e.get("drafts_accepted") or 0, e.get("drafts_offered") or 0)})
    return out


blocks = {}
for d in sorted(p for p in RUNS.iterdir() if p.is_dir() and not p.name.startswith("00")):
    rs, mon = reqs(d), monitor(d)
    for q in rs:
        s = [m for m in mon if q["t0"] <= m["t"] <= q["t1"] + 2]
        q["other_cores"] = sum(m["other"] for m in s) / max(2.0, q["t1"] - q["t0"]) if s else None
        q["thr_bad"] = sorted({hex(m["thr"] & BAD) for m in s if m["thr"] & BAD})
        tops = {}
        for m in s:
            for it in filter(None, m["top"].split(";")):
                n, v = it.rsplit(":", 1)
                tops[n] = tops.get(n, 0) + float(v)
        q["top"] = sorted(tops.items(), key=lambda kv: -kv[1])[:2]
    blocks[d.name] = rs
allq = [q for rs in blocks.values() for q in rs]
base = statistics.median([q["other_cores"] for q in allq if q["other_cores"] is not None])
for q in allq:
    q["disturbed"] = bool(q["thr_bad"]) or (q["other_cores"] is not None and q["other_cores"] > base + 1.0)
print(f"requests {len(allq)}; all reused 0: {all(q['reused'] == 0 for q in allq)}; all 256 generated: "
      f"{all(q['gen'] == 256 for q in allq)}; other-process CPU median {base:.2f} cores, disturbed above {base + 1:.2f}")


def m3(v):
    return (statistics.median(v), min(v), max(v)) if v else None


def f(t, d=1):
    return "-" if not t else f"{t[0]:.{d}f} [{t[1]:.{d}f}-{t[2]:.{d}f}]"


table = {}
print(f"\n{'block':16} {'prompt':>7} {'n':>2} {'cl':>2}  {'prefill all':>22} {'prefill clean':>22}  {'decode all':>18}  ttft  acc")
for b, rs in blocks.items():
    for pt in sorted({q["prompt"] for q in rs}):
        g = [q for q in rs if q["prompt"] == pt]
        c = [q for q in g if not q["disturbed"]]
        table[(b, pt)] = (g, c)
        acc = sum(q["acc"][0] for q in g) / max(1, sum(q["acc"][1] for q in g))
        print(f"{b:16} {pt:>7} {len(g):>2} {len(c):>2}  {f(m3([q['prefill'] for q in g])):>22} "
              f"{f(m3([q['prefill'] for q in c])):>22}  {f(m3([q['decode'] for q in g])):>18}  "
              f"{statistics.median(q['ttft'] for q in g):5.1f} {acc:.0%}")
print("\ndisturbed requests:")
for q in allq:
    if q["disturbed"]:
        print(f"  {q['block']} {q['label']}: other {q['other_cores']:.2f} cores {q['top']}, throttle {q['thr_bad']}, "
              f"prefill {q['prefill']:.1f}, decode {q['decode']:.1f}")


def med(b, pt, key="prefill", clean=False):
    g, c = table.get((b, pt), ([], []))
    v = [q[key] for q in (c if clean else g)]
    return statistics.median(v) if v else None


print("\npairs (all runs; clean in brackets):")
for x, y in (("03-C", "02-B"), ("06-C", "05-B"), ("07-C", "08-B"), ("02-B", "01-A"), ("04-D", "01-A"),
             ("03-C", "01-A"), ("09-B-share-off", "08-B")):
    for pt in (4096, 16384):
        for key in ("prefill", "decode"):
            a, b_ = med(x, pt, key), med(y, pt, key)
            ac, bc = med(x, pt, key, True), med(y, pt, key, True)
            if a and b_:
                print(f"  {x:15} vs {y:6} {pt:>6} {key:7}: {a:7.1f} vs {b_:7.1f}  {a / b_ - 1:+6.1%}"
                      + (f"  [clean {ac / bc - 1:+6.1%}]" if ac and bc else ""))
json.dump({k: v for k, v in blocks.items()}, open(W / "annotated.json", "w", encoding="utf-8"), indent=1, default=str)
