"""Assign each line of kernel-log-events.txt to the run whose window (start of pre-run-state.txt to the end line of post-run-state.txt, or of
pre-run-state.txt for ab/) contains it.   python3 -I scripts/events_by_run.py      (paths relative to the script)"""
import os, re, datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "scripts" else HERE
def t(s): return dt.datetime.fromisoformat(s)
wins = []
for kind in ("runs", "ab"):
    for d in sorted(os.listdir(os.path.join(ROOT, kind))):
        pre = open(os.path.join(ROOT, kind, d, "pre-run-state.txt"), encoding="utf-8").read()
        post_p = os.path.join(ROOT, kind, d, "post-run-state.txt")
        post = open(post_p, encoding="utf-8").read() if os.path.exists(post_p) else pre
        s = re.search(r"^start: (\S+)", pre, re.M).group(1); e = re.search(r"^end: (\S+)", post, re.M).group(1)
        wins.append((t(s), t(e), f"{kind}/{d}"))
wins.sort()
ev = [l for l in open(os.path.join(ROOT, "kernel-log-events.txt"), encoding="utf-8") if not l.startswith("#")]
def when(l):
    m = re.match(r"(\w+) (\d+) (\d\d):(\d\d):(\d\d)", l)
    return dt.datetime(2026, 10, int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), tzinfo=wins[0][0].tzinfo)
def kinds(l):
    if "bcachefs" in l: return "bcachefs read error (recovered)"
    if "Write-error" in l: return "swap write error (zram)"
    if "page allocation stall" in l: return "allocation stall"
    if "page allocation failure" in l: return "allocation failure"
    if "Failed to create a DMA mapping" in l: return "NVIDIA DMA mapping"
    return None   # the NVIDIA driver's follow-up lines of the same event
print("| run | start-end (PDT) | allocation failures | allocation stalls | zram swap write errors | bcachefs read errors (recovered) | NVIDIA driver: DMA mapping failed |\n| --- | --- | ---: | ---: | ---: | ---: | ---: |")
outside = 0
for s, e, name in wins:
    c = {k: 0 for k in ("allocation failure", "allocation stall", "swap write error (zram)", "bcachefs read error (recovered)", "NVIDIA DMA mapping")}
    for l in ev:
        if s <= when(l) <= e and kinds(l): c[kinds(l)] += 1
    print(f"| {name} | {s:%H:%M:%S}-{e:%H:%M:%S} | {c['allocation failure']} | {c['allocation stall']} | {c['swap write error (zram)']} | {c['bcachefs read error (recovered)']} | {c['NVIDIA DMA mapping']} |")
print(f"\nevents in the file: {len(ev)}; inside no run window: {sum(1 for l in ev if not any(s <= when(l) <= e for s, e, _ in wins))}")
