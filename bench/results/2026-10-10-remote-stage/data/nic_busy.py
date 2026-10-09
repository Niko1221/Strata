"""nic_busy.py FILE [MB/s] — per-second rates from cumulative 'epoch rx tx' samples; the longest busy stretch."""
import sys

rows = [tuple(map(float, l.split()[:3])) for l in open(sys.argv[1]) if l[0].isdigit()]
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
rates = [((t1 - t0), (r1 - r0) / (t1 - t0) / 1e6, (x1 - x0) / (t1 - t0) / 1e6)
         for (t0, r0, x0), (t1, r1, x1) in zip(rows, rows[1:])]
busy = [i for i, (_, r, x) in enumerate(rates) if r > 20 or x > 20]   # a prompt is going through
best = (0, 0)
start = None
for i in range(len(rates) + 1):
    on = i < len(rates) and (rates[i][1] > 20 or rates[i][2] > 20)
    if on and start is None:
        start = i
    if not on and start is not None:
        if i - start > best[1] - best[0]:
            best = (start, i)
        start = None
a, b = best
seg = rates[a:b]
print(f"samples {len(rates)}; longest active stretch {b - a} s")
print(f"  seconds at >= {thr:.0f} MB/s: rx {sum(1 for _, r, _ in seg if r >= thr)}, "
      f"tx {sum(1 for _, _, x in seg if x >= thr)}")
print(f"  bytes: rx {sum(d * r for d, r, _ in seg) / 1e3:.2f} GB, tx {sum(d * x for d, _, x in seg) / 1e3:.2f} GB")
print(f"  peak rx {max(r for _, r, _ in seg):.0f} MB/s, tx {max(x for _, _, x in seg):.0f} MB/s")
