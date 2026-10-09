"""P-state / SM clock distribution of the 3070 while the decode bench ran (nvidia-smi -lms 100 samples)."""
import collections
import sys

t0, t1 = sys.argv[2], sys.argv[3]   # "23:09:22" "23:11:16"
rows = []
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    f = [x.strip() for x in line.split(",")]
    if len(f) < 6 or not f[2].endswith("MHz"):
        continue
    hms = f[0].split()[1][:8]
    if t0 <= hms <= t1:
        rows.append((f[1], int(f[2].split()[0]), int(f[3].split()[0]), f[4], f[5]))
print(len(rows), "samples")
print("pstate:", dict(collections.Counter(r[0] for r in rows)))
b = collections.Counter((r[1] // 300) * 300 for r in rows)
print("SM MHz buckets:", dict(sorted(b.items())))
print("mem MHz:", dict(collections.Counter(r[2] for r in rows)))
u = [int(r[3].split()[0]) for r in rows]
print("util mean", sum(u) / len(u), "max", max(u))
print("first 40:", " ".join(f"{r[0]}/{r[1]}" for r in rows[:40]))
