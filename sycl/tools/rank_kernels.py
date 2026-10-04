# rank_kernels.py LOG - compact ranking of unitrace's "Device Timing Summary" table
import re, sys
lines = open(sys.argv[1]).read().split("\n")
i = [k for k, l in enumerate(lines) if "Device Timing Summary" in l][0]
rows = []
for l in lines[i:]:
    m = re.search(r'"(.*)",\s*(\d+),\s*(\d+),\s*([\d.]+),\s*(\d+),\s*(\d+),\s*(\d+)', l)
    if not m: continue
    name, calls, total, pct, avg, mn, mx = m.groups()
    n = re.sub(r"dpct_kernel_name<|strata::kernels::|\(anonymous namespace\)::", "", name)
    n = re.sub(r"\(.*", "", n)
    rows.append((int(total), int(calls), float(pct), int(avg), n[:72]))
rows.sort(reverse=True)
tot = sum(r[0] for r in rows)
print(f"kernels {len(rows)}, device time {tot/1e6:.0f} ms over the run")
print(f"{'ms':>8} {'%':>5} {'calls':>6} {'avg us':>7}  kernel")
for t, c, p, a, n in rows[:int(sys.argv[2]) if len(sys.argv) > 2 else 22]:
    print(f"{t/1e6:8.1f} {p:5.1f} {c:6d} {a/1000:7.1f}  {n}")
