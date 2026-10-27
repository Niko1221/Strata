"""Compare two runs' per-layer row-0 h hashes: which layer and device first differ, and in which pass.

Reads either prefix (`tr hhash` from run_dev, `tr shash` from forward_serial); a pass starts at a `tr step T` line.
"""
import sys


def passes(path):
    out, cur = [], None
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("tr step T "):
            cur = {}
            out.append(cur)
        elif (line.startswith("tr hhash ") or line.startswith("tr shash ")) and cur is not None:
            f = line.split()
            cur[(int(f[3]), int(f[5]))] = f[6]
    return out


a = passes(sys.argv[1])
b = passes(sys.argv[2])
print("passes: %d vs %d" % (len(a), len(b)))
shown = 0
for i in range(min(len(a), len(b))):
    ka, kb = a[i], b[i]
    keys = sorted(set(ka) | set(kb))
    diff = [k for k in keys if ka.get(k) != kb.get(k)]
    if not diff:
        print("pass %d: %d layers, identical" % (i, len(keys)))
        continue
    print("pass %d: %d of %d layers differ; first dev %d il %d" % (i, len(diff), len(keys), diff[0][0], diff[0][1]))
    print("   differing: %s%s" % (diff[:12], " ..." if len(diff) > 12 else ""))
    shown += 1
    if shown >= 4:
        break
