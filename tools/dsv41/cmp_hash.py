"""Compare two runs' per-layer row-0 h hashes (DS4_STEPCHECK=3): which layer and device first differ, and in which pass."""
import sys


def passes(path):
    out = []
    cur = None
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("tr step T "):
            cur = {}
            out.append(cur)
        elif line.startswith("tr hhash ") and cur is not None:
            f = line.split()
            cur[(int(f[2]), int(f[4]))] = f[5]
    return out


a = passes(sys.argv[1])
b = passes(sys.argv[2])
print("passes: %d vs %d" % (len(a), len(b)))
for i in range(min(len(a), len(b))):
    ka, kb = a[i], b[i]
    keys = sorted(set(ka) | set(kb))
    diff = [k for k in keys if ka.get(k) != kb.get(k)]
    if not diff:
        print("pass %d: %d layers, identical" % (i, len(keys)))
        continue
    print("pass %d: first difference at dev %d il %d" % (i, diff[0][0], diff[0][1]))
    print("   %d of %d (dev, layer) differ" % (len(diff), len(keys)))
    print("   differing: %s%s" % (diff[:10], " ..." if len(diff) > 10 else ""))
    break
