"""How big is the difference?  Compares two runs' `tr shash dev <d> il <l> <hash> <v0> <v1> <sum>` lines: for the first
passes, per layer, the relative difference of the sum and of the first value."""
import sys


def passes(path):
    out, cur = [], None
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("tr step T "):
            cur = {}
            out.append(cur)
        elif line.startswith("tr shash ") and cur is not None:
            f = line.split()
            cur[(int(f[3]), int(f[5]))] = (float(f[7]), float(f[8]), float(f[9]), f[6])
    return out


a = passes(sys.argv[1])
b = passes(sys.argv[2])
stop = int(sys.argv[3]) if len(sys.argv) > 3 else 2
for i in range(min(len(a), len(b), stop)):
    print("pass %d" % i)
    for k in sorted(a[i]):
        x, y = a[i][k], b[i].get(k)
        if y is None:
            continue
        dsum = abs(x[2] - y[2]) / max(1e-9, abs(x[2]))
        same = "same" if x[3] == y[3] else "DIFF"
        print("  il %2d  sum %.9g vs %.9g  rel %.2e   v0 %.9g vs %.9g   %s" % (k[1], x[2], y[2], dsum, x[0], y[0], same))
