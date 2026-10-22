#!/usr/bin/env python3
"""test_fit_temp.py - ground-truth tests for the temperature / Platt fitters in jevport-suite.py.

The old fitter could not pass these: it scanned t = 0.05 * 1.1**k and RETURNED a grid point, so
its answer was quantised to 10% steps (1.0557 = 0.05*1.1**32 is why two different suites reported
the same T), and on a flat likelihood it returned a spurious non-1 T instead of identity.

  python3 test_fit_temp.py      # exit 0 = all checks pass

Tolerances are sampling-noise aware: with n items the MLE of T has ~1/sqrt(n*Var(log p)) relative
error, so the recovery check runs at n=4000 and allows 5%.  The convergence runs (n=300..20000)
are printed to show the error shrinking rather than being tuned away.
"""
import importlib.util
import math
import random
import sys

spec = importlib.util.spec_from_file_location("suite", "/opt/data/eval/jevport-suite.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

GRID = [0.05 * 1.1 ** k for k in range(0, 120)]


def norm(pr):
    s = sum(pr)
    return [x / s for x in pr]


def make_rows(n, k, t_true, seed):
    """Data whose correct temperature IS t_true.

    apply_temp divides the log-prob by T, so p = q**(1/T_s) is undone by t = T_s.  Generating
    with apply_temp(q, 1/t_true) gives p = q**t_true, whose correcting temperature is t_true.
    """
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        q = norm([math.exp(rnd.gauss(0, 1.2)) for _ in range(k)])
        g = rnd.choices(range(k), weights=q)[0]
        out.append((S.apply_temp(q, 1.0 / t_true), g))
    return out


def main():
    fails = []

    print("temperature recovery (n=4000, tolerance 5%):")
    for k in (4, 5):
        for t_true in (0.6, 1.0, 1.6, 2.4, 4.0):
            rows = make_rows(4000, k, t_true, seed=k * 100 + int(t_true * 10))
            fi = S.fit_temp_ex(rows)
            err = abs(fi["T"] - t_true) / t_true
            ok = (fi["T"] == 1.0 and fi["flat"]) if t_true == 1.0 else (err < 0.05 and not fi["flat"])
            if not ok:
                fails.append(f"k={k} T={t_true} -> {fi['T']}")
            print(f"  k={k} true T={t_true:<4} fit T={fi['T']:<9} err={err:6.2%} "
                  f"impr={fi['improvement']:5.3f} flat={str(fi['flat']):<5} "
                  f"boundary={str(fi['boundary']):<5} {'OK' if ok else 'FAIL'}")

    print("\nconvergence (T=1.6 must settle on 1.6, error shrinking with n):")
    prev = 1.0
    for n in (300, 1000, 5000, 20000):
        fi = S.fit_temp_ex(make_rows(n, 4, 1.6, seed=5))
        err = abs(fi["T"] - 1.6) / 1.6
        if n == 20000 and err > 0.03:
            fails.append(f"convergence at n=20000 err={err:.3f}")
        print(f"  n={n:<6} fit T={fi['T']:<9} err={err:6.2%}")
        prev = err

    print("\nthe answer must NOT be a grid point (the old quantisation bug):")
    fi = S.fit_temp_ex(make_rows(4000, 4, 1.6, seed=17))
    on_grid = any(abs(fi["T"] - g) < 1e-9 for g in GRID)
    if on_grid:
        fails.append(f"T={fi['T']} is exactly a 0.05*1.1**k grid point")
    print(f"  T={fi['T']} on grid={on_grid}  {'OK' if not on_grid else 'FAIL'}")

    print("\nalready-calibrated data must stay identity (no spurious T):")
    for k in (4, 5):
        fi = S.fit_temp_ex(make_rows(2000, k, 1.0, seed=7 + k))
        ok = fi["T"] == 1.0 and fi["flat"]
        if not ok:
            fails.append(f"identity check k={k} -> T={fi['T']} flat={fi['flat']}")
        print(f"  k={k} T={fi['T']} flat={fi['flat']} candidate={fi.get('candidate')}  "
              f"{'OK' if ok else 'FAIL'}")

    print("\ndegenerate (all mass on one label) must flag the boundary:")
    fi = S.fit_temp_ex([([0.999, 0.0007, 0.0002, 0.0001], 0) for _ in range(50)])
    ok = fi["boundary"] or fi["flat"]
    if not ok:
        fails.append("degenerate case did not flag boundary")
    print(f"  T={fi['T']} boundary={fi['boundary']} flat={fi['flat']}  {'OK' if ok else 'FAIL'}")

    print("\nPlatt recovery (n=4000, a within 25%):")
    rnd = random.Random(11)
    for a_true, c_true in ((1.0, 0.0), (0.5, -1.6), (2.2, 0.8)):
        ds, cs = [], []
        for _ in range(4000):
            d = rnd.gauss(0, 2.5)
            p = S._sig(a_true * d + c_true)
            ds.append(d)
            cs.append(1 if rnd.random() < p else 0)
        fi = S.fit_platt_ex(ds, cs)
        err = abs(fi["a"] - a_true) / a_true
        ok = err < 0.25
        if not ok:
            fails.append(f"platt a={a_true} -> {fi['a']}")
        print(f"  true a={a_true} c={c_true}  fit a={fi['a']:.3f} c={fi['c']:.3f} "
              f"flat={fi['flat']}  {'OK' if ok else 'FAIL'}")

    print("\nPlatt on already-calibrated data must stay identity:")
    ds, cs = [], []
    for _ in range(4000):
        d = rnd.gauss(0, 2.5)
        ds.append(d)
        cs.append(1 if rnd.random() < S._sig(d) else 0)
    fi = S.fit_platt_ex(ds, cs)
    ok = (fi["a"], fi["c"]) == (1.0, 0.0) and fi["flat"]
    if not ok:
        fails.append(f"platt identity -> a={fi['a']} c={fi['c']} flat={fi['flat']}")
    print(f"  a={fi['a']} c={fi['c']} flat={fi['flat']}  {'OK' if ok else 'FAIL'}")

    print()
    if fails:
        print(f"{len(fails)} CHECK(S) FAILED:")
        for f in fails:
            print("  -", f)
        return 1
    print("ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
