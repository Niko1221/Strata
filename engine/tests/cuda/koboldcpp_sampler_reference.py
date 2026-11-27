"""Numeric parity of Strata's sampler chain against koboldcpp's sampling, distribution against distribution.

Strata's CUDA sampler implements tfs, typical, top_a, xtc, dynamic temperature and smoothing with koboldcpp's semantics
(`kcpp_src/gpttype_adapter.cpp`: `sample_top_a`, `sample_tail_free`, `sampler_typical`, `sample_xtc`,
`sample_entropy`, `sample_temperature`, applied in `SampleLogits`'s order). This runs the same chain, written here
from those functions' definitions, on fixed logit rows and requires the engine's final distribution
(`sampler_parity --dist`: the probability of every token after the chain's last step) to match it.

    python engine/tests/cuda/koboldcpp_sampler_reference.py BUILD/bin/sampler_parity [--path both|split|one-block]

Exit status 1 when a case is outside the tolerance. The chain below is an independent statement of the same math
(it shares no code with koboldcpp, which is AGPL-licensed).
"""
from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import sys

ORDER_TOP_A, ORDER_TFS, ORDER_TYP = "top_a", "tfs", "typical"
XTC_BAN = -999.0                 # sample_xtc: what it takes off the tokens it removes


class Case:
    """One logit row and the parameters of the chain over it (the engine's names)."""

    def __init__(self, name, logits, top_k=20, top_p=1.0, min_p=0.0, temperature=1.0, tfs=0.0, typical=0.0, top_a=0.0,
                 xtc_prob=0.0, xtc_thresh=0.1, dyna_low=0.0, dyna_high=0.0, dyna_exp=1.0, smooth_factor=0.0,
                 smooth_curve=1.0):
        self.__dict__.update(locals())
        del self.self
        self.n = len(logits)

    def line(self):
        c = self
        return (f"{c.n} {c.top_k} {c.top_p!r} {c.min_p!r} {c.temperature!r} {c.tfs!r} {c.typical!r} {c.top_a!r} "
                f"{c.xtc_prob!r} {c.xtc_thresh!r} {c.dyna_low!r} {c.dyna_high!r} {c.dyna_exp!r} {c.smooth_factor!r} "
                f"{c.smooth_curve!r}")


# ---- the chain -------------------------------------------------------------------------------
# A candidate list is [(token id, logit)], `probs` their softmax. Each step is the definition it names; `sorted`
# says whether the list is in falling logit order, which decides how the next softmax treats it.


def softmax(cands, sorted_flag):
    """sample_softmax: sort when not sorted; p = exp(l - max) / sum."""
    if not sorted_flag:
        cands.sort(key=lambda c: -c[1])
        sorted_flag = True
    m = cands[0][1]
    if not sorted_flag:
        m = max(c[1] for c in cands)
    e = [math.exp(c[1] - m) for c in cands]
    s = sum(e)
    return [x / s for x in e], sorted_flag


def top_k(cands, k):
    """sample_top_k: k <= 0 keeps all."""
    if k <= 0 or k >= len(cands):
        return cands
    return cands[:k]


def top_a(cands, a, sorted_flag):
    """sample_top_a (min_keep 1): keep the tokens with p >= a * p_max^2."""
    if a <= 0.0 or len(cands) <= 1:
        return cands, sorted_flag
    p, sorted_flag = softmax(cands, sorted_flag)
    thr = a * p[0] * p[0]
    for i in range(len(cands)):
        if p[i] < thr and i >= 1:
            return cands[:i], sorted_flag
    return cands, sorted_flag


def tail_free(cands, z, sorted_flag):
    """sample_tail_free (min_keep 1): cut where the normalized second differences' running sum passes z.
    koboldcpp's default is 1 (off); the engine also reads 0 as off."""
    if z <= 0.0 or z >= 1.0 or len(cands) <= 2:
        return cands, sorted_flag
    p, sorted_flag = softmax(cands, sorted_flag)
    d2 = [abs((p[i] - p[i + 1]) - (p[i + 1] - p[i + 2])) for i in range(len(cands) - 2)]
    s = sum(d2)
    if s > 1e-6:
        d2 = [x / s for x in d2]
    else:
        d2 = [1.0 / len(d2)] * len(d2)
    run = 0.0
    for i, x in enumerate(d2):
        run += x
        if run > z and i >= 1:
            return cands[:i], sorted_flag
    return cands, sorted_flag


def typical(cands, t, sorted_flag):
    """sampler_typical (min_keep 1): by |-.log(p) - entropy|, keep the least of them until their mass passes t.
    koboldcpp's default is 1 (off); the engine also reads 0 as off."""
    if t <= 0.0 or t >= 1.0:
        return cands, sorted_flag
    p, sorted_flag = softmax(cands, sorted_flag)
    ent = -sum(x * math.log(x) for x in p if x > 0.0)
    shifted = [abs(-math.log(x) - ent) if x > 0.0 else float("inf") for x in p]
    order = sorted(range(len(cands)), key=shifted.__getitem__)
    run, last = 0.0, len(order)
    for i, idx in enumerate(order):
        run += p[idx]
        if run > t and i >= 0:
            last = i + 1
            break
    return [cands[idx] for idx in order[:last]], False


def top_p(cands, t, sorted_flag):
    """sample_top_p (min_keep 1): the smallest prefix whose mass reaches t."""
    if t >= 1.0:
        return cands, sorted_flag
    p, sorted_flag = softmax(cands, sorted_flag)
    run = 0.0
    for i in range(len(cands)):
        run += p[i]
        if run >= t and i + 1 >= 1:
            return cands[:i + 1], sorted_flag
    return cands, sorted_flag


def min_p(cands, t, sorted_flag):
    """sample_min_p (min_keep 1): keep p >= t * p_max, the list in logit order if it is not already."""
    if t <= 0.0 or not cands:
        return cands, sorted_flag
    if not sorted_flag:
        floor = max(c[1] for c in cands) + math.log(t)
        kept = [c for c in cands if c[1] >= floor]
        if len(kept) >= 1:
            return kept, False
    _, sorted_flag = softmax(cands, sorted_flag)       # sorts the list; its probabilities are not what is read
    floor = cands[0][1] + math.log(t)
    for i in range(1, len(cands)):
        if cands[i][1] < floor and i >= 1:
            return cands[:i], sorted_flag
    return cands, sorted_flag


def smooth(cands, factor, curve):
    """The smoothing both temperature steps share: h the top logit, l' = -(k*f*(l-h)^2) + (s*f*(l-h)^3) + h."""
    if not (factor > 0.0 and len(cands) > 1):
        return cands
    h = max(c[1] for c in cands)
    k, s = (3.0 - curve) / 2.0, (curve - 1.0) / 2.0
    return [(i, -(k * factor * (l - h) ** 2) + (s * factor * (l - h) ** 3) + h) for i, l in cands]


def temperature(cands, temp, factor, curve):
    """sample_temperature (temp > 0)."""
    if temp <= 0:
        return cands[:1]
    return smooth([(i, l / temp) for i, l in cands], factor, curve)


def entropy_temp(cands, lo, hi, expo, factor, curve):
    """sample_entropy: t' = lo + (hi - lo) * (H / H_max)^exponent over the list's softmax."""
    if len(cands) <= 1:
        return cands
    h_max = -math.log(1.0 / len(cands))
    m = max(c[1] for c in cands)
    e = [math.exp(l - m) for _, l in cands]
    s = sum(e)
    p = [x / s for x in e]
    ent = -sum(x * math.log(x) for x in p if x > 0.0)
    t = lo + (hi - lo) * (ent / h_max) ** expo
    return smooth([(i, l / t) for i, l in cands], factor, curve)


def xtc(cands, thresh, applies, sorted_flag):
    """sample_xtc: read final probabilities over every candidate (smoothing may reorder them); keep the least likely
    candidate at or above thresh and ban the rest of that set."""
    if thresh > 0.5 or not applies or len(cands) <= 1:
        return cands, sorted_flag
    p, sorted_flag = softmax(cands, sorted_flag)
    last = len(cands)
    for i in range(len(cands)):
        if p[i] < thresh:
            last = i
            break
    if last > 1:
        cands = [(i, l + XTC_BAN if k < last - 1 else l) for k, (i, l) in enumerate(cands)]
        sorted_flag = False
    return cands, sorted_flag


def reference(case):
    """The chain's final distribution over the vocabulary: koboldcpp's order, each step as defined above
    (top_k, top_a, tfs, typical, top_p, min_p, the temperature step, xtc last)."""
    c = case
    cands = sorted(enumerate(c.logits), key=lambda x: -x[1])
    cands, s = top_k(cands, c.top_k), True
    cands, s = top_a(cands, c.top_a, s)
    cands, s = tail_free(cands, c.tfs, s)
    cands, s = typical(cands, c.typical, s)
    cands, s = top_p(cands, c.top_p, s)
    cands, s = min_p(cands, c.min_p, s)
    step = (lambda: entropy_temp(cands, c.dyna_low, c.dyna_high, c.dyna_exp, c.smooth_factor, c.smooth_curve)) \
        if c.dyna_low != c.dyna_high else (lambda: temperature(cands, c.temperature, c.smooth_factor, c.smooth_curve))
    cands = step()
    cands, s = xtc(cands, c.xtc_thresh, c.xtc_prob > 0.0, s)
    p, _ = softmax(cands, False)
    out = [0.0] * c.n
    for (i, _), x in zip(cands, p):
        out[i] = x
    return out


# ---- the engine's side -----------------------------------------------------------------------
def engine_dists(exe, cases, path):
    """`sampler_parity --dist`: one line of probabilities per case, in the order sent."""
    text = "".join(c.line() + "\n" + " ".join(repr(x) for x in c.logits) + "\n" for c in cases)
    env = dict(os.environ)
    env.pop("STRATA_OLD_SAMPLER", None)
    env.pop("STRATA_SAMPLER_ONE_BLOCK", None)
    if path == "one-block":
        env["STRATA_SAMPLER_ONE_BLOCK"] = "1"
    else:
        env.pop("STRATA_SAMPLER_ONE_BLOCK", None)
    run = subprocess.run([exe, "--dist"], input=text, text=True, capture_output=True, encoding="utf-8", env=env)
    if run.returncode != 0:
        raise SystemExit(f"sampler_parity --dist exited {run.returncode}: {run.stderr[-600:]}")
    lines = [line for line in run.stdout.splitlines() if line.strip()]
    if len(lines) != len(cases):
        raise SystemExit(f"sampler_parity --dist answered {len(lines)} lines for {len(cases)} cases: {run.stderr[-400:]}")
    return [[float(x) for x in line.split()] for line in lines]


def cases_for(seed):
    """The cases: each new sampler alone over a few logit rows, then all of them together."""
    rng = random.Random(seed)
    n = 400
    out = []

    def row(shape):
        logits = [rng.gauss(0.0, 1.0) for _ in range(n)]
        for i in range(shape):                             # a few strong candidates up front
            logits[i] += 3.0 + 1.5 * (shape - i) / max(1, shape)
        return [float(x) for x in logits]                   # ties would sort differently in the two implementations

    shapes = (2, 6, 20)
    for shape in shapes:
        base = row(shape)
        out.append(Case(f"disabled feature baseline, shape {shape}", base, top_a=0.0, tfs=0.0, typical=0.0,
                        xtc_prob=0.0, dyna_low=0.0, dyna_high=0.0, smooth_factor=0.0))
        out.append(Case(f"top_k only, shape {shape}", base))
        out.append(Case(f"temperature 0.7, shape {shape}", base, temperature=0.7))
        out.append(Case(f"deterministic temperature, shape {shape}", base, temperature=0.0))
        out.append(Case(f"top_a 0.2, shape {shape}", base, top_a=0.2))
        out.append(Case(f"top_a 0.05, shape {shape}", base, top_a=0.05))
        out.append(Case(f"tfs 0.95, shape {shape}", base, tfs=0.95))
        out.append(Case(f"tfs 0.5, shape {shape}", base, tfs=0.5))
        out.append(Case(f"typical 0.5, shape {shape}", base, typical=0.5))
        out.append(Case(f"typical 0.9, shape {shape}", base, typical=0.9))
        out.append(Case(f"xtc (thresh 0.1, always applied), shape {shape}", base, xtc_prob=1.0, xtc_thresh=0.1))
        out.append(Case(f"dynamic temperature 0.5..1.5, shape {shape}", base, dyna_low=0.5, dyna_high=1.5))
        out.append(Case(f"dynamic temperature 0.2..2.0, exponent 0.5, shape {shape}", base, dyna_low=0.2, dyna_high=2.0,
                        dyna_exp=0.5))
        out.append(Case(f"smoothing 0.3, shape {shape}", base, temperature=0.7, smooth_factor=0.3))
        out.append(Case(f"smoothing 0.5, curve 0.0, shape {shape}", base, temperature=0.7, smooth_factor=0.5,
                        smooth_curve=0.0))
        out.append(Case(f"smoothing 0.4, curve 2.0, shape {shape}", base, temperature=0.7, smooth_factor=0.4,
                        smooth_curve=2.0))
        out.append(Case(f"all six, temperature 1.0, shape {shape}", base, top_a=0.1, tfs=0.9, typical=0.8, top_p=0.95,
                        min_p=0.02, xtc_prob=1.0, smooth_factor=0.2, top_k=40))
        out.append(Case(f"all six, temperature 0.7, shape {shape}", base, temperature=0.7, top_a=0.1, tfs=0.9,
                        typical=0.8, top_p=0.95, min_p=0.02, xtc_prob=1.0, smooth_factor=0.2, top_k=20))
        out.append(Case(f"all six + dynamic temperature, shape {shape}", base, top_a=0.1, tfs=0.9, typical=0.8,
                        top_p=0.95, min_p=0.02, xtc_prob=1.0, smooth_factor=0.2, dyna_low=0.5, dyna_high=1.3,
                        top_k=64))
    # xtc reads the probabilities the temperature step leaves: the same row at several temperatures, and
    # after the dynamic temperature and the smoothing
    base = row(20)
    for temp in (0.5, 0.7, 1.0, 1.5):
        out.append(Case(f"xtc with temperature {temp}", base, temperature=temp, xtc_prob=1.0, xtc_thresh=0.1,
                        top_k=40))
    out.append(Case("xtc after the dynamic temperature", base, xtc_prob=1.0, xtc_thresh=0.1, top_k=40,
                    dyna_low=0.5, dyna_high=1.5))
    out.append(Case("xtc after smoothing", base, xtc_prob=1.0, xtc_thresh=0.1, top_k=40, temperature=0.8,
                    smooth_factor=0.4))

    # With a zero smoothing curve, the temperature transform is nonmonotonic. XTC must sort by the final
    # probabilities across the entire candidate list, including candidates no longer in logit order.
    reorder = [-0.08 * i for i in range(n)]
    out.append(Case("xtc after nonmonotonic smoothing", reorder, top_k=64, top_p=1.0, typical=0.99,
                    xtc_prob=1.0, xtc_thresh=0.1, smooth_factor=0.5, smooth_curve=0.0, temperature=1.0))

    # Multiple merge blocks with high candidates in distant blocks. A one-block implementation can still be
    # checked against this row, while the default split path must merge all these partial top-k lists correctly.
    wide = [rng.gauss(-2.0, 1.0) for _ in range(8192)]
    for token, value in ((3, 16.0), (4094, 15.0), (4097, 14.0), (8190, 13.0)):
        wide[token] = value
    out.append(Case("wide split merge, candidates across blocks", wide, top_k=64, top_p=0.9, temperature=0.8))

    # Large logit gaps stress stable softmax and the zero-mass tails in the filters.
    extreme = [0.0] * n
    extreme[:8] = [1000.0, 50.0, 0.0, -50.0, -100.0, -500.0, -1000.0, -2000.0]
    out.append(Case("extreme logits, entropy filters", extreme, top_k=20, top_p=0.99, temperature=0.5,
                    tfs=0.95, typical=0.95, top_a=0.01))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("exe", help="the sampler_parity program, built with BUILD_TESTING=ON")
    ap.add_argument("--path", choices=("both", "split", "one-block"), default="both",
                    help="exercise the default split-merge sampler, one-block sampler, or both")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tolerance", type=float, default=2e-5, help="largest allowed |reference - engine| per token")
    args = ap.parse_args()
    cases = cases_for(args.seed)
    paths = ("split", "one-block") if args.path == "both" else (args.path,)
    bad = 0
    print(f"{len(cases)} cases per path, n_vocab {cases[0].n}, tolerance {args.tolerance:g}\n")
    print(f"{'path':10s} {'case':62s} {'vocab':>6s} {'kept':>5s} {'max |diff|':>11s} {'sum':>8s}")
    for path in paths:
        got = engine_dists(args.exe, cases, path)
        for case, dist in zip(cases, got):
            ref = reference(case)
            if len(dist) != len(ref):
                raise SystemExit(f"{path}: {case.name}: engine returned {len(dist)} probabilities, expected {len(ref)}")
            diff = max(abs(a - b) for a, b in zip(ref, dist))
            kept = sum(1 for x in ref if x > 0.0)
            ok = diff <= args.tolerance
            bad += not ok
            print(f"{path:10s} {case.name[:62]:62s} {len(case.logits):6d} {kept:5d} {diff:11.3e} {sum(dist):8.5f} {'' if ok else '  *** OUT OF TOLERANCE ***'}")
    print(f"\n{'PARITY OK' if not bad else f'PARITY FAIL ({bad} cases)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
