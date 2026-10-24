"""Paired A/B statistics for the benchmark tools: wrappers over scipy and statsmodels, nothing
reimplemented here.

tools/ab_engine.py (--stats) and tools/strata_bench.py (--compare) call this.  Both tools work
without the packages - the tables print as before and say what is missing.  Install with:
    pip install -r requirements-bench.txt

The question it answers for one measured quantity: both arms were run on the same stimulus (the
same prompt slot in the same round, or the same run x question across two sessions); is the
paired difference a real speed change of at least a floor, noise, or too few pairs to say?
The Wilcoxon signed-rank test (scipy) ranks the differences - robust to the heavy tails the
decode numbers carry; Holm (statsmodels) corrects for testing several quantities at once; the
Hodges-Lehmann shift carries a confidence interval, and the interval - not the median alone -
decides the verdict.

All numbers are log-ratios of new/old (speed noise is multiplicative): a positive shift means
the new arm is faster, and the shift exponentiates back to a plain factor.
"""
from __future__ import annotations

import math
import statistics
import warnings


def available() -> bool:
    try:
        import scipy.stats                        # noqa: F401
        import statsmodels.api                    # noqa: F401
    except Exception:                             # noqa: BLE001 - an old/partial install still fails here
        return False
    return True


def signed_rank(diffs: list[float]) -> dict:
    """One two-sided Wilcoxon signed-rank test (scipy) plus the Hodges-Lehmann shift and its
    95% CI.  Zero differences are dropped (scipy's wilcox zero method).  The CI comes from
    inverting the same test: the statistic only changes between Walsh averages, so those gaps
    are the only candidates worth testing."""
    nz = [x for x in diffs if x == x and x != 0.0]
    out = {"n": len(nz), "p": None, "hl": None, "ci_lo": None, "ci_hi": None}
    if len(nz) < 2:
        return out
    from scipy.stats import wilcoxon
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # small-sample notes are expected here
        try:
            out["p"] = float(wilcoxon(nz, alternative="two-sided").pvalue)
            walsh = sorted((a + b) / 2.0 for i, a in enumerate(nz) for b in nz[i:])
            out["hl"] = statistics.median(walsh)
            if len(walsh) <= 4000:                # ~88 pairs and below
                keep = [i for i in range(len(walsh) - 1)
                        if wilcoxon([x - (walsh[i] + walsh[i + 1]) / 2.0 for x in nz]).pvalue > 0.05]
                if keep:
                    out["ci_lo"], out["ci_hi"] = walsh[keep[0]], walsh[keep[-1] + 1]
        except Exception:                         # noqa: BLE001 - a CI is a bonus; keep p and hl
            pass
    return out


def holm(ps: list[float | None], alpha: float = 0.05) -> list[float]:
    """Holm-Bonferroni adjusted p-values (statsmodels); None counts as 1.  Strictly stronger
    than plain Bonferroni with the same guarantee."""
    from statsmodels.stats.multitest import multipletests
    vals = [1.0 if p is None else p for p in ps]
    _, adj, _, _ = multipletests(vals, alpha=alpha, method="holm")
    return [float(x) for x in adj]


def significance(st: dict, p_adj: float, alpha: float = 0.05) -> str:
    """The test result by itself - no floor involved anywhere in here; the practical bar lives in
    verdict().  Can these pairs rule out "no change"?  Both p-values are printed so a reader can
    judge with any alpha they like."""
    if st["p"] is None:
        return "too few pairs to test"
    return (f"yes (p={st['p']:.4f}, Holm {p_adj:.4f})" if p_adj <= alpha
            else f"no (p={st['p']:.4f}, Holm {p_adj:.3f})")


def power_note(diffs: list[float], per_test_alpha: float, floor: float) -> dict | None:
    """What this many pairs can actually detect: the pair spread seen in these very numbers, the
    smallest effect reaching 80% power at the current n, the n a floor effect needs, and the n
    that clears the combinatorial p floor 2**(1-n) at all (normal approximation for the rank
    test with the ARE pi/3 inflation - a guide, not a promise; cross-session drift can add
    spread the pair numbers do not show)."""
    d = [x for x in diffs if x == x and x != 0.0]
    if len(d) < 2:
        return None
    sd = statistics.stdev(d)
    if sd <= 0:
        return None
    from scipy.stats import norm
    za, zb = norm.ppf(1.0 - per_test_alpha / 2.0), norm.ppf(0.8)
    infl = math.sqrt(math.pi / 3.0)
    n_floor = next((n for n in range(1, 40) if 2.0 ** (1 - n) <= per_test_alpha), 99)
    return {"n": len(d), "sd": sd,
            "mde": (za + zb) * sd * infl / math.sqrt(len(d)),
            "n_need": max(len(d), math.ceil(((za + zb) * sd * infl / math.log1p(floor)) ** 2)),
            "n_floor": n_floor}


def verdict(st: dict, p_adj: float, floor: float, alpha: float = 0.05) -> str:
    """A magnitude statement that makes no statistical claim of its own - the test result lives in
    significance() and the two must be read together: "over the floor" says only that the change is
    big enough to matter IF it is real; whether it is real is the other column's answer.  floor is a
    fraction the caller sets (--floor-decode / --floor-prefill, e.g. 0.05 for 5%), never a test."""
    if st["p"] is None or st["hl"] is None:
        return "not testable (too few pairs)"
    if p_adj > alpha:
        return "size unknown (test not passed)"
    lo, hi = st["ci_lo"], st["ci_hi"]
    word = "faster" if st["hl"] > 0 else "slower"
    if lo is None:
        return f"{word}: CI not computed"
    cross = math.log1p(floor) if st["hl"] > 0 else math.log1p(-floor)
    if (st["hl"] > 0 and lo > cross) or (st["hl"] < 0 and hi < cross):
        return f"{word}, over the floor"
    if (st["hl"] > 0 and hi < cross) or (st["hl"] < 0 and lo > cross):
        return f"{word}, under the floor"
    return f"{word}, CI crosses the floor"


def factor(shift: float | None, digits: int = 3) -> str:
    """A log-ratio shift as 'x1.07' (a speed-up factor); '-' when not measured."""
    return "-" if shift is None else f"x{math.exp(shift):.{digits}f}"
