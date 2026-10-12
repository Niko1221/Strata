"""Independent numerical comparison helpers for frozen next-logit fixtures.

A pass here proves closeness under explicit tolerances, not exact equality of
bit patterns, sampling traces, or whole-model quality.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from collections.abc import Sequence


@dataclass(frozen=True)
class LogitReport:
    count: int
    max_absolute_error: float
    max_normalized_error: float
    argmax_expected: int
    argmax_actual: int
    reference_top_margin: float
    kl_expected_to_actual: float
    pass_elementwise: bool
    top1_stable_within_tolerance: bool


def compare_logits(actual: Sequence[float], expected: Sequence[float],
                   *, atol: float, rtol: float) -> LogitReport:
    if not isinstance(actual, Sequence) or not isinstance(expected, Sequence):
        raise TypeError("Logits must be sequence types")
    if len(actual) == 0 or len(actual) != len(expected):
        raise ValueError("Logit vectors must be nonempty and equal length")
    if (not all(type(v) in (int, float) and math.isfinite(v) for v in (*actual, *expected))
            or not all(type(x) in (int, float) and math.isfinite(x) and x >= 0
                       for x in (atol, rtol))):
        raise ValueError("NaN/Inf/non-numeric logits or invalid tolerances")
    a = [float(x) for x in actual]
    e = [float(x) for x in expected]
    max_abs = max(abs(x - y) for x, y in zip(a, e))
    norms = [abs(x - y) / (atol + rtol * abs(y)) if atol + rtol * abs(y) > 0
             else (0.0 if x == y else float("inf")) for x, y in zip(a, e)]
    max_norm = max(norms)
    ei = max(range(len(e)), key=lambda i: e[i])
    ai = max(range(len(a)), key=lambda i: a[i])
    second = max((e[i] for i in range(len(e)) if i != ei), default=e[ei])
    margin = e[ei] - second
    max_allowed = max(atol + rtol * abs(y) for y in e)
    # Close logits can legitimately swap argmax on a near tie; report it,
    # don't silently equate it with large-margin rank corruption.
    stable = ai == ei or margin <= 2 * max_allowed
    # Numerically stable KL(p||q); computed from both log-sum-exp values.
    ez = max(e)
    az = max(a)
    log_pe = [v - ez for v in e]
    log_qa = [v - az for v in a]
    log_zp = math.log(sum(math.exp(x) for x in log_pe))
    log_zq = math.log(sum(math.exp(x) for x in log_qa))
    kl = sum(math.exp(lp - log_zp) * ((lp - log_zp) - (lq - log_zq))
             for lp, lq in zip(log_pe, log_qa))
    return LogitReport(len(e), max_abs, max_norm, ei, ai, margin,
                       max(kl, 0.0), max_norm <= 1.0, stable)


def assert_logits_close(actual: Sequence[float], expected: Sequence[float],
                        *, atol: float, rtol: float, require_clear_margin_top1=True) -> LogitReport:
    report = compare_logits(actual, expected, atol=atol, rtol=rtol)
    if not report.pass_elementwise:
        raise AssertionError(f"Logits diverged: max abs={report.max_absolute_error:.5g}, "
                             f"normalized={report.max_normalized_error:.5g}")
    if require_clear_margin_top1 and not report.top1_stable_within_tolerance:
        raise AssertionError("Clear-margin top-1 changed")
    return report
