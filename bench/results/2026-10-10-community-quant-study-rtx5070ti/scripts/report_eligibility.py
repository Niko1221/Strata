"""Explicit per-axis eligibility gates for the community report compiler."""
from __future__ import annotations

import math

REQUIRED_GYRO_S_SLICES = frozenset(
    {"heplus-low-audit", "heplus-xhigh-audit", "teb-low", "teb-xhigh"}
)
REQUIRED_CLOSEOUT_CHECKS = frozenset(
    {
        "controller_terminal",
        "controller_restored",
        "controller_chain_up",
        "guard_terminal",
        "guard_restored",
        "guard_chain_up",
        "restore_chain_up",
        "stock_digests_unchanged",
        "health_probe",
        "exact_keyed_chat",
        "tool_probe",
        "vision_probe",
    }
)


def gyro_s_scoring_gate(campaign: dict) -> dict:
    """Check scored-slice and restoration validity without requiring a clean whole run.

    An invalid auxiliary slice (for example, a dataset download failure) and a
    nonzero controller exit do not invalidate independently valid scored axes.
    Required score slices and the full restoration/closeout proof must pass.
    """
    reasons: list[str] = []
    audit = campaign.get("scoring_slice_audit") or {}
    slices = audit.get("required_slices") or {}
    for name in sorted(REQUIRED_GYRO_S_SLICES):
        status = slices.get(name) or {}
        if status.get("present") is not True or status.get("valid") is not True:
            reasons.append(f"required scored slice invalid or absent: {name}")
    if audit.get("eligible") is not True:
        reasons.append("scored-slice audit did not mark the required axes eligible")
    if audit.get("invalid_required_slices"):
        reasons.append("one or more required scored slices are invalid")
    if audit.get("missing_required_slices"):
        reasons.append("one or more required scored slices are missing")

    closeout = campaign.get("controller_closeout_audit") or {}
    checks = closeout.get("checks") or {}
    for name in sorted(REQUIRED_CLOSEOUT_CHECKS):
        if checks.get(name) is not True:
            reasons.append(f"restore/closeout proof missing or failed: {name}")
    if closeout.get("eligible") is not True:
        reasons.append("controller/guard closeout audit is not eligible")
    if any(value is not True for value in checks.values()):
        reasons.append("one or more recorded controller/guard checks failed")
    return {"eligible": not reasons, "reasons": reasons}


def decode_axis_status(perf_by_depth: dict, depth: str = "0") -> dict:
    """Return a zero-background decode value only when the cell is eligible."""
    cell = perf_by_depth.get(str(depth)) if isinstance(perf_by_depth, dict) else None
    if not isinstance(cell, dict):
        return {"eligible": False, "value": None, "reason": "missing measured profile"}
    if cell.get("comparison_eligible") is False:
        return {
            "eligible": False,
            "value": None,
            "reason": cell.get("comparison_ineligibility_reason")
            or "speed attribution is not verified for this setup",
        }
    if cell.get("eligible_for_composite") is False:
        return {
            "eligible": False,
            "value": None,
            "reason": cell.get("composite_ineligibility_reason")
            or "zero-background decode cell is not eligible for the composite",
        }
    value = cell.get("tg_mean")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return {"eligible": False, "value": None, "reason": "missing finite decode mean"}
    return {"eligible": True, "value": float(value), "reason": ""}
