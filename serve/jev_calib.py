"""serve/jev_calib.py - M3.2: jevcpp calibration port (donor src/server.cpp:518-633).

Post-hoc, monotone calibration of the jev readout: identity until a calib.json is
loaded; Platt (a, c) on the noul logit-difference; temperature T on choice/score
label log-probs (log-sum-exp form).  Argmax is always preserved - calibration only
reshapes probabilities, never picks a different label.

The module is self-contained (stdlib only) and thread-safe: one lock guards the
active Calib, one guards the async log queue; the writer flushes every 200 ms.

Wire surface (server.py routes):
    POST /admin/calib      hot-reload + active state as JSON (Bearer admin key)
    calib/data/log.jsonl   one row per answered question (donor B1.1 schema)

Tuple v1 (our port): (alias|quant|none|v1|L|all) in state_json; per-row tuples in
log.jsonl are (alias|quant|none|v1|L|<qtype>) like the donor's, refit per model.
"""
from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path


class Calib:
    __slots__ = ("loaded", "tuple_str", "T_choice", "T_score", "a_noul", "c_noul")

    def __init__(self, loaded: bool = False, tuple_str: str = "",
                 T_choice: float = 1.0, T_score: float = 1.0,
                 a_noul: float = 1.0, c_noul: float = 0.0):
        self.loaded = loaded
        self.tuple_str = tuple_str
        self.T_choice = T_choice
        self.T_score = T_score
        self.a_noul = a_noul
        self.c_noul = c_noul


# ---- active calibrator + lock ----
# P0-1 (Opus plan): keyed by READOUT letter, not one global Calib - an image fit
# must never change text decisions.  "" (empty string) = the donor's text/legacy
# tuple: calib.json keeps working for L exactly as before.  Unknown keys -> None
# (identity), so a missing per-mode file cannot silently reuse another mode's fit.
_ACTIVE: dict[str, "Calib | None"] = {"": None}
_MU = threading.Lock()


def active(mode: str = "") -> Calib | None:
    with _MU:
        return _ACTIVE.get(mode)


def _set_for_test(c: Calib | None, mode: str = "") -> None:
    with _MU:
        _ACTIVE[mode] = c


# ---- file load (donor calib_load :565-586) ----
CALIB_DIR = Path(os.environ.get("STRATA_JEV_CALIB_DIR", "calib"))


def load(path: Path) -> Calib:
    """Parse one calib.json; missing/corrupt -> not loaded (identity).
    B1.4 tuple refusal: a file whose tuple names a DIFFERENT readout letter than
    the mode it is being loaded for is refused (logged, identity) — copying
    calib-D.json to calib-S.json would otherwise apply silently (Opus G13)."""
    c = Calib()
    try:
        j = json.loads(Path(path).read_text())
        stem = path.stem
        want = stem.split("calib-")[-1] if stem.startswith("calib-") else ""
        if len(want) != 1 or want not in "LVDS":
            want = ""                            # legacy calib.json / v1.json: no mode binding
        got = str(j.get("tuple", "")).split("|")[4] if str(j.get("tuple", "")).count("|") >= 5 else ""
        if want and got and got != want:
            print(f"[jev-calib] refusing {path.name}: tuple readout '{got}' != '{want}' "
                  f"(B1.4 mismatch; identity until a matching file is fitted)", flush=True)
            return c
        if "T_choice" in j:
            c.T_choice = float(j["T_choice"])
        if "T_score" in j:
            c.T_score = float(j["T_score"])
        if "a_noul" in j:
            c.a_noul = float(j["a_noul"])
        if "c_noul" in j:
            c.c_noul = float(j["c_noul"])
        if "tuple" in j:
            c.tuple_str = str(j["tuple"])
        c.loaded = True
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return c


def swap(dir: Path | None = None) -> dict[str, bool]:
    """Hot-reload from CALIB_DIR (or the argument): the donor's calib.json (legacy
    text tuple, key "") plus per-readout files calib-<mode>.json for L/V/D/S.
    Returns which modes loaded (state_json shows it)."""
    d = Path(dir) if dir is not None else CALIB_DIR
    loaded: dict[str, bool] = {}
    with _MU:
        for mode, names in (("", ("calib.json", "v1.json")),
                            ("L", ("calib-L.json",)),
                            ("V", ("calib-V.json",)),
                            ("D", ("calib-D.json",)),
                            ("S", ("calib-S.json",))):
            best: Calib | None = None
            for name in names:
                c = load(d / name)
                if c.loaded:
                    best = c
                    break
            _ACTIVE[mode] = best
            loaded[mode or "text(legacy)"] = best is not None
    return loaded


# ---- apply (donor calib_apply :603-629; P0-1: per readout mode) ----
def apply(qtype: str, pr: list[float], mode: str = "") -> tuple[list[float], str]:
    """Calibrate label-normalized probabilities in place; -> (pr, calib_id).
    `mode`: the readout letter ("" text legacy / L / V / D / S).
    Fallback is STRICT: only the text modes ("" and L) may fall back to the
    legacy calib.json fit; V/D/S use their own calib-<mode>.json or identity —
    an image mode silently inheriting the TEXT fit is the exact cross-tuple
    contamination the per-tuple split exists to prevent (Opus R1)."""
    c = active(mode)
    if c is None and mode in ("", "L"):
        c = active("")
    if c is None or not c.loaded or not pr:
        return pr, "identity"
    if qtype == "noul":
        p = min(max(pr[1] if len(pr) > 1 else 0.0, 1e-9), 1 - 1e-9)
        d = math.log(p / (1 - p))
        q = 1.0 / (1.0 + math.exp(-(c.a_noul * d + c.c_noul)))
        if len(pr) > 1:
            pr[1] = q
        pr[0] = 1.0 - q
        s = sum(pr)
        if s > 0:
            for i in range(len(pr)):
                pr[i] /= s
        return pr, "platt_v1"
    T = c.T_score if qtype == "score" else c.T_choice
    if T <= 0.01 or abs(T - 1.0) < 1e-9:
        return pr, "identity"
    lm = math.log(max(max(pr), 1e-300))
    lse = 0.0
    for x in pr:
        lse += math.exp((math.log(max(x, 1e-300)) - lm) / T)
    lse = math.log(lse)
    for i in range(len(pr)):
        pr[i] = math.exp((math.log(max(pr[i], 1e-300)) - lm) / T - lse)
    s = sum(pr)
    if s > 0:
        for i in range(len(pr)):
            pr[i] /= s
    return pr, ("T_score_v1" if qtype == "score" else "T_choice_v1")


# ---- async log (donor B1.1: calib_log_thread :540-557) ----
_LOGQ: list[str] = []
_LOGMU = threading.Lock()
_MAXQ = 4096
ADMIN_KEY = os.environ.get("STRATA_JEV_ADMIN_KEY", "")

# ---- mode gating (donor --modes chat,decision; A0.1) ----
# Which capability surfaces this process exposes. Comma-separated env var:
#   STRATA_JEV_MODES="decision"          -> decision-only (no chat, no then.generate)
#   STRATA_JEV_MODES="chat,decision"     -> both (default)
# Unknown tokens fail-closed at load (ValueError -> serve refuses to start).
# Surfaces: "decision" gates /v1/systemone + the chat jev extension;
#           "chat"     gates plain /v1/chat generation + then.generate narration.
def _parse_modes(raw: str) -> frozenset:
    toks = [t.strip() for t in raw.split(",") if t.strip()]
    if not toks:
        return frozenset()
    bad = [t for t in toks if t not in ("chat", "decision")]
    if bad:
        raise ValueError(f"STRATA_JEV_MODES: unknown mode(s) {bad}; valid: chat, decision")
    return frozenset(toks)

MODES: frozenset = _parse_modes(os.environ.get("STRATA_JEV_MODES", "chat,decision"))


def log_enqueue(row: dict) -> None:
    line = json.dumps(row, separators=(",", ":"))
    with _LOGMU:
        if len(_LOGQ) >= _MAXQ:
            return
        _LOGQ.append(line)


def log_flush(dir: Path | None = None) -> int:
    """Drain the queue to <CALIB_DIR>/data/log-<mode>.jsonl (one file per readout
    tuple; the donor's fit.py pools a single log by type, so a mixed file would
    fit text+image rows into one calibrator — Opus G3); -> rows written."""
    d = Path(dir) if dir is not None else CALIB_DIR
    (d / "data").mkdir(parents=True, exist_ok=True)
    with _LOGMU:
        rows, _LOGQ[:] = _LOGQ[:], []
    if not rows:
        return 0
    by_mode = {}
    for line in rows:
        try:
            mode = json.loads(line).get("tuple", "|").split("|")[4] if line.count("|") >= 5 else ""
        except ValueError:
            mode = ""
        by_mode.setdefault(mode or "", []).append(line)
    n = 0
    for mode, lines in by_mode.items():
        with open(d / "data" / f"log{'-' + mode if mode else ''}.jsonl", "a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        n += len(lines)
    return n


def log_thread(stop: threading.Event | None = None):
    """Background flusher (200 ms), exits when stop is set and queue is empty."""
    while not (stop.wait(0.2) if stop is not None else False):
        log_flush()
    log_flush()


def tuple_of(readout_mode: str, qtype: str) -> str:
    alias = os.environ.get("STRATA_JEV_ALIAS", "local")
    quant = os.environ.get("STRATA_JEV_QUANT", "iq3_xxs")
    return f"{alias}|{quant}|none|v1|{readout_mode}|{qtype}"


def state_json() -> dict:
    out: dict = {"loaded": False, "modes": {}}
    for mode in ("", "L", "V", "D", "S"):
        c = active(mode)
        if c is None or not c.loaded:
            out["modes"][mode or "text(legacy)"] = {"loaded": False}
            continue
        if mode == "":
            out.update({"loaded": True, "tuple": c.tuple_str, "T_choice": c.T_choice,
                        "T_score": c.T_score, "a_noul": c.a_noul, "c_noul": c.c_noul})
        out["modes"][mode or "text(legacy)"] = {
            "loaded": True, "tuple": c.tuple_str, "T_choice": c.T_choice,
            "T_score": c.T_score, "a_noul": c.a_noul, "c_noul": c.c_noul}
    return out
