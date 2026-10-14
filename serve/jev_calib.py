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
_ACTIVE: dict[str, Calib | None] = {"c": None}
_MU = threading.Lock()


def active() -> Calib | None:
    with _MU:
        return _ACTIVE["c"]


def _set_for_test(c: Calib | None) -> None:
    with _MU:
        _ACTIVE["c"] = c


# ---- file load (donor calib_load :565-586) ----
CALIB_DIR = Path(os.environ.get("STRATA_JEV_CALIB_DIR", "calib"))


def load(path: Path) -> Calib:
    """Parse one calib.json; missing/corrupt -> not loaded (identity)."""
    c = Calib()
    try:
        j = json.loads(Path(path).read_text())
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


def swap(dir: Path | None = None) -> Calib:
    """Hot-reload from CALIB_DIR (or the argument): calib.json, then v1.json."""
    d = Path(dir) if dir is not None else CALIB_DIR
    best: Calib | None = None
    for name in ("calib.json", "v1.json"):
        c = load(d / name)
        if c.loaded:
            best = c
            break
    with _MU:
        _ACTIVE["c"] = best
    return best if best is not None else Calib()


# ---- apply (donor calib_apply :603-629) ----
def apply(qtype: str, pr: list[float]) -> tuple[list[float], str]:
    """Calibrate label-normalized probabilities in place; -> (pr, calib_id)."""
    c = active()
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


def log_enqueue(row: dict) -> None:
    line = json.dumps(row, separators=(",", ":"))
    with _LOGMU:
        if len(_LOGQ) >= _MAXQ:
            return
        _LOGQ.append(line)


def log_flush(dir: Path | None = None) -> int:
    """Drain the queue to <CALIB_DIR>/data/log.jsonl; -> rows written."""
    d = Path(dir) if dir is not None else CALIB_DIR
    (d / "data").mkdir(parents=True, exist_ok=True)
    with _LOGMU:
        rows, _LOGQ[:] = _LOGQ[:], []
    if not rows:
        return 0
    with open(d / "data" / "log.jsonl", "a", encoding="utf-8") as f:
        for line in rows:
            f.write(line + "\n")
    return len(rows)


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
    c = active()
    if c is None or not c.loaded:
        return {"loaded": False, "tuple": "", "T_choice": 1.0,
                "T_score": 1.0, "a_noul": 1.0, "c_noul": 0.0}
    return {"loaded": True, "tuple": c.tuple_str, "T_choice": c.T_choice,
            "T_score": c.T_score, "a_noul": c.a_noul, "c_noul": c.c_noul}
