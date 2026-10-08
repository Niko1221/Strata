"""serve/usage.py - token usage kept across starts, and what those tokens would have cost at API prices.

The server's own totals (GET /metrics) start at zero with every start; this keeps the same counts per day in a
small JSON file next to the run config (strata-<model>.usage.json), so the Monitor can show today and all time.
A pricing block turns the tokens into money (strata-<model>.pricing.json, or the web page's Token pricing card):
prices per million tokens, with a separate price for prompt tokens the conversation cache reused - the same shape
the API providers bill.  DEFAULT_PRICING is the starting point; every field can be set to one's own.

Electricity is billed the same way.  The sampler (serve/telemetry.py) hands every GPU power reading to
record_energy(), which keeps watt-hours per day: the whole box is what one pays for, so the card's own watt-hours
are added to `other_watts` (CPU, board, drives) times the seconds sampled, and `price_per_kwh` is what a
kilowatt-hour costs where one lives.  A machine that cannot read the card's power still counts the seconds, so
only the card's share of the bill goes missing.
Nothing here can stop the server: a file that cannot be read or written only means the numbers start over.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

# the reference prices the pricing card starts from: per million tokens for the tokens, per kWh for the power,
# CNY.  Every field can be changed.  `other_watts` is the box without the card - CPU, board, drives - the card's
# own draw being measured.
DEFAULT_PRICING = {"currency": "CNY", "input_per_mtok": 2.0, "cached_input_per_mtok": 0.4, "output_per_mtok": 8.0,
                   "price_per_kwh": 0.55, "other_watts": 100.0}

CURRENCIES = {"CNY": "¥", "USD": "$", "EUR": "€", "JPY": "¥", "GBP": "£"}

TOKEN_KEYS = ("input_per_mtok", "cached_input_per_mtok", "output_per_mtok")
POWER_KEYS = ("price_per_kwh", "other_watts")

# what a day holds: the token counts, then the power the sampler saw (watt-hours on the card, how much of it was
# drawn while the GPU worked, and the seconds sampled - the last one feeds the fixed part of the bill too)
COUNT_KEYS = ("requests", "prompt_tokens", "reused", "output_tokens", "energy_wh", "busy_wh", "seconds")
COUNT_FLOATS = ("energy_wh", "busy_wh", "seconds")     # the rest of them are whole counts


def count_value(key: str, value) -> float | int:
    """One stored counter as its own type: the watt-hours and the seconds are fractional, the counts are not.
    A missing, unreadable or negative value reads as 0 (a JSON file one edited by hand must not stop anything)."""
    try:
        n = float(value or 0.0)
    except (TypeError, ValueError):
        n = 0.0
    if n != n:                                          # NaN would spread through every sum
        n = 0.0
    return round(max(0.0, n), 4) if key in COUNT_FLOATS else int(max(0.0, n))


def clean_pricing(d, base: dict | None = None) -> dict:
    """The pricing to use: `base` (the current or default) with the fields `d` names changed.  A field sent as
    null or "" goes back to the default; anything else than a number >= 0 (or a known currency) is a ValueError
    naming what is expected."""
    out = dict(DEFAULT_PRICING if base is None else base)
    if d is None:
        return out
    if not isinstance(d, dict):
        raise ValueError("pricing must be an object")
    for key, value in d.items():
        if value is None or value == "":
            out[key] = DEFAULT_PRICING[key]
            continue
        if key == "currency":
            if value not in CURRENCIES:
                raise ValueError(f"currency: one of {', '.join(CURRENCIES)}, not {value!r}")
            out[key] = value
        elif key in TOKEN_KEYS:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not (value >= 0) or value != value:
                raise ValueError(f"{key}: a number of at least 0 per million tokens, not {value!r}")
            out[key] = float(value)
        elif key in POWER_KEYS:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not (value >= 0) or value != value:
                raise ValueError(f"{key}: a number of at least 0, not {value!r}")
            out[key] = float(value)
        else:
            raise ValueError(f"unknown pricing key {key!r}")
    return out


def cost_of(counts: dict, pricing: dict | None = None) -> dict:
    """What `counts` (prompt_tokens / reused / output_tokens / energy_wh / seconds) costs at `pricing`: the prompt
    tokens the cache reused are billed as cached input, the rest as input; the card's watt-hours, plus the rest of
    the box at `other_watts` over the seconds sampled, are billed as electricity.  The amounts are rounded to the
    micro-unit of the currency so sums stay exact enough for a readout."""
    p = dict(DEFAULT_PRICING if pricing is None else pricing)
    reused = max(0, int(counts.get("reused") or 0))
    inp = max(0, int(counts.get("prompt_tokens") or 0) - reused)
    out = max(0, int(counts.get("output_tokens") or 0))
    ic = inp / 1e6 * float(p["input_per_mtok"])
    cc = reused / 1e6 * float(p["cached_input_per_mtok"])
    oc = out / 1e6 * float(p["output_per_mtok"])
    gpu_wh = max(0.0, float(counts.get("energy_wh") or 0.0))
    secs = max(0.0, float(counts.get("seconds") or 0.0))
    kwh = (gpu_wh + float(p["other_watts"]) * secs / 3600.0) / 1000.0
    ec = kwh * float(p["price_per_kwh"])
    return {"currency": p.get("currency", "CNY"),
            "input_cost": round(ic, 6), "cached_cost": round(cc, 6), "output_cost": round(oc, 6),
            "token_cost": round(ic + cc + oc, 6),
            "gpu_energy_wh": round(gpu_wh, 4), "energy_kwh": round(kwh, 6),
            "energy_cost": round(ec, 6), "total_cost": round(ic + cc + oc + ec, 6)}


class UsageStore:
    """Per-day token counts and watt-hours, kept in a JSON file when one is given (next to the run config) and in
    memory always.  record() runs once per finished request; the file is small and rewritten whole, which at a few
    requests a second costs nothing.  record_energy() runs with the power sampler, so its writes are spaced."""

    SAVE_EVERY_S = 10.0                                 # how often the power path may rewrite the file

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.lock = threading.Lock()
        self.days: dict[str, dict] = {}
        self._saved_at = 0.0
        if self.path:
            try:
                d = json.loads(self.path.read_text(encoding="utf-8-sig"))
                if isinstance(d, dict) and isinstance(d.get("days"), dict):
                    self.days = {k: {kk: count_value(kk, vv) for kk, vv in v.items() if kk in COUNT_KEYS}
                                 for k, v in d["days"].items() if isinstance(v, dict)}
            except (OSError, ValueError):
                self.days = {}

    @staticmethod
    def _today() -> str:
        return time.strftime("%Y-%m-%d")

    def record(self, prompt_tokens: int, reused: int, output_tokens: int) -> None:
        with self.lock:
            d = self.days.setdefault(self._today(), {k: 0 for k in COUNT_KEYS})
            d["requests"] += 1
            d["prompt_tokens"] += int(prompt_tokens)
            d["reused"] += int(reused)
            d["output_tokens"] += int(output_tokens)
            self._save()

    def record_energy(self, watts: float | None, seconds: float, busy: bool = False) -> None:
        """One reading from the power sampler: `watts` held for `seconds`.  A reading that is missing or unreadable
        still counts the seconds - the fixed part of the bill, everything but the card, keeps accruing on its own -
        only the card's share goes missing.  `busy` marks a reading taken while the GPU was working; it splits the
        card's own watt-hours, the bill itself does not care."""
        w: float | None
        try:
            w = float(watts)
        except (TypeError, ValueError):
            w = None
        else:
            if w != w or w < 0:
                w = None
        secs = max(0.0, float(seconds or 0.0))
        with self.lock:
            d = self.days.setdefault(self._today(), {k: 0 for k in COUNT_KEYS})
            d["seconds"] = round(float(d.get("seconds") or 0.0) + secs, 4)
            if w is not None:
                wh = w * secs / 3600.0
                d["energy_wh"] = round(float(d.get("energy_wh") or 0.0) + wh, 4)
                if busy:
                    d["busy_wh"] = round(float(d.get("busy_wh") or 0.0) + wh, 4)
            self._save_soon()

    def flush(self) -> None:
        """Writes what the power path has kept in memory only (the server, as it stops)."""
        with self.lock:
            self._save()

    def _save_soon(self) -> None:
        """record_energy() runs about once a second: rewriting the file every time would be some 86k small writes
        a day, so the power path keeps the memory copy current and the file at most this often.  A request's own
        record() still saves at once, and so does the server as it stops."""
        if time.time() - self._saved_at < self.SAVE_EVERY_S:
            return
        self._save()

    def _save(self) -> None:
        self._saved_at = time.time()
        if not self.path:
            return
        try:
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps({"days": self.days}, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as e:
            print(f"[strata] could not save the usage stats: {e}", flush=True)

    def view(self, pricing: dict | None = None) -> dict:
        """GET /usage and the Monitor's card: today, the last 14 days newest first, and the all-time sum -
        each with its cost at `pricing`."""
        with self.lock:
            days = {k: dict(v) for k, v in self.days.items()}
        empty = {k: 0 for k in COUNT_KEYS}
        total = {k: count_value(k, sum(float(d.get(k) or 0.0) for d in days.values())) for k in COUNT_KEYS}

        def row(name: str, counts: dict) -> dict:
            return {"date": name, **{k: count_value(k, counts.get(k)) for k in COUNT_KEYS},
                    "cost": cost_of(counts, pricing)}

        return {"today": row(self._today(), days.get(self._today(), empty)),
                "days": [row(k, v) for k, v in sorted(days.items(), reverse=True)[:14]],
                "total": row("total", total)}
