"""serve/usage.py - token usage kept across starts, and what those tokens would have cost at API prices.

The server's own totals (GET /metrics) start at zero with every start; this keeps the same counts per day in a
small JSON file next to the run config (strata-<model>.usage.json), so the Monitor can show today and all time.
A pricing block turns the tokens into money (strata-<model>.pricing.json, or the web page's Token pricing card):
prices per million tokens, with a separate price for prompt tokens the conversation cache reused - the same shape
the API providers bill.  DEFAULT_PRICING is the starting point; every field can be set to one's own.
Nothing here can stop the server: a file that cannot be read or written only means the numbers start over.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

# the reference prices the pricing card starts from (per million tokens, CNY); every field can be changed
DEFAULT_PRICING = {"currency": "CNY", "input_per_mtok": 2.0, "cached_input_per_mtok": 0.4, "output_per_mtok": 8.0}

CURRENCIES = {"CNY": "¥", "USD": "$", "EUR": "€", "JPY": "¥", "GBP": "£"}

COUNT_KEYS = ("requests", "prompt_tokens", "reused", "output_tokens")


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
        elif key in ("input_per_mtok", "cached_input_per_mtok", "output_per_mtok"):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not (value >= 0) or value != value:
                raise ValueError(f"{key}: a number of at least 0 per million tokens, not {value!r}")
            out[key] = float(value)
        else:
            raise ValueError(f"unknown pricing key {key!r}")
    return out


def cost_of(counts: dict, pricing: dict | None = None) -> dict:
    """What `counts` (prompt_tokens / reused / output_tokens) costs at `pricing`: the prompt tokens the cache
    reused are billed as cached input, the rest as input; the amounts are rounded to the micro-unit of the
    currency so sums stay exact enough for a readout."""
    p = dict(DEFAULT_PRICING if pricing is None else pricing)
    reused = max(0, int(counts.get("reused") or 0))
    inp = max(0, int(counts.get("prompt_tokens") or 0) - reused)
    out = max(0, int(counts.get("output_tokens") or 0))
    ic = inp / 1e6 * float(p["input_per_mtok"])
    cc = reused / 1e6 * float(p["cached_input_per_mtok"])
    oc = out / 1e6 * float(p["output_per_mtok"])
    return {"currency": p.get("currency", "CNY"),
            "input_cost": round(ic, 6), "cached_cost": round(cc, 6), "output_cost": round(oc, 6),
            "total_cost": round(ic + cc + oc, 6)}


class UsageStore:
    """Per-day token counts, kept in a JSON file when one is given (next to the run config) and in memory
    always.  record() runs once per finished request; the file is small and rewritten whole, which at a few
    requests a second costs nothing."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.lock = threading.Lock()
        self.days: dict[str, dict] = {}
        if self.path:
            try:
                d = json.loads(self.path.read_text(encoding="utf-8-sig"))
                if isinstance(d, dict) and isinstance(d.get("days"), dict):
                    self.days = {k: {kk: int(vv or 0) for kk, vv in v.items() if kk in COUNT_KEYS}
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

    def _save(self) -> None:
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
        total = {k: sum(d.get(k, 0) for d in days.values()) for k in COUNT_KEYS}

        def row(name: str, counts: dict) -> dict:
            return {"date": name, **{k: int(counts.get(k, 0)) for k in COUNT_KEYS},
                    "cost": cost_of(counts, pricing)}

        return {"today": row(self._today(), days.get(self._today(), empty)),
                "days": [row(k, v) for k, v in sorted(days.items(), reverse=True)[:14]],
                "total": row("total", total)}
