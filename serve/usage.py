"""serve/usage.py - token usage kept across starts, and what those tokens would have cost at API prices.

The server's own totals (GET /metrics) start at zero with every start; this keeps the same counts per day in a
small JSON file next to the run config (strata-<model>.usage.json), so the Monitor can show today and all time.
A pricing block turns the tokens into money (strata-<model>.pricing.json, or the web page's Token pricing card):
prices per million tokens, with a separate price for prompt tokens the conversation cache reused - the same shape
the API providers bill.  DEFAULT_PRICING is the starting point; every field can be set to one's own.

One folder usually holds the usage file of every model one has run, so the Monitor can also add them up (the
default) or tell them apart ("By model" in the web app's Settings).  That reads the other files and never writes to
them; each model is named after the "model_name" of the run config beside its file, which is what one calls it.

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

# one usage file per model, next to its run config: strata-<model>.usage.json <- strata-<model>.json
USAGE_SUFFIX = ".usage.json"


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


def read_days(path) -> dict[str, dict]:
    """The per-day counts in one usage file.  A file that is missing, unreadable or half written reads as the days
    it does hold (nothing here may stop the server), and every count goes through count_value()."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    if not isinstance(d, dict) or not isinstance(d.get("days"), dict):
        return {}
    return {k: {kk: count_value(kk, vv) for kk, vv in v.items() if kk in COUNT_KEYS}
            for k, v in d["days"].items() if isinstance(v, dict)}


def model_name(usage_path) -> str:
    """What the model behind one usage file is called: the "model_name" of the run config it sits next to
    (strata-<model>.usage.json <-> strata-<model>.json), or the file's own stem when there is no readable one.
    A config one edited by hand, or a file someone copied here, still names itself."""
    p = Path(usage_path)
    stem = p.name[:-len(USAGE_SUFFIX)] if p.name.endswith(USAGE_SUFFIX) else p.stem
    try:
        cfg = json.loads(p.with_name(stem + ".json").read_text(encoding="utf-8-sig"))
        name = cfg.get("model_name") if isinstance(cfg, dict) else None
    except (OSError, ValueError):
        name = None
    return name.strip() if isinstance(name, str) and name.strip() else stem


def summed(days_list) -> dict[str, dict]:
    """Several models' per-day counts added up: a day then holds what every model did on it.  This is what the
    Monitor shows when it is not showing the models apart - the whole folder's bill."""
    out: dict[str, dict] = {}
    for days in days_list:
        for day, counts in days.items():
            row = out.setdefault(day, {k: 0 for k in COUNT_KEYS})
            for k in COUNT_KEYS:
                v = float(counts.get(k) or 0.0)
                row[k] = round(row[k] + v, 4) if k in COUNT_FLOATS else int(row[k] + v)
    return out


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


def days_view(days: dict[str, dict], pricing: dict | None = None, today: str | None = None) -> dict:
    """The rows the Monitor draws from one set of per-day counts (one model's, or the folder's added up): today,
    the last 14 days newest first, and the all-time sum - each with its cost at `pricing`."""
    today = today or time.strftime("%Y-%m-%d")
    empty = {k: 0 for k in COUNT_KEYS}
    total = {k: count_value(k, sum(float(d.get(k) or 0.0) for d in days.values())) for k in COUNT_KEYS}

    def row(name: str, counts: dict) -> dict:
        return {"date": name, **{k: count_value(k, counts.get(k)) for k in COUNT_KEYS},
                "cost": cost_of(counts, pricing)}

    return {"today": row(today, days.get(today, empty)),
            "days": [row(k, v) for k, v in sorted(days.items(), reverse=True)[:14]],
            "total": row("total", total)}


class UsageStore:
    """Per-day token counts and watt-hours, kept in a JSON file when one is given (next to the run config) and in
    memory always.  record() runs once per finished request; the file is small and rewritten whole, which at a few
    requests a second costs nothing.  record_energy() runs with the power sampler, so its writes are spaced."""

    SAVE_EVERY_S = 10.0                                 # how often the power path may rewrite the file

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.lock = threading.Lock()
        self.days: dict[str, dict] = read_days(self.path) if self.path else {}
        self._saved_at = 0.0

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

    def snapshot(self) -> dict[str, dict]:
        """A copy of the per-day counts, taken under the lock: the power path records while a caller is adding
        days up, and a dict that changes under an iteration is not worth the risk."""
        with self.lock:
            return {k: dict(v) for k, v in self.days.items()}

    def view(self, pricing: dict | None = None) -> dict:
        """GET /usage and the Monitor's card: today, the last 14 days newest first, and the all-time sum of this
        model - each with its cost at `pricing`."""
        return days_view(self.snapshot(), pricing, self._today())


class ModelsUsage:
    """The models sharing the running model's folder, read from their own usage files: one entry per model_name,
    which is what "By model" on the Monitor's card shows, and what the card's default adds up.

    The model the server is running is not read from its file here - its counts live in the UsageStore and are
    fresher, because the power sampler keeps them in memory between saves.  Everything else is a few small files
    per folder, read at most once per poll: /metrics is asked once a second and the files change only when a
    request or a save lands, so days are kept until mtime or size moves.  Nothing is written."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._cache: dict[str, tuple[int, int, dict[str, dict]]] = {}

    def _days(self, p: Path) -> dict[str, dict]:
        try:
            st = p.stat()
        except OSError:
            return {}
        hit = self._cache.get(str(p))
        if hit and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
            return hit[2]
        days = read_days(p)
        self._cache[str(p)] = (st.st_mtime_ns, st.st_size, days)
        return days

    def models(self, live_days: dict[str, dict] | None = None, live_name: str = "") -> list[tuple[str, dict]]:
        """Every model's per-day counts, the running one (from memory) included.  Two run configs that carry the
        same model_name - the same model at another context or quantisation - are one model and add up; a file
        whose config is gone names itself after the file.  Unsorted: the caller knows the prices to sort by."""
        found: dict[str, dict] = {}
        if self.path and self.path.parent.is_dir():
            for p in sorted(self.path.parent.glob("*" + USAGE_SUFFIX)):
                if self.path and p.name == self.path.name:
                    continue                            # the live model: the caller hands its counts in
                name, days = model_name(p), self._days(p)
                found[name] = summed([found[name], days]) if name in found else days
        if live_name or live_days:
            name, days = live_name or "this model", dict(live_days or {})
            found[name] = summed([found[name], days]) if name in found else days
        return list(found.items())
