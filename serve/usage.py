"""Persistent token value and GPU work accounting. No prompts, API keys or external requests."""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

DEFAULT_RATES = {"input_per_million": 0.15, "output_per_million": 0.47,
                 "cached_input_per_million": 0.016, "electricity_per_kwh": None,
                 "electricity_currency": "USD", "usd_to_electricity_currency": None}
SOURCE = "https://openrouter.ai/qwen/qwen3.8-flash"
CHECKED = "2026-10-13"
COUNTERS = ("requests", "input_tokens", "cached_tokens", "output_tokens", "active_s", "energy_wh",
            "measured_s", "unmeasured_s", "cpu_s", "cpu_measured_s", "cpu_unmeasured_s", "decode_s", "decode_tokens")


def empty_counts():
    return {**dict.fromkeys(COUNTERS, 0), "energy_method": None}


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def clean_rates(changes, current):
    if not isinstance(changes, dict) or not changes:
        raise ValueError("rates must be a non-empty object")
    out = dict(current)
    for key, value in changes.items():
        if key not in DEFAULT_RATES:
            raise ValueError(f"unknown rate {key!r}")
        if key == "electricity_currency":
            if not isinstance(value, str) or not re.fullmatch(r"[A-Z]{3}", value):
                raise ValueError("electricity_currency: a three-letter currency code, such as USD or BDT")
        elif key in ("electricity_per_kwh", "usd_to_electricity_currency") and value is None:
            pass
        elif not number(value) or (key == "usd_to_electricity_currency" and value == 0):
            raise ValueError(f"{key}: a finite nonnegative price, or a positive exchange rate")
        out[key] = value
    return out


def process_cpu_times(owners):
    """User + kernel CPU seconds of engines and encoder, keyed by process lifetime."""
    try:
        import psutil
    except ImportError:
        return None
    pids = set()
    for owner in owners:
        for worker in getattr(owner, "engines", None) or [owner]:
            proc = getattr(worker, "proc", None)
            if proc is not None and proc.poll() is None:
                pids.add(proc.pid)
    if not pids:
        return None
    out = {}
    try:
        for pid in pids:
            proc = psutil.Process(pid)
            used = proc.cpu_times()
            out[f"{pid}:{proc.create_time()}"] = used.user + used.system
    except (psutil.Error, OSError):
        return None
    return out


def priced(counts, rates, pending=()):
    inputs = counts["input_tokens"] + sum(row.get("input_tokens", 0) for row in pending)
    outputs = counts["output_tokens"] + sum(row.get("output_tokens", 0) for row in pending)
    cached = min(inputs, counts["cached_tokens"])
    value = (inputs * rates["input_per_million"] + outputs * rates["output_per_million"]) / 1_000_000
    cache_value = ((inputs - cached) * rates["input_per_million"] + cached * rates["cached_input_per_million"] +
                   outputs * rates["output_per_million"]) / 1_000_000
    energy = counts["energy_wh"] / 1000 if counts["measured_s"] > 0 or counts["active_s"] == 0 else None
    tariff = rates["electricity_per_kwh"]
    cost = energy * tariff if energy is not None and tariff is not None else 0.0 if tariff == 0 else None
    fx = 1 if rates["electricity_currency"] == "USD" else rates["usd_to_electricity_currency"]
    cost_usd = cost / fx if cost is not None and fx else None
    return {**{key: counts[key] for key in COUNTERS}, "input_tokens": inputs, "output_tokens": outputs,
            "cached_tokens": cached, "total_tokens": inputs + outputs, "energy_kwh": energy,
            "cache_hit_pct": cached / inputs * 100 if inputs else None,
            "decode_tok_s": counts["decode_tokens"] / counts["decode_s"] if counts["decode_s"] > 0 else None,
            "energy_method": counts["energy_method"],
            "cpu_s": counts["cpu_s"] if counts["cpu_measured_s"] > 0 or counts["active_s"] == 0 else None,
            "api_value_usd": value, "cache_adjusted_usd": cache_value, "electricity_cost": cost,
            "electricity_cost_usd": cost_usd, "profit_usd": value - cost_usd if cost_usd is not None else None,
            "profit_partial": counts["unmeasured_s"] > 0}


class UsageLedger:
    def __init__(self, path, model, sample=None, clock=time.monotonic, wall_clock=time.time):
        self.path, self.model = Path(path), model
        self.sample, self.clock, self.wall_clock = sample, clock, wall_clock
        self.lock = threading.RLock()
        self.active = set()
        self._at, self._wall, self._reading = None, None, {}
        self._stop = threading.Event()
        self._thread = None
        self._last_save = 0.0
        self.persistence_error = None
        self.state = {**empty_counts(), "version": 2, "model": model, "since": wall_clock(),
                      "energy_since": wall_clock(), "daily_since": wall_clock(), "daily": {},
                      "undated": empty_counts(), "seeded": False, "rates": dict(DEFAULT_RATES),
                      "rates_updated_at": None}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict) or data.get("version") not in (1, 2) or data.get("model") != model:
                raise ValueError("the usage ledger has an unsupported format or belongs to another model")
            for key in ("since", "energy_since", "requests", "input_tokens", "cached_tokens", "output_tokens",
                        "active_s", "energy_wh", "measured_s", "unmeasured_s"):
                if not number(data.get(key)):
                    raise ValueError(f"the usage ledger has an invalid {key}")
            data["rates"] = clean_rates(data.get("rates"), DEFAULT_RATES)
            if data["version"] == 1:
                old = {**empty_counts(), **{k: data[k] for k in COUNTERS[:8]},
                       "energy_method": data.get("energy_method"), "cpu_unmeasured_s": data["active_s"]}
                data.update(version=2, daily_since=wall_clock(), daily={}, undated=old,
                            cpu_s=0.0, cpu_measured_s=0.0, cpu_unmeasured_s=data["active_s"])
            if not isinstance(data.get("daily"), dict) or not isinstance(data.get("undated"), dict):
                raise ValueError("the usage ledger has invalid daily history")
            for day, counts in [(None, data), (None, data["undated"]), *data["daily"].items()]:
                if day is not None and (not isinstance(day, str) or
                                        not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day)):
                    raise ValueError("the usage ledger has an invalid date")
                if isinstance(counts, dict):
                    counts.setdefault("decode_s", 0.0)
                    counts.setdefault("decode_tokens", 0)
                if not isinstance(counts, dict) or any(not number(counts.get(k)) for k in COUNTERS):
                    raise ValueError("the usage ledger has invalid usage counters")
            self.state.update(data)

    def _save(self, state=None):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(state or self.state, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)
        self.persistence_error = None

    def _persist(self):
        try:
            self._save()
        except OSError as error:
            if self.persistence_error != str(error):
                print(f"[strata] usage totals could not be saved: {error}", flush=True)
            self.persistence_error = str(error)

    def seed(self, totals):
        """Import existing session counts once. Historical GPU energy cannot be reconstructed."""
        with self.lock:
            if self.state["seeded"]:
                return
            duration = (float(totals.get("prompt_ms", 0)) + float(totals.get("decode_ms", 0))) / 1000
            old = {**empty_counts(), "requests": int(totals.get("requests", 0)),
                   "input_tokens": int(totals.get("prompt_tokens", 0)), "cached_tokens": int(totals.get("reused", 0)),
                   "output_tokens": int(totals.get("output_tokens", 0)), "active_s": duration,
                   "unmeasured_s": duration, "cpu_unmeasured_s": duration}
            self.state.update(old, since=totals.get("since", self.wall_clock()), undated=dict(old), seeded=True)
            self._save()

    def _read(self):
        try:
            reading = self.sample() if self.sample else {}
            out = {key: reading.get(key) if number(reading.get(key)) else None
                   for key in ("power_w", "energy_mj")}
            cpu = reading.get("cpu_times")
            out["cpu_times"] = dict(cpu) if isinstance(cpu, dict) and cpu and all(number(v) for v in cpu.values()) else None
            return out
        except (OSError, RuntimeError, ValueError, TypeError, AttributeError):
            return {"power_w": None, "energy_mj": None, "cpu_times": None}

    @staticmethod
    def _add(counts, delta, fraction=1):
        for key, value in delta.items():
            if key == "energy_method":
                if value:
                    old = counts[key]
                    counts[key] = value if old in (None, value) else "mixed"
            else:
                counts[key] += value * fraction

    def _day(self, stamp):
        day = datetime.fromtimestamp(stamp).date().isoformat()
        return self.state["daily"].setdefault(day, empty_counts())

    def _distribute(self, delta, start, end):
        """Split elapsed work at local midnight; tokens settle on completion day."""
        cursor = start
        while cursor < end:
            tomorrow = datetime.fromtimestamp(cursor).date() + timedelta(days=1)
            boundary = time.mktime(tomorrow.timetuple())
            stop = min(end, boundary)
            if stop <= cursor:
                stop = end
            self._add(self._day(cursor), delta, (stop - cursor) / (end - start))
            cursor = stop

    def _advance(self):
        now, wall = self.clock(), self.wall_clock()
        reading = self._read()
        if self.active and self._at is not None:
            dt = max(0.0, now - self._at)
            delta = {"active_s": dt, "energy_wh": 0.0, "measured_s": 0.0, "unmeasured_s": 0.0,
                     "cpu_s": 0.0, "cpu_measured_s": 0.0, "cpu_unmeasured_s": 0.0, "energy_method": None}
            previous, energy = self._reading.get("energy_mj"), reading.get("energy_mj")
            power, previous_power = reading.get("power_w"), self._reading.get("power_w")
            if dt and previous is not None and energy is not None and energy >= previous:
                delta.update(energy_wh=(energy - previous) / 3_600_000, measured_s=dt, energy_method="energy_counter")
            elif dt and power is not None and previous_power is not None:
                delta.update(energy_wh=(power + previous_power) / 2 * dt / 3600,
                             measured_s=dt, energy_method="sampled_power")
            else:
                delta["unmeasured_s"] = dt
            cpu, old_cpu = reading.get("cpu_times"), self._reading.get("cpu_times")
            if dt and cpu and old_cpu and cpu.keys() == old_cpu.keys() and all(cpu[k] >= old_cpu[k] for k in cpu):
                delta.update(cpu_s=sum(cpu[k] - old_cpu[k] for k in cpu), cpu_measured_s=dt)
            else:
                delta["cpu_unmeasured_s"] = dt
            self._add(self.state, delta)
            if dt:
                start = self._wall if 0 < wall - self._wall and abs(wall - self._wall - dt) < 2 else wall - dt
                self._distribute(delta, start, wall)
        self._at, self._wall, self._reading = now, wall, reading

    def begin(self, key):
        with self.lock:
            self._advance()
            self.active.add(key)

    def end(self, key):
        with self.lock:
            if key not in self.active:
                return
            self._advance()
            self.active.discard(key)
            self._persist()

    def record(self, input_tokens, cached_tokens, output_tokens, *, decode_ms=None):
        with self.lock:
            inputs = max(0, int(input_tokens))
            delta = {"requests": 1, "input_tokens": inputs,
                     "cached_tokens": min(max(0, int(cached_tokens)), inputs),
                     "output_tokens": max(0, int(output_tokens))}
            if number(decode_ms) and decode_ms > 0:
                delta.update(decode_s=decode_ms / 1000, decode_tokens=delta["output_tokens"])
            self._add(self.state, delta)
            self._add(self._day(self.wall_clock()), delta)
            self.state["seeded"] = True
            self._persist()

    def backfill_decode(self, history):
        """Import available completed-call timings once, without changing token or money totals."""
        with self.lock:
            if self.state.get("decode_history_imported"):
                return
            seen = set()
            for row in history:
                stamp, duration = row.get("time"), row.get("duration_s")
                outputs, milliseconds = row.get("output_tokens"), row.get("decode_ms")
                if not all(number(value) for value in (stamp, duration, outputs, milliseconds)) or milliseconds <= 0:
                    continue
                key = (stamp, duration, outputs, milliseconds)
                if key in seen:
                    continue
                seen.add(key)
                end = stamp + duration
                day = datetime.fromtimestamp(end).date().isoformat()
                if datetime.fromtimestamp(stamp).date().isoformat() != day:
                    continue
                counts = self.state["undated"] if end < self.state["daily_since"] else self.state["daily"].get(day)
                if counts is None or counts["decode_tokens"] + outputs > counts["output_tokens"]:
                    continue
                delta = {"decode_s": milliseconds / 1000, "decode_tokens": outputs}
                self._add(self.state, delta)
                self._add(counts, delta)
            self.state["decode_history_imported"] = True
            self._persist()

    def configure(self, changes):
        with self.lock:
            new = {**self.state, "rates": clean_rates(changes, self.state["rates"]),
                   "rates_updated_at": self.wall_clock()}
            self._save(new)
            self.state = new
            return self.snapshot()

    def snapshot(self, live=()):
        with self.lock:
            if self.active:
                self._advance()
            s, rates = self.state, dict(self.state["rates"])
            pending = [row for row in live if row["key"] in self.active]
            local = datetime.fromtimestamp(self.wall_clock()).astimezone()
            today = local.date().isoformat()
            daily = [{"date": day, **priced(counts, rates, pending if day == today else ())}
                     for day, counts in sorted({today: empty_counts(), **s["daily"]}.items(), reverse=True)]
            return {**priced(s, rates, pending), "model": self.model, "since": s["since"],
                    "energy_since": s["energy_since"], "daily_since": s["daily_since"], "daily": daily,
                    "daily_since_date": datetime.fromtimestamp(s["daily_since"]).date().isoformat(),
                    "undated": priced(s["undated"], rates), "today": today,
                    "timezone": local.tzname(), "timezone_offset_minutes": int(local.utcoffset().total_seconds() / 60),
                    "active": bool(self.active),
                    "power_w": self._reading.get("power_w") if self.active else None,
                    "rates": rates, "rates_updated_at": s["rates_updated_at"], "source": SOURCE,
                    "pricing_checked": CHECKED, "pricing_model": "Qwen3.8 Flash · OpenRouter",
                    "persistence_error": self.persistence_error}

    def start(self):
        if self._thread:
            return
        def poll():
            while not self._stop.wait(1):
                with self.lock:
                    if self.active:
                        self._advance()
                        if self.clock() - self._last_save >= 5:
                            self._persist()
                            self._last_save = self.clock()
        self._thread = threading.Thread(target=poll, daemon=True, name="strata-usage")
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        with self.lock:
            if self.active:
                self._advance()
                self.active.clear()
            self._persist()
