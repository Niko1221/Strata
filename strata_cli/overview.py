"""The monitor's Overview (view 1): the browser Monitor's page redone for the terminal.

A model-state bar, eight tiles with sparklines (speed, GPU load, unified memory, GPU temperature, power, PCIe, CPU, disk), bars for context
fill, the KV cache hit of the newest request that reports reuse, and system RAM, and the recent requests. It reads what the browser Monitor reads: the server's `/metrics`
(live state, engine facts, the hardware sample and a minute of history per series, the last requests) as the app's
apps/host/src/lib/native-monitor.ts maps it: the context used is the current request's prompt + generated tokens while one runs, else the
last request's; the speed is the live rate while generating, else the last request's.

`build(metrics)` turns the raw metrics into plain numbers (None where the server did not report one: never a made-up zero);
`render(...)` draws them. Neither needs Textual.
"""

from __future__ import annotations

import time
from typing import Any

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from strata_cli import nav

GIB = 2 ** 30
BLUE = "#5aa9ff"
AMBER = "#f5c56b"
RED = "#f28b82"
ACCENT = "#34d399"
GREY = "#59636b"
STATES = ("Idle", "Reading", "Generating", "Queued")     # Error is not printed as a legend: it joins the line, in red, only when the model has one
# the classic Windows console draws with its 16 palette colours; these are the names that map onto them (glyphs="solid")
SOLID_ACCENT, SOLID_BLUE, SOLID_AMBER, SOLID_RED, SOLID_TRACK = "bright_green", "bright_blue", "yellow", "red", "bright_black"


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return float(v)


def _gib(v: Any) -> float | None:
    n = _num(v)
    return None if n is None else n / GIB


def _series(history: dict, key: str, scale: float = 1.0) -> list[float | None]:
    raw = history.get(key) if isinstance(history, dict) else None
    if not isinstance(raw, list):
        return []
    return [(None if _num(x) is None else _num(x) / scale) for x in raw[-60:]]


def request_speed(r: dict) -> float | None:
    out, ms = _num(r.get("output_tokens")), _num(r.get("decode_ms"))
    if r.get("decode_tok_s") is not None and _num(r.get("decode_tok_s")) is not None:
        return _num(r.get("decode_tok_s"))
    return round(out / (ms / 1000.0), 1) if out and out >= 2 and ms and ms > 0 else None   # one token has no decode span (a probe showed 1,250 tok/s)


def prefill_speed(r: dict) -> float | None:
    """The prompt reading speed of one finished request: the tokens the engine actually read over the time it spent
    reading them, the same arithmetic its own request line prints (a reused prompt token is never read again).  None when
    the server sent no prompt timing for that request."""
    if _num(r.get("prefill_tok_s")) is not None:
        return _num(r.get("prefill_tok_s"))
    ms, read = _num(r.get("prompt_ms")), _num(r.get("prompt_read"))
    if read is None and _num(r.get("prompt_tokens")) is not None:
        read = _num(r.get("prompt_tokens")) - (_num(r.get("reused")) or 0)
    return round(read / (ms / 1000.0), 1) if read and ms and ms > 0 else None


def active_requests(metrics: dict, live: dict) -> list[dict]:
    """The requests the engine is working on right now, in the same shape as the finished ones so the request table can draw both.
    With `--parallel` the server lists every occupied batch slot (`live.slots`), so each one gets a row; on the single-turn path it
    reports only the request it is running, and a request still queued is a count rather than a row (the state line already says how
    many are queued).  Nothing is invented: the speed is the live rate the server gives for that request, and what it does not report
    (`reused` for a request that has not finished reading its prompt) stays missing."""
    sampled, mean_pre = _num(metrics.get("time")), _num(live.get("prefill_tok_s_mean"))

    def row(state: str, prompt, read_rate, generated, gen_rate, elapsed):
        return {"time": (sampled - elapsed) if (sampled is not None and elapsed is not None) else None, "state": state,
                "prompt": prompt, "reused": None, "output": generated or None, "prefill": read_rate, "tps": gen_rate,
                "duration": elapsed, "active": True}

    rows = []
    for s in (live.get("slots") if isinstance(live.get("slots"), list) else []):
        if not isinstance(s, dict) or s.get("state") not in ("reading", "decoding"):
            continue
        elapsed, read = _num(s.get("elapsed_s")), _num(s.get("prompt_read"))
        reading = s.get("state") == "reading"
        # a slot's own reading speed: the tokens it has read over the time it has been running, the engine's mean only when that slot
        # sends no read count; a decoding slot's prefill is not attributed to it, so its READ cell stays missing
        rate = round(read / elapsed, 1) if (reading and read and elapsed and elapsed > 0) else (mean_pre if reading else None)
        rows.append(row("Read" if reading else "Gen", _num(s.get("prompt_tokens")), rate, _num(s.get("generated")),
                        _num(s.get("tok_s")), elapsed))
    if rows:
        return rows
    state = str(live.get("state") or "")                      # reading / generating: the one request on the single-turn path
    if state not in ("reading", "generating"):
        return []
    return [row("Read" if state == "reading" else "Gen", _num(live.get("prompt_tokens")), mean_pre, _num(live.get("generated")),
                _num(live.get("tok_s")) if state == "generating" else None, _num(live.get("elapsed_s")))]


def _cards(hw: dict, static: dict) -> list[dict]:
    """Each graphics card's own readings when the server lists two or more (hardware.gpus); [] for one card or none: the Overview then
    draws exactly what it always drew."""
    raw = hw.get("gpus") if isinstance(hw.get("gpus"), list) else []
    cards = [{"index": c.get("index"), "name": c.get("name"), "util": _num(c.get("util")), "mem_used": _gib(c.get("mem_used")),
              "mem_total": _gib(c.get("mem_total")), "temp": _num(c.get("temp")), "power": _num(c.get("power")),
              "power_limit": _num(c.get("power_limit"))} for c in raw[:16] if isinstance(c, dict)]
    return cards if len(cards) >= 2 else []


def _total(values: list) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def _most(values: list) -> float | None:
    present = [v for v in values if v is not None]
    return max(present) if present else None


def build(metrics: dict | None) -> dict | None:
    """The numbers the Overview draws, or None when the server does not serve /metrics (an old server: the caller falls back)."""
    if not isinstance(metrics, dict) or not isinstance(metrics.get("live"), dict):
        return None
    live, engine = metrics["live"], metrics.get("engine") if isinstance(metrics.get("engine"), dict) else {}
    hw = metrics.get("hardware") if isinstance(metrics.get("hardware"), dict) else {}
    static = metrics.get("hardware_static") if isinstance(metrics.get("hardware_static"), dict) else {}
    history = metrics.get("history") if isinstance(metrics.get("history"), dict) else {}
    recent = [r for r in (metrics.get("requests") or []) if isinstance(r, dict)]
    state_raw = str(live.get("state") or "idle")
    queued = _num(live.get("queued")) or 0
    err = bool(engine.get("error")) or state_raw == "error"
    if err:
        state = "Error"
    elif state_raw == "reading":
        state = "Reading"
    elif state_raw == "generating":
        state = "Generating"
    elif queued > 0 and state_raw not in ("reading", "generating"):
        state = "Queued"
    else:
        state = "Idle"
    active = state in ("Reading", "Generating")
    prompt_read, prompt_total = _num(live.get("prompt_read")), _num(live.get("prompt_total"))
    generated, max_tokens = _num(live.get("generated")), _num(live.get("max_tokens"))
    if state == "Reading" and prompt_total:
        done, total = prompt_read or 0.0, prompt_total
    elif state == "Generating" and max_tokens:
        done, total = generated or 0.0, max_tokens
    else:
        done, total = None, None
    last = recent[0] if recent else None
    max_ctx = _num(engine.get("max_context"))
    if active and _num(live.get("prompt_tokens")) is not None and generated is not None:
        used = _num(live.get("prompt_tokens")) + generated
    elif last and _num(last.get("prompt_tokens")) is not None and _num(last.get("output_tokens")) is not None:
        used = _num(last["prompt_tokens"]) + _num(last["output_tokens"])
    else:
        used = None
    if state == "Generating" and _num(live.get("tok_s")) is not None:
        speed, speed_note = _num(live.get("tok_s")), "now"
    elif state == "Reading" and _num(live.get("prefill_tok_s_mean")) is not None:
        speed, speed_note = _num(live.get("prefill_tok_s_mean")), "reading the prompt"
    else:
        speed, speed_note = (request_speed(last) if last else None), "last request"
    slots, total_slots = _num(engine.get("expert_slots")), _num(engine.get("experts"))
    cache_mib = _num(engine.get("expert_cache_mib"))
    gen, genmax, width = _num(hw.get("gpu_pcie_gen")), _num(hw.get("gpu_pcie_gen_max")), _num(hw.get("gpu_pcie_width"))
    model = {
        "model": engine.get("model"), "max_context": max_ctx, "state": state, "state_raw": state_raw, "queued": int(queued),
        "progress": (done, total), "phase_text": {"Idle": "Waiting for a request", "Reading": "Reading prompt", "Generating": "Generating a reply",
                                                 "Queued": "Requests are queued", "Error": "Model unavailable"}[state],
        "context": (used, max_ctx),
        "speed": (speed, speed_note, _series(history, "tok_s") if speed_note == "now" or not _series(history, "prefill_tok_s_mean") else
                  _series(history, "tok_s")),
        "gpu": {"cards": _cards(hw, static),
                "name": static.get("gpu_name"), "util": _num(hw.get("gpu_util")), "util_hist": _series(history, "gpu_util"),
                "vram_used": _gib(hw.get("gpu_mem_used")), "vram_total": _gib(hw.get("gpu_mem_total")) or
                (_num(static.get("gpu_vram_total_mib")) / 1024.0 if _num(static.get("gpu_vram_total_mib")) else None),
                "vram_hist": _series(history, "gpu_mem_used", GIB), "temp": _num(hw.get("gpu_temp")), "temp_hist": _series(history, "gpu_temp"),
                "power": _num(hw.get("gpu_power")), "power_limit": _num(hw.get("gpu_power_limit")), "power_hist": _series(history, "gpu_power"),
                "pcie_rx": _num(hw.get("gpu_pcie_rx_mb")), "pcie_hist": _series(history, "gpu_pcie_rx_mb"),
                "pcie_gen": gen if gen is not None else genmax, "pcie_width": width, "count": _num(static.get("gpu_count"))},
        "cpu": {"load": _num(hw.get("cpu_pct") if hw.get("cpu_pct") is not None else hw.get("cpu")), "hist": _series(history, "cpu"),
                "cores": _num(static.get("cores")), "threads": _num(static.get("threads")), "sockets": _num(static.get("sockets")),
                "numa": _num(static.get("numa_nodes"))},
        "disk": {"read": _num(hw.get("disk_read_mb")), "write": _num(hw.get("disk_write_mb")), "hist": _series(history, "disk_read_mb")},
        "ram": (_gib(hw.get("ram_used")), _gib(hw.get("ram_total"))), "ram_hist": _series(history, "ram_used", GIB),
        "experts": {"slots": slots, "total": total_slots, "gb": (cache_mib / 1024.0 if cache_mib is not None else None)},
        # what the request ended as: the server's `finish`.  "error" and "timeout" are the ones that say the reply never came, so they
        # are the red Error rows; a client that walked away ("cancel", "disconnect") is the amber Stopped
        "recent": [{"time": _num(r.get("time")), "state": ("Error" if (r.get("error") or (r.get("finish_reason") or r.get("finish")) in ("error", "timeout"))
                                                           else "Stopped" if (r.get("finish_reason") or r.get("finish"))
                                                           in ("cancel", "cancelled", "abort", "disconnect") else "Done"),
                    "prompt": _num(r.get("prompt_tokens")), "reused": _num(r.get("reused")), "output": _num(r.get("output_tokens")),
                    "prefill": prefill_speed(r), "tps": request_speed(r),
                    "duration": _num(r.get("duration_s"))} for r in recent[:10]],      # newest first, as the server sends them
        "active": active_requests(metrics, live),
        "sampled_at": _num(metrics.get("time")),
    }
    cards = model["gpu"]["cards"]
    if cards:                                  # two or more cards: the tiles say it for the whole computer, the Cards strip says it per card
        g = model["gpu"]
        g["util"] = _most([c["util"] for c in cards])                          # the busiest card
        g["vram_used"] = _total([c["mem_used"] for c in cards])                # all cards together
        g["vram_total"] = _total([c["mem_total"] for c in cards])
        g["vram_hist"] = [g["vram_used"]] * 60 if g["vram_used"] is not None else []     # the server keeps a history of card 0 only: a flat line, not a wrong one
        g["temp"] = _most([c["temp"] for c in cards])                          # the hottest card
        g["power"] = _total([c["power"] for c in cards])
        g["power_limit"] = _total([c["power_limit"] for c in cards])
        g["count"] = float(len(cards))
    return model


# ------------------------------------------------------------------------------------------------ drawing

def solid_spark(values: list, width: int, lo: float | None, hi: float | None) -> Text:
    """The classic console's sparkline (glyphs="solid"): only solid blocks, no dithered shades (ClearType fringes those with colour):
    a thin dim ▄ for the lowest quarter, ▄ for the middle, █ for the top."""
    vals = values[-width:]
    present = [v for v in vals if v is not None]
    if not present:
        return Text(" " * width)
    out = Text(" " * (width - len(vals)))
    low = min(present) if lo is None else lo
    high = max(present) if hi is None else hi
    if high - low < 1e-9:
        high = low + 1.0
    for v in vals:
        if v is None:
            out.append(" ")
            continue
        f = (v - low) / (high - low)
        if f < 0.25:
            out.append("▄", style=SOLID_TRACK)
        elif f < 0.6:
            out.append("▄")
        else:
            out.append("█")
    return out


def spark(values: list[float | None], width: int, glyphs: str, lo: float | None = None, hi: float | None = None) -> str | Text:
    """One row of block characters for the last `width` samples; blanks where a sample is missing; ASCII ramp when `glyphs` is ascii,
    solid blocks (a Text) in the classic console's solid look."""
    if glyphs == "solid":
        return solid_spark(values, width, lo, hi)
    ramp = ".:-=+*#@" if glyphs == "ascii" else "▁▂▃▄▅▆▇█"
    vals = values[-width:]
    present = [v for v in vals if v is not None]
    if not present:
        return " " * width
    low = min(present) if lo is None else lo
    high = max(present) if hi is None else hi
    if high - low < 1e-9:
        high = low + 1.0
    cells = []
    for v in vals:
        if v is None:
            cells.append(" ")
        else:
            cells.append(ramp[min(7, max(0, int((v - low) / (high - low) * 7.999)))])
    return " " * (width - len(cells)) + "".join(cells)


def bar(frac: float | None, width: int, glyphs: str, color: str, empty: str = GREY) -> Text:
    t = Text(no_wrap=True)
    if glyphs == "solid":                       # solid █ and a thin ▄ track: no dithered shades in the classic console
        n = 0 if frac is None else int(round(max(0.0, min(1.0, frac)) * width))
        t.append("█" * n, style=color)
        t.append("▄" * (width - n), style=SOLID_TRACK)
        return t
    full, hollow = ("#", ".") if glyphs == "ascii" else ("█", "░")
    if frac is None:
        t.append(hollow * width, style=GREY)
        return t
    n = int(round(max(0.0, min(1.0, frac)) * width))
    t.append(full * n, style=color)
    t.append(hollow * (width - n), style=empty)
    return t


def short_gpu(name: str | None) -> str:
    """The card's name without the maker prefixes that eat a tile's width ("NVIDIA GeForce RTX 5070" -> "RTX 5070")."""
    if not name:
        return "graphics card"
    for prefix in ("NVIDIA GeForce ", "NVIDIA ", "AMD Radeon ", "AMD ", "Intel(R) ", "Intel "):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def blink(now: float) -> bool:
    """The bright second of the running requests' pulse, dim the next: the monitor repaints every `--interval` (one second by
    default), so an active request's status reads as blinking."""
    return int(now) % 2 == 0


def fmt(v: float | None, digits: int = 0, unit: str = "") -> str:
    if v is None:
        return "--"
    return f"{v:,.{digits}f}{unit}"


def tokens(v: float | None) -> str:
    return "--" if v is None else f"{v:,.0f}"


def k(v: float | None) -> str:
    if v is None:
        return "--"
    return f"{v / 1024:.0f}K" if v >= 1024 else f"{v:.0f}"


def _tile(title: str, big: str, sub: str, series: str, width: int, *, accent: str, box_style, big_style: str = "bold") -> Panel:
    body = Text(no_wrap=True, overflow="crop")
    body.append(big + "\n", style=big_style)
    body.append(sub + "\n", style="dim")
    if isinstance(series, Text):                # the solid look's sparkline carries its own styles
        styled = Text(style=accent)
        styled.append_text(series)
        body.append_text(styled)
    else:
        body.append(series, style=accent)
    return Panel(body, title=Text(title, style="dim"), title_align="left", box=box_style, padding=(0, 1), width=width,
                 border_style="dim")


def _cards_strip(cards: list[dict], glyphs: str, accent: str, ascii_mode: bool) -> Text:
    """One line per card (up to 4): a small load bar, load %, memory, temperature and power; "+ N more cards" beyond that."""
    t = Text(no_wrap=True, overflow="crop")
    t.append("Cards" + chr(10), style=f"bold {accent}")
    shown = cards[:4]
    names = [short_gpu(c.get("name")) for c in shown]
    pad = min(16, max(len(n) for n in names))
    for c, name in zip(shown, names):
        util = c["util"]
        used, total = c["mem_used"], c["mem_total"]
        t.append(f"GPU {c['index'] if c['index'] is not None else '?'}  {name[:pad]:<{pad}}  ")
        t.append_text(bar((util / 100.0) if util is not None else None, 10, glyphs, accent))
        mem = f"{fmt(used, 1)} / {fmt(total, 0)} GB" if used is not None and total is not None else "-- GB"
        temp = f"{fmt(c['temp'])} {'C' if ascii_mode else '°C'}"
        t.append(f"  {fmt(util):>3} %   {mem:<14}  {temp:>6}   {fmt(c['power']):>4} W" + chr(10))
    if len(cards) > 4:
        t.append(f"+ {len(cards) - 4} more cards: press 5" + chr(10), style="dim")
    return t


def render(model: dict, *, width: int, height: int, connection: str, connection_state: str = "connected", frozen: bool = False,
           glyphs: str = "unicode", color: bool = True, theme: str = "dark", connect: bool = False, view_label: str = "Overview",
           now: float | None = None, bullet: str = "●") -> Group:
    now = time.time() if now is None else now
    ascii_mode = glyphs == "ascii"
    solid = glyphs == "solid"                   # the classic Windows console without Cascadia Mono: solid blocks, square boxes, palette colours
    box_style = box.SQUARE if (ascii_mode or solid) else box.ROUNDED
    accent = ("#0c8a60" if theme == "light" else ACCENT) if color else "default"
    blue, amber, red = BLUE, AMBER, RED
    if solid and color:
        accent, blue, amber, red = SOLID_ACCENT, SOLID_BLUE, SOLID_AMBER, SOLID_RED
    plain = (lambda s: "default") if not color else (lambda s: {BLUE: blue, AMBER: amber, RED: red}.get(s, s))
    state = model["state"]
    state_color = plain({"Idle": "default", "Reading": BLUE, "Generating": accent, "Queued": AMBER, "Error": RED}[state])
    age = None if model.get("sampled_at") is None else now - model["sampled_at"]
    marker = ("FROZEN" if frozen else {"connected": "LIVE", "stale": "STALE", "connecting": "CONNECTING", "disconnected": "DISCONNECTED",
                                       "auth_required": "AUTH REQUIRED", "incompatible": "INCOMPATIBLE",
                                       "telemetry_unavailable": "LIVE"}.get(connection_state, "STALE"))
    if marker == "LIVE" and age is not None and age > 6:
        marker = "STALE"
    ctx_used, ctx_max = model["context"]
    done, total = model["progress"]
    pct = int(round(100 * done / total)) if done is not None and total else None

    # ---- heading, tabs, rule
    head = Text(no_wrap=True, overflow="crop")
    if view_label == "Overview":
        view_label = "DGX Spark"
    head.append(f"STRATA  {view_label}   ", style=f"bold {accent}")
    head.append(connection + "   ", style=f"bold {accent}")
    label = f"{model.get('model') or ''} {k(ctx_max) if ctx_max else ''}".strip()
    head.append(label, style="bold")
    right = Text()
    right.append((bullet + " " if not ascii_mode else "* ") + model["phase_text"] + (f" · {pct}%" if pct is not None and state in ("Reading", "Generating") else ""),
                 style=state_color if state != "Idle" else "dim")
    right.append("   ")
    right.append(f"[{marker}]", style=f"bold {accent}" if marker == "LIVE" else "bold yellow")
    pad = max(2, width - len(head.plain) - len(right.plain))
    head.append(" " * pad)
    head.append_text(right)
    tabs = nav.styled(0, min(width, 110), connect, accent=accent, color=color)
    rule = Text(("-" if ascii_mode else "─") * min(width, 110), style="dim")

    # ---- model state bar
    bar_w = max(20, min(width, 110) - 2)
    st = Text(no_wrap=True, overflow="crop")
    st.append("Model state  ", style=f"bold {accent}")
    for n in STATES + (("Error",) if state == "Error" else ()):      # the sample "Error" is not on the line: a real one arrives in red
        txt = f"[{n}]" if n == state else n
        st.append(txt + "  ", style=(state_color if n == state and state != "Idle" else "bold" if n == state else "dim"))
    if done is not None and total:
        right_txt = f"{tokens(done)} / {tokens(total)} tokens · {pct}%"
    elif model["queued"]:
        right_txt = f"{model['queued']} queued"
    else:
        right_txt = ""
    st.append(" " * max(2, min(width, 110) - len(st.plain) - len(right_txt)) + right_txt, style="dim")
    state_bar = bar((done / total) if done is not None and total else None, bar_w, glyphs, state_color if state != "Idle" else accent)

    # ---- tiles
    g = model["gpu"]
    cards = g.get("cards") or []
    ncards = len(cards)
    tile_w = max(18, (min(width, 110) - 3) // 4)
    inner = tile_w - 4
    spd, spd_note, spd_hist = model["speed"]
    util_scale = dict(lo=0, hi=100)
    e = model["experts"]
    ram_used, ram_total = model["ram"]
    tiles1 = [
        _tile("Speed", f"{fmt(spd, 1)} tok/s" if spd is not None else "-- tok/s", spd_note, spark(spd_hist, inner, glyphs), tile_w, accent=accent, box_style=box_style),
        _tile("GPU load", f"{fmt(g['util'])} %", f"busiest of {ncards} cards" if ncards else
              short_gpu(g["name"]) if g["count"] in (None, 1) else f"{short_gpu(g['name'])} +{int(g['count']) - 1}",
              spark(g["util_hist"], inner, glyphs, **util_scale), tile_w, accent=accent, box_style=box_style),
        # the memory the model really shares on a unified-memory machine (GB10 and friends): the system's, with the expert
        # cache's size under it - a discrete card's per-card memory is the Cards strip's job, not this tile's
        _tile("Unified Memory", f"{fmt(ram_used, 1)} / {fmt(ram_total, 0)} GB",
              (f"{int(e['slots']):,} - {e['gb']:.1f} GB" if e["slots"] is not None and e["gb"] is not None else
               f"{int(e['slots']):,} experts" if e["slots"] is not None else ""),
              spark(model["ram_hist"], inner, glyphs, lo=0, hi=ram_total), tile_w, accent=accent, box_style=box_style),
        _tile("GPU temp", f"{fmt(g['temp'])} °C" if not ascii_mode else f"{fmt(g['temp'])} C", "hottest card" if ncards else "",
              spark(g["temp_hist"], inner, glyphs, lo=20, hi=100), tile_w, accent=plain(AMBER) if color else "default", box_style=box_style),
    ]
    c, d = model["cpu"], model["disk"]
    gen = f"Gen{int(g['pcie_gen'])}" if g["pcie_gen"] else "PCIe"
    link = f"{gen} x{int(g['pcie_width'])}" if g["pcie_width"] else gen
    cpu_sub = (f"{int(c['sockets'])} CPUs · {int(c['cores'])} cores" if c.get("sockets") and c["sockets"] > 1 and c["cores"] else
               f"{int(c['cores'])} cores · {int(c['threads'])} threads" if c["cores"] and c["threads"] else
               f"{int(c['threads'])} threads" if c["threads"] else "")
    tiles2 = [
        _tile("Power", f"{fmt(g['power'])} W", f"of {fmt(g['power_limit'])} W limit" if g["power_limit"] else "", spark(g["power_hist"], inner, glyphs, lo=0),
              tile_w, accent=accent, box_style=box_style),
        _tile("PCIe", link, f"to GPU {fmt(g['pcie_rx'])} MB/s" if g["pcie_rx"] is not None else "to GPU --", spark(g["pcie_hist"], inner, glyphs, lo=0), tile_w,
              accent=accent, box_style=box_style),
        _tile("CPU", f"{fmt(c['load'])} %", cpu_sub, spark(c["hist"], inner, glyphs, lo=0, hi=100), tile_w, accent=accent, box_style=box_style),
        _tile("Disk read", f"{fmt(d['read'], 1)} MB/s", f"write {fmt(d['write'], 1)} MB/s" if d["write"] is not None else "write --",
              spark(d["hist"], inner, glyphs, lo=0), tile_w, accent=accent, box_style=box_style),
    ]

    def row(tiles: list[Panel]) -> Table:
        t = Table.grid(padding=(0, 1))
        for _ in tiles:
            t.add_column(width=tile_w)
        t.add_row(*tiles)
        return t

    # ---- bars and requests
    # the two speeds apart: READ t/s is the prompt read, GEN t/s the answer (the engine's own request line prints both).  A narrow
    # terminal gives whole columns up rather than print numbers cut in half: below 100 columns the duration goes, below 90 the reused
    # count too - on a short screen the prompt and the two speeds are what matters.  The third number in each entry is that column's
    # widest cell, so the table's own width is known before it is drawn: TIME to the second, and two spaces between columns
    fixed_cols = [("TIME", "left", 8), ("STATUS", "left", 6)]
    value_cols = [("PROMPT", "right", 6, lambda r: tokens(r["prompt"]))]
    if width >= 90:
        value_cols.append(("REUSED", "right", 6, lambda r: tokens(r["reused"])))
    value_cols += [("OUTPUT", "right", 6, lambda r: tokens(r["output"])), ("READ t/s", "right", 8, lambda r: fmt(r["prefill"], 1)),
                   ("GEN t/s", "right", 7, lambda r: fmt(r["tps"], 1))]
    if width >= 100:
        value_cols.append(("DURATION", "right", 8, lambda r: fmt(r["duration"], 1, " s")))
    table_w = (sum(max(w, len(h)) for h, _, w in fixed_cols) + sum(max(w, len(h)) for h, _, w, _ in value_cols)
               + 2 * (len(fixed_cols) + len(value_cols) - 1))
    # the bars are the shorter half, so they give the room back to the table rather than have it elide a number (never below 14: a bar
    # that short says nothing either).  Eight is what the two-column grid holding the table and the bars spends: 2 spaces on each side
    # of each column
    left_w = max(14, min(max(20, min(width, 110) // 3) if width >= 100 else max(18, min(width, 110) // 4), width - 8 - table_w))
    used_ctx = (ctx_used / ctx_max) if ctx_used is not None and ctx_max else None
    ram_frac = (ram_used / ram_total) if ram_used is not None and ram_total else None

    def bar_caption(long_label: str, short_label: str, value: str, label_style: str = "", value_style: str = "") -> Text:
        """One bar's caption: the full wording when the column has room for label and value together, the short word when the request
        table has taken the space - a caption cut in half ("KV Cache Hit   0 /") tells you less than the short word does."""
        t = Text(no_wrap=True, overflow="crop")
        t.append(long_label if len(long_label) + len(value) <= left_w else short_label + " ", style=label_style)
        t.append(value, style=value_style)
        return t

    def room(short_label: str, value: str) -> bool:
        """Will that caption and value fit with only the short label?  If not, the value part is left out rather than cut in half."""
        return len(short_label) + 1 + len(value) <= left_w

    ctx_value = f"{k(ctx_used)} / {k(ctx_max)}"
    if used_ctx is not None and room("Context", ctx_value + f"  {used_ctx * 100:.0f}%"):
        ctx_value += f"  {used_ctx * 100:.0f}%"
    # the KV cache hit of the newest request the server reported reuse for: the prompt tokens the cache already held
    # over that request's context.  A running request reports no reuse yet (the server only knows it once the prompt
    # has been read), so the newest finished one carries the number; none reports it: "-- / --", never a made-up 0%.
    hit_reused, hit_prompt = next(((r["reused"], r["prompt"]) for r in model["recent"]
                                   if r["reused"] is not None and r["prompt"]), (None, None))
    hit_frac = (hit_reused / hit_prompt) if hit_reused is not None and hit_prompt else None
    hit_value = f"{k(hit_reused)} / {k(hit_prompt)}"
    if hit_frac is not None and room("KV hit", hit_value + f"  {hit_frac * 100:.0f}%"):
        hit_value += f"  {hit_frac * 100:.0f}%"
    lines = [
        bar_caption("Context fill  ", "Context", ctx_value, label_style=f"bold {accent}"),
        bar(used_ctx, left_w, glyphs, accent),
        bar_caption("KV Cache Hit    ", "KV hit", hit_value, value_style="dim"),
        bar(hit_frac, left_w, glyphs, accent),
        bar_caption("System RAM        ", "RAM", f"{fmt(ram_used, 1)} / {fmt(ram_total, 0)} GB", value_style="dim"),
        bar(ram_frac, left_w, glyphs, plain(RED) if ram_frac is not None and ram_frac > 0.9 else accent),
    ]

    req = Table(box=None, padding=(0, 1), show_header=True, header_style="dim", pad_edge=False)
    for title, just, _ in fixed_cols:                # (`head` is the page heading's Text: never rebind it in a loop)
        req.add_column(title, justify=just, no_wrap=True)
    for title, just, _, _ in value_cols:
        req.add_column(title, justify=just, no_wrap=True)
    strip_rows = (1 + min(ncards, 4) + (1 if ncards > 4 else 0) + 1) if ncards else 0
    # ten requests at most, and only as many as the terminal has room for: the ones still running on top, then the finished ones newest
    # first (the order the server sends them).  Four running rows at the most - past that, --parallel leaves no room for the log
    show = max(2, min(10, height - 20 - strip_rows))
    running = (model.get("active") or [])[:4]
    rows = running + list(model["recent"])[:max(0, 10 - len(running))]
    pulse = blink(now)                     # an active request's status flashes: bold colour, then dim
    for r in rows[:show]:
        stamp = time.strftime("%H:%M:%S", time.localtime(r["time"])) if r["time"] else "--"
        if r.get("active"):
            status = Text(r["state"], style=(f"bold {plain(BLUE if r['state'] == 'Read' else accent)}" if pulse else "dim"))
        else:
            status = Text(r["state"], style=plain(accent) if r["state"] == "Done" else plain(RED if r["state"] == "Error" else AMBER))
        vals = [stamp, status, tokens(r["prompt"]), tokens(r["reused"]), tokens(r["output"]), fmt(r["prefill"], 1), fmt(r["tps"], 1)]
        if width >= 100:
            vals.append(fmt(r["duration"], 1, " s"))
        req.add_row(*vals)
    bottom = Table.grid(padding=(0, 2))
    bottom.add_column(width=left_w)
    bottom.add_column()
    # no requests: a line under the heading, not a table row - the row would stretch TIME to its length and squeeze the numbers
    right = Group(Text("Recent requests", style=f"bold {accent}"),
                  req if rows else Text("No requests yet", style="dim"))
    bottom.add_row(Group(*lines), right)

    hint = Text(f"{nav.keys(connect)} views" + "   / filter   Space freeze   r refresh   t theme   b open in browser   ? help   q quit", style="dim")
    parts: list[Any] = [head, tabs, rule, st, state_bar, Text(""), row(tiles1), row(tiles2)]
    if ncards:
        parts += [Text(""), _cards_strip(cards, glyphs, accent, ascii_mode)]
    parts += [bottom, Text(""), hint]
    return Group(*parts)
