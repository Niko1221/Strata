#!/usr/bin/env python3
"""Live Textual dashboard for Strata's local monitoring API.

Expected endpoints:
  GET /metrics  -> Strata JSON metrics
  GET /status   -> current request status
  GET /slots    -> per-slot state

This dashboard only displays telemetry returned by Strata. Missing fields are shown
as unavailable instead of being inferred.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Static

BASE_URL = os.environ.get("STRATA_URL", "http://127.0.0.1:8000").rstrip("/")
POLL_SECONDS = float(os.environ.get("STRATA_MONITOR_INTERVAL", "1.0"))
HTTP_TIMEOUT = float(os.environ.get("STRATA_MONITOR_TIMEOUT", "2.0"))


def fetch_json(path: str) -> Any:
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        headers={"Accept": "application/json", "User-Agent": "strata-tui-monitor/1.0"},
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as response:
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(f"HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


async def fetch_all() -> tuple[dict, dict, list, list[str]]:
    paths = ("/metrics", "/status", "/slots")

    async def one(path: str):
        try:
            result = await asyncio.to_thread(fetch_json, path)
            return path, result, None
        except Exception as exc:
            return path, None, f"{path}: {type(exc).__name__}: {exc}"

    results = await asyncio.gather(*(one(path) for path in paths))
    values: dict[str, Any] = {}
    errors: list[str] = []
    for path, result, error in results:
        if error:
            errors.append(error)
        else:
            values[path] = result

    metrics = values.get("/metrics") or {}
    status = values.get("/status") or {}
    slots = values.get("/slots") or []
    return metrics, status, slots if isinstance(slots, list) else [], errors


def number(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def integer(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def mib(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) / (1024 ** 2):,.1f} GiB"
    except (TypeError, ValueError):
        return "—"


def pct(used: Any, total: Any) -> str:
    try:
        used_f, total_f = float(used), float(total)
        if total_f <= 0:
            return "—"
        return f"{100 * used_f / total_f:.1f}%"
    except (TypeError, ValueError):
        return "—"


def bar(value: Any, width: int = 24) -> str:
    try:
        n = max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return "—" * width
    filled = round(width * n / 100)
    return "█" * filled + "░" * (width - filled)


def sparkline(values: Any, width: int = 48) -> str:
    if not isinstance(values, list):
        return "—"
    nums = []
    for value in values:
        try:
            if value is not None:
                nums.append(float(value))
        except (TypeError, ValueError):
            continue
    if not nums:
        return "—"
    nums = nums[-width:]
    lo, hi = min(nums), max(nums)
    glyphs = "▁▂▃▄▅▆▇█"
    if hi == lo:
        return glyphs[0] * len(nums)
    return "".join(glyphs[round((n - lo) / (hi - lo) * (len(glyphs) - 1))] for n in nums)


def local_time(timestamp: Any) -> str:
    try:
        return datetime.fromtimestamp(float(timestamp)).strftime("%H:%M:%S")
    except (TypeError, ValueError, OSError):
        return "—"


def get_state(metrics: dict, status: dict, errors: list[str]) -> tuple[str, str]:
    if not metrics:
        return "API ERROR", "Cannot read /metrics"
    live = metrics.get("live") or {}
    state = str(live.get("state") or "unknown").strip().upper()
    queued = int(live.get("queued") or status.get("queued") or 0)

    # Trust explicit live state first. /status.phase can describe the last request
    # even when busy=false, so only use it to refine state while busy is true.
    busy = status.get("busy")
    phase = str(status.get("phase") or live.get("phase") or "").lower()
    if state in {"READING", "PREFILL", "PROCESSING"}:
        state = "READING"
    elif state in {"GENERATING", "DECODING"}:
        state = "GENERATING"
    elif state in {"IDLE", "UNKNOWN", "NONE"}:
        if busy is True:
            if any(word in phase for word in ("read", "prefill", "prompt")):
                state = "READING"
            elif phase:
                state = "GENERATING"
            else:
                state = "BUSY"
        elif queued:
            state = "IDLE"
        elif state in {"UNKNOWN", "NONE"}:
            state = "IDLE"
    elif state in {"ERROR", "FAILED"}:
        state = "ERROR"

    sub = []
    if queued:
        sub.append(f"{queued} queued")
    if errors:
        sub.append("partial telemetry unavailable")
    return state, " · ".join(sub) if sub else "Ctrl+C / q to return"


class StrataMonitor(App):
    TITLE = "Strata Live Monitor"
    SUB_TITLE = BASE_URL
    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }
    Header {
        background: $primary;
        color: $text;
    }
    #banner {
        height: 4;
        padding: 0 2;
        margin: 1 1 0 1;
        border: round $success;
        background: $panel;
        content-align: left middle;
    }
    #top {
        height: 13;
        margin: 1 1 0 1;
    }
    .panel {
        border: round $panel-lighten-2;
        background: $panel;
        padding: 1 2;
        width: 1fr;
        height: 1fr;
    }
    #state { border-title-color: $success; }
    #model { border-title-color: $accent; }
    #hardware {
        height: 8;
        margin: 1 1 0 1;
        border: round $panel-lighten-2;
        background: $panel;
        padding: 1 2;
    }
    #history {
        height: 7;
        margin: 1 1 0 1;
        border: round $panel-lighten-2;
        background: $panel;
        padding: 1 2;
    }
    #requests-title {
        height: 1;
        margin: 1 2 0 2;
        color: $text-muted;
        text-style: bold;
    }
    #requests {
        height: 1fr;
        min-height: 7;
        margin: 0 1;
        background: $panel;
    }
    DataTable > .datatable--header {
        background: $primary;
        color: $text;
        text-style: bold;
    }
    DataTable > .datatable--cursor {
        background: $accent 25%;
    }
    #footnote {
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    Footer {
        background: $panel;
    }
    """

    BINDINGS = [
        ("q", "quit", "Return"),
        ("escape", "quit", "Return"),
        ("r", "refresh_now", "Refresh"),
    ]

    def __init__(self):
        super().__init__()
        self.last_ok = 0.0
        self.last_metrics: dict = {}
        self.last_status: dict = {}
        self.last_slots: list = []
        self.last_errors: list[str] = []
        self.polling = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("  STRATA  /  LIVE TELEMETRY     Connecting…", id="banner")
        with Horizontal(id="top"):
            yield Static("REQUEST  •  LIVE", id="state", classes="panel")
            yield Static("MODEL  •  CONTEXT", id="model", classes="panel")
        yield Static("HARDWARE  •  APPLE SILICON", id="hardware", classes="panel")
        yield Static("THROUGHPUT  •  RECENT HISTORY", id="history", classes="panel")
        yield Static("RECENT REQUESTS", id="requests-title")
        yield DataTable(id="requests")
        yield Static("r refresh · q / Esc return to switcher", id="footnote")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#history", Static).border_title = "THROUGHPUT  •  RECENT HISTORY"
        table = self.query_one("#requests", DataTable)
        table.add_columns("Time", "Duration", "Finish", "Prompt", "Output", "Decode tk/s")
        table.cursor_type = "row"
        self.set_interval(POLL_SECONDS, self.refresh_data)
        self.run_worker(self.refresh_data(), group="poll", exclusive=True)

    async def action_refresh_now(self) -> None:
        await self.refresh_data()

    async def refresh_data(self) -> None:
        if self.polling:
            return
        self.polling = True
        try:
            metrics, status, slots, errors = await fetch_all()
            if metrics:
                self.last_ok = time.time()
                self.last_metrics = metrics
                self.last_status = status
                self.last_slots = slots
            self.last_errors = errors
            self.render_dashboard()
        finally:
            self.polling = False

    def render_dashboard(self) -> None:
        m = self.last_metrics
        s = self.last_status
        if not m:
            details = "\n".join(self.last_errors) or "Waiting for /metrics…"
            self.query_one("#banner", Static).update(
                f"[bold red]!  STRATA  /  API UNAVAILABLE[/]  [dim]{details}[/]"
            )
            return

        live = m.get("live") or {}
        engine = m.get("engine") or {}
        hw = m.get("hardware") or {}
        static = m.get("hardware_static") or {}
        totals = m.get("totals") or {}
        state, note = get_state(m, s, self.last_errors)
        state_color = {
            "IDLE": "green", "GENERATING": "bright_green", "READING": "yellow",
            "QUEUED": "yellow", "ERROR": "red", "API ERROR": "red",
            "BUSY": "yellow", "UNKNOWN": "yellow",
        }.get(state, "cyan")

        active = state in {"READING", "GENERATING", "BUSY"}
        queued = live.get("queued", s.get("queued", 0)) or 0
        rate = live.get("tok_s") if active else None
        if rate is None and active:
            rate = hw.get("tok_s")
        last_req = (m.get("requests") or [{}])[0]
        speed_label = "Live decode" if active and rate is not None else "Last decode"
        speed = rate if rate is not None else last_req.get("decode_tok_s")
        elapsed = live.get("elapsed_s")
        generated = live.get("generated") if active else last_req.get("output_tokens")
        prefill = live.get("prefill_tok_s_mean") if active else (
            (last_req.get("prompt_tokens", 0) / (last_req.get("prompt_ms", 0) / 1000))
            if last_req.get("prompt_ms") else None
        )

        state_symbol = {
            "IDLE": "●", "GENERATING": "▶", "READING": "◌", "ERROR": "!",
            "API ERROR": "!", "BUSY": "◌", "UNKNOWN": "?"
        }.get(state, "●")
        self.query_one("#banner", Static).update(
            f"[bold {state_color}]{state_symbol}  STRATA  /  {state}[/]  "
            f"[dim]{note}[/]\n"
            f"[bold]{engine.get('model', 'Model unavailable')}[/]  "
            f"[dim]•  Updated {datetime.now().strftime('%H:%M:%S')}  •  Refresh {POLL_SECONDS:g}s[/]"
        )

        progress = ""
        pr, pt = live.get("prompt_read"), live.get("prompt_total")
        if active and pr is not None and pt:
            progress = f"\nPrompt read: {integer(pr)} / {integer(pt)} ({100 * float(pr) / float(pt):.1f}%)"
        ctx_line = ""
        if active:
            prompt_count = live.get("prompt_tokens")
            if prompt_count is None:
                prompt_count = live.get("prompt_total")
            gen_count = live.get("generated") or 0
            try:
                used = int(prompt_count or 0) + int(gen_count)
                limit = int(engine.get("max_context") or engine.get("context") or 0)
                if limit:
                    cp = min(100, 100 * used / limit)
                    ctx_line = f"\nActive context: {used:,} / {limit:,} ({cp:.1f}%)\n{bar(cp)}"
            except (ValueError, TypeError):
                pass

        self.query_one("#state", Static).update(
            f"[bold {state_color}]{state_symbol}  {state}[/]   "
            f"[bold]QUEUE[/] {integer(queued)}\n"
            f"\n[bold cyan]{number(speed)}[/] [dim]tk/s · {speed_label.lower()}[/]\n"
            f"Output  {integer(generated)} tokens   ·   Elapsed  {number(elapsed, 1)} s\n"
            f"Prefill  {number(prefill)} tk/s{progress}"
        )
        self.query_one("#state", Static).border_title = "REQUEST  •  LIVE"

        ctx = integer(engine.get("max_context") or engine.get("context"))
        model_text = (
            f"[bold]Model[/]  {engine.get('model', '—')}\n"
            f"Engine  {engine.get('version', engine.get('engine', '—'))}\n"
            f"Context limit  {ctx} tokens\n"
            f"KV cache  {engine.get('kv', '—')}  ·  Speculative MTP {engine.get('mtp_max', '—')}\n"
            f"Expert cache  {engine.get('expert_cache_mib', '—')} MiB\n"
            f"{ctx_line or 'Context usage: shown during active requests only'}"
        )
        self.query_one("#model", Static).update(model_text)
        self.query_one("#model", Static).border_title = "MODEL  •  CONTEXT"

        gpu_used, gpu_total = hw.get("gpu_mem_used"), hw.get("gpu_mem_total")
        ram_used, ram_total = hw.get("ram_used"), hw.get("ram_total")
        hardware_text = (
            f"[bold]{static.get('gpu_name', 'GPU')}[/]  util {number(hw.get('gpu_util'), 0)}%  ·  "
            f"memory {mib(gpu_used)} / {mib(gpu_total)} ({pct(gpu_used, gpu_total)})\n"
            f"CPU {number(hw.get('cpu'), 1)}%  ·  RAM {mib(ram_used)} / {mib(ram_total)} ({pct(ram_used, ram_total)})\n"
            f"Disk read {number(hw.get('disk_read_mb'), 2)} MB  ·  write {number(hw.get('disk_write_mb'), 2)} MB\n"
            f"GPU temp/power: not reported by this API"
        )
        self.query_one("#hardware", Static).update(hardware_text)
        self.query_one("#hardware", Static).border_title = "HARDWARE  •  APPLE SILICON"

        history = m.get("history") or {}
        self.query_one("#history", Static).update(
            f"[bold]Recent samples  [dim](latest →)[/][/]\n"
            f"Decode tk/s  {sparkline(history.get('tok_s'))}  current {number(hw.get('tok_s'), 1)}\n"
            f"GPU util     {sparkline(history.get('gpu_util'))}  current {number(hw.get('gpu_util'), 0)}%\n"
            f"CPU          {sparkline(history.get('cpu'))}  current {number(hw.get('cpu'), 1)}%\n"
            f"RAM          {sparkline(history.get('ram_used'))}  current {pct(ram_used, ram_total)}"
        )

        self.query_one("#history", Static).border_title = "THROUGHPUT  •  RECENT HISTORY"
        table = self.query_one("#requests", DataTable)
        table.clear()
        for req in (m.get("requests") or [])[:8]:
            table.add_row(
                local_time(req.get("time")),
                f"{number(req.get('duration_s'))} s",
                str(req.get("finish") or "—"),
                integer(req.get("prompt_tokens")),
                integer(req.get("output_tokens")),
                number(req.get("decode_tok_s")),
            )
        if not (m.get("requests") or []):
            table.add_row("—", "—", "No history", "—", "—", "—")

        stamp = datetime.fromtimestamp(self.last_ok).strftime("%H:%M:%S") if self.last_ok else "never"
        footer_note = f"● Connected  ·  Last metrics {stamp}"
        if self.last_errors:
            footer_note += " · Partial endpoint errors: " + "; ".join(self.last_errors)
        self.query_one("#footnote", Static).update(footer_note)


if __name__ == "__main__":
    StrataMonitor().run()
