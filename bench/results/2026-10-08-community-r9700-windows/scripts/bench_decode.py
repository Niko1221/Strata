#!/usr/bin/env python3
"""Matched decode probe against an otherwise idle local Strata server.

Measures decode throughput the engine itself reports, not the HTTP wall clock, and
never lets a cancelled/failed request contribute to a statistic.

Unit of measurement
    Everything comes from the engine's own stderr summary lines, which are written
    to the engine log (serve/server.py points the engine's stderr at that file):
      * "1/token" cost  ->  `strata serve: ... generated in <ms> ms (<tok/s>)`
        This includes every engine-side cost of the request: per-window GPU work,
        expert tier traffic, host planning and the MTP verify/draft overhead.
      * per-window split -> `strata decode timing: ...` (env STRATA_DECODE_TIMING=1)
        verify (GPU-reach wait + per-layer host [plan actq jobs CPU] + stage)
        + commit/emit + draft, plus per-layer-window CPU experts / VRAM hits / PCIe.
      * per-GPU-stage split -> `strata decode GPU stages (ms/window): ...`
        (env STRATA_VERIFY_PROFILE=1), split into GDN layers (k=0) and QSA (k=1).

    CAVEAT, read before comparing numbers across arms: profiling is not free.
    src/core/verify.cpp disables the shared-stream fork/join while the stage
    profiler is on (`sh_fork = sh_stream_env && !prof_on_ && ...`, verify.cpp:907
    and :1078), and each stamp point adds a tiny kernel. So a run with
    STRATA_VERIFY_PROFILE=1 is a *different execution* from a production run.
    Use it for the RELATIVE attribution of a window; take absolute tok/s from a
    profile-off arm.

Repeat, median and range policy
    >= 3 repeats are required per arm; the script exits non-zero rather than
    report a median from fewer. Per arm the report gives median, min, max and the
    range (极差) over the *usable* repeats only.
    Repeats of an identical prompt are deliberately used: repeat 1 sees a fresh
    prompt and a cold expert cache, repeats 2..N reuse the conversation cache and
    a warm expert cache. Both are reported; the warm ones are the steady state.

Cancel/exclusion rule
    Any request that was cancelled, aborted, errored, truncated, or whose summary
    line is missing or unparseable is recorded with usable=false and a
    reject_reason, and is excluded from every statistic.

Confounders to watch when comparing arms
    context length (prompt+generated), expert-cache hit rate, draft accept rate,
    warm vs cold expert cache, and whether profiling was on.
"""
from __future__ import annotations

import argparse
import atexit
import contextlib
import json
import os
import random
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ----------------------------------------------------------------------------- parsing

RE_PROMPT = re.compile(
    r"strata serve: prompt (?P<prompt>\d+) tokens = (?P<reused>\d+) reused \+ (?P<read>\d+) read "
    r"in (?P<pms>\d+) ms \((?P<pts>[0-9.]+) tok/s\), "
    r"(?P<gen>\d+) generated in (?P<dms>\d+) ms \((?P<dts>[0-9.]+) tok/s\), "
    r"drafts accepted (?P<dacc>\d+) of (?P<dtot>\d+), (?P<ckpt>\d+) checkpoints"
)
RE_HIT = re.compile(
    r"strata serve: decode expert cache hit rate: (?P<pct>[0-9.]+)% "
    r"\((?P<hits>\d+) hits / (?P<lookups>\d+) lookups\)"
)
RE_RAM = re.compile(
    r"strata serve: resident RAM: (?P<gib>[0-9.]+) GiB of experts in RAM, "
    r"(?P<exch>\d+) exchanged with the VRAM tier, (?P<blobs>\d+) blob reads from the file"
)
RE_SUFFIX = re.compile(
    r"strata serve: suffix drafts: (?P<win>\d+) windows, (?P<acc>\d+) of (?P<tot>\d+) drafts accepted"
)
RE_DECODE_TIMING = re.compile(
    r"strata decode timing: (?P<windows>\d+) windows, avg T (?P<avg_t>[0-9.]+), "
    r"(?P<tpw>[0-9.]+) tokens/window, (?P<msw>[0-9.]+) ms/window = verify (?P<verify>[0-9.]+) "
    r"\(GPU-reach wait (?P<wait>[0-9.]+) \+ per-layer host (?P<host>[0-9.]+) "
    r"\[plan (?P<plan>[0-9.]+) actq (?P<actq>[0-9.]+) jobs (?P<jobs>[0-9.]+) CPU (?P<cpu>[0-9.]+)\] "
    r"\+ stage (?P<stage>[0-9.]+)\) \+ commit/emit (?P<commit>[0-9.]+) \+ draft (?P<draft>[0-9.]+); "
    r"per layer-window: CPU experts (?P<ce>[0-9.]+) \((?P<ce_entries>[0-9.]+) entries\), "
    r"VRAM hits (?P<vh>[0-9.]+), PCIe (?P<pcie>[0-9.]+)"
)
# "strata decode GPU stages (ms/window): GDN layers: name v name v ... | QSA layers: ... | total X ms/window over N windows"
RE_STAGES_HEAD = re.compile(
    r"strata decode GPU stages \(ms/window\):\s*(?P<body>.*?)"
    r"\s*\|\s*total (?P<total>[0-9.]+) ms/window over (?P<windows>\d+) windows"
)
# The trailing "| total ..." must come off before the body is split on "|", or the
# non-greedy body swallows it (the last "|" in the line wins).
RE_STAGES_TAIL = re.compile(
    r"\s*\|\s*total (?P<total>[0-9.]+) ms/window over (?P<windows>\d+) windows\s*$"
)
RE_STAGE_TOKEN = re.compile(r"(?P<name>[A-Za-z][A-Za-z0-9()+\-]*|\(gap\))\s+(?P<ms>[0-9.]+)")

REJECT_WORDS = ("cancel", "cancelled", "canceled", "abort", "aborted", "error", "failed", "fail ")


@dataclass
class StageGroup:
    """One layer group of the per-GPU-stage profile."""

    name: str
    stages: dict[str, float] = field(default_factory=dict)


@dataclass
class DecodeTiming:
    """`strata decode timing:` - the split of one decode window."""

    windows: int
    avg_t: float
    tokens_per_window: float
    ms_per_window: float
    verify_ms: float
    wait_ms: float
    host_ms: float
    plan_ms: float
    actq_ms: float
    jobs_ms: float
    cpu_ms: float
    stage_ms: float
    commit_ms: float
    draft_ms: float
    cpu_experts_per_layer: float
    entries_per_layer: float
    vram_hits_per_layer: float
    pcie_per_layer: float


@dataclass
class StageProfile:
    total_ms_per_window: float
    windows: int
    groups: list[StageGroup]


@dataclass
class RequestRecord:
    label: str
    repeat: int
    line_no: int
    byte_offset: int
    raw_line: str
    prompt_tokens: int
    reused: int
    read: int
    prefill_ms: int
    prefill_tps: float
    generated: int
    decode_ms: int
    decode_tps: float
    drafts_accepted: int
    drafts_total: int
    checkpoints: int
    usable: bool = True
    reject_reason: str | None = None
    hit_pct: float | None = None
    hit_hits: int | None = None
    hit_lookups: int | None = None
    hit_rate_attribution: str = "none"
    ram_gib: float | None = None
    exchanged: int | None = None
    blob_reads: int | None = None
    suffix_windows: int | None = None
    suffix_accepted: int | None = None
    suffix_total: int | None = None
    decode_timing: DecodeTiming | None = None
    stage_profile: StageProfile | None = None
    warm_cold: str = "n/a"

    @property
    def accept_pct(self) -> float | None:
        return 100.0 * self.drafts_accepted / self.drafts_total if self.drafts_total else None

    @property
    def context(self) -> int:
        return self.prompt_tokens + self.generated

    @property
    def ms_per_token(self) -> float | None:
        return self.decode_ms / self.generated if self.generated else None


def _reject_reason(raw: str, m: re.Match[str] | None) -> str | None:
    low = raw.lower()
    for w in REJECT_WORDS:
        if w in low:
            return f"line mentions {w!r}"
    if m is None:
        return "summary line did not parse"
    if int(m.group("gen")) <= 0:
        return "generated <= 0"
    if int(m.group("dms")) <= 0:
        return "decode_ms <= 0"
    return None


# The stage names as src/core/verify.cpp:1178-1182 defines them.  Some contain spaces
# ("hc-read0", "hc0 norm", "VRAM hits") and some are empty placeholders, so the body is
# parsed by matching this list in order rather than by splitting on whitespace.
STAGE_NAMES = [
    "hc-read0", "q8+qkv/q-idx gemv", "conv", "ab", "z", "rec", "q8+kv-idx",
    "k/v+norm-rope", "kv+idx append", "q+q-idx", "scores+topk", "kv-resolve",
    "attention", "gate", "out-proj", "hc-read1+router", "shared+quant",
    "waitA", "VRAM hits", "waitB", "PCIe grp", "waitCPU", "copy+combine",
    "(gap)", "head", "hc0 norm", "hc0 down", "hc0 up",
]
RE_STAGE_ONE = re.compile(
    "(" + "|".join(re.escape(n) for n in sorted(STAGE_NAMES, key=len, reverse=True)) + r")\s+([0-9.]+)")


def parse_stages(body: str) -> StageProfile | None:
    """Split the stage profile body into its per-group stage dictionaries."""
    groups: list[StageGroup] = []
    total = 0.0
    windows = 0
    tail = RE_STAGES_TAIL.search(body)
    if tail:
        total = float(tail.group("total"))
        windows = int(tail.group("windows"))
        body = body[: tail.start()]
    chunks = re.split(r"(GDN layers:|QSA layers:)", body)
    cur: StageGroup | None = None
    for chunk in chunks:
        if chunk == "GDN layers:":
            cur = StageGroup("GDN"); groups.append(cur); continue
        if chunk == "QSA layers:":
            cur = StageGroup("QSA"); groups.append(cur); continue
        if cur is None or not chunk.strip():
            continue
        for tok in chunk.split("|"):
            for m in RE_STAGE_ONE.finditer(tok):
                cur.stages[m.group(1)] = float(m.group(2))
    if not groups:
        return None
    return StageProfile(total_ms_per_window=total, windows=windows, groups=groups)


def parse_log(text: str, base_offset: int = 0) -> list[RequestRecord]:
    """Parse engine stderr into one record per request summary line.

    Detail lines (hit rate, RAM tier, suffix drafts, profilers) attach to the most
    recent summary line.  Lines that are not completions of a request are recorded
    with usable=false when they look like a completion, and ignored otherwise.
    """
    records: list[RequestRecord] = []
    cur: RequestRecord | None = None
    pending_offset = base_offset
    # DETAIL LINES THAT PRECEDE THEIR OWN SUMMARY LINE.
    # The engine writes `strata decode timing:` and `strata decode GPU stages` for a request
    # BEFORE it writes that request's `strata serve: prompt ...` line (tools/opt/logs/profile.log
    # lines 50-53 is the reference order).  Attaching details only to `cur` therefore DROPPED
    # every one of them - the window columns (wait/pool/host/commit), avg T, tokens/window, the
    # CPU-expert and VRAM-hit counts - which is why a baseline could legitimately say "no decode
    # timing lines captured" while the lines were sitting in the log.  Hold them for the record
    # that is about to be created.
    early: dict = {}

    def apply_early(rec: RequestRecord) -> None:
        for key, val in early.items():
            setattr(rec, key, val)
        early.clear()

    for i, ln in enumerate(text.splitlines(), 1):
        m = RE_PROMPT.search(ln)
        if m:
            cur = RequestRecord(
                label="", repeat=0, line_no=i, byte_offset=pending_offset, raw_line=ln.strip(),
                prompt_tokens=int(m.group("prompt")), reused=int(m.group("reused")),
                read=int(m.group("read")), prefill_ms=int(m.group("pms")),
                prefill_tps=float(m.group("pts")), generated=int(m.group("gen")),
                decode_ms=int(m.group("dms")), decode_tps=float(m.group("dts")),
                drafts_accepted=int(m.group("dacc")), drafts_total=int(m.group("dtot")),
                checkpoints=int(m.group("ckpt")),
            )
            reason = _reject_reason(ln, m)
            if reason:
                cur.usable, cur.reject_reason = False, reason
            apply_early(cur)
            records.append(cur)
            continue
        hit = RE_HIT.search(ln)
        if hit and cur is not None and cur.hit_pct is None:
            cur.hit_pct = float(hit.group("pct"))
            cur.hit_hits = int(hit.group("hits"))
            cur.hit_lookups = int(hit.group("lookups"))
            cur.hit_rate_attribution = "adjacent"
            continue
        ram = RE_RAM.search(ln)
        if ram and cur is not None:
            cur.ram_gib = float(ram.group("gib"))
            cur.exchanged = int(ram.group("exch"))
            cur.blob_reads = int(ram.group("blobs"))
            continue
        sfx = RE_SUFFIX.search(ln)
        if sfx and cur is not None and cur.suffix_windows is None:
            cur.suffix_windows = int(sfx.group("win"))
            cur.suffix_accepted = int(sfx.group("acc"))
            cur.suffix_total = int(sfx.group("tot"))
            continue
        dt = RE_DECODE_TIMING.search(ln)
        if dt:
            rec = DecodeTiming(
                windows=int(dt.group("windows")), avg_t=float(dt.group("avg_t")),
                tokens_per_window=float(dt.group("tpw")), ms_per_window=float(dt.group("msw")),
                verify_ms=float(dt.group("verify")), wait_ms=float(dt.group("wait")),
                host_ms=float(dt.group("host")), plan_ms=float(dt.group("plan")),
                actq_ms=float(dt.group("actq")), jobs_ms=float(dt.group("jobs")),
                cpu_ms=float(dt.group("cpu")), stage_ms=float(dt.group("stage")),
                commit_ms=float(dt.group("commit")), draft_ms=float(dt.group("draft")),
                cpu_experts_per_layer=float(dt.group("ce")),
                entries_per_layer=float(dt.group("ce_entries")),
                vram_hits_per_layer=float(dt.group("vh")),
                pcie_per_layer=float(dt.group("pcie")),
            )
            if cur is None or cur.decode_timing is not None:
                early["decode_timing"] = rec      # this line belongs to the NEXT summary line
            else:
                cur.decode_timing = rec
            continue
        if "strata decode GPU stages" in ln:
            body = ln.split("(ms/window):", 1)[-1]
            sp = parse_stages(body)
            if sp:
                if cur is None or cur.stage_profile is not None:
                    early["stage_profile"] = sp   # same: precedes its summary line
                else:
                    cur.stage_profile = sp
            continue
    return records


# ----------------------------------------------------------------------------- stats

def median_range(values: list[float]) -> dict:
    vals = [v for v in values if v is not None and v == v]
    if not vals:
        return {"n": 0, "median": None, "min": None, "max": None, "range": None}
    return {
        "n": len(vals),
        "median": round(statistics.median(vals), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "range": round(max(vals) - min(vals), 4),
    }


# ----------------------------------------------------------------------------- prompt

FILLER_WORDS = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota",
    "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau",
]


def synth_prompt(target_tokens: int, seed: int = 1234) -> str:
    """Deterministic synthetic prompt of roughly `target_tokens` tokens.

    Built from a seeded LCG so every arm and repeat sends byte-identical text.
    The achieved token count is whatever the engine reports; the requested count is
    only a target, because the harness cannot know the tokenizer's segmentation.
    """
    rng = seed
    parts: list[str] = []
    approx = 0
    while approx < target_tokens:
        rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
        w1 = FILLER_WORDS[rng % len(FILLER_WORDS)]
        rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
        w2 = FILLER_WORDS[rng % len(FILLER_WORDS)]
        rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
        n = rng % 10000
        parts.append(f"{w1}-{w2}-{n:04d}")
        approx += 2
    text = " ".join(parts)
    return ("Read the following numeric record and reply with a single short sentence.\n"
            "RECORD: " + text + "\nReply now.")


# ----------------------------------------------------------------------------- driver

def post_completion(base_url: str, model: str, content: str, max_tokens: int, seed: int,
                    timeout: float, api_key: str | None) -> dict:
    body = {
        "model": model, "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens, "temperature": 0, "top_k": 1, "top_p": 1, "min_p": 0,
        "seed": seed, "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def wait_for_log_records(log_path: Path, offset: int, want: int, deadline_s: float = 5.0) -> list[RequestRecord]:
    """Re-read the log until `want` new summary records are visible (stderr may lag the HTTP reply)."""
    end = time.monotonic() + deadline_s
    while True:
        recs = read_new_records(log_path, offset)
        if len(recs) >= want or time.monotonic() >= end:
            return recs
        time.sleep(0.05)


def read_new_records(log_path: Path, offset: int) -> list[RequestRecord]:
    if not log_path.exists():
        return []
    with log_path.open("rb") as fh:
        fh.seek(offset)
        text = fh.read().decode("utf-8", errors="replace")
    return parse_log(text, base_offset=offset)


def port_in_use(base_url: str) -> bool:
    """True if something is LISTENING on the URL's host:port, whatever it is.

    `wait_for_server` above answers a different question ("is a usable Strata server
    reachable") and answering it is not enough to decide whether to start one: a server
    someone else started is perfectly reachable.  This is the raw socket check that
    tells "free" from "occupied", so `--start-server` can refuse rather than attach.
    """
    u = urllib.parse.urlsplit(base_url)
    host = u.hostname or "127.0.0.1"
    port = u.port or (443 if u.scheme == "https" else 80)
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            return s.connect_ex((host, port)) == 0
    except OSError:
        return False


# ----------------------------------------------------------------------------- GPU window lock

GPU_LOCK_PATH = Path("tools/opt/gpu-window.json")


def _pid_alive(pid: int) -> bool:
    """Is `pid` still running?  SAFE ON WINDOWS: os.kill(pid, 0) is NOT - on Windows any
    signal other than CTRL_C/CTRL_BREAK unconditionally TerminateProcess()es the target."""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            h = k32.OpenProcess(0x00100000, False, pid)   # SYNCHRONIZE
            if not h:
                return False
            k32.CloseHandle(h)
            return True
        except Exception:  # noqa: BLE001 - an unreadable pid is treated as dead
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def gpu_lock_acquire(label: str, ttl_s: float, path: Path = GPU_LOCK_PATH) -> dict | None:
    """Claim the one-GPU window, or refuse.  Returns the claim dict to pass to release.

    THE POINT IS THAT "the GPU is yours" IS AN INTENT, NOT A FACT.  Two members each
    believing they hold the window load two 25 GB engines onto one 32 GB card and both
    runs are ruined.  A file with an owner, a pid and a TTL turns the intent into
    something a run can CHECK before it spends a window.
    """
    now = time.time()
    if path.exists():
        try:
            held = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            held = {}
        started = float(held.get("started_epoch") or 0)
        age = now - started
        alive = _pid_alive(int(held.get("pid") or 0))
        if alive and age <= ttl_s:
            print(f"[gpu-lock] REFUSED: {path} is held by {held.get('owner')} "
                  f"(pid {held.get('pid')}, label {held.get('label')!r}, {age / 60:.1f} min old).\n"
                  f"[gpu-lock] Two engines cannot share this card; wait for that run to end "
                  f"(or delete the file if you know it is stale).", flush=True)
            return None
        print(f"[gpu-lock] taking over a stale claim ({held.get('owner')} pid {held.get('pid')}, "
              f"alive={alive}, {age / 60:.1f} min old)", flush=True)
    claim = {"owner": os.environ.get("STRATA_GPU_OWNER") or f"pid{os.getpid()}",
             "host": socket.gethostname(), "pid": os.getpid(), "label": label,
             "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "started_epoch": now, "ttl_s": ttl_s}
    path.parent.mkdir(parents=True, exist_ok=True)
    # ATOMIC CREATE.  A read-then-write would let two members that check at the same
    # instant both believe they hold the card - the exact collision the lock exists to stop.
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"[gpu-lock] REFUSED: {path} appeared while we were claiming it (another member won "
              f"the race); retry later.", flush=True)
        return None
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(claim, indent=2) + "\n")
    print(f"[gpu-lock] claimed {path} for {claim['owner']} ({label})", flush=True)
    return claim


def gpu_lock_acquire_wait(label: str, ttl_s: float, wait_s: float,
                          path: Path = GPU_LOCK_PATH) -> dict | None:
    """`gpu_lock_acquire`, retried until the card frees or `wait_s` runs out.

    WITH BOTH SIDES HOLDING THE LOCK, WAITING IS SAFE AND AUTOMATIC: whoever gets it first
    runs and the other is refused, so a campaign no longer needs a human to hand the GPU
    over.  The poll interval is jittered because two processes started together would
    otherwise wake together forever.
    """
    deadline = time.time() + max(0.0, wait_s)
    while True:
        claim = gpu_lock_acquire(label, ttl_s, path)
        if claim is not None:
            return claim
        if time.time() >= deadline:
            print(f"[gpu-lock] gave up after waiting {wait_s:.0f}s", flush=True)
            return None
        time.sleep(30.0 + random.random() * 20.0)


def gpu_lock_release(claim: dict | None, path: Path = GPU_LOCK_PATH) -> None:
    """Release only OUR claim - never someone else's."""
    if not claim:
        return
    try:
        held = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        held = {}
    if int(held.get("pid") or 0) == int(claim.get("pid") or -1) and \
       held.get("started_epoch") == claim.get("started_epoch"):
        path.unlink(missing_ok=True)
        print(f"[gpu-lock] released {path}", flush=True)


def engine_processes(exe_name: str) -> int:
    """How many engines named `exe_name` are running.

    THE PORT CHECK ALONE IS NOT ENOUGH, AND THAT COST A WINDOW.  `serve\\server.py` binds
    its HTTP port only once the engine reports ready, so an engine that is mid-load - or
    one whose server has been killed and left the engine orphaned - holds ~25 GB of VRAM
    with NO listener on the port.  A run that only probes the port starts a second engine
    and two 25 GB engines do not fit in 32 GB.  Count the processes as well.
    """
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {exe_name}", "/NH"],
                                 capture_output=True, text=True, timeout=30).stdout or ""
        else:
            out = subprocess.run(["pgrep", "-c", "-f", exe_name],
                                 capture_output=True, text=True, timeout=30).stdout or ""
            return int(out.strip() or 0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0
    return out.lower().count(exe_name.lower())


def kill_tree(proc: subprocess.Popen) -> None:
    """Terminate a started server AND EVERYTHING IT STARTED.

    `proc.terminate()` kills only the shell that `shell=True` created.  The python
    `serve/server.py` under it, and the 23 GB `strata.exe` under THAT, survive as
    orphans holding the whole card - which happened three times in this session and
    each time cost a window to a member who then had to prove the leftover was not a
    live run.  Kill the tree.
    """
    if proc is None:
        return
    pid = proc.pid
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                           capture_output=True, text=True, timeout=60)
        else:
            proc.terminate()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(Exception):
            proc.kill()


def wait_for_server(base_url: str, deadline_s: float) -> bool:
    """Wait until the server is actually able to serve, not merely listening.

    serve/server.py binds the HTTP port while the engine is still loading the model,
    so a bare TCP/200 probe passes far too early (and a request then fails, or the
    process gets terminated mid-load).  Require /v1/models to return a model list.
    """
    end = time.monotonic() + deadline_s
    last = ""
    while time.monotonic() < end:
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=10) as r:
                data = json.load(r)
            if r.status == 200 and (data.get("data") or []):
                return True
            last = "the model list is empty (still loading)"
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = str(exc)
        time.sleep(3.0)
    print(f"[bench] readiness probe gave up: {last}", flush=True)
    return False


# ----------------------------------------------------------------------------- config copies

# Config keys that are read from the JSON by serve/server.py itself (cfg[...]) and must
# never be appended to the engine's argument list.
TOP_LEVEL_KEYS = {
    "exe", "cwd", "tokenizer", "model_name", "log", "lib_dirs", "port", "backend",
    "args", "env", "lazy_load", "min_free_vram_mib", "idle_unload_s", "before_load",
}


def emit_config_copy(src: Path, dst: Path, sets: list[str], allow_multi: bool) -> list[str]:
    """Write a copy of `src` at `dst` with `key=value` arg overrides applied.

    Refuses to touch the repository's canonical config, and refuses a diff of more
    than one variable unless `allow_multi`, so A/B arms stay single-variable.
    """
    if dst.resolve() == src.resolve():
        raise SystemExit(f"refusing to overwrite the source config in place: {dst}")
    if src.name == "strata-iq3_xxs.json" and dst.parent.resolve() == src.parent.resolve():
        raise SystemExit("refusing to write a modified copy next to the canonical config; use tools/opt/")
    cfg = json.loads(src.read_text(encoding="utf-8-sig"))
    args: list[str] = list(cfg.get("args") or [])
    env: dict = dict(cfg.get("env") or {})
    changed: list[str] = []
    for s in sets:
        key, _, val = s.partition("=")
        if not _:
            raise SystemExit(f"--set expects KEY=VALUE, got {s!r}")
        if key.startswith("env:"):
            name = key[4:]
            if env.get(name) != val:
                changed.append(f"env {name}")
            env[name] = val
            continue
        if key in TOP_LEVEL_KEYS:
            # a top-level config key (log, port, model_name, ...), not an engine argument
            if cfg.get(key) != val:
                changed.append(f"{key} {cfg.get(key)!r} -> {val!r}")
            cfg[key] = val
            continue
        if key in args:
            idx = args.index(key) + 1
            if idx < len(args) and not args[idx].startswith("--"):
                if args[idx] != val:
                    changed.append(f"{key} {args[idx]} -> {val}")
                args[idx] = val
            else:
                changed.append(f"{key} (absent) -> {val}")
                args.insert(idx, val)
        else:
            changed.append(f"{key} (absent) -> {val}")
            args.append(key)
            if val != "":
                args.append(val)
    if len(changed) > 1 and not allow_multi:
        raise SystemExit(f"refusing a multi-variable diff ({len(changed)}: {changed}); pass --allow-multi-var")
    cfg["args"] = args
    # EVERY ARM NEEDS ITS OWN ENGINE LOG.  Inheriting the canonical config's `log` makes the
    # arm append to the shared log while the harness watches a different file, so the arm
    # silently collects nothing (and pollutes the shared log).  If --set did not name `log`,
    # derive one next to the arm's own config copy.
    if not any(s.partition("=")[0] == "log" for s in sets):
        derived = (Path("tools/opt/logs") / (dst.stem.replace("cfg-", "") + ".log")).as_posix()
        cfg["log"] = str((Path.cwd() / derived).resolve()) if not Path(derived).is_absolute() else derived
        changed.append(f"log -> {cfg['log']} (per-arm, derived)")
    cfg["env"] = env
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(cfg, indent=4) + "\n", encoding="utf-8")
    return changed


# ----------------------------------------------------------------------------- main

def build_arm_plan(args) -> list[dict]:
    """One entry per (label, prompt target) pair to be measured."""
    if args.arm_label and args.prompt_tokens:
        return [{"label": args.arm_label, "prompt_tokens": args.prompt_tokens}]
    plan = []
    for spec in args.arm:
        label, _, val = spec.partition("=")
        plan.append({"label": label, "prompt_tokens": int(val)})
    return plan


def run_arm(args, plan: list[dict], log_path: Path, prompt_cache: dict[int, str]) -> list[RequestRecord]:
    records: list[RequestRecord] = []
    for arm in plan:
        content = prompt_cache.get(arm["prompt_tokens"])
        if content is None:
            content = synth_prompt(arm["prompt_tokens"], args.seed)
            prompt_cache[arm["prompt_tokens"]] = content
        for repeat in range(1, args.repeats + 1):
            offset = log_path.stat().st_size if log_path.exists() else 0
            t0 = time.monotonic()
            try:
                resp = post_completion(args.base_url, args.model, content, args.max_tokens,
                                       args.seed, args.timeout, args.api_key)
            except Exception as exc:  # noqa: BLE001 - any transport failure is a rejected sample
                print(f"[bench] {arm['label']} repeat {repeat}: request failed: {exc}", flush=True)
                continue
            wall = time.monotonic() - t0
            recs = wait_for_log_records(log_path, offset, 1)
            if not recs:
                print(f"[bench] {arm['label']} repeat {repeat}: no engine timing line appeared", flush=True)
                continue
            rec = recs[0]
            rec.label = arm["label"]
            rec.repeat = repeat
            rec.warm_cold = args.warm_cold
            usage = resp.get("usage") or {}
            print(f"[bench] {arm['label']} r{repeat}: {rec.decode_tps} tok/s decode, "
                  f"{rec.generated} tok, prompt {rec.prompt_tokens} (reused {rec.reused}), "
                  f"hit {rec.hit_pct}%, accept {rec.accept_pct and round(rec.accept_pct, 1)}%, "
                  f"wall {wall:.1f}s, usable={rec.usable}"
                  + (f" [{rec.reject_reason}]" if rec.reject_reason else ""), flush=True)
            records.append(rec)
    return records


def summarize(records: list[RequestRecord]) -> dict:
    usable = [r for r in records if r.usable]
    rejected = [r for r in records if not r.usable]
    warm = [r for r in usable if r.repeat > 1]
    return {
        "n_total": len(records),
        "n_usable": len(usable),
        "n_rejected": len(rejected),
        "rejected": [{"label": r.label, "repeat": r.repeat, "reason": r.reject_reason} for r in rejected],
        "decode_tps": median_range([r.decode_tps for r in usable]),
        "decode_tps_warm": median_range([r.decode_tps for r in warm]),
        "prefill_tps": median_range([r.prefill_tps for r in usable]),
        "ms_per_token": median_range([r.ms_per_token for r in usable if r.ms_per_token]),
        "hit_pct": median_range([r.hit_pct for r in usable if r.hit_pct is not None]),
        "accept_pct": median_range([r.accept_pct for r in usable if r.accept_pct is not None]),
        "context": median_range([float(r.context) for r in usable]),
        "generated": median_range([float(r.generated) for r in usable]),
        "prompt_tokens": median_range([float(r.prompt_tokens) for r in usable]),
    }


def md_table(records: list[RequestRecord]) -> str:
    cols = ["label", "rep", "usable", "prompt", "reused", "ctx", "gen", "decode_ms",
            "decode_tps", "hit_%", "accept_%", "ckpt", "exchanged", "ms/token"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in records:
        out.append("| " + " | ".join([
            r.label, str(r.repeat), "yes" if r.usable else f"**no** ({r.reject_reason})",
            str(r.prompt_tokens), str(r.reused), str(r.context), str(r.generated),
            str(r.decode_ms), f"{r.decode_tps:.1f}",
            f"{r.hit_pct:.1f}" if r.hit_pct is not None else "n/a",
            f"{r.accept_pct:.1f}" if r.accept_pct is not None else "n/a",
            str(r.checkpoints), str(r.exchanged if r.exchanged is not None else "n/a"),
            f"{r.ms_per_token:.2f}" if r.ms_per_token else "n/a",
        ]) + " |")
    return "\n".join(out)


def timing_table(records: list[RequestRecord]) -> str:
    rows = [r for r in records if r.usable and r.decode_timing]
    if not rows:
        return "_no `strata decode timing` lines captured (start the engine with `STRATA_DECODE_TIMING=1`)_"
    cols = ["label", "rep", "windows", "T", "tok/win", "ms/win", "verify", "wait", "host",
            "plan", "actq", "jobs", "CPU", "stage", "commit", "draft"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        d = r.decode_timing
        out.append("| " + " | ".join([
            r.label, str(r.repeat), str(d.windows), f"{d.avg_t:.2f}", f"{d.tokens_per_window:.2f}",
            f"{d.ms_per_window:.2f}", f"{d.verify_ms:.2f}", f"{d.wait_ms:.2f}", f"{d.host_ms:.2f}",
            f"{d.plan_ms:.2f}", f"{d.actq_ms:.2f}", f"{d.jobs_ms:.2f}", f"{d.cpu_ms:.2f}",
            f"{d.stage_ms:.2f}", f"{d.commit_ms:.2f}", f"{d.draft_ms:.2f}",
        ]) + " |")
    return "\n".join(out)


def stage_table(records: list[RequestRecord]) -> str:
    rows = [r for r in records if r.usable and r.stage_profile]
    if not rows:
        return "_no `strata decode GPU stages` lines captured (start the engine with `STRATA_VERIFY_PROFILE=1`)_"
    names: list[str] = []
    for r in rows:
        for g in r.stage_profile.groups:
            for n in g.stages:
                if n not in names:
                    names.append(n)
    out = []
    for r in rows:
        sp = r.stage_profile
        for g in sp.groups:
            out.append(f"\n**{r.label} repeat {r.repeat} - {g.name} layers** "
                       f"(total {sp.total_ms_per_window:.2f} ms/window over {sp.windows} windows)\n")
            out.append("| stage | ms/window | % of group |")
            out.append("|---|---|---|")
            gsum = sum(g.stages.values())
            for n in names:
                if n in g.stages:
                    out.append(f"| {n} | {g.stages[n]:.3f} | {100.0 * g.stages[n] / gsum:.1f}% |"
                               if gsum else f"| {n} | {g.stages[n]:.3f} | n/a |")
    return "\n".join(out)


def selftest() -> int:
    good = ("strata serve: prompt 4096 tokens = 4090 reused + 6 read in 100 ms (40.0 tok/s), "
            "128 generated in 2000 ms (64.0 tok/s), drafts accepted 90 of 110, 1 checkpoints")
    cancelled = ("strata serve: prompt 4096 tokens = 0 reused + 4096 read in 100 ms (40.0 tok/s), "
                 "50 generated in 900 ms (55.5 tok/s), drafts accepted 40 of 60, 1 checkpoints [cancelled]")
    truncated = "strata serve: prompt 4096 tokens = 0 reused + 1 read in 4 ms"
    hit = "strata serve: decode expert cache hit rate: 93.3% (2201209 hits / 2359680 lookups)"
    stages = ("strata decode GPU stages (ms/window): GDN layers: hc-read0 0.10 attention 1.20 "
              "(gap) 0.05 | QSA layers: attention 2.00 gate 0.50 | total 3.85 ms/window over 12 windows")
    dt = ("strata decode timing: 12 windows, avg T 4.00, 1.80 tokens/window, 30.00 ms/window = "
          "verify 28.00 (GPU-reach wait 20.00 + per-layer host 5.00 [plan 2.00 actq 1.00 jobs 0.50 "
          "CPU 1.50] + stage 3.00) + commit/emit 1.50 + draft 0.50; per layer-window: CPU experts "
          "0.10 (0.20 entries), VRAM hits 0.90, PCIe 0.05")
    recs = parse_log("\n".join([good, hit, stages, dt, cancelled, truncated]))
    checks = [
        ("two records", len(recs) == 2),
        ("good usable", recs[0].usable),
        ("hit attached", recs[0].hit_pct == 93.3),
        ("stage profile", recs[0].stage_profile is not None
         and recs[0].stage_profile.total_ms_per_window == 3.85),
        ("two stage groups", recs[0].stage_profile and len(recs[0].stage_profile.groups) == 2),
        ("gdn attention", recs[0].stage_profile and recs[0].stage_profile.groups[0].stages["attention"] == 1.2),
        ("decode timing", recs[0].decode_timing is not None and recs[0].decode_timing.wait_ms == 20.0),
        ("cancelled rejected", not recs[1].usable),
        ("accept pct", abs(recs[0].accept_pct - 81.818181) < 1e-3),
        ("stats exclude cancelled", summarize(recs)["n_usable"] == 1),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'ok' if passed else 'FAIL'}] {name}")
        ok &= bool(passed)
    # determinism: the same seed must give byte-identical prompts
    same = synth_prompt(64, 7) == synth_prompt(64, 7) and synth_prompt(64, 7) != synth_prompt(64, 8)
    print(f"  [{'ok' if same else 'FAIL'}] synth_prompt deterministic")
    ok &= same
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true", help="run the parser self-test and exit")
    p.add_argument("--dry-run", action="store_true", help="print the plan, touch nothing")
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--model", default=None, help="model id; default: the first id the server lists")
    p.add_argument("--log-path", type=Path, default=Path("strata-iq3_xxs.log"),
                   help="the engine log the server points the engine's stderr at")
    p.add_argument("--prompt-tokens", type=int, default=4096)
    p.add_argument("--prompt-file", type=Path, default=None, help="send this exact text instead of a synthetic prompt")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--arm", action="append", default=[], metavar="LABEL=PROMPT_TOKENS",
                   help="repeatable; each is measured in order against the same running server")
    p.add_argument("--arm-label", default=None, help="tag for the single-arm mode")
    p.add_argument("--warm-cold", default="n/a", help="recorded verbatim in the report")
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-md", type=Path, default=None)
    p.add_argument("--api-key", default=os.environ.get("STRATA_API_KEY"))
    p.add_argument("--start-server", default=None, metavar="CMD",
                   help="command that starts the local server; the harness starts it, waits for "
                        "/v1/models, and terminates it in a finally block when the run ends")
    p.add_argument("--engine-name", default=os.environ.get("STRATA_ENGINE_NAME", "strata.exe"),
                   help="engine process image name counted by the pre-check (default: strata.exe)")
    p.add_argument("--gpu-lock", action="store_true",
                   default=os.environ.get("STRATA_GPU_LOCK", "0") not in ("", "0"),
                   help="claim tools/opt/gpu-window.json for the run and refuse if another member "
                        "holds it (env STRATA_GPU_LOCK=1 enables it too)")
    p.add_argument("--gpu-lock-ttl", type=float, default=float(os.environ.get("STRATA_GPU_LOCK_TTL", 3600)),
                   help="seconds before an unclaimed-looking lock is treated as stale")
    p.add_argument("--gpu-lock-wait", type=float, default=float(os.environ.get("STRATA_GPU_LOCK_WAIT", 0)),
                   help="with --gpu-lock: wait up to this many seconds for a held lock to free "
                        "(default 0 = refuse immediately)")
    p.add_argument("--start-timeout", type=float, default=420.0,
                   help="seconds to wait for --start-server to answer /v1/models (model load is slow)")
    p.add_argument("--emit-config-copy", nargs=2, metavar=("SRC", "DST"),
                   help="write a modified copy of a config JSON to DST and exit")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="with --emit-config-copy: change one engine arg (or env:NAME=VALUE)")
    p.add_argument("--allow-multi-var", action="store_true")
    args = p.parse_args()

    if args.selftest:
        return selftest()

    if args.emit_config_copy:
        src, dst = Path(args.emit_config_copy[0]), Path(args.emit_config_copy[1])
        changed = emit_config_copy(src, dst, args.set, args.allow_multi_var)
        print(json.dumps({"wrote": str(dst), "changed": changed}, indent=2))
        return 0

    plan = build_arm_plan(args)
    if not plan:
        raise SystemExit("nothing to do: pass --arm-label/--prompt-tokens, or --arm LABEL=TOKENS")
    if args.repeats < 3:
        raise SystemExit("refusing to run: >= 3 repeats are required for a median (--repeats)")
    if args.prompt_file:
        plan = [dict(a, prompt_file=str(args.prompt_file)) for a in plan]
    if args.dry_run:
        print(json.dumps({
            "plan": plan, "repeats": args.repeats, "max_tokens": args.max_tokens,
            "base_url": args.base_url, "log_path": str(args.log_path),
            "note": "dry run: no request is sent and no file is written",
        }, indent=2))
        return 0

    if args.model is None and args.dry_run:
        args.model = "<discovered at run time>"

    stamp = time.strftime("%Y%m%d-%H%M%S")
    label = args.arm_label or "multi"
    out_json = args.out_json or Path("tools/opt/results") / f"{stamp}-{label}.json"
    out_md = args.out_md or out_json.with_suffix(".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    prompt_cache: dict[int, str] = {}
    if args.prompt_file:
        text = args.prompt_file.read_text(encoding="utf-8")
        prompt_cache = {a["prompt_tokens"]: text for a in plan}

    # the server must be up before the model id can be discovered, so probe first
    #
    # A HARD PRECONDITION, AND IT IS THE DIFFERENCE BETWEEN A VALID ARM AND A SILENTLY
    # POISONED ONE.  `--start-server` means "this harness owns the engine's lifecycle".
    # When the port is ALREADY occupied, the old code attached to whatever answered and
    # never started the command it was given - so an A/B arm measured someone else's
    # engine, running someone else's config, and nothing in the output said so.  An
    # occupied port is an ENVIRONMENT ERROR here, not a ready server.
    # THE WINDOW LOCK COMES FIRST: it is the cooperative serialisation, and with
    # --gpu-lock-wait it waits for the other member's run to END rather than refusing.
    # The two pre-checks below stay where they are because they cover the UNCOOPERATIVE
    # case - an engine nobody holds a lock for (a leftover, or a member not yet using it).
    claim = None
    if args.gpu_lock:
        claim = gpu_lock_acquire_wait(args.arm_label or "bench_decode", args.gpu_lock_ttl, args.gpu_lock_wait)
        if claim is None:
            return 3
        # RELEASE ON EVERY EXIT PATH, INCLUDING THE PRE-CHECKS BELOW.  They run AFTER the claim
        # and `raise SystemExit`, which skips the try/finally around the run - so a refusal used
        # to leave the lock held by a dead pid and block the next member (it did exactly that once).
        atexit.register(gpu_lock_release, claim)
    if args.start_server and port_in_use(args.base_url):
        raise SystemExit(
            f"refusing to run: {args.base_url} is already occupied by another server.\n"
            f"  --start-server was given, so this harness must own the engine's lifecycle; attaching to a\n"
            f"  server someone else started would silently measure THEIR engine (a different config) instead\n"
            f"  of the arm you asked for, and every number in the report would be wrong without saying so.\n"
            f"  Stop that server (or wait for its run to end), or drop --start-server to attach deliberately.")
    # THE SECOND HALF OF THE PRE-CHECK: an engine that has not bound its port yet, or whose
    # server died and left it orphaned, is invisible to the port probe and still holds the card.
    if args.start_server:
        n_eng = engine_processes(args.engine_name)
        if n_eng > 0:
            raise SystemExit(
                f"refusing to run: {n_eng} '{args.engine_name}' process(es) are already running.\n"
                f"  The port probe cannot see an engine that is still loading (serve/server.py binds only\n"
                f"  once the engine is ready) or one orphaned by a dead server, and two engines do not fit\n"
                f"  on this card - both runs would be ruined. Wait for it to exit, or stop it yourself.")
    # THIRD PRE-CHECK: the log we watch must be the log the engine will write.  Left unchecked,
    # a mismatch produces an arm that runs to completion and reports nothing (the failure mode
    # that made two whole arms look "done" while empty).
    if args.start_server and not args.prompt_file:
        m_cfg = re.search(r"--config\s+(\S+)", args.start_server)
        if m_cfg:
            try:
                declared = json.loads(Path(m_cfg.group(1)).read_text(encoding="utf-8-sig")).get("log")
            except (OSError, ValueError):
                declared = None
            if declared and Path(declared).resolve() != args.log_path.resolve():
                raise SystemExit(
                    f"refusing to run: --log-path {args.log_path} is not the log the engine will write.\n"
                    f"  The config {m_cfg.group(1)} declares log = {declared}\n"
                    f"  The engine's stderr goes to its config's log, so this arm would read a file nothing\n"
                    f"  writes and report zero usable requests. Point --log-path at {declared}, or give the\n"
                    f"  arm's config its own log (bench_decode.py --emit-config-copy does this automatically).")
    server_up = wait_for_server(args.base_url, 1.0)
    started = None
    if not server_up:
        if not args.start_server:
            raise SystemExit(f"no server at {args.base_url}; start one, or pass --start-server")
        print(f"[bench] starting: {args.start_server}", flush=True)
        started = subprocess.Popen(args.start_server, shell=True)
        if not wait_for_server(args.base_url, args.start_timeout):
            kill_tree(started)
            raise SystemExit("the server did not come up in time")
        print("[bench] server is up", flush=True)

    try:
        if args.model is None:
            with urllib.request.urlopen(args.base_url.rstrip("/") + "/v1/models", timeout=30) as r:
                data = json.load(r)
            args.model = data["data"][0]["id"]
            print(f"[bench] model = {args.model}", flush=True)
        records = run_arm(args, plan, args.log_path, prompt_cache)
    finally:
        if started is not None:
            print("[bench] stopping the server we started (and its engine)", flush=True)
            kill_tree(started)
            # AND PROVE IT.  `serve/server.py` re-execs itself, which BREAKS the process tree that
            # `taskkill /T` walks, so the 23 GB engine can survive its own cleanup - observed, and
            # it then blocks the very next arm's pre-check.  While WE hold the window lock no other
            # engine may legally be running, so escalate to a kill by image name and verify again.
            left = engine_processes(args.engine_name)
            if left:
                for _ in range(10):
                    time.sleep(3)
                    left = engine_processes(args.engine_name)
                    if not left:
                        break
            if left:
                print(f"[bench] escalating: killing {left} surviving '{args.engine_name}' by name", flush=True)
                with contextlib.suppress(OSError, subprocess.SubprocessError):
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/F", "/IM", args.engine_name],
                                       capture_output=True, text=True, timeout=60)
                    else:
                        subprocess.run(["pkill", "-f", args.engine_name],
                                       capture_output=True, text=True, timeout=60)
                left = engine_processes(args.engine_name)
            if left:
                print(f"[bench] WARNING: {left} '{args.engine_name}' process(es) still survive - "
                      f"do not start another arm until they are gone", flush=True)
        gpu_lock_release(claim)

    stats = summarize(records)

    # AN EMPTY ARM MUST FAIL LOUDLY.  Writing a JSON whose stats are all null and exiting 0
    # let a whole sweep "finish" while producing no data: the arm looked completed and the
    # next one started.  The usual cause is a log the harness did not read (e.g. the arm's
    # config still pointed `log` at another file), so the message names both the log it
    # watched and the engine log the config declares.
    if stats["n_total"] == 0:
        cfg_log = None
        if args.start_server:
            m = re.search(r"--config\s+(\S+)", args.start_server)
            if m:
                try:
                    cfg_log = json.loads(Path(m.group(1)).read_text(encoding="utf-8-sig")).get("log")
                except (OSError, ValueError):
                    cfg_log = None
        size = args.log_path.stat().st_size if args.log_path.exists() else None
        raise SystemExit(
            f"ARM PRODUCED NO DATA: --log-path {args.log_path} "
            f"({'missing' if size is None else f'{size} bytes'}) yielded no request summary line, "
            f"so this arm has nothing to report.\n"
            f"  The engine's log is the `log` field of the config it was started with"
            + (f"; that config declares {cfg_log!r}\n" if cfg_log else ".\n")
            + (f"  MISMATCH: watch {cfg_log!r} (or give the arm config its own `log`).\n"
               if cfg_log and Path(cfg_log).resolve() != args.log_path.resolve() else "")
            + "  Nothing was written; do not treat this arm as measured.")
    payload = {
        "run": {
            "timestamp": stamp, "base_url": args.base_url, "model": args.model,
            "requested_prompt_tokens": args.prompt_tokens, "requested_max_tokens": args.max_tokens,
            "repeats_requested": args.repeats, "seed": args.seed, "arm_label": args.arm_label,
            "warm_cold": args.warm_cold, "log_path": str(args.log_path),
            "prompt_file": str(args.prompt_file) if args.prompt_file else None,
        },
        "stats": stats,
        "requests": [asdict(r) for r in records],
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md = [f"# bench_decode - {label}", "",
          f"- run: {stamp}  base_url: {args.base_url}  model: {args.model}",
          f"- requested: prompt ~{args.prompt_tokens} tokens, max_tokens {args.max_tokens}, "
          f"{args.repeats} repeats, seed {args.seed}, warm/cold: {args.warm_cold}",
          f"- usable {stats['n_usable']} of {stats['n_total']} (rejected {stats['n_rejected']})", "",
          "## Measured conditions", "", md_table(records), "",
          "## Median and range (usable repeats only)", "",
          "| metric | n | median | min | max | range |", "|---|---|---|---|---|---|"]
    for key in ("decode_tps", "decode_tps_warm", "prefill_tps", "ms_per_token", "hit_pct",
                "accept_pct", "prompt_tokens", "generated", "context"):
        s = stats[key]
        md.append(f"| {key} | {s['n']} | {s['median']} | {s['min']} | {s['max']} | {s['range']} |")
    md += ["", "## Window split (`strata decode timing`, ms/window)", "", timing_table(records),
           "", "## Per-GPU-stage split (`STRATA_VERIFY_PROFILE=1`, ms/window)", "",
           stage_table(records), ""]
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(out_json), "md": str(out_md), "stats": stats}, indent=2))
    return 0 if stats["n_usable"] >= args.repeats else 1


if __name__ == "__main__":
    sys.exit(main())
