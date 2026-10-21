#!/usr/bin/env python3
"""tools/gyro_ab.py - one repeatable A/B measurement of a Strata model config.

WHAT IT DOES
  It starts the engine with a given config, sends a fixed prompt, and measures
  the same run again and again.  Every number in the report comes from that
  run: the prompt speed, the decode speed, the expert cache hit rate, the bytes
  the drive had to read, the time the decode waited for them, and the split of
  the decode's wall time (the GPU's stage time, the CPU expert pool, the wait
  for the GPU, the commit and the draft).  Two variants are compared with
  `--compare`.

WHY IT EXISTS
  A speed claim needs a number, and the number has to come from the same
  machine, the same config, the same prompt and the same cache state.  This
  script is that setup, so that "it got faster" can be shown and not claimed.

    python tools/gyro_ab.py --variant base                      # the default config
    python tools/gyro_ab.py --variant nomtp --extra-arg --mtp --extra-arg off
    python tools/gyro_ab.py --variant env1 --env STRATA_IO_PREFETCH=0
    python tools/gyro_ab.py --compare bench/results/<date>-gyro-titanx

THE CACHE STATE
  The OS page cache holds the model's 58 GB file across an engine restart, so a
  second run starts with warm pages and a first run does not.  `--cache-state
  cold` (the default) evicts the model's pages before the engine starts: it
  writes a large file once and reads it back, which fills the page cache with
  something else.  No root and no `drop_caches` are needed.  Every result
  records which state it ran in.

THE WORKLOAD
  Two phases, each repeated `--repeats` times:
    prompt   a 4096-token prompt (a fixed text), 16 tokens out: the prompt read
    decode   a 1024-token prompt, 512 tokens out: the decode, long enough that
             the expert cache is in the state a real conversation puts it in
  Every repeat uses its own text, so no prompt is served from a cache: each one
  is read from the drive, as it is the first time.  One warmup request runs
  before the repeats and is not measured, so the caches are in their steady
  state.

WHERE THE RESULTS GO
  bench/results/<date>-<name>/<variant>/ : results.json (every number),
  engine.log (the engine's own log), server.log (the server's), summary.md
  (the table) and the config that was measured.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

# ------------------------------------------------------------------------------------------------ the prompt
RECORD = "record{:06d}: alpha beta gamma delta epsilon zeta eta theta.\n"
QUESTION = ("Read the records above and answer this question about them in plain words: "
            "how many records are there, and what does each record hold?\n")
EVICT_FILE = Path("/home/mo/.cache/strata-page-cache-evict.bin")   # on the disk, never on a tmpfs (that is RAM)


def build_prompt(n_tokens: int, tag: str, tok=None) -> str:
    """A prompt of about `n_tokens` tokens.  The tag is in the first line, so two repeats share no prefix
    and neither the conversation cache nor the prompt cache can serve one from the other."""
    head = f"[run {tag}]\n"
    body = []
    total = 0
    i = 0
    while total < n_tokens:
        rec = RECORD.format(i)
        body.append(rec)
        total += len(tok.encode(rec)) if tok else 12
        i += 1
    return head + "".join(body) + "\n" + QUESTION


def load_tokenizer(tok_dir: str | None):
    """The model's own tokenizer, for the token count.  None when it is not there: the count is then an estimate."""
    if not tok_dir:
        return None
    d = Path(tok_dir)
    if not (d / "vocab.json").exists():
        return None
    import strata_tokenizer as ST
    vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (d / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((d / "token_type.json").read_text(encoding="utf-8"))
    return ST.Tokenizer(tokens, merges, types)


# ------------------------------------------------------------------------------------------------ the machine
def machine_info() -> dict:
    """What this machine is: the card, the driver, the CPU, the RAM, the drive."""
    info = {"cpu": platform.processor() or platform.machine(), "logical_cpus": os.cpu_count(),
            "platform": platform.platform()}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        if out.returncode == 0 and out.stdout.strip():
            g = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
            info["gpu"] = g[0]
            info["driver"] = g[1]
            info["gpu_memory_total_mib"] = int(g[2])
            info["gpu_memory_used_mib"] = int(g[3])
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    try:
        mem = Path("/proc/meminfo").read_text(encoding="utf-8")
        info["ram_total_mib"] = int(re.search(r"MemTotal:\s+(\d+) kB", mem).group(1)) // 1024
        info["ram_available_mib"] = int(re.search(r"MemAvailable:\s+(\d+) kB", mem).group(1)) // 1024
    except (OSError, AttributeError):
        pass
    try:
        out = subprocess.run(["df", "-h", str(ROOT)], capture_output=True, text=True, timeout=20)
        info["disk"] = out.stdout.strip().splitlines()[-1].split()[0:4]
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def git_rev() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=20)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# ------------------------------------------------------------------------------------------------ the page cache
def evict_page_cache(gib: int = 16) -> dict:
    """Fill the page cache with a file that is not the model, so the model's pages are gone.

    The file is written once and kept: the next run reads it again.  Reading it
    back is what evicts - the write only fills the disk.  No root, no
    `drop_caches` (which needs it).  The file lives on the disk, never on a
    tmpfs (a tmpfs file is RAM, and it would be the thing that fills the
    memory)."""
    out = {"file": str(EVICT_FILE), "gib": gib}
    try:
        EVICT_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not EVICT_FILE.exists() or EVICT_FILE.stat().st_size < gib * 1024 ** 3:
            t0 = time.monotonic()
            with open(EVICT_FILE, "wb") as f:
                chunk = b"\0" * (1 << 20)
                for _ in range(gib * 1024):
                    f.write(chunk)
                f.flush()
                os.fsync(f.fileno())
            out["write_s"] = round(time.monotonic() - t0, 1)
        mem = Path("/proc/meminfo").read_text(encoding="utf-8")
        out["ram_cached_mib_before"] = int(re.search(r"Cached:\s+(\d+) kB", mem).group(1)) // 1024
        t0 = time.monotonic()
        with open(EVICT_FILE, "rb") as f:
            while f.read(1 << 24):
                pass
        out["read_s"] = round(time.monotonic() - t0, 1)
        mem = Path("/proc/meminfo").read_text(encoding="utf-8")
        out["ram_cached_mib_after"] = int(re.search(r"Cached:\s+(\d+) kB", mem).group(1)) // 1024
        out["ok"] = True
    except OSError as e:
        out["ok"] = False
        out["error"] = str(e)
    return out


# ------------------------------------------------------------------------------------------------ the engine's log
RE_PROMPT = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
                       r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)(?:, drafts accepted (\d+) of (\d+))?")
RE_HIT = re.compile(r"decode expert cache hit rate: ([\d.]+)% \((\d+) hits / (\d+) lookups\); "
                    r"(\d+) more read by the GPU over PCIe")
RE_FILE = re.compile(r"the OS read ([\d.]+) MB from storage \((\d+) major faults\) "
                     r"for ([\d.]+) MB of expert reads")
RE_PAGE_CACHE = re.compile(r"page cache held ([\d.]+) MB at the read, not ([\d.]+) MB")
RE_WAIT = re.compile(r"decode waited ([\d.]+) ms in (\d+) fetches")
RE_TIERS = re.compile(r"expert tiers: GPU (\d+) hits this request; since the start RAM (\d+) blobs, "
                      r"files (\d+) blobs ([\d.]+) MB read")
RE_ROUTE_TAIL = re.compile(r"route tail skip: (\d+) missed experts skipped \((\d+) entries\) since the start")
RE_PREFILL = re.compile(r"prefill plan: chunk (\d+), ring (\d+) kept: file set ([\d.]+) GiB does not fit the page cache "
                         r"([\d.]+) GiB -> about ([\d.]+) GB read from the drive per (\d+)-token prompt")
RE_PREFILL_OK = re.compile(r"prefill plan: chunk (\d+), ring (\d+) kept: ([\d.]+) GiB in the page cache")
RE_ENGINE = re.compile(r"session is up \(engine ([^)]+)\)")
RE_LOAD = re.compile(r"(\d+) MiB of weights loaded from packs/\S+ in ([\d.]+) s")
RE_CACHE_SLOTS = re.compile(r"expert cache (\d+) slots, ([\d.]+) GiB of VRAM")
RE_RESIDENT = re.compile(r"resident RAM mode: ([\d.]+) GiB of experts in RAM")
RE_PCIE = re.compile(r"PCIe probe: ([\d.]+) GB/s host->device")
RE_FILETIER = re.compile(r"the file tier reads through the file cache .*the cache holds (\d+)% of the read set, "
                         r"direct reads ([\d.]+) GB/s, cached-path reads ([\d.]+) GB/s")
RE_PAGE_CACHE = re.compile(r"([\d.]+) GiB available for the page cache")
RE_DRAFT = re.compile(r"draft layer loaded, (\d+) MiB of VRAM")
# STRATA_DECODE_TIMING=1: where a request's decode time goes
RE_DEC_TIMING = re.compile(
    r"strata decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window = "
    r"verify ([\d.]+) \(GPU-reach wait ([\d.]+) \+ per-layer host ([\d.]+) \[plan ([\d.]+) actq ([\d.]+) jobs ([\d.]+) "
    r"CPU ([\d.]+)\] \+ stage ([\d.]+)\) \+ commit/emit ([\d.]+) \+ draft ([\d.]+); "
    r"per layer-window: CPU experts ([\d.]+) \(([\d.]+) entries\), VRAM hits ([\d.]+), PCIe ([\d.]+); "
    r"adapt wait ([\d.]+) join ([\d.]+), ([\d.]+) swaps/window")
# STRATA_VERIFY_PROFILE=1: the GPU's own time per stage
RE_GPU_STAGES = re.compile(r"strata decode GPU stages \(ms/window\):(.*)")


def parse_decode_timing(text: str) -> dict | None:
    """The engine's own line about where a decode window's time went."""
    m = None
    for m in RE_DEC_TIMING.finditer(text):
        pass
    if not m:
        return None
    g = m.groups()
    return {"windows": int(g[0]), "avg_T": float(g[1]), "tokens_per_window": float(g[2]), "ms_per_window": float(g[3]),
            "verify_ms": float(g[4]), "gpu_reach_wait_ms": float(g[5]), "per_layer_host_ms": float(g[6]),
            "plan_ms": float(g[7]), "actq_ms": float(g[8]), "jobs_ms": float(g[9]), "cpu_pool_ms": float(g[10]),
            "stage_ms": float(g[11]), "commit_emit_ms": float(g[12]), "draft_ms": float(g[13]),
            "cpu_experts_ms": float(g[14]), "cpu_expert_entries": float(g[15]), "vram_hits_ms": float(g[16]),
            "pcie_ms": float(g[17]), "adapt_wait_ms": float(g[18]), "adapt_join_ms": float(g[19]),
            "swaps_per_window": float(g[20])}


def parse_gpu_stages(text: str) -> dict | None:
    """The GPU's time per stage, in ms per window: 'GDN layers: hc-read0 0.10 ... | QSA layers: ... | total X'."""
    m = None
    for m in RE_GPU_STAGES.finditer(text):
        pass
    if not m:
        return None
    body = m.group(1)
    out: dict = {"raw": body.strip()}
    total = re.search(r"total ([\d.]+) ms/window over (\d+) windows", body)
    if total:
        out["total_ms_per_window"] = float(total.group(1))
        out["windows"] = int(total.group(2))
    for part in body.split("|"):
        part = part.strip()
        if ":" not in part:
            continue
        label, _, rest = part.partition(":")
        label = label.strip()
        if label in ("total",) or "ms/window" in label:
            continue
        stages = {}
        for name, value in re.findall(r"(.+?)\s+([0-9]+\.[0-9]+)(?=\s|$)", rest):
            stages[name.strip()] = float(value)
        if stages:
            out[label] = stages
    return out


def parse_engine_log(text: str) -> dict:
    """The numbers the engine wrote about the last request.  Only what the log says, nothing guessed."""
    out: dict = {}
    hits = RE_HIT.search(text)
    if hits:
        out["expert_hit_pct"] = float(hits.group(1))
        out["expert_hits"] = int(hits.group(2))
        out["expert_lookups"] = int(hits.group(3))
        out["experts_over_pcie"] = int(hits.group(4))
    f = RE_FILE.search(text)
    if f:
        out["os_read_mb"] = float(f.group(1))
        out["major_faults"] = int(f.group(2))
        out["expert_read_mb"] = float(f.group(3))
    pc = RE_PAGE_CACHE.search(text)
    if pc:
        out["page_cache_held_mb"] = float(pc.group(1))
        out["page_cache_needed_mb"] = float(pc.group(2))
    w = RE_WAIT.search(text)
    if w:
        out["decode_wait_ms"] = float(w.group(1))
        out["decode_fetches"] = int(w.group(2))
    t = RE_TIERS.search(text)
    if t:
        out["tier_gpu_hits"] = int(t.group(1))
        out["tier_ram_blobs"] = int(t.group(2))
        out["tier_file_blobs"] = int(t.group(3))
        out["tier_file_mb"] = float(t.group(4))
    r = RE_ROUTE_TAIL.search(text)
    if r:
        out["route_tail_skipped"] = int(r.group(1))
    p = RE_PREFILL.search(text)
    if p:
        out["prefill_chunk"] = int(p.group(1))
        out["prefill_file_set_gib"] = float(p.group(3))
        out["prefill_page_cache_gib"] = float(p.group(4))
        out["prefill_drive_gb"] = float(p.group(5))
    dt = parse_decode_timing(text)
    if dt:
        out["decode_timing"] = dt
    gs = parse_gpu_stages(text)
    if gs:
        out["gpu_stages"] = gs
    return out


def parse_prompt_line(text: str) -> dict | None:
    """The engine's own line about the request: prompt read, decode, drafts."""
    last = None
    for m in RE_PROMPT.finditer(text):
        last = m
    if not last:
        return None
    d = {"prompt_tokens": int(last.group(1)), "prompt_reused": int(last.group(2)), "prompt_read": int(last.group(3)),
         "prompt_ms": float(last.group(4)), "prompt_tok_s": float(last.group(5)),
         "generated": int(last.group(6)), "decode_ms": float(last.group(7)), "decode_tok_s": float(last.group(8))}
    if last.group(9) is not None:
        d["drafts_accepted"] = int(last.group(9))
        d["drafts_offered"] = int(last.group(10))
        d["draft_accept_pct"] = round(100.0 * d["drafts_accepted"] / d["drafts_offered"], 1) if d["drafts_offered"] else None
    return d


def parse_startup(text: str) -> dict:
    """What the engine reported about itself when it started."""
    out: dict = {}
    m = RE_CACHE_SLOTS.search(text)
    if m:
        out["expert_cache_slots"] = int(m.group(1))
        out["expert_cache_gib"] = float(m.group(2))
    m = RE_RESIDENT.search(text)
    if m:
        out["resident_ram_gib"] = float(m.group(1))
    m = RE_PCIE.search(text)
    if m:
        out["pcie_gbs"] = float(m.group(1))
    m = RE_FILETIER.search(text)
    if m:
        out["file_cache_hold_pct"] = int(m.group(1))
        out["direct_read_gbs"] = float(m.group(2))
        out["cached_read_gbs"] = float(m.group(3))
    m = RE_PAGE_CACHE.search(text)
    if m:
        out["page_cache_gib"] = float(m.group(1))
    m = RE_DRAFT.search(text)
    if m:
        out["draft_vram_mib"] = int(m.group(1))
    m = RE_ENGINE.search(text)
    if m:
        out["engine_version"] = m.group(1)
    m = RE_LOAD.search(text)
    if m:
        out["weights_mib"] = int(m.group(1))
        out["weights_load_s"] = float(m.group(2))
    m = RE_PREFILL.search(text)
    if m:
        out["prefill_chunk"] = int(m.group(1))
        out["prefill_file_set_gib"] = float(m.group(3))
        out["prefill_page_cache_gib"] = float(m.group(4))
        out["prefill_drive_gb"] = float(m.group(5))
    m = RE_PREFILL_OK.search(text)
    if m:
        out.setdefault("prefill_chunk", int(m.group(1)))
        out["prefill_fits_page_cache"] = True
    return out


# ------------------------------------------------------------------------------------------------ the server
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http_json(url: str, key: str, body: dict | None = None, timeout: float = 60.0) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET")
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_ready(url: str, key: str, proc: subprocess.Popen, timeout: float) -> dict:
    """Wait until the engine is loaded and answers.  The engine takes a minute or two on a small card."""
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"the server stopped before the model was ready (exit {proc.returncode})")
        try:
            st = http_json(url + "/v1/status", key, timeout=15)
            if st.get("loaded") and st.get("model"):
                return st
            last = str(st.get("model_state") or st.get("loaded"))
        except (urllib.error.URLError, OSError, ValueError) as e:
            last = str(e)
        time.sleep(2.0)
    raise SystemExit(f"the model was not ready after {timeout:.0f} s (last: {last})")


def chat(url: str, key: str, prompt: str, max_tokens: int, thinking: bool, model: str | None) -> dict:
    """One request, streamed, so the time to the first token is measured here and not guessed."""
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": max_tokens,
            "stream": True, "chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    if model:
        body["model"] = model
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    t0 = time.monotonic()
    ttft = None
    text = []
    timings = None
    usage = None
    finish = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                ev = json.loads(body)
            except ValueError:
                continue
            if ev.get("timings"):
                timings = ev["timings"]
            if ev.get("usage"):
                usage = ev["usage"]
            ch = ev.get("choices") or []
            if ch:
                finish = ch[0].get("finish_reason") or finish
                delta = ch[0].get("delta") or {}
                piece = delta.get("content") or ""
                if piece and ttft is None:
                    ttft = time.monotonic() - t0
                text.append(piece)
    total = time.monotonic() - t0
    return {"ttft_s": round(ttft, 3) if ttft else None, "total_s": round(total, 3),
            "timings": timings, "usage": usage, "finish": finish,
            "answer": "".join(text), "answer_chars": len("".join(text))}


def engine_pid(server_pid: int) -> int | None:
    """The engine process: the server's child."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        if ppid == server_pid:
            return int(entry.name)
    return None


def proc_io(pid: int | None) -> dict:
    """The process's own I/O counters: what it read from the disk, and how many reads it made."""
    if not pid:
        return {}
    try:
        out = {}
        for line in (Path(f"/proc/{pid}/io")).read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition(":")
            out[k.strip()] = int(v.strip())
        return out
    except (OSError, ValueError):
        return {}


# ------------------------------------------------------------------------------------------------ the run
WORKLOADS = {
    # name: (prompt tokens, tokens out, thinking)
    #  prompt  a 4096-token prompt and 16 tokens out: the prompt read on its own
    #  decode  a 1024-token prompt and 512 tokens out: the decode, long enough that the expert cache
    #          is in the state a real conversation puts it in (a 37-token answer sees 73% hits, a long one 95%)
    "prompt": (4096, 16, False),
    "decode": (1024, 512, True),
}


def run_variant(args, variant: str, cfg: dict, outdir: Path) -> dict:
    """One variant: start the engine, run the workload, stop the engine, write the numbers."""
    outdir.mkdir(parents=True, exist_ok=True)
    port = args.port or free_port()
    key = cfg.get("api_key") or ""
    run_cfg = dict(cfg)
    run_cfg["port"] = port
    run_cfg["host"] = "127.0.0.1"
    run_cfg["log"] = str(outdir / "engine.log")
    run_cfg["cwd"] = str(ROOT)
    run_cfg["model_name"] = f"{cfg.get('model_name', 'model')}-{variant}"
    if args.extra_arg:
        run_cfg["args"] = list(run_cfg["args"]) + list(args.extra_arg)
    env_overrides = dict(args.env)
    if args.phase_env:
        env_overrides.setdefault("STRATA_DECODE_TIMING", "1")
        env_overrides.setdefault("STRATA_VERIFY_PROFILE", "1")
    if env_overrides:
        run_cfg["env"] = {**(cfg.get("env") or {}), **env_overrides}
    cfg_path = outdir / "config.json"
    cfg_path.write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    result = {"variant": variant, "config": run_cfg, "port": port, "started": datetime.now(timezone.utc).isoformat(),
              "phase_env": bool(args.phase_env), "runs": [], "cache_state": args.cache_state}

    if args.cache_state == "cold":
        result["cache_eviction"] = evict_page_cache(args.evict_gib)

    log_path = outdir / "engine.log"
    url = f"http://127.0.0.1:{port}"
    env = dict(os.environ)
    server_log = open(outdir / "server.log", "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "serve.server", "--engine", "strata", "--config", str(cfg_path),
                             "--port", str(port)], cwd=str(ROOT), stdout=server_log, stderr=subprocess.STDOUT, env=env)
    try:
        t_ready = time.monotonic()
        status = wait_ready(url, key, proc, args.ready_timeout)
        result["ready_s"] = round(time.monotonic() - t_ready, 1)
        result["status"] = status
        if log_path.exists():
            result["startup"] = parse_startup(log_path.read_text(encoding="utf-8", errors="replace"))

        tok = load_tokenizer(cfg.get("tokenizer"))
        pid = engine_pid(proc.pid)
        result["engine_pid"] = pid

        if args.warmup:
            # the caches are in their steady state before the first measured request: the warmup is a full
            # request of the same fixture as the first workload, so the expert cache and the page cache are
            # where a user's second request finds them (without this the first measured repeat is the
            # outlier and the repeats after it are not comparable with another run's)
            wl_name = next(iter(WORKLOADS)) if not args.workloads else args.workloads.split(",")[0]
            p_tok, out_tok, wl_thinking = WORKLOADS[wl_name]
            wp = build_prompt(p_tok, f"{variant}-warmup", tok)
            chat(url, key, wp, out_tok, wl_thinking, run_cfg.get("model_name"))
            time.sleep(2.0)

        for rep in range(1, args.repeats + 1):
            for wl_name, (p_tok, out_tok, wl_thinking) in WORKLOADS.items():
                if args.workloads and wl_name not in args.workloads:
                    continue
                tag = f"{variant}-r{rep}"
                prompt = build_prompt(p_tok, tag, tok)
                before = log_path.stat().st_size if log_path.exists() else 0
                io0 = proc_io(pid)
                r = chat(url, key, prompt, out_tok, args.thinking or wl_thinking, run_cfg.get("model_name"))
                io1 = proc_io(pid)
                time.sleep(1.0)                      # the engine writes its own lines a moment later
                text = ""
                if log_path.exists():
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        f.seek(before)
                        text = f.read()
                rec = {"workload": wl_name, "repeat": rep, "tag": tag,
                       "prompt_chars": len(prompt), "prompt_tokens_estimate": len(tok.encode(prompt)) if tok else None,
                       **r}
                pl = parse_prompt_line(text)
                if pl:
                    rec.update(pl)
                rec.update({k: v for k, v in parse_engine_log(text).items() if k not in rec})
                if io0 and io1:
                    rec["proc_io"] = {k: io1.get(k, 0) - io0.get(k, 0) for k in ("rchar", "read_bytes", "syscr", "wchar")}
                rec.pop("answer", None)
                result["runs"].append(rec)
                dt = rec.get("decode_timing") or {}
                print(f"[gyro_ab] {variant} {wl_name} r{rep}: prompt {rec.get('prompt_tok_s')} tok/s, "
                      f"decode {rec.get('decode_tok_s')} tok/s, hit {rec.get('expert_hit_pct')}%, "
                      f"wait {rec.get('decode_wait_ms')} ms, "
                      f"stage {dt.get('stage_ms')} cpu {dt.get('cpu_pool_ms')} draft {dt.get('draft_ms')}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        server_log.close()
    result["ended"] = datetime.now(timezone.utc).isoformat()
    result["summary"] = summarize(result["runs"])
    (outdir / "results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    write_summary(outdir, result)
    return result


def median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def summarize(runs: list[dict]) -> dict:
    """The medians of a variant, per workload.

    Two states are kept apart: the first repeat (the page cache and the expert
    cache are still cold) and the later ones (the state a user lives in).  A
    variant that only looks better in the first repeat is not better."""
    out = {}
    for wl in sorted({r["workload"] for r in runs}):
        rs = [r for r in runs if r["workload"] == wl]
        cold = [r for r in rs if r["repeat"] == 1]
        warm = [r for r in rs if r["repeat"] > 1]
        dts = [r.get("decode_timing") for r in rs if r.get("decode_timing")]
        out[wl] = {"workload": wl, "n": len(rs),
                   "prompt_tok_s": median([r.get("prompt_tok_s") for r in rs]),
                   "decode_tok_s": median([r.get("decode_tok_s") for r in rs]),
                   "cold_prompt_tok_s": median([r.get("prompt_tok_s") for r in cold]),
                   "cold_decode_tok_s": median([r.get("decode_tok_s") for r in cold]),
                   "warm_prompt_tok_s": median([r.get("prompt_tok_s") for r in warm]),
                   "warm_decode_tok_s": median([r.get("decode_tok_s") for r in warm]),
                   "ttft_s": median([r.get("ttft_s") for r in rs]),
                   "total_s": median([r.get("total_s") for r in rs]),
                   "generated": median([r.get("generated") for r in rs]),
                   "expert_hit_pct": median([r.get("expert_hit_pct") for r in rs]),
                   "os_read_mb": median([r.get("os_read_mb") for r in rs]),
                   "expert_read_mb": median([r.get("expert_read_mb") for r in rs]),
                   "decode_wait_ms": median([r.get("decode_wait_ms") for r in rs]),
                   "decode_fetches": median([r.get("decode_fetches") for r in rs]),
                   "draft_accept_pct": median([r.get("draft_accept_pct") for r in rs]),
                   "page_cache_held_mb": median([r.get("page_cache_held_mb") for r in rs]),
                   "stage_ms": median([d.get("stage_ms") for d in dts]),
                   "gpu_reach_wait_ms": median([d.get("gpu_reach_wait_ms") for d in dts]),
                   "cpu_pool_ms": median([d.get("cpu_pool_ms") for d in dts]),
                   "commit_emit_ms": median([d.get("commit_emit_ms") for d in dts]),
                   "draft_ms": median([d.get("draft_ms") for d in dts]),
                   "ms_per_window": median([d.get("ms_per_window") for d in dts]),
                   "tokens_per_window": median([d.get("tokens_per_window") for d in dts]),
                   "vram_hits_ms": median([d.get("vram_hits_ms") for d in dts]),
                   "pcie_ms": median([d.get("pcie_ms") for d in dts]),
                   "cpu_experts_ms": median([d.get("cpu_experts_ms") for d in dts])}
        dts_all = [r.get("decode_timing") for r in rs if r.get("decode_timing")]
        out[wl]["ms_per_window_cold"] = median([d.get("ms_per_window") for d in
                                                [r.get("decode_timing") for r in cold] if d])
        out[wl]["ms_per_window_warm"] = median([d.get("ms_per_window") for d in
                                                [r.get("decode_timing") for r in warm] if d])
        out[wl]["wait_cpu_ms_cold"] = median([d.get("cpu_pool_ms") for d in
                                              [r.get("decode_timing") for r in cold] if d])
        out[wl]["wait_cpu_ms_warm"] = median([d.get("cpu_pool_ms") for d in
                                              [r.get("decode_timing") for r in warm] if d])
    return out


def write_summary(outdir: Path, result: dict) -> None:
    """The table in summary.md: one row per workload, the medians of the repeats."""
    s = result.get("summary", {})
    lines = [f"# {result['variant']}", "",
             f"- started: {result.get('started')}",
             f"- ready in: {result.get('ready_s')} s",
             f"- page cache at the start: {result.get('cache_state')}"
             + (f" ({result.get('cache_eviction', {}).get('read_s')} s to evict)"
                if result.get("cache_eviction") else ""),
             f"- phase timers: {'on' if result.get('phase_env') else 'off'}",
             f"- machine: {json.dumps(machine_info())}",
             f"- git: {git_rev()}",
             f"- engine: {result.get('status', {}).get('engine')}",
             f"- config: `config.json` in this folder", ""]
    if result.get("startup"):
        lines += ["## what the engine said when it started", "",
                  "```", json.dumps(result["startup"], indent=2), "```", ""]
    if s:
        cols = ["workload", "n", "prompt_tok_s", "decode_tok_s", "ttft_s", "generated", "expert_hit_pct",
                "os_read_mb", "expert_read_mb", "decode_wait_ms", "decode_fetches", "draft_accept_pct",
                "ms_per_window", "stage_ms", "gpu_reach_wait_ms", "cpu_pool_ms", "commit_emit_ms", "draft_ms"]
        lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for wl, v in s.items():
            lines.append("| " + " | ".join(str(v.get(c)) for c in cols) + " |")
    lines += ["", "## every run", "",
              "| workload | repeat | prompt tok/s | decode tok/s | ttft s | os read MB | wait ms | hit % | "
              "stage ms/window | cpu ms/window | draft ms/window |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in result.get("runs", []):
        d = r.get("decode_timing") or {}
        lines.append(f"| {r.get('workload')} | {r.get('repeat')} | {r.get('prompt_tok_s')} | {r.get('decode_tok_s')} | "
                     f"{r.get('ttft_s')} | {r.get('os_read_mb')} | {r.get('decode_wait_ms')} | "
                     f"{r.get('expert_hit_pct')} | {d.get('stage_ms')} | {d.get('cpu_pool_ms')} | {d.get('draft_ms')} |")
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare(folder: Path) -> int:
    """Two or more variants, side by side.

    The cold first repeat and the warm later ones are kept apart: the caches
    fill as a run goes on, and a variant that only looks better cold is not
    better.  The base is the variant named "base-both" when it is there, else
    the first one in the folder."""
    variants = []
    for p in sorted(folder.glob("*/results.json")):
        variants.append(json.loads(p.read_text(encoding="utf-8")))
    if not variants:
        print(f"no results in {folder}")
        return 1
    base = next((v for v in variants if v["variant"].startswith("base")), variants[0])
    wls = sorted({wl for v in variants for wl in v.get("summary", {})})
    for wl in wls:
        print(f"\n## {wl}")
        cols = ["variant", "n", "cold prompt tok/s", "warm prompt tok/s", "cold decode tok/s", "warm decode tok/s",
                "warm vs base", "wait_cpu cold ms", "wait_cpu warm ms", "hit %", "draft %"]
        print("| " + " | ".join(cols) + " |")
        print("|" + "---|" * len(cols))
        b = base.get("summary", {}).get(wl) or {}
        for v in variants:
            row = v.get("summary", {}).get(wl)
            if not row:
                continue
            key = "warm_prompt_tok_s" if "prompt" in wl else "warm_decode_tok_s"
            bkey = "warm_prompt_tok_s" if "prompt" in wl else "warm_decode_tok_s"
            ref = b.get(bkey)
            val = row.get(key)
            delta = f"{100.0 * (val - ref) / ref:+.1f} %" if (ref and val) else "-"
            print("| " + " | ".join(str(x) for x in [
                v["variant"], row.get("n"), row.get("cold_prompt_tok_s"), row.get("warm_prompt_tok_s"),
                row.get("cold_decode_tok_s"), row.get("warm_decode_tok_s"), delta,
                row.get("wait_cpu_ms_cold"), row.get("wait_cpu_ms_warm"), row.get("expert_hit_pct"),
                row.get("draft_accept_pct")]) + " |")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="strata-gyro-s.json", help="the model config to measure (default: %(default)s)")
    ap.add_argument("--variant", default="base", help="the name of this run (default: %(default)s)")
    ap.add_argument("--repeats", type=int, default=3, help="how often each workload runs (default: %(default)s)")
    ap.add_argument("--name", default="gyro-titanx", help="the name of the result folder (default: %(default)s)")
    ap.add_argument("--out", default=None, help="where the results go (default: bench/results/<date>-<name>)")
    ap.add_argument("--port", type=int, default=0, help="the port for the server (default: a free one)")
    ap.add_argument("--extra-arg", action="append", default=[], help="an extra engine argument (repeatable)")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="an engine env var (repeatable)")
    ap.add_argument("--workloads", default="prompt,decode", help="which workloads to run (default: %(default)s)")
    ap.add_argument("--thinking", action="store_true", help="let the model think (off by default: a fixed answer)")
    ap.add_argument("--ready-timeout", type=float, default=1800.0, help="how long the model may take to load")
    ap.add_argument("--cache-state", choices=["cold", "warm"], default="cold",
                    help="the page cache before the run: cold evicts the model's pages (default: %(default)s)")
    ap.add_argument("--evict-gib", type=int, default=16, help="how much to read through the page cache to evict it")
    ap.add_argument("--phase-env", action="store_true",
                    help="turn the engine's phase timers on (STRATA_DECODE_TIMING, STRATA_VERIFY_PROFILE); "
                         "they slow the decode a little, so use them to compare, not to measure speed")
    ap.add_argument("--no-warmup", dest="warmup", action="store_false", help="skip the warmup request")
    ap.add_argument("--compare", default=None, help="a result folder to compare against")
    args = ap.parse_args(argv)

    if args.compare:
        return compare(Path(args.compare))

    env = {}
    for item in args.env:
        k, _, v = item.partition("=")
        env[k.strip()] = v.strip()
    args.env = env

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg.setdefault("cwd", str(ROOT))

    outdir = Path(args.out) if args.out else ROOT / "bench" / "results" / f"{datetime.now().strftime('%Y-%m-%d')}-{args.name}"
    variant_dir = outdir / args.variant
    if variant_dir.exists():
        shutil.rmtree(variant_dir)
    result = run_variant(args, args.variant, cfg, variant_dir)
    print(json.dumps(result.get("summary", {}), indent=2))
    print(f"\nresults: {variant_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
