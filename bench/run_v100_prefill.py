#!/usr/bin/env python3
"""GENI prefill benchmark for the Strata V100 fork.

This module measures prefill latency and throughput with exact token ids. It speaks the
engine's own stdin/stdout serve protocol (src/program/generate.cpp, the --serve block). It
sends the token ids verbatim. Thus the requested prompt length equals the engine's prompt
length: no template, no probe, and no calibration.

The HTTP harness bench/run_v100_bench.py calibrates template overhead and token marginals
with extra probe requests. This module does not calibrate. Use this module for exact-length
arms.

PROTOCOL (generate.cpp --serve):
  in : GEN  <max_new> [key=val ...] <id,id,...>
       GENI <max_new> <embeddings_file> [key=val ...] <id,id,...>
       STOP / QUIT
  out: INFO key=value ...            (once, before READY)
       READY <max_context> stop
       RESUME <n>                    prompt tokens taken from a checkpoint (0 = none)
       PP <reached> <total> <ms> <fresh tok/s>   one per prompt chunk
       REUSED <n>                    the prompt is read
       T <token_id>                  one generated token
       DONE <generated> <prompt> <prompt_ms> <decode_ms> <finish> <accepted> <offered>
            <reused> [hits] [lookups] [ram blobs] [file blobs] [file MB] [prompt read]
       ERR <message>

Text-only requests use GENI with an empty embeddings file. The file holds 0 strata-vision
records. The GENI parser accepts it and keeps the identity M-RoPE table. Thus the --vision
reservation stays byte-identical to a production run without starting strata-vision. Use
--request-mode gen when the arm binary has no --vision.

REUSE. Every request is a distinct, seeded prompt. The first tokens carry a unique tag. The
engine starts with --prompt-cache 0, --prompt-cache-every 0, --prompt-cache-root 0 and
--conversation-cache-mib 0, and without any --conversation-cache-disk* flag. Thus resume is
structurally 0. Every row asserts reused == 0 and aborts on a violation.

Usage (baseline arm, five lengths, one repeat):
  python bench/run_v100_prefill.py --arm-label baseline \\
      --exe /tmp/strata-prefill-baseline --targets 2048,4096,8192,16384,32768 --repeats 1

Diagnostic arm (phase instrumentation; timings are NOT comparable to the clean arm):
  python bench/run_v100_prefill.py --arm-label baseline-timing \\
      --exe /tmp/strata-prefill-baseline --repeats 1 --env STRATA_PREFILL_TIMING=1
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import queue
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

# The only lengths this project measures (assignment: 2048,4096,8192,16384,32768).
ALLOWED_TARGETS = (2048, 4096, 8192, 16384, 32768)

IMAGE_PAD = 248056   # qwen4exp.ple.image_token_id (generate.cpp kImagePad)

# The live runtime settings the arms must share (strata-q2_0.json + the user service env).
REQUIRED_ARG_PAIRS = (
    ("--kv", "int8"),
    ("--max-context", "524288"),
    ("--rope-scaling", "yarn"),
    ("--rope-scale", "2"),
    ("--spec", "8"),
    ("--vram-reserve-mib", "700"),
    ("--layer-split", "20"),
    ("--prefill", "auto"),
    ("--expert-cache", "auto"),
)
REQUIRED_FLAGS = ("--vision", "--mtp", "--expert-profile")
REQUIRED_ENV = {"STRATA_EXPERT_PAIR": "1"}
# The user service's Environment= lines (runtime consistency; overridable with --env).
SERVICE_ENV = {"STRATA_ARENA_LOCK": "0", "STRATA_WATCHDOG_S": "300"}
# A/B diagnostic knobs that must never leak into a clean timing arm.
DIAGNOSTIC_ENV = ("STRATA_PREFILL_TIMING", "STRATA_TRACE", "STRATA_DECODE_TIMING", "STRATA_SPLIT_TIMING")

NVIDIA_FIELDS = (
    "index", "temperature.gpu", "clocks.current.sm", "power.draw", "enforced.power.limit",
    "clocks_throttle_reasons.active", "clocks_throttle_reasons.sw_thermal_slowdown",
    "clocks_throttle_reasons.hw_thermal_slowdown", "clocks_throttle_reasons.hw_slowdown",
    "clocks_throttle_reasons.sw_power_cap",
)
NVIDIA_MIN_FIELDS = ("index", "temperature.gpu", "clocks.current.sm", "power.draw")

WORDS = (
    "attention cache layer kernel tensor token sequence prefill decode batch stride block "
    "memory bandwidth latency throughput shard expert router projection residual norm rotary "
    "context window prompt sample greedy verify draft acceptance queue pipeline stream slot "
    "index table hash map reduce scan prefix suffix segment scratch barrier launch grid warp "
    "the model reads a prompt in chunks and writes the key value state into the cache while "
    "each layer attends over the positions it has already seen and then projects the result "
    "a deterministic transform maps an integer through a modulo and returns the remainder"
).split()


# --------------------------------------------------------------------------- PII
def make_scrubber() -> "callable":
    home = str(Path.home())
    user = os.environ.get("USER") or getpass.getuser()

    def scrub(value):
        if not isinstance(value, str):
            return value
        value = value.replace(home, "$HOME")
        if user:
            value = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(user) + r"(?![A-Za-z0-9_])", "$USER", value)
        return value

    return scrub


SCRUB = make_scrubber()


def sanitize(obj):
    """Recursively replace the home directory and user name in every recorded string."""
    if isinstance(obj, str):
        return SCRUB(obj)
    if isinstance(obj, dict):
        return {SCRUB(k) if isinstance(k, str) else k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- tokenizer
def load_tokenizer(directory: Path):
    from strata_tokenizer import Tokenizer
    vocab = json.loads((directory / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for token, number in vocab.items():
        tokens[number] = token
    merges = (directory / "merges.txt").read_text().splitlines()
    types = json.loads((directory / "token_type.json").read_text())
    return Tokenizer(tokens, merges, types)


def corpus_text(rng: random.Random, lines: int) -> str:
    out = []
    for i in range(lines):
        kind = rng.randrange(5)
        if kind == 0:
            out.append(f"def task_{i:05d}(value: int) -> int: return (value * {rng.randrange(1, 97)} "
                       f"+ {rng.randrange(0, 100003)}) % 100003")
        elif kind == 1:
            out.append(" ".join(rng.choice(WORDS) for _ in range(rng.randrange(9, 22))) + ".")
        elif kind == 2:
            out.append(f"// buffer {i:05d}: stride={rng.randrange(2, 64)} offset={rng.randrange(0, 4096)} "
                       f"flags=0x{rng.randrange(0, 1 << 32):08x} count={rng.randrange(1, 1 << 16)}")
        elif kind == 3:
            out.append(f"SELECT id, name, score FROM records WHERE score > {rng.randrange(0, 1000)} "
                       f"ORDER BY score DESC LIMIT {rng.randrange(1, 50)};")
        else:
            out.append(f"Stage {i:05d}: the layer attends over {rng.randrange(1, 4096)} positions, "
                       f"reduces {rng.randrange(1, 64)} heads and writes {rng.randrange(1, 128)} rows "
                       f"into slot {rng.randrange(0, 1 << 20)} with weight {rng.random():.6f}.")
    return "\n".join(out) + "\n"


def build_pool(tokenizer, seed: int, need: int, cache_path: Path | None) -> list[int]:
    """A deterministic pool of realistic text token ids, at least `need` long.

    The pool is a prefix of one deterministic corpus stream, so the same (seed, need) always
    yields the same ids.  A cache written for a DIFFERENT need is rejected: accepting a longer
    pool would move every prompt slice and break the cross-arm prompt-hash parity."""
    if cache_path is not None and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text())
            if (cached.get("seed") == seed and cached.get("need") == need
                    and len(cached.get("ids", [])) >= need):
                return cached["ids"]
        except (OSError, ValueError):
            pass
    rng = random.Random(seed)
    ids: list[int] = []
    lines = 64
    while len(ids) < need:
        ids.extend(tokenizer.encode(corpus_text(rng, lines), parse_special=False))
        lines = max(lines, int(lines * 1.5))
    if cache_path is not None:
        try:
            cache_path.write_text(json.dumps({"seed": seed, "need": need, "ids": ids}))
        except OSError:
            pass
    return ids


def build_prompt(tokenizer, pool: list[int], seed: int, target: int, repeat: int) -> list[int]:
    """Exactly `target` realistic token ids, identical for the same (seed, target, repeat) across arms."""
    tag = f"[bench {seed:08x} t{target} r{repeat}] "
    prefix = tokenizer.encode(tag, parse_special=False)
    if len(prefix) >= target:
        raise SystemExit(f"target {target} is too small for the request tag")
    span = len(pool) - target
    if span < 0:
        raise SystemExit(f"token pool is shorter than target {target}")
    h = int.from_bytes(hashlib.sha256(f"{seed}:{target}:{repeat}".encode()).digest()[:8], "big")
    start = h % span
    body = pool[start:start + (target - len(prefix))]
    ids = prefix + body
    if len(ids) != target:
        raise SystemExit(f"prompt build produced {len(ids)} ids, wanted {target}")
    if IMAGE_PAD in ids:   # GENI reserves this id for image cells; natural text must never produce it
        raise SystemExit(f"prompt for target {target} contains the image-pad token {IMAGE_PAD}")
    return ids


def ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def file_sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


# --------------------------------------------------------------------------- telemetry
class GpuSampler:
    """1 Hz nvidia-smi sampler: temperature, SM clock, power, power limit, thermal slowdown."""

    def __init__(self, gpus: list[int], interval: float, csv_path: Path):
        self.gpus = [str(g) for g in gpus]
        self.interval = interval
        self.csv_path = csv_path
        self.samples: list[dict] = []
        self.errors: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fields = NVIDIA_FIELDS
        self._csv = None

    def start(self):
        self._csv = self.csv_path.open("w", encoding="utf-8")
        self._csv.write("epoch_s,mono_s,gpu,temp_c,sm_mhz,power_w,power_limit_w,"
                        "throttle_active,sw_thermal,hw_thermal,hw_slowdown,sw_power_cap\n")
        self._csv.flush()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self._csv is not None:
            self._csv.close()

    def _query(self) -> list[str] | None:
        for fields in (self._fields, NVIDIA_MIN_FIELDS):
            cmd = ["nvidia-smi", "-i", ",".join(self.gpus),
                   "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"]
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            except (OSError, subprocess.TimeoutExpired) as e:
                self.errors.append(f"nvidia-smi failed: {e}")
                return None
            if out.returncode == 0:
                self._fields = tuple(fields)
                return [ln for ln in out.stdout.splitlines() if ln.strip()]
            self.errors.append(f"nvidia-smi rc={out.returncode}: {out.stderr.strip()[:200]}")
        return None

    @staticmethod
    def _num(value: str):
        value = value.strip()
        if value in ("", "N/A", "[N/A]", "Not Active", "Active"):
            return value if value in ("Not Active", "Active") else None
        try:
            return float(value)
        except ValueError:
            return None

    def _sample_once(self):
        rows = self._query()
        if rows is None:
            return
        now = time.monotonic()
        epoch = time.time()
        with self._lock:
            for row in rows:
                parts = [p.strip() for p in row.split(",")]
                if len(parts) < 4:
                    continue
                gpu = parts[0]
                temp, sm, power = self._num(parts[1]), self._num(parts[2]), self._num(parts[3])
                limit = self._num(parts[4]) if len(parts) > 4 else None
                active = parts[5] if len(parts) > 5 else ""
                sw_thermal = parts[6] if len(parts) > 6 else ""
                hw_thermal = parts[7] if len(parts) > 7 else ""
                hw_slow = parts[8] if len(parts) > 8 else ""
                sw_cap = parts[9] if len(parts) > 9 else ""
                sample = {"t": now, "epoch": epoch, "gpu": gpu, "temp_c": temp, "sm_mhz": sm,
                          "power_w": power, "power_limit_w": limit, "throttle_active": active,
                          "sw_thermal": sw_thermal, "hw_thermal": hw_thermal,
                          "hw_slowdown": hw_slow, "sw_power_cap": sw_cap}
                self.samples.append(sample)
                self._csv.write(f"{epoch:.3f},{now:.3f},{gpu},{temp},{sm},{power},{limit},"
                                f"{active},{sw_thermal},{hw_thermal},{hw_slow},{sw_cap}\n")
            self._csv.flush()

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._sample_once()
            except Exception as e:                       # a sampler must never kill a run
                self.errors.append(f"sample error: {e}")
            self._stop.wait(self.interval)

    def latest_temps(self) -> dict[str, float] | None:
        with self._lock:
            if not self.samples:
                return None
            last = {}
            for s in reversed(self.samples):
                if s["gpu"] not in last:
                    last[s["gpu"]] = s
                if len(last) == len(self.gpus):
                    break
            temps = {}
            for g, s in last.items():
                if s["temp_c"] is None:
                    return None
                temps[g] = float(s["temp_c"])
            return temps

    def stats(self, t0: float, t1: float) -> dict:
        with self._lock:
            picked = [s for s in self.samples if t0 <= s["t"] <= t1]
        out: dict[str, dict] = {}
        for gpu in self.gpus:
            rows = [s for s in picked if s["gpu"] == gpu]
            if not rows:
                out[gpu] = {"samples": 0}
                continue
            temps = [s["temp_c"] for s in rows if s["temp_c"] is not None]
            powers = [s["power_w"] for s in rows if s["power_w"] is not None]
            clocks = [s["sm_mhz"] for s in rows if s["sm_mhz"] is not None]
            thermal = [s for s in rows if s["sw_thermal"] == "Active" or s["hw_thermal"] == "Active"]
            out[gpu] = {
                "samples": len(rows),
                "min_temp_c": min(temps) if temps else None,
                "max_temp_c": max(temps) if temps else None,
                "mean_temp_c": round(statistics.fmean(temps), 2) if temps else None,
                "max_power_w": max(powers) if powers else None,
                "max_sm_mhz": max(clocks) if clocks else None,
                "thermal_slowdown_samples": len(thermal),
                "max_throttle_mask": max((s["throttle_active"] for s in rows), default=""),
            }
        return out


def wait_cooldown(sampler: GpuSampler, target_c: float, stable_s: float, max_s: float, poll_s: float) -> dict:
    """Wait until every card is <= target_c continuously for stable_s; bounded by max_s."""
    t0 = time.monotonic()
    stable_since = None
    last_temps = None
    while True:
        temps = sampler.latest_temps()
        now = time.monotonic()
        if temps is None:
            if now - t0 >= max_s:
                return {"met": False, "waited_s": round(now - t0, 1), "temps_c": None,
                        "reason": "no telemetry"}
            time.sleep(poll_s)
            continue
        last_temps = temps
        ok = all(t <= target_c for t in temps.values())
        if ok:
            if stable_since is None:
                stable_since = now
            if now - stable_since >= stable_s:
                return {"met": True, "waited_s": round(now - t0, 1), "temps_c": temps}
        else:
            stable_since = None
        if now - t0 >= max_s:
            return {"met": False, "waited_s": round(now - t0, 1), "temps_c": last_temps,
                    "reason": "bounded wait exhausted"}
        time.sleep(poll_s)


# --------------------------------------------------------------------------- engine io
class EngineIO:
    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self.q: "queue.Queue[str | None]" = queue.Queue()
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()

    def _pump(self):
        try:
            for line in self.proc.stdout:
                self.q.put(line.rstrip("\n"))
        except Exception:
            pass
        finally:
            self.q.put(None)

    def next_line(self, timeout: float) -> str:
        try:
            item = self.q.get(timeout=max(0.1, timeout))
        except queue.Empty:
            raise TimeoutError("no protocol line within the timeout")
        if item is None:
            raise EOFError("the engine closed its stdout")
        return item


def wait_ready(io: EngineIO, timeout: float, on_line) -> dict:
    info = {}
    t0 = time.monotonic()
    while True:
        line = io.next_line(timeout - (time.monotonic() - t0))
        on_line(line)
        if line.startswith("INFO "):
            for kv in line.split()[1:]:
                k, _, v = kv.partition("=")
                info[k] = int(v) if v.lstrip("-").isdigit() else v
        elif line.startswith("READY"):
            return info
        elif line.startswith("ERR "):
            raise RuntimeError(f"engine error during startup: {line[4:]}")


def run_request(io: EngineIO, request: str, timeout: float, on_line) -> dict:
    io.proc.stdin.write(request + "\n")
    io.proc.stdin.flush()
    ev = {"resume": None, "reused": None, "pp": [], "tokens": [], "done": None, "err": None}
    deadline = time.monotonic() + timeout
    while True:
        line = io.next_line(deadline - time.monotonic())
        on_line(line)
        if line.startswith("T "):
            ev["tokens"].append(int(line[2:]))
        elif line.startswith("DONE "):
            ev["done"] = line.split()[1:]
            return ev
        elif line.startswith("ERR "):
            ev["err"] = line[4:]
            return ev
        elif line.startswith("RESUME "):
            ev["resume"] = int(line.split()[1])
        elif line.startswith("REUSED "):
            ev["reused"] = int(line.split()[1])
        elif line.startswith("PP "):
            ev["pp"].append(line.split()[1:])


def quit_engine(proc: subprocess.Popen, io: EngineIO, logf) -> None:
    try:
        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
    except (BrokenPipeError, ValueError, OSError):
        pass
    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    finally:
        try:
            logf.close()
        except OSError:
            pass


# --------------------------------------------------------------------------- config/argv
def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    if not isinstance(cfg.get("args"), list):
        raise SystemExit(f"{path}: no args list")
    return cfg


def drop_flag(args: list[str], *flags: str) -> list[str]:
    """Remove `flags` (and the value of each, unless it was written as --flag=value) from an argv list."""
    out: list[str] = []
    skip = False
    for i, a in enumerate(args):
        if skip:
            skip = False
            continue
        if a.split("=", 1)[0] in flags:
            if "=" not in a and i + 1 < len(args):
                skip = True
            continue
        out.append(a)
    return out


def flag_value(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


def engine_argv(cfg: dict, exe: str, layer_split: str | None = None) -> list[str]:
    """The server's own construction (serve/server.py engine_args) plus the no-reuse flags.

    `layer_split` overrides the config's split for a diagnostic arm; None keeps the config's
    fixed setting (20 on this machine).  Disk offload is always removed and conversation
    reuse is always disabled."""
    args = list(cfg["args"])
    gpus = cfg.get("gpu") or [0]
    if len(gpus) > 1 and "--layer-split" not in args:
        args += ["--layer-split", str(cfg.get("layer_split") or "auto")]
    if layer_split is not None:
        args = drop_flag(args, "--layer-split")
        args += ["--layer-split", str(layer_split)]
    args = drop_flag(args, "--conversation-cache-disk", "--conversation-cache-disk-gib",
                     "--conversation-cache-disk-slots", "--conversation-cache-disk-min-free-mib")
    # no conversation reuse: no checkpoints, no RAM parking, no disk tier
    args += ["--prompt-cache", "0", "--prompt-cache-every", "0", "--prompt-cache-root", "0",
             "--conversation-cache-mib", "0"]
    return [exe, "--serve", *args]


def engine_env(cfg: dict, gpus: list[int], overrides: list[str], drop_diagnostics: bool) -> dict:
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    for k, v in SERVICE_ENV.items():
        env.setdefault(k, v)
    for k, v in (cfg.get("env") or {}).items():
        env[str(k)] = str(v)
    lib_dirs = [d for d in (cfg.get("lib_dirs") or []) if Path(d).is_dir()]
    if lib_dirs:
        var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[var] = os.pathsep.join(lib_dirs + ([env[var]] if env.get(var) else []))
    if drop_diagnostics:
        for k in DIAGNOSTIC_ENV:
            env.pop(k, None)
    applied = {}
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--env wants KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        env[k] = v
        applied[k] = v
    env["_HARNESS_OVERRIDES"] = json.dumps(applied)   # recorded, harmless to the engine
    return env


def settings_check(argv: list[str], env: dict, expected_split: str = "20") -> dict:
    checks: dict[str, object] = {}
    for flag, want in REQUIRED_ARG_PAIRS:
        want = expected_split if flag == "--layer-split" else want
        checks[f"{flag} {want}"] = (flag_value(argv, flag) == want)
    for flag in REQUIRED_FLAGS:
        checks[flag] = flag in argv
    for k, v in REQUIRED_ENV.items():
        checks[f"env {k}={v}"] = env.get(k) == v
    checks["no conversation disk"] = not any(a.startswith("--conversation-cache-disk") for a in argv)
    checks["prompt-cache 0"] = argv.count("--prompt-cache") == 1 and flag_value(argv, "--prompt-cache") == "0"
    checks["conversation-cache-mib 0"] = flag_value(argv, "--conversation-cache-mib") == "0"
    return {"ok": all(checks.values()), "checks": checks}


# --------------------------------------------------------------------------- compare
def compare_results(a_path: Path, b_path: Path) -> dict:
    a, b = json.loads(a_path.read_text()), json.loads(b_path.read_text())
    out = {"a": {"label": a.get("arm_label"), "exe": a.get("build", {}).get("engine_sha256")},
           "b": {"label": b.get("arm_label"), "exe": b.get("build", {}).get("engine_sha256")},
           "targets": {}}
    for target in sorted({r["target"] for r in a.get("rows", [])} & {r["target"] for r in b.get("rows", [])}):
        ra = sorted((r for r in a["rows"] if r["target"] == target and not r.get("error")), key=lambda r: r["repeat"])
        rb = sorted((r for r in b["rows"] if r["target"] == target and not r.get("error")), key=lambda r: r["repeat"])
        if not ra or not rb:
            continue
        pairs_a = [(r["repeat"], r["prompt_sha256"]) for r in ra]
        pairs_b = [(r["repeat"], r["prompt_sha256"]) for r in rb]
        if pairs_a != pairs_b:
            raise ValueError(f"unpaired prompts at {target} tokens: repeats and prompt hashes must match")
        ma = statistics.median(r["prompt_ms"] for r in ra)
        mb = statistics.median(r["prompt_ms"] for r in rb)
        ta = statistics.median(r["prompt_tps"] for r in ra)
        tb = statistics.median(r["prompt_tps"] for r in rb)
        parity = [r["output_ids"] for r in ra] == [r["output_ids"] for r in rb]
        out["targets"][str(target)] = {
            "a_prompt_ms_median": ma, "b_prompt_ms_median": mb,
            "a_prompt_tps_median": ta, "b_prompt_tps_median": tb,
            "prompt_ms_change_pct": round(100.0 * (mb - ma) / ma, 2) if ma else None,
            "prompt_tps_change_pct": round(100.0 * (tb - ta) / ta, 2) if ta else None,
            "output_parity": parity,
        }
    return out


# --------------------------------------------------------------------------- main
def parse_envs(values: list[str]) -> list[str]:
    return list(values or [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "strata-q2_0.json")
    ap.add_argument("--exe", default=None, help="arm engine binary (default: the config's exe)")
    ap.add_argument("--arm-label", default="arm")
    ap.add_argument("--targets", default="2048,4096,8192,16384,32768",
                    help="comma-separated subset of 2048,4096,8192,16384,32768")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--max-new", type=int, default=1, help="greedy output tokens per request")
    ap.add_argument("--request-mode", choices=("geni", "gen"), default="geni")
    ap.add_argument("--layer-split", default=None,
                    help="diagnostic override of the config's layer split (default: the config's 20)")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="runtime env override for A/B (repeatable)")
    ap.add_argument("--expect-sha256", default=None, help="assert the arm binary's sha256")
    ap.add_argument("--out", type=Path, default=REPO / "bench/results/2026-10-03-int8-prefill-direct")
    ap.add_argument("--sample-interval-s", type=float, default=1.0)
    ap.add_argument("--cooldown-c", type=float, default=55.0)
    ap.add_argument("--cooldown-stable-s", type=float, default=10.0)
    ap.add_argument("--cooldown-max-s", type=float, default=900.0)
    ap.add_argument("--cooldown-poll-s", type=float, default=1.0)
    ap.add_argument("--no-cooldown", action="store_true")
    ap.add_argument("--startup-timeout-s", type=float, default=2400.0)
    ap.add_argument("--request-timeout-s", type=float, default=3600.0)
    ap.add_argument("--no-settings-strict", action="store_true")
    ap.add_argument("--allow-other-engine", action="store_true")
    ap.add_argument("--keep-pool", action="store_true", help="cache the token pool JSON in --out")
    ap.add_argument("--rebuild-pool", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build prompts and print the plan; start no engine")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("A.json", "B.json"))
    args = ap.parse_args()

    if args.compare:
        print(json.dumps(sanitize(compare_results(*args.compare)), indent=2))
        return 0

    targets = [int(t) for t in args.targets.split(",") if t.strip()]
    bad = [t for t in targets if t not in ALLOWED_TARGETS]
    if bad or not targets:
        raise SystemExit(f"--targets must be a subset of {ALLOWED_TARGETS}; got {bad or 'nothing'}")
    if args.repeats < 1 or args.max_new < 1:
        raise SystemExit("--repeats and --max-new must be >= 1")

    cfg = load_config(args.config)
    exe = str(args.exe or cfg["exe"])
    gpus = [int(g) for g in (cfg.get("gpu") or [0])]
    argv = engine_argv(cfg, exe, args.layer_split)
    env = engine_env(cfg, gpus, parse_envs(args.env), drop_diagnostics=True)
    check = settings_check(argv, env, args.layer_split or str(cfg.get("layer_split") or "20"))

    tokenizer_dir = Path(cfg.get("tokenizer") or (Path(cfg["args"][cfg["args"].index("--pack") + 1]) / "tokenizer"))
    tokenizer = load_tokenizer(tokenizer_dir)

    args.out.mkdir(parents=True, exist_ok=True)   # the pool cache and the dry-run CSV live here
    need = max(targets) * args.repeats + 4096
    cache_path = None if args.rebuild_pool else (args.out / f"token-pool-seed{args.seed}.json")
    if not args.keep_pool:
        cache_path = None
    pool = build_pool(tokenizer, args.seed, need, cache_path)

    plan = []
    for target in targets:
        for repeat in range(1, args.repeats + 1):
            ids = build_prompt(tokenizer, pool, args.seed, target, repeat)
            plan.append({"target": target, "repeat": repeat, "len": len(ids),
                         "sha256": ids_sha256(ids), "first_ids": ids[:12], "last_ids": ids[-4:]})

    exe_sha = file_sha256(Path(exe))
    print(f"[harness] arm={args.arm_label} exe={SCRUB(exe)} sha256={exe_sha}", file=sys.stderr)
    print(f"[harness] pool={len(pool)} tokens; {len(plan)} prompts; "
          f"settings_ok={check['ok']}", file=sys.stderr)
    if not check["ok"]:
        for name, ok in check["checks"].items():
            if not ok:
                print(f"[harness] SETTINGS MISMATCH: {name}", file=sys.stderr)
    if args.expect_sha256 and exe_sha != args.expect_sha256:
        raise SystemExit(f"engine sha256 {exe_sha} != --expect-sha256 {args.expect_sha256}")

    if args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
        sampler = GpuSampler(gpus, args.sample_interval_s, args.out / f"gpu-{args.arm_label}-dry.csv")
        sampler.start()
        time.sleep(1.2)
        sampler.stop()
        dry = {
            "arm_label": args.arm_label, "argv": sanitize(argv), "env_overrides": parse_envs(args.env),
            "service_env": SERVICE_ENV, "settings_check": check,
            "prompts": plan, "pool_tokens": len(pool), "tokenizer_dir": sanitize(str(tokenizer_dir)),
            "telemetry_sample": sanitize(sampler.samples[:2]), "telemetry_errors": sampler.errors,
        }
        print(json.dumps(dry, indent=2))
        return 0

    if not args.no_settings_strict and not check["ok"]:
        raise SystemExit("runtime settings do not match the required arm settings; fix or pass --no-settings-strict")
    if shutil.which("nvidia-smi") is None:
        raise SystemExit("nvidia-smi is required for the thermal gate and telemetry")

    args.out.mkdir(parents=True, exist_ok=True)
    label = args.arm_label
    log_path = args.out / f"engine-{label}.log"
    protocol_path = args.out / f"protocol-{label}.jsonl"
    gpu_csv = args.out / f"gpu-{label}.csv"
    emb_path = (args.out / "geni-empty.sve").resolve()
    emb_path.write_bytes(b"")

    protocol_f = protocol_path.open("a", encoding="utf-8")

    def on_line(line: str):
        protocol_f.write(json.dumps({"t": round(time.time(), 3), "line": line}) + "\n")
        protocol_f.flush()

    result: dict = {
        "protocol_version": 1,
        "harness": "bench/run_v100_prefill.py",
        "harness_sha256": file_sha256(Path(__file__)),
        "started_epoch_s": time.time(),
        "arm_label": label,
        "build": {
            "engine_path": SCRUB(exe), "engine_sha256": exe_sha,
            "engine_version": None, "git_commit": None, "git_describe": None,
            "gpu_names": [], "driver_version": None, "python": sys.version.split()[0],
        },
        "runtime": {
            "argv": sanitize(argv), "cwd": SCRUB(str(cfg.get("cwd") or REPO)),
            "env_overrides": parse_envs(args.env), "service_env": SERVICE_ENV,
            "config_env": sanitize(cfg.get("env") or {}),
            "request_mode": args.request_mode, "max_new": args.max_new,
            "layer_split": args.layer_split or str(cfg.get("layer_split") or "20"),
            "embeddings_file": SCRUB(str(emb_path)), "embeddings_sha256": file_sha256(emb_path),
            "prompt_cache": 0, "conversation_cache_mib": 0, "disk_offload": False,
            "sample_interval_s": args.sample_interval_s,
            "cooldown": {"target_c": args.cooldown_c, "stable_s": args.cooldown_stable_s,
                         "max_s": args.cooldown_max_s, "enabled": not args.no_cooldown},
        },
        "settings_check": check,
        "prompt_plan": {"seed": args.seed, "targets": targets, "repeats": args.repeats,
                        "pool_tokens": len(pool), "prompts": plan},
        "engine_info": {},
        "startup_wall_s": None,
        "rows": [],
        "summary": {},
        "errors": [],
        "status": "ok",
    }

    try:
        version = json.loads((Path(exe).parent / "BUILD.json").read_text()).get("version")
        result["build"]["engine_version"] = version
    except (OSError, ValueError):
        pass
    for key, cmd in (("git_commit", ["git", "rev-parse", "HEAD"]),
                     ("git_describe", ["git", "describe", "--tags", "--always"])):
        try:
            result["build"][key] = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                                                  timeout=20).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        names, drivers = [], set()
        for row in q.stdout.splitlines():
            if "," in row:
                n, d = row.split(",", 1)
                names.append(n.strip())
                drivers.add(d.strip())
        result["build"]["gpu_names"] = names
        result["build"]["driver_version"] = ",".join(sorted(drivers)) or None
    except (OSError, subprocess.TimeoutExpired):
        pass

    if not args.allow_other_engine:
        try:
            ps = subprocess.run(["pgrep", "-af", "strata --serve"], capture_output=True, text=True, timeout=10)
            others = [ln for ln in ps.stdout.splitlines() if ln.strip() and str(os.getpid()) not in ln]
            if others:
                print(f"[harness] WARNING: another serve engine looks alive:\n  " +
                      "\n  ".join(others), file=sys.stderr)
        except (OSError, subprocess.TimeoutExpired):
            pass

    sampler = GpuSampler(gpus, args.sample_interval_s, gpu_csv)
    sampler.start()
    proc = None
    logf = None
    io = None
    try:
        t_spawn = time.monotonic()
        logf = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(argv, cwd=cfg.get("cwd") or str(REPO), env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=logf, text=True, encoding="utf-8", bufsize=1)
        io = EngineIO(proc)
        info = wait_ready(io, args.startup_timeout_s, on_line)
        result["startup_wall_s"] = round(time.monotonic() - t_spawn, 1)
        result["engine_info"] = info
        print(f"[harness] READY after {result['startup_wall_s']}s; context={info.get('context')} "
              f"kv={info.get('kv')} disk={info.get('conversation_cache_disk')}", file=sys.stderr)
        if str(info.get("conversation_cache_disk")) not in ("0", "None"):
            raise SystemExit("engine reports a live conversation disk tier; refusing to measure")

        seq = 0
        for target in targets:
            for repeat in range(1, args.repeats + 1):
                seq += 1
                ids = build_prompt(tokenizer, pool, args.seed, target, repeat)
                row = {"seq": seq, "target": target, "repeat": repeat, "prompt_sha256": ids_sha256(ids),
                       "error": None}
                if not args.no_cooldown:
                    gate = wait_cooldown(sampler, args.cooldown_c, args.cooldown_stable_s,
                                         args.cooldown_max_s, args.cooldown_poll_s)
                else:
                    gate = {"met": None, "waited_s": 0.0, "temps_c": None, "reason": "disabled"}
                row["cooldown"] = sanitize(gate)
                if gate["met"] is False:
                    print(f"[harness] WARNING: cooldown not met for {target}x{repeat}: {gate}", file=sys.stderr)
                csv_ids = ",".join(map(str, ids))
                # temperature=0 is the engine default; written out so the protocol log proves greediness
                if args.request_mode == "geni":
                    request = f"GENI {args.max_new} temperature=0 {emb_path} {csv_ids}"
                else:
                    request = f"GEN {args.max_new} temperature=0 {csv_ids}"
                t0 = time.monotonic()
                try:
                    ev = run_request(io, request, args.request_timeout_s, on_line)
                except (TimeoutError, EOFError, OSError) as e:
                    row["error"] = f"{type(e).__name__}: {e}"
                    row["request_wall_s"] = round(time.monotonic() - t0, 3)
                    row["gpu_stats"] = sanitize(sampler.stats(t0, time.monotonic()))
                    result["rows"].append(row)
                    result["errors"].append(f"row {seq} ({target}x{repeat}): {row['error']}")
                    break
                t1 = time.monotonic()
                row["request_wall_s"] = round(t1 - t0, 3)
                row["output_ids"] = ev["tokens"]
                row["resume"] = ev["resume"]
                row["reused"] = ev["reused"]
                row["pp"] = ev["pp"]
                row["gpu_stats"] = sanitize(sampler.stats(t0, t1))
                if ev["err"]:
                    row["error"] = ev["err"]
                    result["errors"].append(f"row {seq} ({target}x{repeat}): {ev['err']}")
                elif ev["done"] is None:
                    row["error"] = "no DONE line"
                    result["errors"].append(f"row {seq} ({target}x{repeat}): no DONE line")
                else:
                    d = ev["done"]
                    row["generated"] = int(d[0])
                    row["prompt_tokens"] = int(d[1])
                    row["prompt_ms"] = float(d[2])
                    row["decode_ms"] = float(d[3])
                    row["finish"] = d[4]
                    row["draft_accepted"] = int(d[5])
                    row["draft_offered"] = int(d[6])
                    row["done_reused"] = int(d[7])
                    row["prompt_read"] = int(d[13]) if len(d) > 13 else None
                    fresh = row["prompt_tokens"] - row["done_reused"]
                    row["prompt_tps"] = round(1000.0 * fresh / row["prompt_ms"], 2) if row["prompt_ms"] else None
                    if row["prompt_tokens"] != target:
                        row["error"] = f"engine prompt_tokens {row['prompt_tokens']} != requested {target}"
                    if row["done_reused"] != 0 or (row["resume"] or 0) != 0 or (row["reused"] or 0) != 0:
                        row["error"] = f"conversation reuse detected (resume={row['resume']} reused={row['reused']} done_reused={row['done_reused']})"
                    if row["generated"] < 1:
                        row["error"] = "no generated token"
                if row["error"]:
                    result["errors"].append(f"row {seq} ({target}x{repeat}): {row['error']}")
                result["rows"].append(row)
                print(f"[harness] {target} r{repeat}: prompt={row.get('prompt_tokens')} "
                      f"reused={row.get('done_reused')} prompt_ms={row.get('prompt_ms')} "
                      f"tps={row.get('prompt_tps')} tok0={row['output_ids'][:1]} err={row['error']}", file=sys.stderr)

        # summary per target over the clean rows
        for target in targets:
            rows = [r for r in result["rows"] if r["target"] == target and not r["error"]]
            if not rows:
                result["summary"][str(target)] = {"repeats": 0}
                continue
            pms = [r["prompt_ms"] for r in rows]
            tps = [r["prompt_tps"] for r in rows]
            result["summary"][str(target)] = {
                "repeats": len(rows),
                "prompt_tokens": rows[0]["prompt_tokens"],
                "prompt_ms_all": pms,
                "prompt_ms_median": statistics.median(pms),
                "prompt_tps_all": tps,
                "prompt_tps_median": statistics.median(tps),
                "output_first_ids": [r["output_ids"][:1] for r in rows],
            }
    except (TimeoutError, EOFError, RuntimeError, OSError) as e:
        result["errors"].append(f"fatal: {type(e).__name__}: {e}")
        result["status"] = "error"
    except SystemExit as e:
        result["errors"].append(f"fatal: {e}")
        result["status"] = "error"
    finally:
        if io is not None and proc is not None and proc.poll() is None:
            quit_engine(proc, io, logf)
        elif logf is not None:
            try:
                logf.close()
            except OSError:
                pass
        sampler.stop()
        protocol_f.close()
        result["finished_epoch_s"] = time.time()
        if result["errors"] and result["status"] == "ok":
            result["status"] = "partial" if result["rows"] else "error"
        result["sampler_errors"] = sampler.errors
        result_path = args.out / f"result-{label}.json"
        result_path.write_text(json.dumps(sanitize(result), indent=2) + "\n")
        (args.out / f"summary-{label}.json").write_text(json.dumps(sanitize(result["summary"]), indent=2) + "\n")
        print(f"[harness] wrote {SCRUB(str(result_path))}", file=sys.stderr)

    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
