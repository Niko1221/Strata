#!/usr/bin/env python3
# strata_bench_rtx5080.py - one-command community benchmark and report for a Strata install
# on this Windows machine (RTX 5080 16 GB, Ryzen 7 7800X3D, 64 GB RAM, self-built CUDA 13.4 engine).
#
# What it does, in order:
#   1  preflight  - validates the install (config, engine exe, pack tokenizer, venv), refuses to
#                   start while another strata engine is running, snapshots hardware/environment,
#                   copies config + BUILD.json into the results folder, optionally hashes artifacts
#   2  server     - starts serve/server.py fresh on --port (default 18080), waits for /health loaded
#   3  sweep      - one warm-up request, then the speed sweep: 4,096 / 32,768 / 128,000 prompt
#                   tokens, --runs repetitions each; deterministic filler + unique nonce, streamed;
#                   TTFT measured client-side, engine timings read from /metrics after each request
#   4  telemetry  - samples RAM / VRAM / temperature / power every two seconds into telemetry.jsonl
#   5  needle     - runs tools/needle_bench.py (32k/128k x depths 10/50/90) into needles.json
#   6  report     - stops the server, writes summary.json and a README.md that follows the report
#                   template in docs/COMMUNITY_BENCHMARKS.md
#
# Usage (from this folder, with the install's venv python - start-benchmark.bat does this):
#   start-benchmark.bat                                              full run, results in .\results\
#   start-benchmark.bat --dry-run                                    checks only, no server, no data
#   start-benchmark.bat --targets 4096 --runs 1 --skip-needle --no-hash    quick smoke test
#
# See README-Anleitung.md (German) for the full walkthrough. The generated report is English on
# purpose: it is meant to be posted to the upstream repository as a community benchmark.
#
# Notes on how the numbers are produced (mirrors the wording rules of docs/COMMUNITY_BENCHMARKS.md):
#   prompt tok/s = freshly read prompt tokens / (prompt_ms / 1000), from the engine record
#   decode tok/s = engine_generated / (decode_ms / 1000), from the engine record
#   TTFT         = client-side seconds from sending the request to the first nonempty delta
#   never: generated tokens divided by the whole request time

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

SCRIPT_VERSION = "1.0.0"

# RAM floor for the safety guard: below this available host RAM the sweep stops between requests
RAM_FLOOR_MIB = 3072
# sampler period in seconds; slower than 1 Hz keeps nvidia-smi overhead out of the measurement
SAMPLE_INTERVAL_S = 2.0

# fixed parameters of the measurement; changing them changes comparability with existing reports
WARMUP_TEXT = "Reply with exactly the word READY."
WARMUP_MAX_TOKENS = 16
ENDING = ("\n\nWrite a detailed explanation of the code above. Discuss deterministic integer transforms, "
          "modulo arithmetic, testing, naming, complexity, and maintainability. Write at least 600 words.")


def log(message: str) -> None:
    print("[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), message), flush=True)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, progress=None) -> str:
    digest = hashlib.sha256()
    done = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            done += len(chunk)
            if progress is not None:
                progress(done)
    return digest.hexdigest()


def median_stats(values: list) -> dict:
    numbers = [v for v in values if v is not None]
    if not numbers:
        return {"median": None, "min": None, "max": None, "n": 0}
    return {"median": statistics.median(numbers), "min": min(numbers), "max": max(numbers), "n": len(numbers)}


def fmt_stats(values: list, digits: int = 1) -> str:
    s = median_stats(values)
    if s["n"] == 0:
        return "not measured"
    pattern = "{:,." + str(digits) + "f}"
    return (pattern + " [" + pattern + "\u2013" + pattern + "]").format(s["median"], s["min"], s["max"])


def fmt_int_stats(values: list) -> str:
    s = median_stats(values)
    if s["n"] == 0:
        return "not measured"
    return "{:,d} [{:,d}\u2013{:,d}]".format(round(s["median"]), s["min"], s["max"])


def thousands(number) -> str:
    return format(int(number), ",d")


# --------------------------------------------------------------------------------------------- #
# HTTP client                                                                                   #
# --------------------------------------------------------------------------------------------- #
class Client:
    def __init__(self, base: str, api_key: str = "", timeout: float = 30.0):
        self.base = base.rstrip("/")
        self.key = api_key
        self.timeout = timeout

    def headers(self) -> dict:
        head = {"Content-Type": "application/json"}
        if self.key:
            head["Authorization"] = "Bearer " + self.key
        return head

    def get(self, path: str, timeout: float | None = None):
        request = urllib.request.Request(self.base + path, headers=self.headers())
        with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def post_stream(self, path: str, body: dict, timeout: float):
        # returns (chunks, text_parts, ttft_seconds, total_seconds, finish_reason)
        request = urllib.request.Request(self.base + path, data=json.dumps(body).encode("utf-8"),
                                         headers=self.headers())
        chunks: list = []
        text_parts: list = []
        first_at = None
        finish = None
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if not line.startswith("data:"):
                    continue  # keep-alives, empty comment lines
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                if not payload:
                    continue
                chunk = json.loads(payload)
                chunks.append(chunk)
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                    if piece:
                        if first_at is None:
                            first_at = time.perf_counter() - started
                        text_parts.append(piece)
                    finish = choice.get("finish_reason") or finish
        total = time.perf_counter() - started
        return chunks, "".join(text_parts), first_at, total, finish


# --------------------------------------------------------------------------------------------- #
# telemetry sampler                                                                             #
# --------------------------------------------------------------------------------------------- #
class Sampler(threading.Thread):
    def __init__(self, interval: float, gpu_index: int, telemetry_path: Path):
        super().__init__(daemon=True)
        self.interval = interval
        self.gpu_index = gpu_index
        self.telemetry_path = telemetry_path
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.records: list = []
        self.low_ram = False
        self.started_at = None
        self.finished_at = None

    def gpu_sample(self):
        try:
            done = subprocess.run(
                ["nvidia-smi", "-i", str(self.gpu_index), "--query-gpu=memory.used,temperature.gpu,power.draw,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15)
            if done.returncode != 0:
                return None
            parts = [p.strip() for p in done.stdout.strip().splitlines()[0].split(",")]
            values = []
            for part in parts:
                try:
                    values.append(float(part))
                except ValueError:
                    values.append(None)
            return values
        except Exception:
            return None

    def sample_once(self) -> dict:
        import psutil  # imported here so a missing psutil fails in preflight, not in the thread
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        gpu = self.gpu_sample()
        row = {
            "t": round(time.time(), 2),
            "ram_available_mib": round(vm.available / (1024 * 1024), 1),
            "ram_percent": vm.percent,
            "swap_used_mib": round(swap.used / (1024 * 1024), 1),
            "vram_used_mib": gpu[0] if gpu else None,
            "gpu_temp_c": gpu[1] if gpu else None,
            "gpu_power_w": gpu[2] if gpu else None,
            "gpu_util_pct": gpu[3] if gpu else None,
        }
        with self.lock:
            self.records.append(row)
            if row["ram_available_mib"] < RAM_FLOOR_MIB:
                self.low_ram = True
        with self.telemetry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        return row

    def run(self) -> None:
        self.started_at = time.time()
        self.sample_once()
        while not self.stop_event.wait(self.interval):
            try:
                self.sample_once()
            except Exception as exc:  # sampling must never kill the benchmark
                log("sampler warning: %s" % exc)
        self.finished_at = time.time()

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=30)

    def mark(self) -> int:
        with self.lock:
            return len(self.records)

    def min_available_since(self, index: int):
        with self.lock:
            subset = [r["ram_available_mib"] for r in self.records[index:]]
        return min(subset) if subset else None

    def summary(self) -> dict:
        with self.lock:
            records = list(self.records)
        if not records:
            return {"samples": 0}
        def values(key):
            return [r[key] for r in records if r.get(key) is not None]
        first_swap = next((r["swap_used_mib"] for r in records), None)
        return {
            "samples": len(records),
            "span_s": round((self.finished_at or time.time()) - (self.started_at or time.time()), 1),
            "ram_available_mib": median_stats(values("ram_available_mib")),
            "ram_percent_max": max(values("ram_percent")) if values("ram_percent") else None,
            "swap_first_mib": first_swap,
            "swap_max_mib": max(values("swap_used_mib")) if values("swap_used_mib") else None,
            "vram_used_mib": median_stats(values("vram_used_mib")),
            "gpu_temp_c_max": max(values("gpu_temp_c")) if values("gpu_temp_c") else None,
            "gpu_power_w_max": max(values("gpu_power_w")) if values("gpu_power_w") else None,
        }


# --------------------------------------------------------------------------------------------- #
# nvidia-smi helpers                                                                            #
# --------------------------------------------------------------------------------------------- #
def nvidia_smi_static(gpu_index: int) -> dict:
    fields = ["name", "memory.total", "driver_version", "vbios_version",
              "pcie.link.gen.max", "pcie.link.width.max", "pcie.link.gen.current", "pcie.link.width.current",
              "power.limit", "power.default_limit", "power.max_limit", "clocks.max.sm", "clocks.max.mem",
              "clocks.current.sm", "temperature.gpu"]
    try:
        done = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_index), "--query-gpu=" + ",".join(fields),
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        if done.returncode != 0:
            return {"error": done.stderr.strip() or "nvidia-smi failed", "raw": done.stdout}
        parts = [p.strip() for p in done.stdout.strip().splitlines()[0].split(",")]
        return dict(zip(fields, parts))
    except Exception as exc:
        return {"error": str(exc)}


# --------------------------------------------------------------------------------------------- #
# preflight: environment, hardware, provenance                                                  #
# --------------------------------------------------------------------------------------------- #
def collect_environment() -> dict:
    import psutil
    cpu_name = None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        pass
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    release, win_version, _csd, _ptype = platform.win32_ver()
    build = win_version.split(".")[-1] if win_version else "?"
    os_name = "Windows %s (build %s)" % (release, build) if release else platform.platform()
    return {
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script_version": SCRIPT_VERSION,
        "os": os_name,
        "os_platform": platform.platform(),
        "os_version": win_version,
        "os_edition": getattr(platform, "win32_edition", lambda: None)() or None,
        "python": sys.version,
        "python_executable": sys.executable,
        "cpu_name": cpu_name,
        "cpu_logical": psutil.cpu_count(logical=True),
        "cpu_physical": psutil.cpu_count(logical=False),
        "ram_total_gib": round(vm.total / (1024 ** 3), 2),
        "swap_total_gib": round(swap.total / (1024 ** 3), 2),
        "hostname": platform.node(),
    }


def collect_processes() -> dict:
    import psutil
    notable = []
    for proc in psutil.process_iter(["name", "memory_info", "pid"]):
        try:
            info = proc.info
            rss = info["memory_info"].rss if info["memory_info"] else 0
            notable.append({"name": info["name"], "pid": info["pid"], "rss_mib": round(rss / (1024 * 1024), 1)})
        except Exception:
            continue
    notable.sort(key=lambda row: row["rss_mib"], reverse=True)
    return {"captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "process_count": len(notable),
            "top_by_rss_mib": notable[:15]}


def find_running_strata(exe_path: Path) -> list:
    # returns [pid, ...] of processes whose executable is the configured engine binary
    import psutil
    found = []
    wanted = str(exe_path).lower()
    for proc in psutil.process_iter(["pid", "name", "exe"]):
        try:
            exe = (proc.info.get("exe") or "").lower()
            if exe and exe == wanted:
                found.append(proc.info["pid"])
        except Exception:
            continue
    return found


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


# fixed ordering of the argument list we want in the report, so the text reads like the other reports
ARG_ORDER = ["--pack", "--native", "--ple-gguf", "--expert-profile", "--expert-cache", "--prefill",
             "--spec", "--mtp", "--max-context", "--kv", "--vision", "--vram-reserve-mib", "--pcie-frac",
             "--spec-min-p", "--pool-workers", "--adapt-every", "--adapt-swaps", "--adapt-decay",
             "--prompt-cache", "--conversation-cache-mib", "--conversation-cache-slots"]


def config_arg(cfg: dict, flag: str):
    args = cfg.get("args") or []
    if flag in args:
        index = args.index(flag)
        if index + 1 < len(args) and not str(args[index + 1]).startswith("--"):
            return args[index + 1]
        return True
    return None


def describe_config_args(cfg: dict) -> str:
    args = cfg.get("args") or []
    used = set()
    ordered = []
    for flag in ARG_ORDER:
        if flag in args:
            index = args.index(flag)
            value = args[index + 1] if index + 1 < len(args) and not str(args[index + 1]).startswith("--") else None
            ordered.append(flag if value is None else "%s %s" % (flag, value))
            used.add(index)
            if value is not None:
                used.add(index + 1)
    for index, item in enumerate(args):
        if index not in used:
            ordered.append(str(item))
    return " ".join(ordered)


def hash_one(path: Path, label: str, hashes: dict) -> None:
    started = time.time()
    size = path.stat().st_size if path.exists() else 0

    last_report = {"done": 0}

    def progress(done: int) -> None:
        if done - last_report["done"] >= 4 * 1024 * 1024 * 1024:
            last_report["done"] = done
            log("  hashing %s: %.1f / %.1f GiB" % (label, done / 2**30, size / 2**30))

    if not path.exists():
        hashes[label] = {"path": str(path), "error": "missing"}
        return
    digest = sha256_file(path, progress)
    hashes[label] = {"path": str(path), "bytes": size, "sha256": digest,
                     "seconds": round(time.time() - started, 1)}
    log("  hashed %s (%.2f GiB) in %.0f s" % (label, size / 2**30, time.time() - started))


def collect_provenance(install_root: Path, data_dir: Path, cfg: dict, out_dir: Path,
                       do_hash: bool, hash_only: bool) -> dict:
    native = config_arg(cfg, "--native")
    ple = config_arg(cfg, "--ple-gguf")
    pack = config_arg(cfg, "--pack")
    mtp = config_arg(cfg, "--mtp")
    profile = config_arg(cfg, "--expert-profile")
    mmproj = (cfg.get("vision") or {}).get("mmproj")
    engine_exe = cfg.get("exe")
    vision_exe = (cfg.get("vision") or {}).get("exe")

    # model repository + pinned revision: read from the installer's HF_REVISIONS table
    repo, revision = None, None
    setup_py = install_root / "setup.py"
    if setup_py.exists():
        text = setup_py.read_text(encoding="utf-8", errors="replace")
        match = re.search(r'"(ISTA-DASLab/[A-Za-z0-9._-]+GGUF)":\s*"([0-9a-f]{40})"', text)
        if match:
            repo, revision = match.group(1), match.group(2)

    provenance = {
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_repo": repo,
        "model_revision": revision,
        "native_gguf": str(native) if native else None,
        "ple_gguf": str(ple) if ple else None,
        "mmproj_gguf": str(mmproj) if mmproj else None,
        "pack_dir": str(pack) if pack else None,
        "mtp_dir": str(mtp) if mtp else None,
        "expert_profile": str(profile) if profile else None,
        "files": {},
    }

    # sizes always; sha256 only when asked
    items = []
    if native:
        items.append(("native", Path(str(native))))
    if ple:
        items.append(("ple", Path(str(ple))))
    if mmproj:
        items.append(("mmproj", Path(str(mmproj))))
    for label, path in items:
        provenance["files"].setdefault(label, {})
        if path.exists():
            provenance["files"][label]["path"] = str(path)
            provenance["files"][label]["bytes"] = path.stat().st_size
        else:
            provenance["files"][label] = {"path": str(path), "error": "missing"}

    # mtp manifest: copy the installer's per-tensor record next to the report
    mtp_manifest = data_dir / "mtp" / "mtp-manifest.json"
    if mtp_manifest.exists():
        try:
            if mtp_manifest.stat().st_size < 20 * 1024 * 1024:
                (out_dir / "mtp-manifest.json").write_bytes(mtp_manifest.read_bytes())
                provenance["mtp_manifest"] = "mtp-manifest.json"
            else:
                provenance["mtp_manifest"] = "exists in data dir, too large to copy (%d bytes)" % mtp_manifest.stat().st_size
        except Exception as exc:
            provenance["mtp_manifest"] = "copy failed: %s" % exc

    if do_hash or hash_only:
        log("hashing artifacts (this reads the files once; --no-hash skips it)")
        if native:
            hash_one(Path(str(native)), "native", provenance["files"])
        if ple:
            hash_one(Path(str(ple)), "ple", provenance["files"])
        if mmproj:
            hash_one(Path(str(mmproj)), "mmproj", provenance["files"])

        artifacts = {}
        for label, candidate in [("engine", engine_exe), ("vision_exe", vision_exe),
                                 ("expert_profile", profile), ("BUILD.json", str(install_root / "engine" / "BUILD.json"))]:
            if candidate:
                hash_one(Path(str(candidate)), label, artifacts)
        if pack:
            for file in sorted(Path(str(pack)).glob("*")):
                if file.is_file():
                    hash_one(file, "pack/" + file.name, artifacts)
        provenance["artifacts"] = artifacts

    # BUILD.json and the sanitized config travel with the report
    build_json = install_root / "engine" / "BUILD.json"
    if build_json.exists():
        (out_dir / "BUILD.json").write_bytes(build_json.read_bytes())
    config_copy = {k: v for k, v in cfg.items() if k not in ("api_key", "config_path")}
    write_json(out_dir / str(Path(cfg.get("config_path", "strata-iq3_xxs.json")).name), config_copy)
    return provenance


# --------------------------------------------------------------------------------------------- #
# tokenizer + prompt construction                                                               #
# --------------------------------------------------------------------------------------------- #
def load_tokenizer(install_root: Path, tokenizer_dir: Path):
    sys.path[:0] = [str(install_root), str(install_root / "tools")]
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate, openai_to_messages
    vocab = json.loads((tokenizer_dir / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for token, number in vocab.items():
        tokens[number] = token
    tokenizer = Tokenizer(tokens,
                          (tokenizer_dir / "merges.txt").read_text(encoding="utf-8").splitlines(),
                          json.loads((tokenizer_dir / "token_type.json").read_text(encoding="utf-8")))
    template = ChatTemplate(tokenizer_dir / "chat_template.jinja")
    return tokenizer, template, openai_to_messages


def make_body(content: str, max_tokens: int) -> dict:
    return {"model": "strata", "messages": [{"role": "user", "content": content}],
            "temperature": 0, "reasoning_effort": "none", "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}}


def count_tokens(tokenizer, template, openai_to_messages, body: dict) -> int:
    messages, tools, kwargs = openai_to_messages(body)
    return len(tokenizer.encode(template.render(messages, tools, **kwargs), parse_special=True))


def build_filler(lines: int) -> list:
    return ["def task_%05d(value: int) -> int: return (value * %d + %d) %% 100003" % (i, (i % 97) + 1, i)
            for i in range(lines)]


def fit_prompt(tokenizer, template, openai_to_messages, prefix: str, ending: str,
               filler: list, target: int, max_tokens: int, tolerance: int = 20):
    # binary search over the number of filler lines so the complete rendered chat prompt lands in
    # [target - tolerance, target]; returns (content, actual_tokens)
    lines = filler
    for _ in range(3):
        low, high = 0, len(lines)
        while low < high:
            middle = (low + high + 1) // 2
            content = prefix + "\n".join(lines[:middle]) + ending
            if count_tokens(tokenizer, template, openai_to_messages, make_body(content, max_tokens)) <= target:
                low = middle
            else:
                high = middle - 1
        content = prefix + "\n".join(lines[:low]) + ending
        actual = count_tokens(tokenizer, template, openai_to_messages, make_body(content, max_tokens))
        if target - tolerance <= actual <= target:
            return content, actual
        if low < len(lines):
            break  # overshoot with the next line; cannot land inside tolerance
        lines = lines * 2  # filler exhausted, grow and retry
    raise RuntimeError("could not build a %d-token prompt (last count %d)" % (target, actual))


# --------------------------------------------------------------------------------------------- #
# one measured request                                                                          #
# --------------------------------------------------------------------------------------------- #
def perform(client: Client, sampler: Sampler, out_dir: Path, rows: list, label: str, group: str,
            content: str, expected: int, max_tokens: int, timeout: float, target=None) -> dict:
    body = make_body(content, max_tokens)
    wire = json.dumps(body).encode("utf-8")
    (out_dir / (label + "-request.json")).write_bytes(wire + b"\n")
    row = {
        "label": label, "group": group, "target_prompt_tokens": target,
        "expected_prompt_tokens": expected, "request_bytes": len(wire),
        "request_sha256": hashlib.sha256(wire).hexdigest(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ok": False, "error": None, "error_type": None,
        "client_ttft_s": None, "client_total_s": None, "finish_reason": None,
        "prompt_tokens": None, "reused_tokens": None, "output_tokens": None,
        "prompt_ms": None, "decode_ms": None, "prompt_tok_s": None, "decode_tok_s": None,
        "drafts_offered": None, "drafts_accepted": None, "hit_rate": None,
        "available_ram_before_mib": None, "min_available_ram_mib": None, "available_ram_after_mib": None,
        "usage": None, "stream_timings": None, "engine": None, "text": None,
    }
    import psutil
    chunks: list = []
    mark = sampler.mark()
    row["available_ram_before_mib"] = round(psutil.virtual_memory().available / (1024 * 1024), 1)
    try:
        before = client.get("/metrics")
        before_requests = (before.get("totals") or {}).get("requests", 0)
        log("start %s (expected %s prompt tokens, max %d output)" %
            (label, thousands(expected) if expected else "?", max_tokens))
        chunks, text, ttft, total, finish = client.post_stream("/v1/chat/completions", body, timeout)

        # wait until the engine is idle and the request counter moved, then take its record
        metrics = None
        deadline = time.time() + 120
        while time.time() < deadline:
            metrics = client.get("/metrics")
            live = metrics.get("live") or {}
            totals = metrics.get("totals") or {}
            if live.get("state") in ("idle", "unloaded") and totals.get("requests", 0) > before_requests:
                break
            time.sleep(0.25)
        engine = (metrics or {}).get("requests", [{}])[0] if metrics else {}

        usage = next((chunk["usage"] for chunk in reversed(chunks) if chunk.get("usage")), None)
        timings = next((chunk["timings"] for chunk in reversed(chunks) if chunk.get("timings")), None)
        row.update({
            "ok": True, "client_ttft_s": ttft, "client_total_s": total, "finish_reason": finish,
            "prompt_tokens": engine.get("prompt_tokens"), "reused_tokens": engine.get("reused"),
            "output_tokens": engine.get("engine_generated"),
            "prompt_ms": engine.get("prompt_ms"), "decode_ms": engine.get("decode_ms"),
            "drafts_offered": engine.get("drafts_offered"), "drafts_accepted": engine.get("drafts_accepted"),
            "hit_rate": engine.get("hit_rate"),
            "usage": usage, "stream_timings": timings, "engine": engine, "text": text,
        })
        if engine.get("prompt_ms"):
            row["prompt_tok_s"] = (engine.get("prompt_tokens", 0) - (engine.get("reused") or 0)) * 1000.0 / engine["prompt_ms"]
        if engine.get("decode_ms"):
            row["decode_tok_s"] = engine.get("engine_generated", 0) * 1000.0 / engine["decode_ms"]

        # validation: the engine must have seen exactly the tokens we counted locally
        if expected is not None and engine.get("prompt_tokens") != expected:
            row["error_type"] = "token_count_mismatch"
            row["error"] = "engine saw %s prompt tokens, local count said %s" % (engine.get("prompt_tokens"), expected)
        if not text:
            row["error_type"] = "empty_output"
            row["error"] = "no nonempty text delta received"
        if engine.get("reused"):
            note = "request reused %s tokens" % engine.get("reused")
            row["error"] = note if row["error"] is None else row["error"] + "; " + note
    except Exception as exc:
        row["error_type"] = type(exc).__name__
        row["error"] = str(exc)
    finally:
        row["available_ram_after_mib"] = round(psutil.virtual_memory().available / (1024 * 1024), 1)
        row["min_available_ram_mib"] = sampler.min_available_since(mark)
    if row["error"] is not None:
        row["ok"] = False
    (out_dir / (label + "-raw.json")).write_text(
        json.dumps({"label": label, "request_sha256": row["request_sha256"], "chunks": chunks, "row": row},
                   indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    rows.append(row)
    write_json(out_dir / "results.json", rows)
    if row["ok"]:
        log("done  %s: ttft %.3f s, total %.3f s, prompt %s tok/s, decode %s tok/s, reused %s" % (
            label, row["client_ttft_s"] or 0, row["client_total_s"] or 0,
            "%.1f" % row["prompt_tok_s"] if row["prompt_tok_s"] else "?", 
            "%.1f" % row["decode_tok_s"] if row["decode_tok_s"] else "?",
            row["reused_tokens"]))
    else:
        log("FAIL  %s: %s" % (label, row["error"]))
    return row


# --------------------------------------------------------------------------------------------- #
# server process control                                                                        #
# --------------------------------------------------------------------------------------------- #
class Server:
    def __init__(self, python_exe: Path, install_root: Path, config: Path, port: int, log_path: Path):
        self.python_exe = python_exe
        self.install_root = install_root
        self.config = config
        self.port = port
        self.log_path = log_path
        self.proc = None
        self.log_handle = None
        self.load_seconds = None

    def command(self) -> list:
        return [str(self.python_exe), "serve/server.py", "--engine", "strata",
                "--config", str(self.config), "--host", "127.0.0.1", "--port", str(self.port)]

    def start(self) -> None:
        self.log_handle = self.log_path.open("w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(self.command(), cwd=str(self.install_root),
                                     stdout=self.log_handle, stderr=subprocess.STDOUT)

    def wait_loaded(self, client: Client, timeout: float) -> None:
        started = time.time()
        last_note = 0.0
        while time.time() - started < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited with code %s; log tail:\n%s" %
                                   (self.proc.returncode, self.log_tail()))
            try:
                health = client.get("/health", timeout=10)
                if health.get("loaded"):
                    self.load_seconds = time.time() - started
                    log("engine loaded after %.0f s (max_context %s)" % (self.load_seconds, health.get("max_context")))
                    return
            except Exception:
                pass
            if time.time() - last_note > 20:
                last_note = time.time()
                log("waiting for model load ... %.0f s" % (time.time() - started))
            time.sleep(2)
        raise RuntimeError("model did not load within %.0f s; log tail:\n%s" % (timeout, self.log_tail()))

    def log_tail(self, lines: int = 25) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(text[-lines:])
        except Exception:
            return "(no log)"

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            # taskkill /T kills the child engine too; TerminateProcess alone would orphan it
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                           capture_output=True, text=True)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                log("warning: server process did not exit")
        if self.log_handle:
            self.log_handle.close()
            self.log_handle = None
        self.proc = None


# --------------------------------------------------------------------------------------------- #
# needle check                                                                                  #
# --------------------------------------------------------------------------------------------- #
def run_needle(python_exe: Path, install_root: Path, base_url: str, out_dir: Path,
               lengths: str, depths: str, api_key: str = "") -> dict:
    command = [str(python_exe), "tools/needle_bench.py", "--url", base_url,
               "--lengths", lengths, "--depths", depths, "--out", str(out_dir / "needles.json")]
    display = list(command)
    if api_key:
        command += ["--api-key", api_key]
        display += ["--api-key", "***"]
    log("needle: %s" % " ".join(display))
    started = time.time()
    proc = subprocess.Popen(command, cwd=str(install_root), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    with (out_dir / "needle-stdout.log").open("w", encoding="utf-8") as handle:
        for line in proc.stdout:
            handle.write(line)
            print("  [needle] " + line.rstrip(), flush=True)
    code = proc.wait()
    log("needle finished with exit code %d after %.0f s" % (code, time.time() - started))
    return {"exit_code": code, "seconds": round(time.time() - started, 1)}


# --------------------------------------------------------------------------------------------- #
# summary                                                                                       #
# --------------------------------------------------------------------------------------------- #
def build_summary(rows: list, targets: list) -> dict:
    summary = {}
    for target in targets:
        subset = [r for r in rows if r["group"] == "speed" and r["target_prompt_tokens"] == target and r["ok"]]
        failed = [r for r in rows if r["group"] == "speed" and r["target_prompt_tokens"] == target and not r["ok"]]
        def values(key):
            return [r[key] for r in subset if r.get(key) is not None]
        summary[str(target)] = {
            "runs_total": len(subset) + len(failed),
            "runs_ok": len(subset),
            "prompt_tokens": median_stats(values("prompt_tokens")),
            "reused_tokens": median_stats(values("reused_tokens")),
            "output_tokens": median_stats(values("output_tokens")),
            "prompt_tok_s": median_stats(values("prompt_tok_s")),
            "decode_tok_s": median_stats(values("decode_tok_s")),
            "client_ttft_s": median_stats(values("client_ttft_s")),
            "client_total_s": median_stats(values("client_total_s")),
            "drafts_offered_sum": sum(v for v in values("drafts_offered") if v) if values("drafts_offered") else None,
            "drafts_accepted_sum": sum(v for v in values("drafts_accepted") if v) if values("drafts_accepted") else None,
            "hit_rate": median_stats(values("hit_rate")),
        }
    return summary


# --------------------------------------------------------------------------------------------- #
# report                                                                                        #
# --------------------------------------------------------------------------------------------- #
def build_readme(state: dict) -> str:
    env = state["environment"]
    hw = state["gpu"]
    prov = state["provenance"]
    cfg = state["config"]
    summary = state["summary"]
    telemetry = state["telemetry"]
    targets = state["targets"]
    date = state["date"]
    engine_version = state["engine_version"]
    quant = state.get("quant") or "IQ3_XXS"
    max_ctx = config_arg(cfg, "--max-context")
    max_ctx_text = thousands(int(max_ctx)) if max_ctx else "the configured"
    commit = state["git_commit"] or "(commit not recorded)"
    host = "a Windows desktop"
    if env.get("hostname"):
        host = "a Windows desktop called %s" % env["hostname"]
    lines: list = []
    add = lines.append

    # headline
    gpu_name = hw.get("name") or "NVIDIA GPU"
    add("# Community benchmark on %s (Windows)" % gpu_name)
    add("")
    add("Measured on %s by <your-github-handle>, on %s. This tests Strata %s (`%s`, "
        "locally built source, CUDA %s) with the original Flash-Next %s, one GPU, and a "
        "%s-token context limit." % (date, host, engine_version or "?", commit[:12],
                                      state["cuda_version"] or "?", quant, max_ctx_text))
    add("")
    decode_medians = [summary.get(str(t), {}).get("decode_tok_s", {}).get("median") for t in targets]
    prompt_medians = [summary.get(str(t), {}).get("prompt_tok_s", {}).get("median") for t in targets]
    if all(v is not None for v in decode_medians) and all(v is not None for v in prompt_medians):
        parts = ["{:,.1f} tok/s at {} prompt tokens".format(v, thousands(t)) for v, t in zip(decode_medians, targets)]
        decode_sentence = ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]
        add("Median decode throughput was **%s**; median prompt throughput was %s. These are synthetic "
            "code-explanation requests with greedy decoding and a 256-token output cap. They do not establish "
            "general answer quality or performance on other workloads." % (
                decode_sentence, " / ".join("{:,.1f}".format(v) for v in prompt_medians) + " tok/s"))
    else:
        add("Not every requested size produced a successful run; see the table below.")
    add("")

    # hardware and software
    add("## Hardware and software")
    add("")
    vram = hw.get("memory.total")
    vram_text = thousands(int(vram)) if vram and str(vram).isdigit() else (vram or "?")
    power_limit = hw.get("power.limit")
    power_default = hw.get("power.default_limit")
    add("- %s; %s MiB reported VRAM%s. PCIe capability: Gen %s, x%s (current link at idle: Gen %s x%s); "
        "GPU clocks were not fixed." % (
            gpu_name, vram_text, "; %.0f W power limit (default %.0f W)" % (float(power_limit), float(power_default))
            if power_limit and power_default else "",
            hw.get("pcie.link.gen.max") or "?", hw.get("pcie.link.width.max") or "?",
            hw.get("pcie.link.gen.current") or "?", hw.get("pcie.link.width.current") or "?"))
    smt_note = (" The OS reports no SMT/hyper-threading (logical equals physical count)."
                if env.get("cpu_logical") == env.get("cpu_physical") else "")
    add("- %s; %s logical CPUs (%s physical).%s" % (env.get("cpu_name") or "CPU", env.get("cpu_logical"),
                                                     env.get("cpu_physical"), smt_note))
    add("- %.2f GiB RAM reported by Windows; %.2f GiB system swap. Storage: <fill in: model storage device and "
        "type, for example a local NVMe SSD>." % (env.get("ram_total_gib", 0), env.get("swap_total_gib", 0)))
    add("- %s%s; NVIDIA driver %s." % (env.get("os") or "Windows",
                                        (" " + env["os_edition"]) if env.get("os_edition") else "",
                                        hw.get("driver_version") or "?"))
    archs = state.get("archs") or []
    arch_text = ", ".join("sm_%s" % a for a in archs) if archs else "not recorded"
    add("- Source: git `%s` at `%s`; engine %s, locally built (BUILD.json: source `%s`, src hash `%s`), CUDA "
        "architecture %s. See [BUILD.json](BUILD.json)."
        % (commit, state["install_root"], engine_version or "?", state.get("engine_source") or "?",
           (state.get("engine_src") or "not recorded")[:16], arch_text))
    add("- CUDA %s; Python %s. Installed Python dependencies are listed in [python-packages.txt](python-packages.txt)."
        % (state["cuda_version"] or "not recorded", env.get("python", "").split()[0]))
    add("- Background workloads at the start of the run are listed in [hardware-windows.json](hardware-windows.json) "
        "(processes by working set) and [environment.json](environment.json). <review: state what was left running>.")
    add("")

    # model and configuration
    add("## Model and configuration")
    add("")
    if prov.get("model_repo"):
        add("Model: `%s`, revision `%s` (the installer's pinned revision):" % (prov["model_repo"], prov["model_revision"]))
    else:
        add("Model: <fill in repository and revision>:")
    add("")
    for label in ("native", "ple", "mmproj"):
        entry = (prov.get("files") or {}).get(label) or {}
        if entry.get("path"):
            name = Path(entry["path"]).name
            if entry.get("sha256"):
                add("- `%s` (%s bytes; sha256 `%s`)" % (name, thousands(entry["bytes"]), entry["sha256"]))
            elif entry.get("bytes"):
                add("- `%s` (%s bytes; hash not computed)" % (name, thousands(entry["bytes"])))
    if any((prov.get("files") or {}).get(k, {}).get("sha256") for k in ("native", "ple", "mmproj")):
        add("")
        add("Local SHA-256 hashes were computed on this machine; all artifacts are in "
            "[model-provenance.json](model-provenance.json).")
    else:
        add("")
        add("SHA-256 hashes were **not** computed for this run (rerun with `--hash-only` to add them).")
    if prov.get("mtp_manifest"):
        add("MTP experts were fetched by the installer; per-tensor hashes are in [mtp-manifest.json](mtp-manifest.json).")
    add("")
    add("Configuration: the machine's daily-use IQ3_XXS config ([%s](%s)):"
        % (state["config_name"], state["config_name"]))
    add("")
    add("```text")
    add(describe_config_args(cfg))
    add("```")
    add("")
    add("Reasoning disabled, temperature 0, maximum 256 generated tokens per speed run. There was no calibration "
        "and no experimental speed projection; the bundled expert profile was used as-is.")
    add("")
    add("```text")
    add(" ".join(state["server_command"]))
    add("```")
    add("")

    # method
    add("## Method")
    add("")
    add("The attached [strata_bench_rtx5080.py](strata_bench_rtx5080.py) ran the whole sequence; per-run data is "
        "in [results.json](results.json), aggregates in [summary.json](summary.json). Server state snapshots before "
        "and after the sweep are in [initial-status.json](initial-status.json) and [final-status.json](final-status.json); "
        "the effective runtime properties are in [props.json](props.json).")
    add("")
    add("It generates a deterministic synthetic Python module, adds a different nonce near the start of every "
        "request so no prompt can reuse a prefix (nonces are before the filler, so earlier long prompts cannot be "
        "reused), and counts the complete rendered chat prompt with Strata's own tokenizer plus chat template. "
        "It adjusts the filler by binary search until the count lands in [target − 20, target] and verifies the "
        "count against the engine's record afterwards. One warm-up request (`Reply with exactly the word READY.`, "
        "16 tokens) preceded the sweep and is excluded from the tables.")
    add("")
    add("All speed runs of one engine load were executed serially in increasing-length order on the same loaded "
        "engine; no engine restart or OS page-cache reset separated them. The expert cache was filled during "
        "model load and adapted during the run (auto policy); it was not reset between runs. Loading time is "
        "excluded; the model load took %s seconds." % state["load_seconds_text"])
    add("")
    all_reused = [r["reused_tokens"] for r in state["rows"] if r["group"] == "speed" and r["ok"]]
    if all_reused and all(v == 0 for v in all_reused):
        add("All %d speed requests processed their entire prompt: **zero reused tokens**." % len(all_reused))
    else:
        reused_note = ", ".join("%s reused %s" % (r["label"], r["reused_tokens"])
                                for r in state["rows"] if r["group"] == "speed" and r["ok"] and r["reused_tokens"])
        add("Some speed requests reused prompt tokens: %s." % (reused_note or "none"))
    add("")
    add("TTFT is measured client-side from immediately before the HTTP request until the first nonempty text "
        "delta, ignoring empty role chunks and keep-alives. With reasoning off, that delta contains answer text. "
        "Total seconds end at stream completion; both include HTTP and frontend tokenization over loopback. "
        "Engine prompt throughput uses freshly read tokens divided by `prompt_ms`; decode throughput uses "
        "`engine_generated / decode_ms`. The JSON preserves the engine timing values and output text per run. "
        "Host RAM, GPU memory, temperature and board power were sampled every two seconds into "
        "[telemetry.jsonl](telemetry.jsonl); aggregates are in [resource-summary.json](resource-summary.json).")
    add("")

    # results
    add("## Results")
    add("")
    add("Each cell below is the median **[minimum\u2013maximum]** across the successful runs. Every request was "
        "capped at 256 generated tokens.")
    add("")
    add("| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range | Total seconds median and range |")
    add("| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |")
    for target in targets:
        block = summary.get(str(target)) or {}
        if block.get("runs_ok"):
            runs = str(block["runs_ok"])
            if block["runs_total"] > block["runs_ok"]:
                runs += " of %d (%d failed)" % (block["runs_total"], block["runs_total"] - block["runs_ok"])
            add("| %s, %s-token prompt | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                quant, thousands(target),
                fmt_int_stats([r["prompt_tokens"] for r in state["rows"] if r["group"] == "speed" and r["target_prompt_tokens"] == target and r["ok"]]),
                fmt_int_stats([r["reused_tokens"] for r in state["rows"] if r["group"] == "speed" and r["target_prompt_tokens"] == target and r["ok"]]),
                fmt_int_stats([r["output_tokens"] for r in state["rows"] if r["group"] == "speed" and r["target_prompt_tokens"] == target and r["ok"]]),
                runs,
                fmt_stats([r["prompt_tok_s"] for r in state["rows"] if r["group"] == "speed" and r["target_prompt_tokens"] == target and r["ok"]]),
                fmt_stats([r["decode_tok_s"] for r in state["rows"] if r["group"] == "speed" and r["target_prompt_tokens"] == target and r["ok"]]),
                fmt_stats([r["client_ttft_s"] for r in state["rows"] if r["group"] == "speed" and r["target_prompt_tokens"] == target and r["ok"]], 3),
                fmt_stats([r["client_total_s"] for r in state["rows"] if r["group"] == "speed" and r["target_prompt_tokens"] == target and r["ok"]], 3)))
        else:
            add("| %s, %s-token prompt | not measured | – | – | 0 of %d (all failed) | not measured | not measured | not measured | not measured |" % (
                quant, thousands(target), block.get("runs_total", 0)))
    add("")
    add("Raw per-run records: [results.json](results.json); the complete request bodies are the "
        "`tokens-*-request.json` files. Calculated aggregates: [summary.json](summary.json).")
    drafts_offered = sum(( (summary.get(str(t)) or {}).get("drafts_offered_sum") or 0) for t in targets)
    drafts_accepted = sum(((summary.get(str(t)) or {}).get("drafts_accepted_sum") or 0) for t in targets)
    if drafts_offered:
        add("")
        add("MTP draft acceptance across the speed runs: %s of %s proposed draft tokens accepted (%.1f%%)."
            % (thousands(drafts_accepted), thousands(drafts_offered), 100.0 * drafts_accepted / max(drafts_offered, 1)))
    hit_rates = [r["hit_rate"] for r in state["rows"]
                 if r["group"] == "speed" and r["ok"] and r.get("hit_rate") is not None]
    if hit_rates:
        add("")
        add("Decode expert-cache hit rates across the speed requests: %.1f%%\u2013%.1f%%."
            % (100 * min(hit_rates), 100 * max(hit_rates)))
    if telemetry.get("samples"):
        add("")
        add("[telemetry.jsonl](telemetry.jsonl) contains %d samples over %.0f seconds, starting before model load:"
            % (telemetry["samples"], telemetry.get("span_s") or 0))
        add("")
        if telemetry.get("vram_used_mib", {}).get("max") is not None:
            add("- GPU memory (system-wide on the selected GPU): peak **%.0f MiB**." % telemetry["vram_used_mib"]["max"])
        if telemetry.get("ram_available_mib", {}).get("min") is not None:
            add("- Host RAM: minimum available **%.2f GiB** (sampled; other workloads included)."
                % (telemetry["ram_available_mib"]["min"] / 1024))
        if telemetry.get("swap_first_mib") is not None and telemetry.get("swap_max_mib") is not None:
            add("- Swap in use: %.1f MiB at the first sample, peak %.1f MiB." % (telemetry["swap_first_mib"], telemetry["swap_max_mib"]))
        if telemetry.get("gpu_temp_c_max") is not None:
            add("- GPU temperature peak: %.0f \u00b0C; board power peak: %.0f W."
                % (telemetry["gpu_temp_c_max"], telemetry.get("gpu_power_w_max") or 0))
    failures = [r for r in state["rows"] if not r["ok"]]
    if failures:
        add("")
        add("Failures and skipped cases: " + "; ".join("%s: %s" % (r["label"], r["error_type"] or r["error"]) for r in failures))
    add("")

    # correctness and limitations
    add("## Correctness and limitations")
    add("")
    needle = state.get("needle") or {}
    entries = needle.get("entries") or []
    errors = needle.get("errors") or []
    if entries:
        found = sum(1 for e in entries if e.get("found"))
        add("The repository's unchanged `tools/needle_bench.py` found **%d of %d needles** across depths %s at "
            "lengths %s. Actual prompt lengths and per-case results are in [needles.json](needles.json)."
            % (found, len(entries) + len(errors), state["needle_depths"], state["needle_lengths"]))
        if errors:
            add("")
            add("Skipped or failed needle cases: " + "; ".join(json.dumps(e) for e in errors))
    elif needle.get("exit_code") is None:
        add("The needle check was skipped for this run (`--skip-needle`).")
    else:
        add("The needle check did not complete (exit code %s); see [needle-stdout.log](needle-stdout.log)."
            % needle.get("exit_code"))
    add("")
    add("This is one machine, one quantization, one configuration, and a small synthetic workload. Long output, "
        "sampled decoding, thinking, coding-task correctness, vision (the vision helper was enabled but these "
        "requests contained no images), tool use, multi-request concurrency, and a sustained thermal run were not "
        "evaluated. The built-in needle check measures recall on these inputs, not general model accuracy. No "
        "same-workload baseline on another engine or GPU was run; other reports' numbers are not a controlled "
        "comparison. The longest speed prompt does not fill the full %s-token context." % max_ctx_text)
    add("")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------- #
# main                                                                                          #
# --------------------------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Strata community benchmark for this machine: runs the speed sweep and writes a "
                    "report package for bench/results/ (format: docs/COMMUNITY_BENCHMARKS.md).")
    parser.add_argument("--install-root", default=r"D:\8_KI\Strata", help="Strata installation folder")
    parser.add_argument("--data-dir", default=r"D:\8_KI\Strata-data", help="Strata data folder (models, packs, mtp)")
    parser.add_argument("--config", default=None, help="engine config json (default: <install-root>/strata-iq3_xxs.json)")
    parser.add_argument("--port", type=int, default=18080, help="port for the benchmark server (default 18080)")
    parser.add_argument("--url", default=None, help="base URL to attach to (default http://127.0.0.1:<port>)")
    parser.add_argument("--attach", action="store_true", help="do not start/stop a server; use the one at --url")
    parser.add_argument("--keep-server", action="store_true", help="leave the started server running at the end")
    parser.add_argument("--out", default=None, help="output folder (default: <script folder>/results)")
    parser.add_argument("--targets", default="4096,32768,128000", help="prompt sizes in tokens, comma-separated")
    parser.add_argument("--runs", type=int, default=3, help="runs per prompt size (default 3)")
    parser.add_argument("--max-tokens", type=int, default=256, help="output cap per speed run (default 256)")
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds per chat request (default 1800)")
    parser.add_argument("--load-timeout", type=float, default=1200.0, help="seconds to wait for model load")
    parser.add_argument("--skip-needle", action="store_true", help="skip tools/needle_bench.py")
    parser.add_argument("--needle-lengths", default="32k,128k", help="needle lengths (default 32k,128k)")
    parser.add_argument("--needle-depths", default="10,50,90", help="needle depths in percent (default 10,50,90)")
    parser.add_argument("--no-hash", action="store_true", help="skip SHA-256 of model/engine artifacts")
    parser.add_argument("--hash-only", action="store_true", help="only hash artifacts and write provenance, then exit")
    parser.add_argument("--api-key", default=None, help="API key (default: $STRATA_API_KEY if set)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for nvidia-smi telemetry (default 0)")
    parser.add_argument("--dry-run", action="store_true", help="preflight checks only; no server, no data")
    parser.add_argument("--allow-running", action="store_true", help="proceed even if the engine is already running")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = parse_args(argv)
    script_dir = Path(__file__).resolve().parent
    install_root = Path(args.install_root)
    data_dir = Path(args.data_dir)
    config_path = Path(args.config) if args.config else install_root / "strata-iq3_xxs.json"
    out_dir = Path(args.out) if args.out else script_dir / "results"
    base_url = args.url or ("http://127.0.0.1:%d" % args.port)
    api_key = args.api_key if args.api_key is not None else os.environ.get("STRATA_API_KEY", "")
    targets = [int(t.strip()) for t in args.targets.split(",") if t.strip()]
    targets = sorted(set(targets))

    # -------- preflight checks -------- #
    log("preflight")
    if not install_root.is_dir():
        log("ERROR: install root not found: %s" % install_root); return 2
    if not config_path.is_file():
        log("ERROR: config not found: %s" % config_path); return 2
    cfg = read_json(config_path)
    cfg["config_path"] = str(config_path)
    if not args.attach:
        for name, module in [("psutil", "psutil"), ("regex", "regex"), ("jinja2", "jinja2")]:
            try:
                __import__(module)
            except ImportError:
                log("ERROR: this python lacks %s. Run with the install's venv python "
                    "(start-benchmark.bat does that): %s" % (name, install_root / ".venv" / "Scripts" / "python.exe"))
                return 2
    tokenizer_dir = Path(cfg.get("tokenizer") or "")
    for needed in [cfg.get("exe"), str(tokenizer_dir / "vocab.json"), str(tokenizer_dir / "chat_template.jinja")]:
        if not needed or not Path(str(needed)).exists():
            log("ERROR: missing %s (is the model prepared?)" % needed); return 2
    python_exe = install_root / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        log("ERROR: venv python not found: %s" % python_exe); return 2
    if not args.skip_needle and not (install_root / "tools" / "needle_bench.py").exists():
        log("ERROR: tools/needle_bench.py not found; use --skip-needle to run without it"); return 2

    gpu = nvidia_smi_static(args.gpu)
    environment = collect_environment()
    engine_build = read_json(install_root / "engine" / "BUILD.json") if (install_root / "engine" / "BUILD.json").exists() else {}
    cuda_dirs = engine_build.get("cuda_dirs") or []
    cuda_version = None
    for folder in cuda_dirs:
        match = re.search(r"v(\d+\.\d+)", str(folder))
        if match:
            cuda_version = match.group(1)
            break
    git_commit = None
    try:
        done = subprocess.run(["git", "-C", str(install_root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=20)
        if done.returncode == 0:
            git_commit = done.stdout.strip()
        dirty = subprocess.run(["git", "-C", str(install_root), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=30)
        if dirty.returncode == 0:
            # untracked files (??) do not change the engine sources; modified tracked files do
            git_dirty = any(not line.startswith("??") for line in dirty.stdout.splitlines())
        else:
            git_dirty = None
    except Exception:
        git_dirty = None
    if git_commit is None and not args.attach:
        log("note: git commit of the installation not readable; the report will say so")

    running = find_running_strata(Path(str(cfg.get("exe"))))
    if running and not args.attach and not args.allow_running:
        log("ERROR: the Strata engine is already running (PID %s). Close it first so the benchmark has "
            "the GPU and RAM to itself, or pass --allow-running." % ", ".join(map(str, running)))
        return 2
    if not args.attach and not port_is_free(args.port):
        log("ERROR: port %d is already in use. Pass --port with a free port." % args.port)
        return 2
    log("install: %s" % install_root)
    log("config:  %s" % config_path)
    log("gpu:     %s" % (gpu.get("name") or gpu.get("error")))
    log("engine:  %s (%s, archs %s, source %s)" % (engine_build.get("version"), git_commit or "?",
                                                   engine_build.get("archs"), engine_build.get("source")))
    if git_dirty:
        log("HINT: the installation has uncommitted changes; say so in the report")

    if args.dry_run:
        for target in targets:
            log("would measure %s prompt tokens x %d runs" % (thousands(target), args.runs))
        log("would run needle check: %s" % ("no" if args.skip_needle else "yes (%s x %s)" % (args.needle_lengths, args.needle_depths)))
        log("would hash artifacts: %s" % ("no" if args.no_hash else "yes"))
        log("would start: %s" % " ".join(Server(python_exe, install_root, config_path, args.port, out_dir / "server-stdout.log").command()))
        log("dry run complete - no server was started, nothing was written")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    client = Client(base_url, api_key=api_key, timeout=30)
    if args.attach:
        health = client.get("/health")
        log("attached to %s (model %s, loaded %s, max_context %s)" % (base_url, health.get("model"),
                                                                      health.get("loaded"), health.get("max_context")))

    # -------- artifacts and provenance -------- #
    write_json(out_dir / "environment.json", environment)
    write_json(out_dir / "hardware-windows.json", {"sampled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                   "nvidia_smi": gpu, "processes": collect_processes()})
    provenance = collect_provenance(install_root, data_dir, cfg, out_dir,
                                    do_hash=not args.no_hash, hash_only=args.hash_only)
    write_json(out_dir / "model-provenance.json", provenance)
    if args.hash_only:
        log("hash-only run complete; provenance written to %s" % (out_dir / "model-provenance.json"))
        return 0
    try:
        done = subprocess.run([str(python_exe), "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120)
        (out_dir / "python-packages.txt").write_text(done.stdout, encoding="utf-8")
    except Exception as exc:
        log("warning: pip freeze failed: %s" % exc)

    # -------- tokenizer -------- #
    log("loading tokenizer from %s" % tokenizer_dir)
    tokenizer, template, openai_to_messages = load_tokenizer(install_root, tokenizer_dir)
    filler = build_filler(12000)

    rows: list = []
    telemetry_path = out_dir / "telemetry.jsonl"
    if telemetry_path.exists():
        telemetry_path.unlink()
    sampler = Sampler(SAMPLE_INTERVAL_S, args.gpu, telemetry_path)

    # -------- server and measurement -------- #
    server = Server(python_exe, install_root, config_path, args.port, out_dir / "server-stdout.log")
    engine_log_path = Path(str(cfg.get("log"))) if cfg.get("log") else None
    engine_log_offset = engine_log_path.stat().st_size if engine_log_path and engine_log_path.exists() else 0
    needle_result = {"exit_code": None, "entries": [], "errors": []}
    summary: dict = {}
    telemetry_summary: dict = {}
    run_started = time.strftime("%Y-%m-%d %H:%M:%S")
    date = run_started[:10]
    exit_code = 0

    sampler.start()
    try:
        if not args.attach:
            log("starting server on port %d (log: %s)" % (args.port, out_dir / "server-stdout.log"))
            server.start()
            server.wait_loaded(client, args.load_timeout)
        write_json(out_dir / "initial-status.json", client.get("/v1/status"))
        try:
            write_json(out_dir / "props.json", client.get("/props"))
        except Exception as exc:
            log("warning: /props failed: %s" % exc)

        # -------- warm-up -------- #
        perform(client, sampler, out_dir, rows, "warmup", "warmup", WARMUP_TEXT, None,
                WARMUP_MAX_TOKENS, args.timeout)

        # -------- speed sweep -------- #
        for target in targets:
            if sampler.low_ram:
                log("stopping sweep: available RAM fell below the %d MiB floor" % RAM_FLOOR_MIB)
                break
            for run in range(1, args.runs + 1):
                if sampler.low_ram:
                    log("stopping sweep: available RAM fell below the %d MiB floor" % RAM_FLOOR_MIB)
                    break
                label = "tokens-%d-run-%d" % (target, run)
                log("fitting a %s-token prompt for %s (tokenizer search, a few seconds)" % (thousands(target), label))
                prefix = "Benchmark nonce: series-%d-trial-%d.\nReview this synthetic Python module:\n" % (target, run)
                content, actual = fit_prompt(tokenizer, template, openai_to_messages, prefix, ENDING,
                                             filler, target, args.max_tokens)
                perform(client, sampler, out_dir, rows, label, "speed", content, actual,
                        args.max_tokens, args.timeout, target=target)

        # -------- needle -------- #
        if not args.skip_needle:
            needle_result.update(run_needle(python_exe, install_root, base_url, out_dir,
                                            args.needle_lengths, args.needle_depths, api_key))
            needles_path = out_dir / "needles.json"
            if needles_path.exists():
                try:
                    data = read_json(needles_path)
                    needle_result["entries"] = [e for e in data if "found" in e]
                    needle_result["errors"] = [e for e in data if "found" not in e]
                except Exception as exc:
                    log("warning: needles.json unreadable: %s" % exc)

        # -------- finish -------- #
        summary = build_summary(rows, targets)
        write_json(out_dir / "summary.json", summary)
        write_json(out_dir / "final-status.json", client.get("/v1/status"))
    except KeyboardInterrupt:
        log("interrupted by user; partial results are kept")
        exit_code = 130
    except Exception as exc:
        log("ERROR: %s" % exc)
        exit_code = 1
    finally:
        if not args.attach and not args.keep_server and server.proc is not None:
            log("stopping server")
            server.stop()
            for pid in find_running_strata(Path(str(cfg.get("exe")))):
                log("killing leftover engine process %d" % pid)
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, text=True)
        sampler.stop()
        if engine_log_path and engine_log_path.exists():
            size = engine_log_path.stat().st_size
            offset = engine_log_offset if size >= engine_log_offset else 0
            with engine_log_path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(offset)
                section = handle.read()
            if len(section) > 5 * 1024 * 1024:
                head = "\n".join(section.splitlines()[:200])
                tail = "\n".join(section.splitlines()[-400:])
                section = head + "\n\n[... middle trimmed, see TRIMMED.md ...]\n\n" + tail
                (out_dir / "TRIMMED.md").write_text(
                    "# Engine log trimmed\n\nThe engine log section was larger than 5 MB and was cut to the first "
                    "200 and last 400 lines.\n", encoding="utf-8")
            (out_dir / "engine-log.txt").write_text(section, encoding="utf-8")
        telemetry_summary = sampler.summary()
        write_json(out_dir / "resource-summary.json", telemetry_summary)
        if not summary and rows:
            summary = build_summary(rows, targets)
            write_json(out_dir / "summary.json", summary)

    # -------- report -------- #
    state = {
        "environment": environment, "gpu": gpu, "provenance": provenance, "config": cfg,
        "config_name": config_path.name, "summary": summary, "telemetry": telemetry_summary,
        "targets": targets, "date": date, "rows": rows, "needle": needle_result,
        "needle_lengths": args.needle_lengths, "needle_depths": args.needle_depths,
        "engine_version": engine_build.get("version"), "archs": engine_build.get("archs"),
        "engine_source": engine_build.get("source"), "engine_src": engine_build.get("src"),
        "cuda_version": cuda_version, "git_commit": git_commit, "install_root": str(install_root),
        "server_command": server.command() if not args.attach else [base_url],
        "load_seconds_text": ("%.0f" % server.load_seconds) if server.load_seconds else "not measured",
        "quant": (cfg.get("model_name") or "iq3_xxs").split("-")[-1].upper(),
    }
    write_json(out_dir / "runmeta.json", {"script_version": SCRIPT_VERSION,
                                          "args": {k: v for k, v in vars(args).items() if k != "api_key"},
                                          "started_at": run_started, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                          "targets": targets, "runs": args.runs, "exit_code": exit_code})
    if exit_code == 0:
        (out_dir / "README.md").write_text(build_readme(state), encoding="utf-8")
        try:
            # the report links the script; ship the exact version that produced the numbers
            shutil.copyfile(Path(__file__).resolve(), out_dir / "strata_bench_rtx5080.py")
        except Exception as exc:
            log("warning: could not copy the script into the results folder: %s" % exc)
        log("")
        log("done. Report package in %s:" % out_dir)
        log("  README.md            <- the report to post (fill the <placeholders> first)")
        log("  results.json / summary.json / needles.json / telemetry.jsonl")
        log("See README-Anleitung.md for how to post it.")
    else:
        log("run incomplete (exit code %d). Partial data is kept in %s; no README.md was generated."
            % (exit_code, out_dir))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
