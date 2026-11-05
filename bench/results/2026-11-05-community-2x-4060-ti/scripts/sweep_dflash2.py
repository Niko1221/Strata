"""Confidence-by-window sweep for DFlash2 versus MTP on the dual-4060-Ti rig.

Serial GPU pipeline: one engine at a time on 127.0.0.1:18310. Each arm starts
from a base config, pins --draft-conf / --spec through user_args, self-checks
the engine's effective command line from the log, sends a warm-up then 2
rounds x 3 tasks, then unloads. Production 18300 and production profiles are
never touched.

Phases:
  prepare   write the per-arm resolved configs (no engine started)
  arm NAME  start server, wait ready, self-check, warm-up + 6 requests, unload
  summarize print the results table from all arms' saved JSON
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = Path("REPO_DIR_PLACEHOLDER")
ENGINE = Path(r"ENGINE_DIR")
CONFIGS = Path(r"DEV_DATA")
OUT = Path(r"DEV_DATA")
PORT = 18310
BASE_URL = f"http://127.0.0.1:{PORT}"
PY = sys.executable

ARMS = {
    "mtp":        {"drafting": "mtp",     "user_args": ["--spec", "4", "--spec-min-p", "0.5"]},
    "df2-c35-s6": {"drafting": "dflash2", "user_args": ["--spec", "6", "--draft-conf", "0.35"]},
    "df2-c50-s6": {"drafting": "dflash2", "user_args": ["--spec", "6", "--draft-conf", "0.5"]},
    "df2-c70-s6": {"drafting": "dflash2", "user_args": ["--spec", "6", "--draft-conf", "0.7"]},
    "df2-c35-s4": {"drafting": "dflash2", "user_args": ["--spec", "4", "--draft-conf", "0.35"]},
    "df2-c35-s8": {"drafting": "dflash2", "user_args": ["--spec", "8", "--draft-conf", "0.35"]},
}

PROMPTS = {
    "code": ("Write Python function merge_sorted(a, b) that merges two sorted lists. "
             "Include concise assertions for empty lists, duplicates, and negative numbers. Do not explain.", 256),
    "fiction": ("写一个中文悬疑短篇的开头，钟表铺老板发现每个失踪者的手表都比真实时间快七分钟。只写故事，不解释。", 384),
    "reasoning": ("Four non-preemptive FIFO jobs A,B,C,D arrive at 0,2,3,8. Durations 7,4,6,3. "
                  "Explain briefly, then return their completion times separated by commas.", 256),
}

# The engine warns below 1024 MiB free on the fuller card (#2065): WDDM pages
# VRAM to RAM and a window may never finish. Raising the reserves keeps the
# decode path out of that regime at the cost of a slightly smaller expert cache.
VRAM_RESERVE_MIB = 2048
VRAM_RESERVE_LATER_MIB = 1536


def load_base(name: str) -> dict:
    cfg = json.loads((CONFIGS / name).read_text(encoding="utf-8-sig"))
    if cfg.get("port") != PORT:
        raise SystemExit(f"{name}: port must be {PORT}")
    cfg["exe"] = str(ENGINE)
    cfg["open_browser"] = False
    return cfg


def resolve(arm: str) -> Path:
    spec = ARMS[arm]
    cfg = load_base("sc117-mtp-dev.json" if spec["drafting"] == "mtp" else "sc117-dflash2-dev.json")
    cfg["drafting"] = spec["drafting"]
    cfg["user_args"] = list(spec["user_args"])
    # one parked conversation per request would accumulate (#2065 wedge);
    # the engine reads the FLAG in args, not the dict key, so patch the flag
    args = cfg["args"]
    if "--conversation-cache-slots" in args:
        args[args.index("--conversation-cache-slots") + 1] = "2"
    else:
        args += ["--conversation-cache-slots", "2"]
    for flag, value in (("--vram-reserve-mib", VRAM_RESERVE_MIB),
                        ("--vram-reserve-later-mib", VRAM_RESERVE_LATER_MIB)):
        if flag in args:
            args[args.index(flag) + 1] = str(value)
        else:
            args += [flag, str(value)]
    folder = OUT / arm
    folder.mkdir(parents=True, exist_ok=True)
    cfg["log"] = str(folder / "engine.log")
    path = folder / "resolved.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def health() -> dict | None:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def wait_ready(timeout_s: float = 420.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = health()
        if state and state.get("loaded"):
            return
        if state is None and _process is not None and _process.poll() is not None:
            raise SystemExit("server exited before readiness")
        time.sleep(3)
    raise SystemExit("server not ready in time")


_process: subprocess.Popen | None = None


def subprocess_stopped() -> bool:
    return _process is not None and _process.poll() is not None


def self_check(arm: str, log_text: str) -> None:
    starts = re.findall(r"engine started: (.+)$", log_text, re.M)
    if not starts:
        raise SystemExit(f"{arm}: no engine-started line in log")
    command = starts[-1]           # the log is append-mode: the LAST start is this arm's
    wanted = ARMS[arm]["user_args"]
    for i in range(0, len(wanted), 2):
        flag, value = wanted[i], wanted[i + 1]
        pattern = re.escape(flag) + r"\s+" + re.escape(value)
        if not re.search(pattern, command):
            raise SystemExit(f"{arm}: effective engine args lack {flag} {value}:\n{command}")
    if "--vram-reserve-mib " + str(VRAM_RESERVE_MIB) not in command:
        raise SystemExit(f"{arm}: vram reserve not applied:\n{command}")
    if ARMS[arm]["drafting"] == "dflash2" and "--draft dflash2" not in command:
        raise SystemExit(f"{arm}: DFlash2 not in effective args:\n{command}")
    if ARMS[arm]["drafting"] == "mtp" and "--mtp " not in command and "--mtp" not in command:
        raise SystemExit(f"{arm}: MTP not in effective args:\n{command}")
    if "--conversation-cache-slots 2" not in command:
        raise SystemExit(f"{arm}: conversation slots not applied:\n{command}")
    (OUT / arm / "selfcheck.txt").write_text(command, encoding="utf-8")


def request(task: str, round_no: int, seed: int) -> dict:
    prompt, cap = PROMPTS[task]
    body = {
        "model": "qwen3.8-flash-next",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": cap,
        "temperature": 0,
        "seed": seed,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{BASE_URL}/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    for attempt in (1, 2, 3):
        try:
            with urllib.request.urlopen(req, timeout=900) as response:
                payload = json.loads(response.read())
            return {"task": task, "round": round_no, "seed": seed,
                    "usage": payload.get("usage"), "timings": payload.get("timings"),
                    "finish": payload.get("choices", [{}])[0].get("finish_reason")}
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == 3:
                return {"task": task, "round": round_no, "seed": seed,
                        "error": f"{type(exc).__name__}: {exc}"}
            time.sleep(20 * attempt)


def unload() -> None:
    req = urllib.request.Request(f"{BASE_URL}/unload", data=b"{}",
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as response:
        json.loads(response.read())


def stall_watch(arm: str) -> None:
    """Abort the whole PYTHON PROCESS (os._exit) when the live request is wedged
    in the #2065 WDDM regime: the engine keeps state=generating with a frozen
    counter. Returning errors up the chain once let a retry re-send into a dead
    engine; os._exit ends the arm for good and the shell loop moves on."""
    try:
        before = urllib.request.urlopen(f"{BASE_URL}/metrics", timeout=10).read()
        generated_1 = json.loads(before).get("live", {}).get("generated")
        time.sleep(20)
        after = urllib.request.urlopen(f"{BASE_URL}/metrics", timeout=10).read()
        live = json.loads(after).get("live", {})
        generated_2 = live.get("generated")
        if (generated_1 is not None and generated_2 is not None
                and live.get("state") == "generating" and generated_2 == generated_1):
            print(f"{arm}: request stalled (generated {generated_2} frozen for 20 s) - aborting arm", flush=True)
            # flush partial rows so the data survives
            import os as _os
            _os._exit(3)
    except urllib.error.URLError:
        pass  # between requests: no live request to watch


def run_arm(arm: str) -> None:
    global _process
    path = resolve(arm)
    log_path = OUT / arm / "server-stdout.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    command = [PY, str(REPO / "serve" / "server.py"), "--engine", "strata",
               "--config", str(path), "--host", "127.0.0.1", "--port", str(PORT),
               "--no-api-key"]
    with log_path.open("w", encoding="utf-8") as log_file:
        _process = subprocess.Popen(command, cwd=str(REPO), env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace")
        try:
            wait_ready()
            time.sleep(2)
            # the engine's own log has the authoritative start line
            engine_log = Path(json.loads(path.read_text(encoding="utf-8-sig"))["log"])
            for _ in range(60):
                if engine_log.is_file() and "engine started:" in engine_log.read_text(encoding="utf-8", errors="replace"):
                    break
                time.sleep(1)
            self_check(arm, engine_log.read_text(encoding="utf-8", errors="replace"))
            # a previous run's partial results are the rows' provenance, not this arm's
            (OUT / arm / "results.json").unlink(missing_ok=True)
            request("code", 0, 99)  # warm-up, excluded from results
            rows = []
            for round_no in (1, 2):
                for task in PROMPTS:
                    stall_watch(arm)
                    record = request(task, round_no, round_no)
                    rows.append(record)
                    # write after every request: a later stall must not lose
                    # the measurements already made
                    (OUT / arm / "results.json").write_text(
                        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
                    time.sleep(2)
            unload()
        finally:
            if _process.poll() is None:
                _process.terminate()
                try:
                    _process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    _process.kill()
            _process = None
    time.sleep(5)


def summarize() -> None:
    print("| arm | task | decode tok/s mean (r1, r2) | offered | accepted | acc% | finish |")
    print("|---|---|---:|---:|---:|---:|---|")
    for arm in ARMS:
        path = OUT / arm / "results.json"
        if not path.is_file():
            continue
        for task in PROMPTS:
            task_rows = [r for r in json.loads(path.read_text(encoding="utf-8")) if r["task"] == task and "timings" in r]
            errored = [r for r in json.loads(path.read_text(encoding="utf-8")) if r["task"] == task and "error" in r]
            if not task_rows and not errored:
                continue
            rates = [r["timings"]["predicted_per_second"] for r in task_rows]
            offered = sum(r["timings"].get("draft_n") or 0 for r in task_rows)
            accepted = sum(r["timings"].get("draft_n_accepted") or 0 for r in task_rows)
            finishes = ",".join(str(r["finish"]) for r in task_rows)
            if errored:
                print(f"| {arm} | {task} | ERROR x{len(errored)} | - | - | - | {finishes} |")
                continue
            mean = sum(rates) / len(rates)
            parts = ",".join(str(rate) for rate in rates)
            acc = round(100.0 * accepted / offered, 1) if offered else None
            print(f"| {arm} | {task} | {mean:.1f} ({parts}) | {offered} | {accepted} | {acc} | {finishes} |")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "arm", "summarize"])
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.phase == "prepare":
        for arm in ARMS:
            print("resolved:", resolve(arm))
        return 0
    if args.phase == "arm":
        if not args.arm:
            raise SystemExit("--arm is required for phase arm")
        run_arm(args.arm)
        return 0
    summarize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
