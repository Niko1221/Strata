#!/usr/bin/env python3
"""Stop sushi, serve Strata on a Strata-only machine, and run the batch gates autonomously.

The box cannot host sushi's model and Strata at once (kernel panic from GPU-wired memory
contention, 2026-10-20).  This script sequences them: it SIGTERMs `sushi serve`, waits for
the memory to come back, writes a batch-test run config (arena 28 GB, ctx 64k - the sizes
the last crash analysis measured safe), starts `serve.server` with STRATA_BD_DEBUG=1 and
STRATA_STATE_HASH=1 so the engine logs the per-window batch traces, then runs
tools/batch_parity_check.py and tools/subagent_concurrency_harness.py.

The server is left RUNNING on both pass and fail (pass: ready to take the omp session;
fail: so the next probe does not pay the model load again).  The captured trace slice and
both exit codes print at the end.  Exit 0 = every gate green.  Standard library only.

    python3 tools/strata_batch_validation.py [--keep-sushi] [--skip-harness]
"""
import argparse, json, os, re, signal, subprocess, sys, time, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = "/opt/homebrew/opt/python@3.13/bin/python3.13"
BASE = "http://127.0.0.1:8180"
CONFIG_IN = REPO / "strata-iq3_xxs.json"
CONFIG_OUT = REPO / "strata-iq3-batchtest.json"
SERVER_LOG = Path("/tmp/batchtest_server.log")


def free_pct() -> float:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage:\s*(\d+)", out)
    return int(m.group(1)) if m else 0.0


def stop_sushi() -> None:
    pids = [int(p) for p in subprocess.run(["pgrep", "-f", "sushi serve"],
                                           capture_output=True, text=True).stdout.split()]
    if not pids:
        print("[gate] sushi is not running; nothing to stop")
        return
    for stage, sig, wait_s in (("SIGTERM", signal.SIGTERM, 45), ("SIGINT", signal.SIGINT, 30),
                               ("SIGKILL", signal.SIGKILL, 20)):
        print(f"[gate] {stage} sushi {pids}", flush=True)
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.time() + wait_s
        while time.time() < deadline:
            if not subprocess.run(["pgrep", "-f", "sushi serve"], capture_output=True, text=True).stdout.split():
                print(f"[gate] sushi stopped (after {stage})")
                return
            time.sleep(2)
    print("[gate] sushi survived TERM/INT/KILL - aborting (panic risk)")
    sys.exit(3)


def wait_memory(need_pct: float) -> None:
    deadline = time.time() + 180
    while time.time() < deadline:
        pct = free_pct()
        print(f"[gate] memory free {pct}%")
        if pct >= need_pct:
            return
        time.sleep(5)
    print("[gate] wired memory never came back — refusing to load Strata (panic guard)")
    sys.exit(3)


def write_config() -> dict:
    cfg = json.loads(CONFIG_IN.read_text())
    cfg["env"]["STRATA_METAL_ARENA_GB"] = "22"        # leaves room for the kept weight copies (below)
    cfg["env"]["STRATA_METAL_KEEP_HOST_WEIGHTS"] = "1"   # bg_6: a prompt copy touched an 1075.9 MB device
    cfg["log"] = str(REPO / "strata-iq3_xxs-batchtest.log")  # buffer that had been given back; keep them
    args = list(cfg["args"])
    for i, a in enumerate(args):
        if a == "--max-context":
            args[i + 1] = "65536"
    cfg["args"] = args
    CONFIG_OUT.write_text(json.dumps(cfg, indent=1))
    return cfg


def health() -> bool:
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def models_ready() -> bool:
    """/health answers while the engine still loads; /v1/models with data means it is READY."""
    try:
        with urllib.request.urlopen(BASE + "/v1/models", timeout=5) as r:
            return bool(json.load(r).get("data"))
    except Exception:
        return False


def start_server(cfg: dict) -> subprocess.Popen:
    if health():
        print("[gate] something already answers :8180 — refusing to double-serve")
        sys.exit(3)
    env = os.environ | cfg.get("env", {}) | {"STRATA_BD_DEBUG": "1", "STRATA_STATE_HASH": "1"}
    log = SERVER_LOG.open("w")
    p = subprocess.Popen([PY, "-m", "serve.server", "--engine", "strata",
                          "--config", str(CONFIG_OUT), "--host", "127.0.0.1", "--port", "8180"],
                         cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT)
    return p


def run_tool(path: Path, *args: str) -> int:
    print(f"[gate] running {path.name} ...", flush=True)
    return subprocess.run([PY, str(path), *args]).returncode


def dump_trace(log_path: Path, since_bytes: int) -> Path:
    out = Path("/tmp/bd_trace.txt")
    with open(log_path, "rb") as f:
        f.seek(since_bytes)
        lines = f.read().decode("utf-8", "replace").splitlines()
    keep = [l for l in lines if any(s in l for s in ("strata bd:", "state hash", "STRATA_STATE_HASH",
                                                     "given back", "no progress", "BADM", "batch admission"))]
    out.write_text("\n".join(keep))
    print(f"[gate] batch traces: {len(keep)} lines -> {out}")
    for l in keep[-12:]:
        print("  " + l[:160])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--keep-sushi", action="store_true", help="do not stop sushi (fails the memory guard)")
    ap.add_argument("--skip-harness", action="store_true", help="parity only, skip the concurrency harness")
    a = ap.parse_args()

    if not a.keep_sushi:
        stop_sushi()
    wait_memory(55.0)
    cfg = write_config()
    engine_log = Path(cfg["log"])
    since = engine_log.stat().st_size if engine_log.exists() else 0
    server = start_server(cfg)
    deadline = time.time() + 420
    while time.time() < deadline and not models_ready():
        if server.poll() is not None:
            print(f"[gate] serve.server died at startup (exit {server.returncode}); see {SERVER_LOG}")
            return 3
        time.sleep(3)
    if not models_ready():
        print("[gate] server never became healthy in 420 s; see " + str(SERVER_LOG))
        return 3
    print("[gate] server healthy; Strata owns the machine")

    rc_parity = run_tool(REPO / "tools/batch_parity_check.py")
    dump_trace(engine_log, since)
    rc_harness = 0 if a.skip_harness else run_tool(REPO / "tools/subagent_concurrency_harness.py",
                                                   BASE, "--subagents", "4")
    print(f"\n[gate] parity exit {rc_parity}, harness exit {rc_harness}")
    print("[gate] the server is LEFT RUNNING on :8180 (point omp at it, or kill it to give the box back)")
    return 0 if rc_parity == 0 and rc_harness == 0 else (1 if rc_parity else 10)


sys.exit(main())
