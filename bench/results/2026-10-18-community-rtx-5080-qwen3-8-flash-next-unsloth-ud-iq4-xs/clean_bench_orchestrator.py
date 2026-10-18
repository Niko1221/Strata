#!/usr/bin/env python3
"""Detached clean community-benchmark orchestrator v2 (2026-10-18).

v2 changes vs v1: speed suite only by default (needles 6/6 already banked from
the v1 window; pass --with-needles to redo them); warm-up stabilizes BOTH the
4K and the 32K prefill paths; status/logs are archived, not appended; fixed
swap/throttle parsing. The v1 suite is discarded (reclaim storm ~109 GiB
swapout during its 32K/128K arms crushed prefill). Stdlib only. Nothing is
uploaded at any point.
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import random
from pathlib import Path

BASE = "http://127.0.0.1:8080"
REPO = Path("<home>/Strata")
VENV_PY = REPO / ".venv/bin/python"
STAGE = Path("<home>/.hermes/cache/scratch/bench-clean")
STAGE.mkdir(parents=True, exist_ok=True)
HOME = "<home>"
T0 = time.time()
WITH_NEEDLES = "--with-needles" in sys.argv
SUITE_OUT = STAGE / os.environ.get("SUITE_OUT", "suite2")
loggers = []


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{stamp()}] {msg}", flush=True)


def set_status(**kw):
    p = STAGE / "status.json"
    st = json.loads(p.read_text()) if p.exists() else {}
    st.update(kw)
    st["updated"] = stamp()
    p.write_text(json.dumps(st, indent=1))


def redact(s):
    return s.replace(HOME, "<home>") if isinstance(s, str) else s


# ----------------------------------------------------------------------------- HTTP
def stream_chat(messages, max_tokens, timeout=1800):
    body = {"model": "strata", "stream": True, "max_tokens": max_tokens,
            "temperature": 0, "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False}, "messages": messages}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out = {"ok": False, "prompt_tps": None, "ttft_s": None}
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    j = json.loads(d)
                except ValueError:
                    continue
                for ch in j.get("choices") or []:
                    dl = ch.get("delta") or {}
                    if (dl.get("content") or "") or (dl.get("reasoning_content") or ""):
                        if out["ttft_s"] is None:
                            out["ttft_s"] = round(time.perf_counter() - t0, 3)
                t = j.get("timings") or {}
                if t:
                    out["prompt_tps"] = t.get("prompt_per_second")
                if out["ttft_s"] is not None and t:
                    out["ok"] = True
    except Exception as e:  # noqa: BLE001
        out["err"] = repr(e)[:200]
    return out


def word_prompt(tokens, rng):
    words = max(50, int((tokens - 40) / 1.07))
    vocab = ("river stone market lantern copper winter garden signal harbor needle orbit "
             "velvet canyon ember ledger quartz meadow anchor violet thunder").split()
    body = " ".join(rng.choice(vocab) for _ in range(words))
    return (f"[warm {rng.randrange(10**9)}]\nHere is a list of words:\n{body}\n\nSummarize "
            f"what kinds of words appear in the list, in five sentences.")


def probe(tokens, rng):
    r = stream_chat([{"role": "user", "content": word_prompt(tokens, rng)}], 8)
    return r.get("prompt_tps") or 0.0


# ----------------------------------------------------------------------------- logging
def start_loggers():
    q = "utilization.gpu,memory.used,power.draw,temperature.gpu,clocks_event_reasons.active"
    p1 = subprocess.Popen(["bash", "-c",
                           f'while :; do echo "$(date +%s),$(nvidia-smi --query-gpu={q} '
                           f'--format=csv,noheader,nounits)" >> {STAGE}/gpu.log; sleep 2; done'],
                          start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p2 = subprocess.Popen(["bash", "-c",
                           f'while :; do echo "$(date +%s) $(grep -E "^(MemAvailable|SwapFree)" /proc/meminfo '
                           f'| tr "\\n" " ") $(grep -E "^(pswpout|pgpgout)" /proc/vmstat | tr "\\n" " ")" '
                           f'>> {STAGE}/mem.log; sleep 5; done'],
                          start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    loggers.extend([p1, p2])


def stop_loggers():
    for p in loggers:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:  # noqa: BLE001
            pass


def wait_idle(max_s=360):
    t0, calm = time.time(), 0
    while time.time() - t0 < max_s:
        try:
            u = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu",
                                         "--format=csv,noheader,nounits"], text=True)
            calm = calm + 1 if int(u.strip()) < 5 else 0
            if calm >= 5:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    return False


def run_cmd(argv, logfile, cwd=REPO, timeout=7200):
    log(f"running: {redact(' '.join(argv))[:170]}")
    with open(logfile, "w") as fh:
        p = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd, timeout=timeout, text=True)
    lines = [l for l in Path(logfile).read_text(errors="replace").splitlines() if l.strip()]
    for l in lines[-3:]:
        log(f"  | {redact(l)[:170]}")
    return p.returncode


def parse_mem():
    out = {}
    try:
        rows = []
        for l in (STAGE / "mem.log").read_text().splitlines():
            f = l.split()
            if len(f) >= 11 and "SwapFree" in l:
                rows.append((int(f[2]), int(f[5]), int(f[10])))
        if rows:
            out["memavail_min_mib"] = min(r[0] for r in rows) // 1024
            out["swapfree_min_mib"] = min(r[1] for r in rows) // 1024
            out["pswpout_delta_gib"] = round((rows[-1][2] - rows[0][2]) * 4 / 1048576, 1)
    except Exception as e:  # noqa: BLE001
        out["err"] = repr(e)[:150]
    return out


# ----------------------------------------------------------------------------- main
def main():
    (STAGE / "status.json").unlink(missing_ok=True)
    for name in ("gpu.log", "mem.log"):
        p = STAGE / name
        if p.exists():
            p.replace(STAGE / (name + ".prev"))
    set_status(phase="grace", started=stamp(), v2=True)
    log("grace sleep 120 s ...")
    time.sleep(120)

    m = json.loads(urllib.request.urlopen(BASE + "/metrics", timeout=20).read())
    eng = m.get("engine", m)
    assert eng.get("version") == "0.1.45", f"engine moved off 0.1.45: {eng.get('version')}"
    assert "ud-iq4_xs" in eng.get("model", ""), eng.get("model")
    (STAGE / "metrics-at-start.json").write_text(json.dumps(eng, indent=1))
    log(f"server ok: engine {eng['version']}, ctx {eng.get('context')}")

    start_loggers()
    set_status(phase="idle-check")
    if not wait_idle():
        log("WARNING: GPU never looked idle; proceeding, logs will show contention")

    set_status(phase="warmup")
    rng = random.Random()
    a = b = 0.0
    for rnd in range(1, 7):                              # 4K path within 10%
        a, b = probe(4000, rng), probe(4000, rng)
        log(f"warm 4K round {rnd}: {a:.0f} / {b:.0f} tok/s")
        if a > 0 and b > 0 and abs(a - b) / max(a, b) < 0.10:
            break
        time.sleep(5)
    s32 = []
    for rnd in range(1, 5):                              # 32K path within 15%
        s32.append(probe(32000, rng))
        log(f"warm 32K round {rnd}: {s32[-1]:.0f} tok/s")
        if len(s32) >= 2 and abs(s32[-2] - s32[-1]) / max(s32[-2], s32[-1]) < 0.15:
            break
        time.sleep(5)
    for _ in range(2):                                   # warm decode paths
        stream_chat([{"role": "user", "content": f"[warm {rng.randrange(10**9)}] Name three "
                     "primary colors, one sentence."}], 100)
    time.sleep(15)
    set_status(warm_4k=[round(a), round(b)], warm_32k=[round(x) for x in s32])
    log(f"warm done at {int(time.time()-T0)}s")

    set_status(phase="suite", suite_started=stamp())
    SUITE_OUT.mkdir(parents=True, exist_ok=True)
    rc = run_cmd([str(VENV_PY), "tools/strata_bench.py", "--url", BASE, "--runs", "5",
                  "--long", "--out", str(SUITE_OUT)], STAGE / "suite2.log")
    folders = sorted(SUITE_OUT.glob("2026-*"))
    set_status(suite_rc=rc, suite_folder=str(folders[-1]) if folders else None,
               suite_finished=stamp())
    if rc != 0 or not folders:
        stop_loggers()
        set_status(phase="FAILED-suite")
        sys.exit(1)
    time.sleep(15)

    rc2 = None
    if WITH_NEEDLES:
        set_status(phase="needles", needles_started=stamp())
        rc2 = run_cmd([str(VENV_PY), "tools/needle_bench.py", "--url", BASE,
                       "--lengths", "32k,128k", "--depths", "10,50,90",
                       "--out", str(STAGE / "needles.json")], STAGE / "needle.log")
        set_status(needles_rc=rc2, needles_finished=stamp())

    stop_loggers()
    set_status(phase="facts")
    facts = {"run_window_utc": [time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(T0)),
                                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())],
             "elapsed_s": round(time.time() - T0, 1), "suite_rc": rc, "needles_rc": rc2,
             "warm_4k": [round(a), round(b)], "warm_32k_final": round(s32[-1]) if s32 else None}
    try:
        facts["strata_commit"] = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                                         text=True).strip()
    except Exception:  # noqa: BLE001
        pass
    try:
        pid = int(subprocess.check_output(
            ["bash", "-c", "pgrep -o -f 'engine/strata --serve'"], text=True).strip())
        facts["engine_cmdline"] = redact(Path(f"/proc/{pid}/cmdline").read_bytes()
                                         .replace(b"\0", b" ").decode().strip())
        env = Path(f"/proc/{pid}/environ").read_bytes().replace(b"\0", b"\n").decode(errors="replace")
        facts["hc_q8"] = [l for l in env.splitlines() if l.startswith("STRATA_HC_Q8")] or ["not set"]
        facts["engine_started"] = subprocess.check_output(["ps", "-o", "lstart=", "-p", str(pid)],
                                                          text=True).strip()
    except Exception:  # noqa: BLE001
        pass
    facts["gpu_now"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version,power.limit,power.draw,pcie.link.gen.current,"
         "pcie.link.width.current", "--format=csv,noheader"], text=True).strip()
    try:
        src = subprocess.check_output(["findmnt", "-no", "SOURCE", "/"], text=True).strip()
        disk = subprocess.check_output(["bash", "-c",
                    f"lsblk -no NAME,ROTA,MODEL {src} 2>/dev/null | tail -1; "
                    "for d in /sys/class/nvme/nvme*/model; do echo NVME $(cat $d); done 2>/dev/null | head -2"],
                    text=True).strip()
        facts["storage"] = redact(disk)
    except Exception:  # noqa: BLE001
        pass
    facts["memory_run"] = parse_mem()
    try:
        g = [l.split(",") for l in (STAGE / "gpu.log").read_text().splitlines() if l.count(",") >= 5]
        util = [int(x[1]) for x in g]
        thr = 0
        for x in g:
            try:
                thr += 1 if int(x[5].strip(), 16) != 0 else 0
            except ValueError:
                pass
        facts["contention"] = {"samples": len(util), "util_min": min(util), "util_max": max(util),
                               "temp_max": max(int(x[4]) for x in g), "throttled_samples": thr}
    except Exception as e:  # noqa: BLE001
        facts["contention_err"] = repr(e)[:150]
    (STAGE / "machine-facts2.json").write_text(json.dumps(facts, indent=1))
    ok = rc == 0 and bool(folders)
    set_status(phase="DONE" if ok else "FAILED", finished=stamp(),
               summary=f"suite rc={rc} needles rc={rc2} elapsed={facts['elapsed_s']}s "
                       f"folder={folders[-1].name if folders else None}")
    log(f"{'DONE' if ok else 'FAILED'} in {facts['elapsed_s']}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        stop_loggers()
        set_status(phase="FAILED", error=redact(repr(e))[:400])
        log(f"FATAL: {e!r}")
        sys.exit(1)
