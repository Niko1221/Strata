# B2 of the 2026-11-10 merge re-evaluation: needle recall, int8 against the exact-cell window,
# on the merged build (0.1.59.1). Reproduces the 2026-10-24-needle-window protocol with its
# committed driver (needle_ab.py): 6 needles at 32K/128K, depths 10/50/90, greedy, thinking off,
# max 40 tokens; the haystack is this repository's text files today (fixed order), the code words
# come from random.Random(7), so both arms see identical prompts. needle_bench.py checks 6/6
# itself and exits 1 otherwise.
#
# Differences from the 2026-10-24 configs, all recorded in the campaign protocol: the merged
# engine build (build-hip-port), ROCm 7.2 lib_dirs, the current repo's expert-profile.bin (the
# old checkout's file no longer exists), port 7603, and logs/results under Strata-data.
#
# Gates this driver enforces beyond needle_bench's own 6/6:
#   - the engine's SHA-256 is printed before each arm;
#   - the window arm's engine log must contain the "ring sized" line, the int8 arm's must not;
#   - no strata.exe may be running when an arm starts (one GPU job at a time), and the engine
#     must be gone before the next arm;
#   - 30 s between engine starts.
import hashlib, json, pathlib, signal, subprocess, sys, time, urllib.request

REPO = pathlib.Path(r"C:\Users\krone\git\kv-quant-strata-0.2")
PY = str(REPO / ".venv" / "Scripts" / "python.exe")   # the repo venv: serve needs `regex`
OUT = pathlib.Path(r"C:\Users\krone\git\Strata-data\bench\2026-11-10-merge-reeval\b2")
URL = "http://127.0.0.1:7603"
ARMS = [
    ("int8", REPO / "bench" / "configs" / "needle-swift-iq3_xxs-merged-int8.json", False),
    ("win", REPO / "bench" / "configs" / "needle-swift-iq3_xxs-merged-win.json", True),
]


def wait_up(seconds=900):
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            with urllib.request.urlopen(URL + "/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except OSError:
            time.sleep(3)
    return False


def stop(server):
    if server.poll() is None:
        server.send_signal(signal.CTRL_BREAK_EVENT)
        try:
            server.wait(timeout=120)
        except subprocess.TimeoutExpired:
            server.kill()


def strata_running():
    # The one-GPU-job gate.  tasklist's bytes are decoded leniently: its output can hold a character
    # outside the console code page (a process name), which cp1252 cannot decode - that killed the
    # first attempt of this driver.  A failing tasklist counts as "something runs", so the gate errs
    # on the safe side.
    try:
        raw = subprocess.run(["tasklist", "/FI", "IMAGENAME eq strata.exe"], capture_output=True).stdout
    except OSError as e:
        print(f"tasklist failed ({e}); the one-GPU-job gate cannot be checked", flush=True)
        return True
    out = raw.decode("utf-8", errors="replace")
    return any(line.lower().lstrip().startswith("strata.exe") for line in out.splitlines())


def wait_engine_gone(seconds=180):
    waited = 0
    while strata_running() and waited < seconds:
        time.sleep(5)
        waited += 5
    return not strata_running()


def arm(tag, cfg, window):
    print(f"=== arm {tag}: config {cfg} ===", flush=True)
    c = json.loads(cfg.read_text(encoding="utf-8-sig"))
    h = hashlib.sha256()
    with open(c["exe"], "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    print(f"arm {tag}: engine {c['exe']} sha256 {h.hexdigest()}", flush=True)
    if strata_running():
        print(f"arm {tag}: a strata.exe is already running; refusing to start (one GPU job at a time)", flush=True)
        return 1
    print(f"arm {tag}: starting server", flush=True)
    server = subprocess.Popen([PY, str(REPO / "serve" / "server.py"), "--engine", "strata",
                               "--config", str(cfg), "--port", "7603", "--no-api-key"],
                              cwd=str(REPO), creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    if not wait_up():
        print(f"arm {tag}: server never came up", flush=True)
        stop(server)
        return 1
    print(f"arm {tag}: server up, running needle_bench (32k,128k x depths 10/50/90)", flush=True)
    t0 = time.time()
    done = subprocess.run([PY, str(REPO / "tools" / "needle_bench.py"), "--url", URL,
                           "--lengths", "32k,128k", "--depths", "10,50,90",
                           "--out", str(OUT / f"needles-{tag}.json")], cwd=str(REPO))
    print(f"arm {tag}: needle_bench exit {done.returncode}, wall {int(time.time() - t0)} s", flush=True)
    stop(server)
    if not wait_engine_gone():
        print(f"arm {tag}: the engine did not exit after the server stopped", flush=True)
        return 1
    log = OUT / f"needle-{tag}.engine.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    ring = "ring sized" in text
    want = window
    print(f"arm {tag}: window ring line in engine log: {'present' if ring else 'absent'} "
          f"(expected {'present' if want else 'absent'})", flush=True)
    rc = done.returncode
    if ring != want:
        rc = rc or 1
    time.sleep(30)   # the campaign rule: 30 s between engine starts
    return rc


OUT.mkdir(parents=True, exist_ok=True)
rc = arm("int8", ARMS[0][1], ARMS[0][2]) or arm("win", ARMS[1][1], ARMS[1][2])
print("=== SUMMARY ===", flush=True)
for tag, _cfg, _window in ARMS:
    p = OUT / f"needles-{tag}.json"
    if not p.exists():
        print(f"{tag}: no results")
        continue
    rows = json.loads(p.read_text(encoding="utf-8"))
    ok = sum(1 for r in rows if r.get("found"))
    n = sum(1 for r in rows if "found" in r)
    print(f"{tag}: {ok}/{n} found")
    for r in rows:
        if "found" in r:
            print(f"  {r['length']:>5} depth {r['depth']:>3}%: {'FOUND' if r['found'] else 'missed'}  "
                  f"({r.get('prompt_tokens', 0):,} tok, {r.get('seconds', 0):.0f} s)"
                  + ("" if r["found"] else f"  answer: {r.get('answer', '')[:80]!r}"))
sys.exit(rc)
