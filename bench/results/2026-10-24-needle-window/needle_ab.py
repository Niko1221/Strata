# Needle-in-a-haystack A/B through the HTTP server: int8 (reference arm) then the exact-cell window
# (--kv q4lm8v + STRATA_KV_WINDOW=1). tools/needle_bench.py asks both arms the same needles: it builds the
# haystack deterministically from this repo's files and picks the code words with random.Random(7), so the
# two arms see identical prompts. STRATA_PREFILL_CPU_SHARE=0 in both arms (the #2318 lesson: with the CPU
# share on, a prompt chunk under 1024 tokens splits MoE experts with the CPU and the split varies run to run).
import json, os, pathlib, signal, subprocess, sys, time, urllib.request
REPO = pathlib.Path(r"C:\Users\krone\git\kv-quant-strata-0.2")
PY = str(REPO / ".venv" / "Scripts" / "python.exe")   # the repo venv: serve needs `regex`
PERF = pathlib.Path(r"C:\Users\krone\AppData\Local\Temp\opencode\perf")
base = json.load(open(PERF / "cfg-new.json", encoding="utf-8-sig"))
URL = "http://127.0.0.1:7601"

def cfg_for(tag, kv, window):
    c = dict(base)
    c["args"] = list(base["args"]) + ["--kv", kv, "--max-context", "133120", "--prefill", "auto"]
    env = {"STRATA_PREFILL_CPU_SHARE": "0"}
    if window:
        env.update({"STRATA_KV_WINDOW": "1", "STRATA_KV_WINDOW_S": "256", "STRATA_KV_WINDOW_R": "64"})
    c["env"] = env
    c["log"] = str(PERF / f"needle-{tag}.engine.log")
    p = PERF / f"cfg-needle-{tag}.json"
    p.write_text(json.dumps(c, indent=1), encoding="utf-8")
    return p


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


def arm(tag, cfg):
    print(f"=== arm {tag}: starting server ({cfg}) ===", flush=True)
    server = subprocess.Popen([PY, str(REPO / "serve" / "server.py"), "--engine", "strata",
                               "--config", str(cfg), "--port", "7601", "--no-api-key"],
                              cwd=str(REPO), creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    if not wait_up():
        print(f"arm {tag}: server never came up", flush=True)
        stop(server)
        return 1
    print(f"arm {tag}: server up, running needle_bench (32k,128k x depths 10/50/90)", flush=True)
    t0 = time.time()
    done = subprocess.run([PY, str(REPO / "tools" / "needle_bench.py"), "--url", URL,
                           "--lengths", "32k,128k", "--depths", "10,50,90",
                           "--out", str(PERF / f"needles-{tag}.json")], cwd=str(REPO))
    print(f"arm {tag}: needle_bench exit {done.returncode}, wall {int(time.time() - t0)} s", flush=True)
    stop(server)
    time.sleep(10)
    return done.returncode


rc = arm("int8", cfg_for("int8", "int8", window=False)) or arm("win", cfg_for("win", "q4lm8v", window=True))
print("=== SUMMARY ===", flush=True)
for tag in ("int8", "win"):
    p = PERF / f"needles-{tag}.json"
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
