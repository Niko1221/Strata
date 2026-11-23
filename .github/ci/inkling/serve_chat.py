#!/usr/bin/env python3
"""serve_chat.py : start serve/server.py on a strata-inkling engine, run the three answer-text chats (chat_check.py), stop both. Own PIDs only.

  python3 serve_chat.py --out DIR --src TREE --exe ENGINE --tok TOKDIR --port N --model-name NAME [--env K=V ...] [--no-chat] -- <engine args...>

TREE is a checkout with terminal/serve/server.py. TOKDIR holds tokenizer.json (+ config / chat_template). The engine args are the strata-inkling
arguments (--native ..., --max-context ..., --draft ...). Prints the chat lines; exit 0 only when the answers are right.
Loopback only. Stops the server (SIGINT, then SIGTERM, then SIGKILL after waits) and any engine child that is still ours."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--src", required=True)
ap.add_argument("--exe", required=True)
ap.add_argument("--tok", required=True)
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model-name", default="inkling-test")
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--chat", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_check.py"))
ap.add_argument("--wait", type=int, default=900, help="seconds to wait for the model to be ready")
ap.add_argument("--no-chat", action="store_true")
ap.add_argument("--post", action="append", default=[], help="extra command (run with the same python as chat), {port} is replaced; exit code must be 0")
ap.add_argument("--py-server", default=sys.executable, help="python for server.py")
ap.add_argument("engine_args", nargs=argparse.REMAINDER)
a = ap.parse_args()
eargs = [x for x in a.engine_args if x != "--"]

os.makedirs(a.out, exist_ok=True)
cfg = {"arch": "inkling2_loop", "exe": a.exe, "args": eargs, "cwd": a.src, "tokenizer": a.tok, "model_name": a.model_name,
       "log": os.path.join(a.out, "srv.log"), "host": "127.0.0.1", "port": a.port}
cfgp = os.path.join(a.out, "cfg.json")
json.dump(cfg, open(cfgp, "w"), indent=1)
env = dict(os.environ)
for kv in a.env:
    k, _, v = kv.partition("=")
    env[k] = v
env["PYTHONUNBUFFERED"] = "1"
sp = subprocess.Popen([a.py_server, os.path.join(a.src, "terminal", "serve", "server.py"), "--engine", "strata", "--config", cfgp, "--port", str(a.port),
                       "--no-api-key"], cwd=os.path.join(a.src, "terminal"), env=env,
                      stdout=open(os.path.join(a.out, "server.out"), "w"), stderr=subprocess.STDOUT)
print("server pid", sp.pid, flush=True)
rc = 1
t0 = time.time()
ready = False
try:
    while time.time() - t0 < a.wait:
        if sp.poll() is not None:
            print("server exited early rc", sp.returncode, flush=True)
            break
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % a.port, timeout=3) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:
            pass
        time.sleep(3)
    print("ready=%s after %.0fs" % (ready, time.time() - t0), flush=True)
    if ready:
        rc = 0
        if not a.no_chat:
            r = subprocess.run([sys.executable, a.chat, str(a.port)], capture_output=True, text=True, timeout=3600)
            print(r.stdout, r.stderr[-500:], flush=True)
            rc = r.returncode
        for cmd in a.post:
            parts = [sys.executable] + [x.replace("{port}", str(a.port)) for x in cmd.split()]
            r2 = subprocess.run(parts, capture_output=True, text=True, timeout=7200)
            print("POST", cmd, "rc", r2.returncode, flush=True)
            print(r2.stdout, r2.stderr[-500:], flush=True)
            rc = rc or r2.returncode
finally:
    if sp.poll() is None:
        try:
            sp.send_signal(signal.SIGINT)
        except Exception:
            sp.terminate()
        for _ in range(40):
            if sp.poll() is not None:
                break
            time.sleep(1)
    if sp.poll() is None:
        sp.terminate()
        for _ in range(20):
            if sp.poll() is not None:
                break
            time.sleep(1)
    if sp.poll() is None:
        sp.kill()
    print("server stopped rc", sp.returncode, flush=True)
    # an engine child that outlived the server: only processes started from our own exe path
    time.sleep(2)
    for d in (os.listdir("/proc") if os.path.isdir("/proc") else []):
        if d.isdigit() and int(d) != os.getpid():
            try:
                cl = open("/proc/%s/cmdline" % d, "rb").read().split(bytes([0]))
            except OSError:
                continue
            if cl and cl[0] == a.exe.encode() and a.exe.encode() in cl[0]:
                print("engine child left, pid", d, "-> SIGTERM", flush=True)
                try:
                    os.kill(int(d), signal.SIGTERM)
                except OSError:
                    pass
sys.exit(rc)
