"""DIAGNOSTIC ONLY (ci/checks-green; dropped before anything goes to main): why a child Python that serves HTTP on 127.0.0.1
does not answer within 30 s on GitHub's macOS runner (tools/test_strata_mcp.py StartStop), while servers in the test process do.

Prints: socket.getfqdn("127.0.0.1") time in this process and in fresh children; then two child servers (http.server's own
server_bind, which calls getfqdn, and one that skips it), each with the time until a TCP connect and an HTTP answer succeed and,
if it hangs, the child's stack (faulthandler) and output."""
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


def t_fqdn():
    t = time.time()
    name = socket.getfqdn("127.0.0.1")
    return name, round(time.time() - t, 3)


print("python", sys.executable, sys.version.split()[0], "platform", sys.platform, flush=True)
print("hostname", socket.gethostname(), flush=True)
for i in range(2):
    print("in-process getfqdn('127.0.0.1') ->", t_fqdn(), flush=True)
for i in range(2):
    t = time.time()
    r = subprocess.run([sys.executable, "-c", "import socket,time;t=time.time();n=socket.getfqdn('127.0.0.1');print(n, round(time.time()-t,3))"],
                       capture_output=True, text=True, timeout=120)
    print(f"child getfqdn: {r.stdout.strip()!r} rc {r.returncode} (child wall {time.time() - t:.2f} s) {r.stderr.strip()[-300:]}", flush=True)
t = time.time()
r = subprocess.run([sys.executable, "-c", "pass"], capture_output=True, text=True, timeout=120)
print(f"child python start: {time.time() - t:.2f} s rc {r.returncode}", flush=True)

SERVER = r'''
import faulthandler, sys, time, socketserver
faulthandler.dump_traceback_later(25, exit=False)
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
port, mode = int(sys.argv[1]), sys.argv[2]
print("child up", flush=True)
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"ok")
class NoFqdn(ThreadingHTTPServer):
    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]
t = time.time()
srv = (ThreadingHTTPServer if mode == "stdlib" else NoFqdn)(("127.0.0.1", port), H)
print(f"bound after {time.time() - t:.3f} s", flush=True)
srv.serve_forever()
'''

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with tempfile.TemporaryDirectory() as d:
    script = Path(d) / "srv.py"
    script.write_text(SERVER)
    for mode in ("stdlib", "nofqdn"):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        log = Path(d) / f"{mode}.log"
        with open(log, "wb") as f:
            p = subprocess.Popen([sys.executable, str(script), str(port), mode], stdout=f, stderr=subprocess.STDOUT)
        t0 = time.time()
        tcp = http = None
        while time.time() - t0 < 40 and (tcp is None or http is None):
            if tcp is None:
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=1).close()
                    tcp = round(time.time() - t0, 2)
                except OSError:
                    pass
            if tcp is not None and http is None:
                try:
                    with opener.open(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                        r.read()
                    http = round(time.time() - t0, 2)
                except Exception as e:  # noqa: BLE001
                    last = repr(e)
            time.sleep(0.1)
        if http is None:
            time.sleep(max(0, 27 - (time.time() - t0)))   # let faulthandler print the child's stack
        p.kill()
        p.wait(10)
        print(f"--- child server {mode}: tcp connect after {tcp} s, http answer after {http} s; its output:", flush=True)
        print(log.read_text(errors="replace")[-3000:], flush=True)
# the real Strata server (serve/server.py, mock engine), as a fresh child: time from start to its ready file and to an HTTP answer
import json  # noqa: E402
REPO = Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory() as d:
    ready = Path(d) / "ready.json"
    env = {k: v for k, v in os.environ.items() if k not in ("STRATA_API_KEY", "STRATA_TELEMETRY_KEY")}
    t0 = time.time()
    p = subprocess.Popen([sys.executable, "-m", "serve.server", "--engine", "mock", "--port", "0", "--ready-file", str(ready)],
                         cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    rec = ready_s = http_s = None
    while time.time() - t0 < 60 and p.poll() is None and rec is None:
        try:
            rec = json.loads(ready.read_text(encoding="utf-8"))
            ready_s = round(time.time() - t0, 2)
        except (OSError, ValueError):
            time.sleep(0.05)
    if rec:
        try:
            with opener.open(rec["api_url"] + "/health", timeout=5) as r:
                r.read()
            http_s = round(time.time() - t0, 2)
        except Exception as e:  # noqa: BLE001
            http_s = repr(e)
    p.kill()
    p.wait(10)
    print(f"--- serve/server.py (mock engine) child: ready file after {ready_s} s, /health answered after {http_s} s; "
          f"stderr tail: {p.stderr.read().decode(errors='replace')[-400:]!r}", flush=True)
print("--- /etc/hosts")
try:
    print(Path("/etc/hosts").read_text())
except OSError as e:
    print(e)
r = subprocess.run(["scutil", "--dns"], capture_output=True, text=True)
print("--- scutil --dns (head)\n" + r.stdout[:1500])
print("--- env HTTP proxy vars:", {k: v for k, v in os.environ.items() if "proxy" in k.lower()})
