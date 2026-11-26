"""Authenticated control-plane endpoint for opt-in LAN clusters.

Groundwork only: it is not wired into `strata` and is not an inference transport.
Run on a node with: python -m strata_control.cluster.agent --port 8791
Set STRATA_CLUSTER_TOKEN to the cluster's shared secret before starting it.
"""
from __future__ import annotations

import argparse
import ctypes
import hmac
import json
import os
import platform
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _ram_gib() -> float:
    try:
        if os.name == "nt":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return round(status.ullTotalPhys / (1024 ** 3), 1)
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / (1024 ** 2), 1)
    except Exception:
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * size / (1024 ** 3), 1)
    except Exception:
        return 0.0


def _nvidia() -> tuple[str, float]:
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True)
        first = done.stdout.strip().splitlines()[0]
        name, mib = [part.strip() for part in first.split(",")[:2]]
        return name, round(float(re.sub(r"[^0-9.]", "", mib)) / 1024, 1)
    except Exception:
        return (os.environ.get("STRATA_GPU_NAME", "") or "none", 0.0)


def resources() -> dict[str, Any]:
    gpu, vram = _nvidia()
    backend = os.environ.get("STRATA_GPU_BACKEND", "cuda" if gpu and gpu != "none" else "cpu")
    return {
        "ok": True,
        "service": "strata-cluster",
        "version": os.environ.get("STRATA_VERSION", "unknown"),
        "hostname": platform.node(),
        "os": platform.platform(),
        "backend": backend,
        "gpu": gpu,
        "vram_gib": vram,
        "ram_gib": _ram_gib(),
        "cpu_cores": os.cpu_count() or 0,
        "status": "healthy",
    }


class Agent:
    """A tiny JSON endpoint with a shared secret, bound to a chosen address."""

    def __init__(self, token: str, host: str = "127.0.0.1", port: int = 0):
        if not token:
            raise ValueError("token must not be empty")
        self.token = token
        self.host = host
        self.port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def _handler(self):
        token = self.token

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code: int, body: dict[str, Any]):
                raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):  # noqa: N802 - stdlib name
                if self.path not in ("/cluster/ping", "/cluster/info"):
                    self.send_error(404)
                    return
                supplied = self.headers.get("X-Cluster-Token", "")
                if not hmac.compare_digest(supplied, token):
                    self.send_error(401)
                    return
                body = {"ok": True, "service": "strata-cluster"} if self.path == "/cluster/ping" else resources()
                self._send(200, body)

            def log_message(self, *args: Any) -> None:
                return

        return Handler

    def start(self) -> int:
        self._server = ThreadingHTTPServer((self.host, self.port), self._handler())
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Strata cluster control-plane agent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--token-env", default="STRATA_CLUSTER_TOKEN")
    args = parser.parse_args()
    token = os.environ.get(args.token_env, "")
    if not token:
        print(f"Set {args.token_env} before starting the cluster agent.", file=os.sys.stderr)
        return 2
    agent = Agent(token, args.host, args.port)
    port = agent.start()
    print(f"cluster agent listening on {args.host}:{port}", flush=True)
    try:
        while True:
            threading.Event().wait(3600)
    except KeyboardInterrupt:
        agent.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
