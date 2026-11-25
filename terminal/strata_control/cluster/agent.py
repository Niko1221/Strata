"""Minimal authenticated control-plane endpoint for opt-in LAN clusters.

Groundwork only: it is not wired into `strata` and is not an inference transport.
The production design requires TLS; this loopback-capable helper exists so the
control protocol can be measured and tested without special hardware.
"""
from __future__ import annotations

import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


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
            def do_GET(self):  # noqa: N802 - stdlib name
                if self.path != "/cluster/ping":
                    self.send_error(404)
                    return
                supplied = self.headers.get("X-Cluster-Token", "")
                if not hmac.compare_digest(supplied, token):
                    self.send_error(401)
                    return
                body = json.dumps({"ok": True, "service": "strata-cluster"}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

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
