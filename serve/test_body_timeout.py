"""A request whose body stops arriving must give its handler thread back, not hold it forever.

A client whose connection dies without a close (a laptop sleeping, a link dropping mid-body) used to
leave the handler blocked in rfile.read() for the life of the server: the request never reached the
engine, /metrics kept reporting "idle", and the thread was lost.  Handler.timeout (HTTP_BODY_S,
STRATA_HTTP_BODY_S) puts a per-read deadline on the socket; a stalled body is a 408.

    python -m unittest serve.test_body_timeout -v
"""
import os

os.environ.setdefault("STRATA_HTTP_BODY_S", "2")      # a short deadline so the test waits seconds, not a minute

import socket
import threading
import time
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parent.parent


class BodyTimeout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(MockEngine(tok, "hello there", max_context=4096), tok,
                          ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _stalled_body(self, transfer: str) -> socket.socket:
        """Announce a body, send a piece of it, and keep the connection open: nothing more arrives."""
        s = socket.create_connection(("127.0.0.1", self.port))
        head = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\n"
                f"Content-Type: application/json\r\n{transfer}\r\n\r\n").encode()
        s.sendall(head + b'{"model": "m", "messages": [{"role": "user", "content":')
        return s

    def test_stalled_body_gives_its_thread_back(self):
        from serve.server import HTTP_BODY_S
        self.assertIsNotNone(HTTP_BODY_S)                    # the deadline is on unless STRATA_HTTP_BODY_S=0
        before = threading.active_count()
        sock = self._stalled_body("Content-Length: 100000")
        try:
            started = time.monotonic() + 5                   # the handler thread must first pick the connection up
            while threading.active_count() <= before and time.monotonic() < started:
                time.sleep(0.1)
            self.assertGreater(threading.active_count(), before, "the stalled request never reached a handler")
            deadline = time.monotonic() + HTTP_BODY_S + 8    # the read expires, the handler answers and exits
            while threading.active_count() > before and time.monotonic() < deadline:
                time.sleep(0.2)
            self.assertLessEqual(threading.active_count(), before,
                                 "the handler of a stalled body still holds its thread after the deadline")
        finally:
            sock.close()


if __name__ == "__main__":
    unittest.main()
