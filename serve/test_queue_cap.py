"""The queue cap ("max_queue"): once that many requests wait beyond the ones running, a new generation request
gets an immediate HTTP 429 with a whole-second Retry-After (1..60) instead of joining the queue.

The real StrataEngine and Service over HTTP with test_parallel's fake engine (no GPU), plus unit tests of
Service.admit / release / retry_after.

    python -m pytest serve/test_queue_cap.py -q
"""
import json
import math
import socket
import threading
import time
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest import mock

from serve.server import EngineDied, QueueFull, Service
from serve.test_parallel import ParallelService

SLOW = "hold this one LONGREPLY"        # the fake engine answers this one slowly (about a second)
PATHS = {
    "/v1/chat/completions": lambda text, stream=False: {
        "messages": [{"role": "user", "content": text}], "max_tokens": 64, "reasoning_effort": "none",
        "stream": stream},
    "/v1/messages": lambda text, stream=False: {
        "messages": [{"role": "user", "content": text}], "max_tokens": 64, "stream": stream},
    "/v1/responses": lambda text, stream=False: {"input": text, "max_output_tokens": 64, "stream": stream},
}


class Unit(unittest.TestCase):
    def service(self, batch=0, max_queue=None, history=()):
        svc = Service.__new__(Service)
        svc.engine = SimpleNamespace(batch=batch)
        svc.status_lock = threading.Lock()
        svc.totals = {"rejected": 0}
        svc.inflight, svc.max_queue = 0, max_queue
        svc.history = __import__("collections").deque({"duration_s": d} for d in history)
        svc.live_reqs, svc.status = {}, {"busy": False}
        return svc

    def test_no_limit_by_default(self):
        svc = self.service()
        for _ in range(50):
            svc.admit()
        self.assertEqual(svc.inflight, 50)
        self.assertEqual(svc.totals["rejected"], 0)

    def test_capacity_is_slots_plus_max_queue(self):
        svc = self.service(batch=2, max_queue=1)
        for _ in range(3):
            svc.admit()
        with self.assertRaises(QueueFull) as c:
            svc.admit()
        self.assertEqual(svc.inflight, 3)                  # a rejected request is not counted
        self.assertEqual(svc.totals["rejected"], 1)
        self.assertIn("1 waiting, max_queue 1", str(c.exception))
        svc.release()
        svc.admit()                                        # a place opened
        self.assertEqual(svc.inflight, 3)

    def test_zero_means_one_at_a_time_without_a_queue(self):
        svc = self.service(batch=0, max_queue=0)
        svc.admit()
        with self.assertRaises(QueueFull):
            svc.admit()

    def test_release_never_goes_below_zero(self):
        svc = self.service()
        svc.release()
        self.assertEqual(svc.inflight, 0)

    def test_retry_after_no_history(self):
        self.assertEqual(self.service().retry_after(), 10)

    def test_retry_after_mean_over_slots_and_clamps(self):
        self.assertEqual(self.service(batch=0, history=[4.0, 6.0]).retry_after(), 5)
        self.assertEqual(self.service(batch=2, history=[4.0, 6.0]).retry_after(), 3)       # 5 / 2 -> ceil
        self.assertEqual(self.service(history=[0.1]).retry_after(), 1)
        self.assertEqual(self.service(history=[500.0]).retry_after(), 60)
        # only the last 8 count
        self.assertEqual(self.service(history=[1000.0] * 5 + [8.0] * 8).retry_after(), 8)

    def test_retry_after_ignores_how_long_the_running_ones_have_gone(self):
        # an overdue request must not pull the estimate to 1 s: litellm would spend its retries in seconds
        svc = self.service(history=[20.0])
        svc.status = {"busy": True, "started": time.time() - 100}
        self.assertEqual(svc.retry_after(), 20)
        svc = self.service(batch=2, history=[20.0])
        svc.live_reqs = {1: ({"started": time.time() - 40}, None), 2: ({"started": time.time() - 1}, None)}
        self.assertEqual(svc.retry_after(), 10)


class QueueCap(unittest.TestCase):
    # ParallelService's set-up, without inheriting (and so re-running) its tests
    tearDown, get = ParallelService.tearDown, ParallelService.get

    def start(self, slots, max_queue=None, **kw):
        ParallelService.start(self, slots, **kw)
        self.svc.max_queue = max_queue

    def post(self, path, body, timeout=60):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, dict(r.headers), r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read().decode()

    def hold(self, n, path="/v1/chat/completions"):
        """n slow requests in flight; returns once the server counts them all"""
        out = []
        threads = [threading.Thread(target=lambda i=i: out.append(self.post(path, PATHS[path](f"{SLOW} {i}"))))
                   for i in range(n)]
        for t in threads:
            t.start()
        deadline = time.time() + 10
        while self.svc.inflight < n and time.time() < deadline:
            time.sleep(0.005)
        self.assertEqual(self.svc.inflight, n)
        return threads, out

    def settle(self):
        deadline = time.time() + 15
        while self.svc.inflight and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.svc.inflight, 0)

    def test_over_the_cap_gets_429_in_each_dialect(self):
        self.start(2, max_queue=1)
        threads, out = self.hold(3)
        for path, make in PATHS.items():
            code, headers, raw = self.post(path, make("one too many"))
            self.assertEqual(code, 429, path)
            ra = headers["Retry-After"]
            self.assertTrue(ra.isdigit() and 1 <= int(ra) <= 60, ra)
            body = json.loads(raw)
            if path == "/v1/messages":
                self.assertEqual(body["type"], "error")
                self.assertEqual(body["error"]["type"], "rate_limit_error")
            else:
                self.assertEqual(body["error"]["type"], "rate_limit_error")
                self.assertEqual(body["error"]["code"], "rate_limit_exceeded")
            self.assertIn("the queue is full (1 waiting, max_queue 1)", body["error"]["message"])
        for t in threads:
            t.join(30)
        self.assertEqual([c for c, _, _ in out], [200, 200, 200])      # the admitted ones finish normally
        self.settle()
        self.assertEqual(self.svc.totals["rejected"], 3)
        self.assertEqual(self.get("/metrics")["totals"]["rejected"], 3)
        live = self.get("/metrics")["live"]
        self.assertEqual((live["inflight"], live["max_queue"]), (0, 1))
        self.assertEqual(self.get("/v1/status")["activity"]["max_queue"], 1)

    def test_a_stream_over_the_cap_is_a_429_json_not_an_event_stream(self):
        self.start(2, max_queue=0)
        threads, _ = self.hold(2)
        for path, make in PATHS.items():
            code, headers, raw = self.post(path, make("streamed", True))
            self.assertEqual(code, 429, path)
            self.assertEqual(headers["Content-Type"], "application/json")
            self.assertIn("Retry-After", headers)
            json.loads(raw)
        for t in threads:
            t.join(30)
        self.settle()

    def test_other_endpoints_are_not_limited(self):
        self.start(2, max_queue=0)
        threads, _ = self.hold(2)
        code, _, _ = self.post("/v1/messages/count_tokens", {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(code, 200)
        self.assertEqual(self.get("/v1/status")["model"], self.svc.model)
        for t in threads:
            t.join(30)
        self.settle()

    def test_max_queue_zero_with_one_at_a_time_engine(self):
        self.start(0, max_queue=0)
        threads, out = self.hold(1)
        code, headers, _ = self.post("/v1/chat/completions", PATHS["/v1/chat/completions"]("second"))
        self.assertEqual(code, 429)
        self.assertIn("Retry-After", headers)
        for t in threads:
            t.join(30)
        self.settle()
        self.assertEqual(out[0][0], 200)
        # the place is free again
        self.assertEqual(self.post("/v1/chat/completions", PATHS["/v1/chat/completions"]("third"))[0], 200)

    def test_one_at_a_time_queue_of_one(self):
        self.start(0, max_queue=1)
        threads, out = self.hold(2)                        # one runs, one waits
        self.assertEqual(self.post("/v1/chat/completions", PATHS["/v1/chat/completions"]("x"))[0], 429)
        for t in threads:
            t.join(30)
        self.assertEqual([c for c, _, _ in out], [200, 200])
        self.settle()

    def test_unset_means_no_rejections(self):
        self.start(2, max_queue=None)
        threads, out = self.hold(6)
        for t in threads:
            t.join(40)
        self.assertEqual([c for c, _, _ in out], [200] * 6)
        self.assertEqual(self.svc.totals["rejected"], 0)
        self.settle()

    def test_counter_returns_to_zero(self):
        self.start(2, max_queue=1)
        path = "/v1/chat/completions"
        # normal end, streamed and not
        self.assertEqual(self.post(path, PATHS[path]("hi"))[0], 200)
        self.assertEqual(self.post(path, PATHS[path]("hi", True))[0], 200)
        self.settle()
        # a 400 (ValueError)
        self.assertEqual(self.post(path, {"messages": "not a list"})[0], 400)
        self.settle()
        # an engine error
        with mock.patch.object(Service, "run", side_effect=EngineDied("boom")):
            self.assertEqual(self.post(path, PATHS[path]("hi"))[0], 503)
        self.settle()
        with mock.patch.object(Service, "run", side_effect=EngineDied("boom")):
            self.post(path, PATHS[path]("hi", True))
        self.settle()

    def test_counter_returns_to_zero_after_a_client_disconnect(self):
        self.start(2, max_queue=1)
        body = json.dumps(PATHS["/v1/chat/completions"](SLOW, True)).encode()
        s = socket.create_connection(("127.0.0.1", self.httpd.server_address[1]))
        s.sendall(b"POST /v1/chat/completions HTTP/1.0\r\nContent-Type: application/json\r\n"
                  b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        s.recv(100)
        self.assertEqual(self.svc.inflight, 1)
        s.close()
        self.settle()


if __name__ == "__main__":
    unittest.main()
