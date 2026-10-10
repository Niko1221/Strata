"""Opt-in OTLP tracing: spans per /v1 request, exported to a collector; no native model or GPU is required."""
import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.server import ByteTokenizer, MockEngine, Service, serve
from serve.frontend import ChatTemplate
from serve.runconfig import check
from serve.tracing import (DEFAULT_ENDPOINT, Tracing, endpoint_from_config, endpoint_of, parse_traceparent)


class ClockEngine(MockEngine):
    """MockEngine plus a DONE line: the engine's own milliseconds, which the prefill and decode spans are built from."""

    def generate(self, ids, *args, **kwargs):
        def replay():
            self.last = {"generated": None, "prompt_tokens": len(ids), "prompt_ms": 120.0,
                         "decode_ms": 90.0, "reused": 0, "finish": "stop"}
            n = 0
            for t in super(ClockEngine, self).generate(ids, *args, **kwargs):
                n += 1
                yield t
            self.last["generated"] = n
        return replay()


class Collector:
    """A stand-in for the OTLP collector: it keeps every JSON payload POSTed to /v1/traces and answers 200."""

    def __init__(self):
        self.received = []
        self.httpd = None

    def start(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        collector = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                collector.received.append(json.loads(self.rfile.read(n)))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/traces"

    def stop(self):
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()

    def payloads(self, want=1, timeout=5.0):
        """Wait until `want` traces have arrived, so the exporter's thread gets its moment."""
        deadline = time.time() + timeout
        while len(self.received) < want and time.time() < deadline:
            time.sleep(0.05)
        return list(self.received)

    def spans(self):
        return [s for p in self.payloads() for scope in p["scopeSpans"] for s in scope["spans"]]

    def traces(self):
        out = {}
        for s in self.spans():
            out.setdefault(s["traceId"], []).append(s)
        return out


def make_service(endpoint=None, monitor=False):
    tok = ByteTokenizer()
    svc = Service(ClockEngine(tok, "Hello.", max_context=4096), tok,
                  ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
    if monitor:
        svc.api_monitor = True
    if endpoint is not None:
        svc.tracing = Tracing(endpoint, model="strata-test", version="test")
    return svc


def request(base, path, body, headers=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        response = urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        raw = response.read().decode()
        return (response.status, dict(response.headers),
                json.loads(raw) if "application/json" in response.headers.get("Content-Type", "") else raw)


class TracingOff(unittest.TestCase):
    """Off by default: with no endpoint nothing runs, nothing is exported, and the request path is the release path."""

    def test_off_by_default_exports_nothing(self):
        collector = Collector()
        endpoint = collector.start()
        svc = make_service()                       # no tracing, no monitor
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            code, _, _ = request(base, "/v1/chat/completions",
                                 {"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 8})
            self.assertEqual(code, 200)
        finally:
            httpd.shutdown()
            httpd.server_close()
        collector.stop()
        self.assertEqual(len(collector.received), 0)


class OneTracePerRequest(unittest.TestCase):
    """Tracing on: every /v1 request becomes one OTLP trace carrying the server's own span tree."""

    def setUp(self):
        self.collector = Collector()
        self.endpoint = self.collector.start()
        self.svc = make_service(endpoint=self.endpoint)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.svc.tracing.close()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.collector.stop()

    def test_two_requests_export_two_traces(self):
        for _ in range(2):
            code, _, _ = request(self.base, "/v1/chat/completions",
                                 {"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 8})
            self.assertEqual(code, 200)
        payloads = self.collector.payloads(want=2, timeout=8.0)
        self.assertEqual(len(payloads), 2)                       # one OTLP POST per trace, never a merged batch
        traces = self.collector.traces()
        self.assertEqual(len(traces), 2)
        for trace_id, spans in traces.items():
            root = next(s for s in spans if s["kind"] == 2)
            self.assertEqual(root["name"], "strata.request /v1/chat/completions")
            self.assertNotIn("parentSpanId", root)               # a root with no caller has no parent
            self.assertEqual(root["status"]["code"], 1)          # STATUS_CODE_OK
            attrs = {a["key"]: a["value"] for a in root["attributes"]}
            self.assertEqual(attrs["url.path"]["stringValue"], "/v1/chat/completions")
            self.assertEqual(attrs["http.response.status_code"]["intValue"], "200")
            self.assertEqual(attrs["gen_ai.operation.name"]["stringValue"], "chat")
            self.assertEqual(attrs["gen_ai.provider.name"]["stringValue"], "strata")
            self.assertEqual(attrs["strata.state"]["stringValue"], "completed")
            names = {s["name"] for s in spans}
            self.assertIn("strata.prefill", names)               # the engine's DONE line gives both
            self.assertIn("strata.decode", names)
            for child in spans:
                if child is not root:
                    self.assertEqual(child["kind"], 1)           # SPAN_KIND_INTERNAL
                    self.assertEqual(child["parentSpanId"], root["spanId"])
            prefill = next(s for s in spans if s["name"] == "strata.prefill")
            decode = next(s for s in spans if s["name"] == "strata.decode")
            self.assertLessEqual(int(prefill["endTimeUnixNano"]), int(decode["startTimeUnixNano"]))
            self.assertLess(int(prefill["startTimeUnixNano"]), int(prefill["endTimeUnixNano"]))

    def test_answer_carries_this_request_span(self):
        code, headers, _ = request(self.base, "/v1/chat/completions",
                                           {"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 8})
        self.assertEqual(code, 200)
        sent = headers.get("traceparent")
        self.assertIsNotNone(sent)
        parts = sent.split("-")
        self.assertEqual(len(parts), 4)
        root = next(s for s in self.collector.spans() if s["kind"] == 2)
        self.assertEqual(parts[1], root["traceId"])
        self.assertEqual(parts[2], root["spanId"])

    def test_a_trace_only_record_never_leaves_text(self):
        code, _, _ = request(self.base, "/v1/chat/completions",
                                     {"messages": [{"role": "user", "content": "secret-prompt-xyz"}], "max_tokens": 8})
        self.assertEqual(code, 200)
        self.assertEqual(len(self.svc.api_requests), 0)          # trace-only records are not kept in memory
        self.assertNotIn("secret-prompt-xyz", json.dumps(self.collector.payloads(timeout=8.0)))


class ClientAsParent(unittest.TestCase):
    """W3C both ways: a caller's traceparent becomes our root's parent, and its ids stay in our trace."""

    def test_traceparent_header_links_the_caller(self):
        collector = Collector()
        endpoint = collector.start()
        svc = make_service(endpoint=endpoint)
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        trace_id, client_span = "0123456789abcdef0123456789abcdef", "0011223344556677"
        try:
            code, headers, _ = request(
                base, "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 8},
                headers={"traceparent": f"00-{trace_id}-{client_span}-01"})
            self.assertEqual(code, 200)
            root = next(s for s in collector.spans() if s["kind"] == 2)
            self.assertEqual(root["traceId"], trace_id)          # the caller's trace, not a new one
            self.assertEqual(root["parentSpanId"], client_span)  # the caller's span is our parent
            self.assertNotEqual(root["spanId"], client_span)     # this request is its own span inside it
            sent = headers.get("traceparent").split("-")
            self.assertEqual(sent[1], trace_id)
            self.assertEqual(sent[2], root["spanId"])            # the answer names OUR span
        finally:
            svc.tracing.close()
            httpd.shutdown()
            httpd.server_close()
            collector.stop()


class LateSpansAndInstantRequests(unittest.TestCase):
    """Two shapes the settle path really produces: a record whose spans enter the queue after the answer was
    written, and a request whose first token settles in the same rounded millisecond as its start."""

    def test_a_span_queued_after_the_answer_is_exported(self):
        collector = Collector()
        endpoint = collector.start()
        svc = make_service(endpoint=endpoint)
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            code, _, _ = request(base, "/v1/chat/completions",
                                  {"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 8})
            self.assertEqual(code, 200)
            # the answer is out; this record comes over a beat later, as server.py's settling thread does
            late = {"path": "/v1/chat/completions", "model": "strata-test", "state": "completed",
                    "started_at": time.time(), "wallclock_s": 1.2, "first_token_s": 0.4,
                    "usage": {"prompt_tokens": 254, "completion_tokens": 7},
                    "timings": {"prompt_n": 254, "prompt_ms": 300.0, "predicted_n": 7, "predicted_ms": 500.0}}
            svc.tracing.begin(late, {})
            svc.tracing.finish(late)
            svc.tracing.close(drain_s=2.0)
            self.assertEqual(len(collector.received), 2)         # no polling: the drain itself must have posted it
            self.assertEqual(len(collector.traces()), 2)
        finally:
            httpd.shutdown()
            httpd.server_close()
            collector.stop()

    def test_an_instant_request_keeps_its_prefill(self):
        """The anchor's floor is the root span's own width floor, so a request that settles in one rounded
        millisecond is not drawn as decoding without prefills."""
        collector = Collector()
        svc = make_service(endpoint=collector.start())
        instant = {"path": "/v1/chat/completions", "model": "strata-test", "state": "completed",
                   "started_at": time.time(), "wallclock_s": 0.0, "first_token_s": 0.0,
                   "usage": {"prompt_tokens": 254, "completion_tokens": 7},
                   "timings": {"prompt_n": 254, "prompt_ms": 120.0, "predicted_n": 7, "predicted_ms": 90.0}}
        try:
            svc.tracing.begin(instant, {})
            svc.tracing.finish(instant)
            svc.tracing.close(drain_s=2.0)
            spans = collector.spans()
            names = {s["name"] for s in spans}
            self.assertIn("strata.prefill", names)
            self.assertIn("strata.decode", names)
            prefill = next(s for s in spans if s["name"] == "strata.prefill")
            root = next(s for s in spans if s["kind"] == 2)
            self.assertLess(int(prefill["startTimeUnixNano"]), int(prefill["endTimeUnixNano"]))
            self.assertLessEqual(int(root["startTimeUnixNano"]), int(prefill["startTimeUnixNano"]))
            self.assertLessEqual(int(prefill["endTimeUnixNano"]), int(root["endTimeUnixNano"]))
        finally:
            collector.stop()



class DeadCollector(unittest.TestCase):
    """A collector that is not there costs a counted drop, never a failed or slowed request."""

    def test_requests_survive_a_dead_endpoint(self):
        svc = make_service(endpoint="http://127.0.0.1:9/v1/traces")   # nothing listens there
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            code, _, reply = request(base, "/v1/chat/completions",
                                             {"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 8})
            self.assertEqual(code, 200)
            self.assertIn("Hello", json.dumps(reply))
            svc.tracing.close(drain_s=5.0)
            self.assertGreater(svc.tracing.dropped, 0)
            self.assertEqual(svc.tracing.exported, 0)
        finally:
            httpd.shutdown()
            httpd.server_close()


class EndpointRules(unittest.TestCase):
    """The value a config, env or --trace-otlp carries: off words, the local default, and a typo that stops start."""

    def test_off_default_and_typos(self):
        self.assertIsNone(endpoint_of(None))
        self.assertIsNone(endpoint_of("off"))
        self.assertIsNone(endpoint_of(""))
        self.assertEqual(endpoint_of("default"), DEFAULT_ENDPOINT)
        self.assertEqual(endpoint_of("http://localhost:4318"), DEFAULT_ENDPOINT)
        self.assertEqual(endpoint_of("http://10.0.0.2:4318/v1/traces"), "http://10.0.0.2:4318/v1/traces")
        with self.assertRaises(ValueError) as e:
            endpoint_of("not a url!")
        self.assertIn("not an OTLP endpoint", str(e.exception))

    def test_cli_beats_config_beats_env(self):
        self.assertEqual(endpoint_from_config({"trace_otlp": "http://1.1.1.1:4318/v1/traces"}, "default"),
                         DEFAULT_ENDPOINT)
        self.assertEqual(endpoint_from_config({"trace_otlp": "http://1.1.1.1:4318/v1/traces"}),
                         "http://1.1.1.1:4318/v1/traces")
        import os
        with _env(STRATA_TRACE_OTLP="http://2.2.2.2:4318/v1/traces"):
            self.assertEqual(endpoint_from_config({}), "http://2.2.2.2:4318/v1/traces")

    def test_runconfig_accepts_only_a_real_endpoint(self):
        self.assertEqual(check("trace_otlp", "default", {}), "default")
        with self.assertRaises(ValueError) as e:
            check("trace_otlp", "not a url!", {})
        self.assertIn("not an OTLP endpoint", str(e.exception))


class _env:
    """Set an environment variable for the duration of a with-block."""

    def __init__(self, **names):
        self.names, self.saved = names, {}

    def __enter__(self):
        import os
        for k, v in self.names.items():
            self.saved[k] = os.environ.get(k)
            os.environ[k] = v
        return self

    def __exit__(self, *exc):
        import os
        for k, old in self.saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
        return False


if __name__ == "__main__":
    unittest.main()
