"""Tests for tools/busbar_gateway.py without any network or provider key: stand-in members that answer or fail.

    python -m unittest tools.test_busbar_gateway

The routing rules checked here are the ones the gateway promises: an 80/20 split sends exactly 80 of 100 requests
to Claude and 20 to GPT, spread evenly rather than in clumps; a member that fails is tried on the next member of
the pool and then on the failover pools; a member that fails `max_fails` times in a row is put on cooldown and
skipped until the cooldown ends.
"""
from __future__ import annotations

import contextlib
import io
import json
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import busbar_gateway as G  # noqa: E402


def graceful_close(sock: socket.socket) -> None:
    """Make a close of this connection wait for the peer to confirm the sent bytes.

    Windows sockets sit on a layered provider: a close while the last writes are
    still in the send queue discards them and aborts the connection (WinError
    10053), which on a local test upstream races the reader ~1 in 60 runs. A
    positive SO_LINGER timeout makes close() finish the graceful shutdown
    (drain, FIN, confirm) and return once the peer confirmed - bounded by the
    timeout, so a test never blocks on a peer that ignores it.
    """
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                    struct.pack("ii", 1, 5))


class FakeGateway(G.Gateway):
    """Replace the HTTP call with a member's `down` flag: True means the upstream answers nothing."""

    def complete(self, body):
        pool = self.registry.resolve(body.get("model"))
        last = "no member answered"
        for owner, member in pool.pick():
            if getattr(member, "down", False):
                owner.note_fail(member)
                last = f"{member.name}: down"
                continue
            owner.note_ok(member, 5)
            return 200, {"model": member.model, "choices": [{"message": {"content": "ok"}}],
                        "usage": {"total_tokens": 5}}
        return 502, {"error": {"message": last}}


def cfg(members, failover=None, **pool_extra):
    spec = {"members": members, "failover": failover or [], **pool_extra}
    return {"default_pool": "fast", "pools": {"fast": spec}}


def registry_from(config):
    return G.Registry(config)


class Split(unittest.TestCase):
    def test_exact_split_over_100_requests(self):
        reg = registry_from(cfg([{"name": "claude", "weight": 80, "base_url": "u", "model": "c"},
                                 {"name": "gpt", "weight": 20, "base_url": "u", "model": "g"}]))
        order = reg.plan("fast", 100)
        self.assertEqual(order.count("claude"), 80)
        self.assertEqual(order.count("gpt"), 20)

    def test_split_is_spread_not_clumped(self):
        reg = registry_from(cfg([{"name": "claude", "weight": 80, "base_url": "u", "model": "c"},
                                 {"name": "gpt", "weight": 20, "base_url": "u", "model": "g"}]))
        first10 = reg.plan("fast", 10)
        self.assertEqual(first10.count("claude"), 8)
        self.assertEqual(first10.count("gpt"), 2)
        # the smaller share is spread out, not saved for the end: it shows up in the first five requests,
        # and no run of claude is longer than four
        self.assertIn("gpt", first10[:5])
        longest = max(len(run) for run in "".join("1" if x == "claude" else "0" for x in first10).split("0") if run)
        self.assertLessEqual(longest, 4)

    def test_weights_are_relative(self):
        reg = registry_from(cfg([{"name": "a", "weight": 4, "base_url": "u", "model": "a"},
                                 {"name": "b", "weight": 1, "base_url": "u", "model": "b"}]))
        order = reg.plan("fast", 5)
        self.assertEqual(order.count("a"), 4)
        self.assertEqual(order.count("b"), 1)

    def test_equal_weights_alternate(self):
        reg = registry_from(cfg([{"name": "a", "weight": 1, "base_url": "u", "model": "a"},
                                 {"name": "b", "weight": 1, "base_url": "u", "model": "b"}]))
        self.assertEqual(reg.plan("fast", 4), ["a", "b", "a", "b"])

    def test_planning_a_pool_does_not_send_anything(self):
        reg = registry_from(cfg([{"name": "a", "weight": 1, "base_url": "u", "model": "a"}]))
        reg.plan("fast", 3)
        self.assertEqual(reg.pools["fast"].counts["a"], 0)


class Failover(unittest.TestCase):
    def test_failing_member_falls_to_the_next(self):
        reg = registry_from(cfg([{"name": "bad", "weight": 90, "base_url": "u", "model": "b"},
                                 {"name": "good", "weight": 10, "base_url": "u", "model": "g"}]))
        reg.pools["fast"].members[0].down = True
        gw = FakeGateway(reg, "")
        status, body = gw.complete({"model": "fast", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        self.assertEqual(body["model"], "g")
        self.assertEqual(reg.pools["fast"].errors["bad"], 1)

    def test_fails_over_to_another_pool(self):
        config = {"pools": {
            "fast": {"members": [{"name": "bad", "weight": 1, "base_url": "u", "model": "b"}],
                     "failover": ["bedrock"]},
            "bedrock": {"members": [{"name": "bedrock", "weight": 1, "base_url": "u", "model": "br"},
                                    {"name": "bedrock2", "weight": 3, "base_url": "u", "model": "br2"}]},
        }}
        reg = registry_from(config)
        reg.pools["fast"].members[0].down = True
        gw = FakeGateway(reg, "")
        status, body = gw.complete({"model": "fast", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        self.assertEqual(body["model"], "br2")   # the heavier failover member is tried first
        # a request that only reached the failover pool as a backup does not move its round robin
        self.assertEqual(reg.pools["bedrock"].current["bedrock2"], 0.0)

    def test_every_member_down_returns_502(self):
        reg = registry_from(cfg([{"name": "a", "weight": 1, "base_url": "u", "model": "a"},
                                 {"name": "b", "weight": 1, "base_url": "u", "model": "b"}]))
        for m in reg.pools["fast"].members:
            m.down = True
        gw = FakeGateway(reg, "")
        status, body = gw.complete({"model": "fast", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 502)
        self.assertIn("down", body["error"]["message"])
        self.assertEqual([m["failed"] for m in reg.pools["fast"].stats()["members"]], [1, 1])

    def test_unknown_pool_is_an_error(self):
        reg = registry_from(cfg([{"name": "a", "weight": 1, "base_url": "u", "model": "a"}]))
        with self.assertRaises(ValueError):
            reg.resolve("nope")
        self.assertIs(reg.resolve(None), reg.pools["fast"])

    def test_missing_failover_pool_is_a_config_error(self):
        with self.assertRaises(ValueError):
            registry_from({"pools": {"fast": {"members": [{"name": "a", "base_url": "u", "model": "a"}],
                                          "failover": ["ghost"]}}})


class Cooldown(unittest.TestCase):
    def test_member_is_put_aside_after_enough_failures(self):
        pool = G.Pool("fast", {"members": [{"name": "bad", "weight": 5, "base_url": "u", "model": "b"},
                                          {"name": "good", "weight": 1, "base_url": "u", "model": "g"}],
                               "max_fails": 2, "cooldown_s": 30})
        bad = pool.members[0]
        good = pool.members[1]
        pool.note_fail(bad)
        pool.note_fail(bad)
        self.assertGreater(pool.until[bad.name], time.time())
        for _ in range(3):
            self.assertIs(pool.pick()[0][1], good)
        stats = pool.stats()["members"]
        self.assertEqual([m["failed"] for m in stats], [2, 0])
        self.assertGreater(stats[0]["cooling_until"], 0)

    def test_cooldown_expires_and_the_member_returns(self):
        pool = G.Pool("fast", {"members": [{"name": "bad", "weight": 5, "base_url": "u", "model": "b"},
                                          {"name": "good", "weight": 1, "base_url": "u", "model": "g"}],
                               "max_fails": 1, "cooldown_s": 0.05})
        pool.note_fail(pool.members[0])
        time.sleep(0.06)
        self.assertIs(pool.pick()[0][1], pool.members[0])

    def test_a_failover_member_on_cooldown_is_skipped(self):
        pool = G.Pool("fast", {"members": [{"name": "a", "weight": 1, "base_url": "u", "model": "a"}]})
        backup = G.Pool("backup", {"members": [{"name": "b1", "weight": 1, "base_url": "u", "model": "b1"},
                                              {"name": "b2", "weight": 1, "base_url": "u", "model": "b2"}],
                                   "max_fails": 1, "cooldown_s": 30})
        pool.failover.append(backup)
        backup.note_fail(backup.members[0])
        names = [m.name for _, m in pool.pick()]
        self.assertEqual(names, ["a", "b2"])

    def test_success_resets_the_failure_count(self):
        pool = G.Pool("fast", {"members": [{"name": "a", "weight": 1, "base_url": "u", "model": "a"}],
                               "max_fails": 2})
        a = pool.members[0]
        pool.note_fail(a)
        pool.note_ok(a, 1)
        self.assertEqual(pool.fails["a"], 0)
        pool.note_fail(a)
        self.assertEqual(pool.until[a.name], 0.0)


class Payload(unittest.TestCase):
    def test_openai_payload(self):
        m = G.Member("gpt", 1, "https://api.openai.com/v1", "gpt-5.2", "k", "openai")
        url, payload, headers = m.request({"messages": [{"role": "user", "content": "hi"}],
                                          "max_tokens": 64, "temperature": 0.2}, False)
        self.assertEqual(url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(json.loads(payload)["model"], "gpt-5.2")
        self.assertEqual(json.loads(payload)["max_tokens"], 64)
        self.assertEqual(headers["Authorization"], "Bearer k")

    def test_anthropic_payload_and_system_message(self):
        m = G.Member("claude", 1, "https://api.anthropic.com/v1", "claude-sonnet-4-5", "k", "anthropic")
        url, payload, headers = m.request({"messages": [{"role": "system", "content": "be brief"},
                                                        {"role": "user", "content": "hi"}],
                                          "max_tokens": 128}, False)
        body = json.loads(payload)
        self.assertEqual(url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(body["system"], "be brief")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(body["max_tokens"], 128)
        self.assertEqual(headers["x-api-key"], "k")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")

    def test_anthropic_answer_becomes_openai_shape(self):
        converted = G.openai_from_anthropic(
            {"content": [{"type": "text", "text": "Hello"}], "usage": {"input_tokens": 3, "output_tokens": 7}},
            "claude-sonnet-4-5")
        self.assertEqual(converted["choices"][0]["message"]["content"], "Hello")
        self.assertEqual(converted["usage"]["total_tokens"], 10)

    def test_bad_weight_is_rejected(self):
        with self.assertRaises(ValueError):
            G.Member("x", 0, "u", "m", "", "openai")
        with self.assertRaises(ValueError):
            G.Member("x", 1, "u", "m", "", "unknown-wire")


class Auth(unittest.TestCase):
    def test_key_check(self):
        gw = G.Gateway(registry_from(cfg([{"name": "a", "weight": 1, "base_url": "u", "model": "a"}])), "s3cret")
        self.assertTrue(gw.authorized("Bearer s3cret"))
        self.assertFalse(gw.authorized("Bearer wrong"))
        self.assertFalse(gw.authorized(""))
        self.assertTrue(G.Gateway(gw.registry, "").authorized(""))


class HttpServer(unittest.TestCase):
    """The real HTTP surface, on a random port, with members that never reach the network."""

    def setUp(self):
        self.registry = registry_from(cfg([{"name": "a", "weight": 1, "base_url": "http://127.0.0.1:1/nope", "model": "a"}]))
        self.server = G.ThreadingHTTPServer(("127.0.0.1", 0), G.make_handler(G.Gateway(self.registry, "key")))
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}/v1"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def _post(self, payload, token="key"):
        req = urllib.request.Request(self.base + "/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode())

    def test_bad_pool_name_answers_400_instead_of_dropping_the_connection(self):
        status, body = self._post({"model": "nope", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 400)
        self.assertIn("no pool named", body["error"]["message"])

    def test_missing_messages_answers_400(self):
        status, body = self._post({"model": "fast"})
        self.assertEqual(status, 400)
        self.assertIn("messages", body["error"]["message"])

    def test_member_that_cannot_be_reached_fails_over_and_reports_502(self):
        status, body = self._post({"model": "fast", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 502)
        self.assertEqual(self.registry.pools["fast"].errors["a"], 1)

    def test_models_and_health_are_listed(self):
        req = urllib.request.Request(self.base + "/models", headers={"Authorization": "Bearer key"})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode())
        self.assertEqual([m["id"] for m in data["data"]], ["fast"])
        self.assertEqual(data["data"][0]["members"], ["a"])
        req = urllib.request.Request(self.base + "/health", headers={"Authorization": "Bearer key"})
        with urllib.request.urlopen(req, timeout=5) as r:
            self.assertTrue(json.loads(r.read().decode())["ok"])

    def test_wrong_key_is_refused(self):
        try:
            urllib.request.urlopen(self.base + "/models", timeout=5)
            self.fail("a request with no key should not be accepted")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 401)


class Plan(unittest.TestCase):
    """`--plan` and the pool lookup share the same rule: a name that is not a pool is an error."""

    def test_planning_an_unknown_pool_is_an_error(self):
        reg = registry_from(cfg([{"name": "a", "weight": 1, "base_url": "u", "model": "a"}]))
        with self.assertRaises(ValueError):
            reg.plan("ghost", 3)

    def test_plan_with_no_name_uses_the_default_pool(self):
        reg = registry_from(cfg([{"name": "a", "weight": 1, "base_url": "u", "model": "a"}]))
        self.assertEqual(reg.plan(None, 2), ["a", "a"])


class PlanCli(unittest.TestCase):
    """The `--plan` flag prints the routing and sends nothing, and a bad pool name is a clean exit."""

    def test_plan_prints_the_split_and_rejects_an_unknown_pool(self):
        config = {"default_pool": "fast", "pools": {"fast": {"members": [
            {"name": "claude", "weight": 80, "base_url": "http://127.0.0.1:1", "model": "c"},
            {"name": "gpt", "weight": 20, "base_url": "http://127.0.0.1:1", "model": "g"}]}}}
        path = Path(tempfile.mktemp(suffix=".json"))
        path.write_text(json.dumps(config), encoding="utf-8")
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                self.assertEqual(G.main(["--config", str(path), "--plan", "--requests", "10"]), 0)
                self.assertEqual(G.main(["--config", str(path), "--plan", "--pool", "ghost"]), 2)
        finally:
            path.unlink()
        self.assertIn("claude", out.getvalue())
        self.assertIn("counts: claude=8, gpt=2", out.getvalue())
        self.assertIn("no pool named", err.getvalue())


class Streaming(unittest.TestCase):
    """The streaming path against a real local upstream that answers with SSE, one wire per provider."""

    class Collector:
        """Stand-in for the client connection: collects the status and the bytes written."""

        def __init__(self):
            self.status = 0
            self.headers = {}
            self.buf = io.BytesIO()

        def send_response(self, status):
            self.status = status

        def send_header(self, key, value):
            self.headers[key] = value

        def end_headers(self):
            pass

        @property
        def wfile(self):
            return self.buf

    def _upstream(self, events):
        """A local upstream that answers POST with SSE: `data: <json>\n\n` for each event."""
        body = b"".join(
            b"data: " + (ev.encode() if isinstance(ev, str) else json.dumps(ev).encode()) + b"\n\n"
            for ev in events)

        class Up(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def setup(self):
                super().setup()
                graceful_close(self.connection)   # the body below must survive the handler's close

            def do_POST(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()   # a real SSE server sends each event at once; without this the
                # stand-in closes the socket with the events still in its write buffer, which on Windows
                # aborts the connection now and then (WinError 10053, once in ~60 runs)
                # SO_LINGER alone is not enough on Windows: give the OS a moment to
                # deliver what the flush sent before the handler's close runs.
                time.sleep(0.25)

        srv = G.ThreadingHTTPServer(("127.0.0.1", 0), Up)
        t = threading.Thread(target=srv.serve_forever)
        t.start()
        return srv, t

    def _stream(self, pool_spec, upstream_port=None, failover_pool=None):
        config = {"default_pool": "fast", "pools": {"fast": {"members": pool_spec}}}
        if failover_pool:
            config["pools"]["failover"] = {"members": failover_pool}
        reg = registry_from(config)
        gw = G.Gateway(reg, "")
        collector = self.Collector()
        gw.stream(collector, {"model": "fast", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
        return collector, reg

    def test_openai_stream_passes_through_and_counts_tokens(self):
        srv, thread = self._upstream([{"choices": [{"delta": {"content": "hello"}}]},
                                     {"choices": [{"delta": {"content": " world"}}]}, "[DONE]"])
        try:
            collector, reg = self._stream(
                [{"name": "gpt", "weight": 1, "wire": "openai",
                  "base_url": f"http://127.0.0.1:{srv.server_port}", "model": "g"}])
            out = collector.buf.getvalue().decode()
            self.assertEqual(collector.status, 200)
            self.assertEqual(collector.headers["Content-Type"], "text/event-stream")
            self.assertIn("[DONE]", out)
            self.assertEqual(reg.pools["fast"].tokens["gpt"], len("hello world"))
            self.assertEqual(reg.pools["fast"].counts["gpt"], 1)
        finally:
            srv.shutdown()
            thread.join()
            srv.server_close()

    def test_anthropic_stream_is_converted_to_openai_chunks(self):
        srv, thread = self._upstream([{"type": "content_block_delta", "delta": {"text": "Hi"}},
                                     {"type": "message_stop"}])
        try:
            collector, reg = self._stream(
                [{"name": "claude", "weight": 1, "wire": "anthropic",
                  "base_url": f"http://127.0.0.1:{srv.server_port}", "model": "c"}])
            out = collector.buf.getvalue().decode()
            chunks = [line[6:] for line in out.split("\n\n") if line.startswith("data:")]
            self.assertEqual(json.loads(chunks[0])["choices"][0]["delta"]["content"], "Hi")
            self.assertEqual(chunks[-1], "[DONE]")
            self.assertEqual(reg.pools["fast"].tokens["claude"], 2)
        finally:
            srv.shutdown()
            thread.join()
            srv.server_close()

    def test_streaming_falls_over_to_the_next_member(self):
        srv, thread = self._upstream([{"choices": [{"delta": {"content": "backup"}}]}, "[DONE]"])
        try:
            collector, reg = self._stream(
                [{"name": "dead", "weight": 5, "wire": "openai", "base_url": "http://127.0.0.1:1", "model": "d"},
                 {"name": "live", "weight": 1, "wire": "openai",
                  "base_url": f"http://127.0.0.1:{srv.server_port}", "model": "l"}])
            self.assertEqual(collector.status, 200)
            self.assertIn("backup", collector.buf.getvalue().decode())
            self.assertEqual(reg.pools["fast"].errors["dead"], 1)
            self.assertEqual(reg.pools["fast"].counts["live"], 1)
        finally:
            srv.shutdown()
            thread.join()
            srv.server_close()


    def test_an_upstream_that_drops_the_connection_fails_over(self):
        """A socket that dies before a status line is a failure, not a crash: the next member answers."""
        srv, thread = self._upstream([{"choices": [{"delta": {"content": "backup"}}]}, "[DONE]"])

        class Drop(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                self.connection.close()          # no status line at all

        dead = G.ThreadingHTTPServer(("127.0.0.1", 0), Drop)
        dead_thread = threading.Thread(target=dead.serve_forever)
        dead_thread.start()
        try:
            collector, reg = self._stream(
                [{"name": "drop", "weight": 5, "wire": "openai",
                  "base_url": f"http://127.0.0.1:{dead.server_port}", "model": "d"},
                 {"name": "live", "weight": 1, "wire": "openai",
                  "base_url": f"http://127.0.0.1:{srv.server_port}", "model": "l"}])
            self.assertEqual(collector.status, 200)
            self.assertIn("backup", collector.buf.getvalue().decode())
            self.assertEqual(reg.pools["fast"].errors["drop"], 1)
            self.assertEqual(reg.pools["fast"].counts["live"], 1)
        finally:
            dead.shutdown()
            dead_thread.join()
            dead.server_close()
            srv.shutdown()
            thread.join()
            srv.server_close()

    def test_streaming_works_through_the_real_http_handler(self):
        """Route and stream together: a real client POSTs to /v1/chat/completions and gets the SSE back."""
        srv, thread = self._upstream([{"choices": [{"delta": {"content": "hi"}}]}, "[DONE]"])
        reg = registry_from(cfg([{"name": "gpt", "weight": 1, "wire": "openai",
                                 "base_url": f"http://127.0.0.1:{srv.server_port}", "model": "g"}]))
        gateway = G.ThreadingHTTPServer(("127.0.0.1", 0), G.make_handler(G.Gateway(reg, "key")))
        gw_thread = threading.Thread(target=gateway.serve_forever)
        gw_thread.start()
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{gateway.server_port}/v1/chat/completions",
                                        data=json.dumps({"model": "fast", "stream": True,
                                                        "messages": [{"role": "user", "content": "hi"}]}).encode(),
                                        headers={"Authorization": "Bearer key", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as r:
                out = r.read().decode()
            self.assertIn("hi", out)
            self.assertIn("[DONE]", out)
            self.assertEqual(reg.pools["fast"].tokens["gpt"], 2)
        finally:
            gateway.shutdown()
            gw_thread.join()
            gateway.server_close()
            srv.shutdown()
            thread.join()
            srv.server_close()

    def test_a_stream_the_upstream_cut_off_short_counts_as_a_failure(self):
        """A member that stops in the middle of a stream is not a member that answered."""

        class Cut(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def setup(self):
                super().setup()
                graceful_close(self.connection)   # the cut must arrive as an EOF, not as an abort

            def do_POST(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.write(b"12\ndata: hello\n")     # the closing chunk never comes
                self.wfile.flush()
                time.sleep(0.25)   # let the reader receive the cut-off stream before the close runs
                self.close_connection = True

        dead = G.ThreadingHTTPServer(("127.0.0.1", 0), Cut)
        thread = threading.Thread(target=dead.serve_forever)
        thread.start()
        try:
            collector, reg = self._stream(
                [{"name": "cut", "weight": 1, "wire": "openai",
                  "base_url": f"http://127.0.0.1:{dead.server_port}", "model": "c"}])
            self.assertEqual(collector.status, 200)
            self.assertEqual(reg.pools["fast"].errors["cut"], 1)
            self.assertEqual(reg.pools["fast"].counts["cut"], 0)
        finally:
            dead.shutdown()
            thread.join()
            dead.server_close()

    def test_a_client_that_walks_away_is_not_blamed_on_the_member(self):
        """The provider answered; only the reader went away, so the member stays healthy."""

        class WalksAway:
            def __init__(self):
                self.status = 0
                self.headers = {}
                self.writes = 0

            def send_response(self, status):
                self.status = status

            def send_header(self, key, value):
                self.headers[key] = value

            def end_headers(self):
                pass

            @property
            def wfile(self):
                self.writes += 1
                if self.writes > 1:
                    raise BrokenPipeError("the client closed the connection")
                return io.BytesIO()

        srv, thread = self._upstream([{"choices": [{"delta": {"content": "hi"}}]}, "[DONE]"])
        try:
            reg = registry_from(cfg([{"name": "gpt", "weight": 1, "wire": "openai",
                                     "base_url": f"http://127.0.0.1:{srv.server_port}", "model": "g"}]))
            gw = G.Gateway(reg, "")
            walker = WalksAway()
            gw.stream(walker, {"model": "fast", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
            self.assertEqual(walker.status, 200)
            self.assertEqual(walker.writes, 2)          # one chunk written, then the client is gone
            self.assertEqual(reg.pools["fast"].counts["gpt"], 1)
            self.assertEqual(reg.pools["fast"].errors["gpt"], 0)
        finally:
            srv.shutdown()
            thread.join()
            srv.server_close()

if __name__ == "__main__":
    unittest.main()
