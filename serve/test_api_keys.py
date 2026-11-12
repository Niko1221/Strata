"""serve/test_api_keys.py - #5004: "api_keys", several API keys each with its own admission limits
(max_concurrent, reserved_capacity), against the mock engine (no GPU, no pack).

    python -m unittest serve.test_api_keys -v
"""
from __future__ import annotations

import contextlib
import http.client
import io
import json
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (ByteTokenizer, ClientGone, KeyLimits, MockEngine, QueueWaitExpired,  # noqa: E402
                          Service, api_key_limits_of, key_matches, matched_key, serve)

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096


class MatchedKey(unittest.TestCase):
    """#5004: the auth check that answers WHICH key passed, without changing what key_matches answers."""

    def test_returns_the_configured_key(self):
        self.assertEqual(matched_key("k1", ["k1", "k2"]), "k1")
        self.assertEqual(matched_key("k2", "k1,k2"), "k2")
        self.assertIsNone(matched_key("k3", ["k1", "k2"]))
        self.assertIsNone(matched_key("", ["k1"]))
        self.assertTrue(key_matches("k2", ["k1", "k2"]))          # unchanged answers
        self.assertFalse(key_matches("k3", ["k1", "k2"]))

    def test_both_header_byte_forms_still_match(self):
        key = "schlüssel"                                          # #725: UTF-8 and Latin-1 both pass
        self.assertEqual(matched_key(key, [key]), key)
        self.assertEqual(matched_key(key.encode().decode("latin-1"), [key]), key)


class Parse(unittest.TestCase):
    """The "api_keys" block: what starts and what stops the start."""

    def limits(self, value, slots=2):
        return api_key_limits_of(value, slots)

    def test_the_shape_the_issue_asked_for(self):
        p = self.limits({"hermes-key": {"name": "hermes", "max_concurrent": 1},
                         "ha-key": {"name": "home assistant", "max_concurrent": 1, "reserved_capacity": 1}})
        self.assertEqual([(x["key"], x["max_concurrent"], x["reserved_capacity"]) for x in p],
                         [("hermes-key", 1, 0), ("ha-key", 1, 1)])
        self.assertEqual([x["name"] for x in p], ["hermes", "home assistant"])

    def test_a_bare_key_is_a_key_without_limits(self):
        p = self.limits({"a": None, "b": {}})
        self.assertEqual([(x["max_concurrent"], x["reserved_capacity"]) for x in p], [(None, 0), (None, 0)])
        self.assertEqual([x["name"] for x in p], ["key 1", "key 2"])

    def test_no_block_is_no_limits(self):
        self.assertEqual(self.limits(None), [])
        self.assertEqual(self.limits({}), [])

    def test_reservations_that_leave_a_key_no_slot_stop_the_start(self):
        with self.assertRaises(ValueError):
            self.limits({"a": {"reserved_capacity": 2}, "b": {"reserved_capacity": 2}}, slots=2)

    def test_bad_entries_stop_the_start(self):
        for bad in (["k1"], "k1", {"": {}}, {"  ": {}}, {"none": {}}, {"k": "max 1"},
                    {"k": {"max_concurrent": 0}}, {"k": {"max_concurrent": True}}, {"k": {"max_concurrent": "1"}},
                    {"k": {"reserved_capacity": -1}}, {"k": {"reserved_capacity": 1.5}},
                    {"k": {"nonsense": 1}}, {"k": {"max_concurrent": 1, "max_concurent": 1}}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.limits(bad)

    def test_config_errors_name_the_place_not_the_key(self):
        secret = "s3cret-key"            # a failed start is read off the console and pasted into reports
        for bad in ({secret: "max 1"}, {secret: {"nonsense": 1}}, {secret: {"max_concurrent": 0}},
                    {secret: {"max_concurrent": "1"}}, {secret: {"reserved_capacity": -1}},
                    {secret: {"reserved_capacity": 1.5}}, {secret: {"name": KeyLimits.OTHER}}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as got:
                    self.limits(bad)
                self.assertNotIn(secret, str(got.exception))

    def test_names_are_unique_and_not_the_shared_bucket(self):
        with self.assertRaises(ValueError):
            self.limits({"a": {"name": "x"}, "b": {"name": "x"}})
        with self.assertRaises(ValueError):
            self.limits({"a": {"name": KeyLimits.OTHER}})


class Gate(unittest.TestCase):
    """KeyLimits: who may be inside at once, and what frees them."""

    def make(self, specs, slots=2):
        return KeyLimits(api_key_limits_of(specs, slots), lambda: slots)

    def test_a_key_at_its_limit_waits_and_the_other_key_does_not(self):
        g = self.make({"hk": {"name": "hermes", "max_concurrent": 1},
                       "ak": {"name": "assistant", "max_concurrent": 1}})
        g.acquire("hk")
        g.acquire("ak")                                    # hermes being busy holds nobody else up
        err = []
        done = threading.Event()

        def second():
            try:
                g.acquire("hk")
                g.release("hk")
            except Exception as e:                         # noqa: BLE001
                err.append(e)
            done.set()

        threading.Thread(target=second, daemon=True).start()
        self.assertFalse(done.wait(0.4), "the second hermes request got in while the first held the key")
        g.release("hk")
        self.assertTrue(done.wait(5), "the second hermes request never went in after the first freed the key")
        self.assertEqual(err, [])

    def test_reserved_capacity_is_out_of_reach_of_the_other_keys(self):
        # slots 2, one reserved for "ha": the keys with no reservation share the other one, together
        g = self.make({"ha": {"name": "ha", "reserved_capacity": 1}})
        g.acquire("plain-1")                               # the shared bucket (reach: 2 - 1)
        with self.assertRaises(QueueWaitExpired):          # a second unreserved key would eat the reserved slot
            g.acquire("plain-2", deadline=time.monotonic() + 0.05)
        g.acquire("ha")                                    # the reserved slot is always its reach
        g.release("plain-1")
        g.release("ha")
        self.assertEqual(sum(g.held.values()), 0)

    def test_max_concurrent_beyond_its_reach_is_the_reach_that_holds(self):
        g = self.make({"ha": {"name": "ha", "reserved_capacity": 1},
                       "hk": {"name": "hermes", "max_concurrent": 2}}, slots=2)
        g.acquire("hk")                                    # reach: 2 - 1 reserved for ha
        with self.assertRaises(QueueWaitExpired):
            g.acquire("hk", deadline=time.monotonic() + 0.05)
        g.release("hk")

    def test_the_queue_timeout_answers_a_wait_that_runs_out(self):
        g = self.make({"hk": {"name": "hermes", "max_concurrent": 1}})
        g.acquire("hk")
        with self.assertRaises(QueueWaitExpired):
            g.acquire("hk", deadline=time.monotonic() + 0.05)

    def test_a_client_that_hangs_up_mid_wait_ends_it(self):
        g = self.make({"hk": {"name": "hermes", "max_concurrent": 1}})
        g.acquire("hk")
        gone = threading.Event()
        gone.set()
        with self.assertRaises(ClientGone):
            g.acquire("hk", cancel=gone)

    def test_counts_return_when_requests_end(self):
        g = self.make({"hk": {"name": "hermes"}})
        for _ in range(3):
            g.acquire("hk")
            g.release("hk")
        g.acquire("plain")
        snap = g.snapshot()
        self.assertEqual(snap["unlisted"]["inflight"], 1)
        self.assertEqual(snap["keys"][0]["inflight"], 0)
        g.release("plain")
        snap = g.snapshot()
        self.assertEqual(sum(r["inflight"] for r in snap["keys"]), 0)
        self.assertEqual(snap["unlisted"]["inflight"], 0)
        self.assertNotIn("hermes-key", json.dumps(snap))    # names and counts only, never a key


class OverHttp(unittest.TestCase):
    """The limits in the running server: keys that carry a policy, a plain key without one, and what
    GET /status says while requests are inside.  The mock answers slowly (delay_s per token) so two
    requests really overlap."""

    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "</think>\n\nok", max_context=CTX, delay_s=0.3), tok,
                           ChatTemplate(ROOT / "serve" / "chat_template.jinja"))
        policies = api_key_limits_of({"hermes-key": {"name": "hermes", "max_concurrent": 1},
                                      "ha-key": {"name": "home assistant", "max_concurrent": 1,
                                                 "reserved_capacity": 1}}, slots=2)
        self.svc.api_key = ["hermes-key", "ha-key", "plain-key"]     # as the start merges api_keys into api_key
        self.svc.key_limits = KeyLimits(policies, lambda: 2)
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.svc.key_limits = None

    def req(self, method, path, body=None, key=None, timeout=30):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        h = {"Content-Type": "application/json"}
        if key:
            h["Authorization"] = f"Bearer {key}"
        data = json.dumps(body).encode() if body is not None else None
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                c.request(method, path, body=data, headers=h)
                r = c.getresponse()
                raw = r.read()
            return r.status, (json.loads(raw) if raw[:1] in (b"{", b"[") else raw), out.getvalue()
        finally:
            c.close()

    def chat(self, key):
        return self.req("POST", "/v1/chat/completions",
                        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64}, key=key)

    def status(self):
        return self.req("GET", "/status", key="ha-key")[1]

    def wait_for_inflight(self, name, want, timeout=10.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            rows = {r["name"]: r["inflight"] for r in self.status()["api_keys"]["keys"]}
            if rows.get(name, 0) >= want:
                return True
            time.sleep(0.01)
        return False

    def test_wrong_key_is_refused_and_a_policy_key_answers(self):
        self.assertEqual(self.chat("nope")[0], 401)
        code, body, _ = self.chat("ha-key")
        self.assertEqual(code, 200)
        self.assertIn("ok", body["choices"][0]["message"]["content"])

    def test_a_key_the_block_does_not_name_is_a_plain_key(self):
        code, _, _ = self.chat("plain-key")
        self.assertEqual(code, 200)
        self.assertEqual(self.status()["api_keys"]["unlisted"]["max"], 1)   # 2 slots - 1 reserved for ha

    def test_two_requests_of_one_key_never_run_at_once_while_another_key_gets_in(self):
        out = {}
        t1 = threading.Thread(target=lambda: out.update(h1=self.chat("hermes-key")), daemon=True)
        t1.start()
        self.assertTrue(self.wait_for_inflight("hermes", 1), "the first hermes request never showed in /status")
        t2 = threading.Thread(target=lambda: out.update(h2=self.chat("hermes-key")), daemon=True)
        t2.start()
        time.sleep(0.5)                                    # the second request is waiting at the key's limit now
        self.assertEqual(self.status()["api_keys"]["keys"][0]["inflight"], 1,
                         "two requests of the capped key were inside at once")
        code, _, _ = self.chat("ha-key")                   # latency-sensitive client: straight in, not queued behind
        self.assertEqual(code, 200)
        t1.join(20)
        t2.join(20)
        self.assertEqual(out["h1"][0], 200)
        self.assertEqual(out["h2"][0], 200)
        self.assertNotIn("hermes-key", json.dumps(self.status()))           # /status shows names only

    def test_the_queue_timeout_answers_a_key_wait_like_any_other_wait(self):
        self.svc.queue_timeout_s = 0.2                     # like --queue-timeout on a latency-sensitive client
        try:
            out = {}
            t1 = threading.Thread(target=lambda: out.update(first=self.chat("hermes-key")), daemon=True)
            t1.start()
            self.assertTrue(self.wait_for_inflight("hermes", 1))
            code, body, _ = self.chat("hermes-key")        # the key is full and the wait runs out: 429
            self.assertEqual(code, 429)
            self.assertEqual(body["error"]["type"], "rate_limit_error")
            t1.join(20)
            self.assertEqual(out["first"][0], 200)
        finally:
            self.svc.queue_timeout_s = None
        self.assertEqual(self.status()["api_keys"]["keys"][0]["inflight"], 0)


if __name__ == "__main__":
    unittest.main()
