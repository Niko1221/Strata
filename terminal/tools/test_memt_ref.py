"""tools/test_memt_ref.py - the memt_sglang reference driver, against a stub server.

    python tools/test_memt_ref.py [-v]

The stub implements the routes the reference README documents (`/memt/identity`,
`/memt/capabilities`, `/memt/control`, `/generate`, SSE included) and checks every
generation payload with the reference repository's own
`memt_sglang.protocol.validate_payload` when that checkout is available, so a
drift in this client's wire format fails here rather than on the GPU host.  That
check needs Python 3.11+ (the reference's own floor); under 3.9 the test skips it.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import memt_ref as R  # noqa: E402

#: The memt_sglang checkout, if the machine has one (it is a separate project).
#: The cross-check below is optional: set MEMT_SGLANG_PATH to its `inference/`
#: directory to have the reference's own payload validator run.
REFERENCE = Path(os.environ["MEMT_SGLANG_PATH"]) if os.environ.get("MEMT_SGLANG_PATH") else None


def reference_validator():
    """The reference's own payload validator, if that checkout is here (it needs Python 3.11+)."""
    if REFERENCE is None:
        return None
    if str(REFERENCE) not in sys.path:
        sys.path.insert(0, str(REFERENCE))
    try:
        from memt_sglang.protocol import validate_payload  # noqa: F401
    except Exception:   # pragma: no cover - no checkout, or Python 3.9 where its syntax fails
        return None
    return validate_payload


IDENTITY = "ab" * 32
ENCODING = "qwen-text-raw-v1"


class Stub:
    """The documented routes over an in-memory session table."""

    def __init__(self):
        self.epoch = "epoch-1"
        self.sessions: dict[str, dict] = {}
        self.exports: dict[str, dict] = {}
        self.payloads: list[dict] = []
        self.control_calls: list[dict] = []
        self.stream_frames = [[7, 8], [9]]
        self.stream_no_done = False
        self.bad_rid = False
        self.bad_finish = False
        self.bad_format = False
        self.bad_import_format = False
        self.publish_status = "published"
        self.all_in_terminal = False
        self.validate = reference_validator()

    # ---- routes
    def identity(self):
        return {"model": "memt-nano", "configuration": "0" * 64, "memt_runtime": {}}

    def capabilities(self):
        return {"protocol": "memt-http-v1", "epoch": self.epoch, "memory_snapshot": "memt-memory-v1",
                "chunk_tokens": 4, "max_prefill_tokens": 64, "snapshot_identity": IDENTITY,
                "memory_encoding": ENCODING}

    def control(self, body):
        self.control_calls.append(body)
        if "epoch" in body and body["epoch"] is not None and body["epoch"] != self.epoch:
            return 409, {"error": "engine epoch changed; create a new session"}
        action = body.get("action")
        session = body.get("session")
        lease = body.get("lease")
        if not isinstance(session, str) or not session:
            return 400, {"error": "no session"}
        if action == "create":
            if session in self.sessions:
                return 409, {"error": "session exists"}
            self.sessions[session] = {"lease": None, "transaction": None, "published": [], "pending": []}
            return 200, {"epoch": self.epoch, "memory_read": body.get("memory_read", True)}
        state = self.sessions.get(session)
        if state is None:
            return 409, {"error": "unknown session"}
        if action == "acquire":
            state["lease"] = "lease-" + session
            return 200, {"lease": state["lease"], "ttl": body.get("ttl", 300)}
        if lease != state["lease"]:
            return 409, {"error": "invalid lease"}
        if action == "begin":
            state["transaction"] = body.get("transaction")
            state["pending"] = []
            state["replace"] = False
            return 200, {"status": "open"}
        if action == "publish":
            if state.get("replace"):
                state["published"] = list(state["pending"])   # an import replaces the memory
            else:
                state["published"].extend(state["pending"])
            state["pending"] = []
            state["transaction"] = None
            state["replace"] = False
            return 200, {"status": self.publish_status}
        if action == "status":
            return 200, {"status": "published" if state["transaction"] is None else "open"}
        if action == "export":
            if body.get("encoding") != ENCODING:
                return 400, {"error": "Snapshot codec does not match the server's HF identity"}
            name = body.get("file")
            if not isinstance(name, str) or not name.endswith(".safetensors"):
                return 400, {"error": "bad snapshot name"}
            if name in self.exports and not body.get("overwrite"):
                return 409, {"error": "exists"}
            self.exports[name] = {"name": name, "identity": IDENTITY, "tensors": 10,
                                  "committed": list(state["published"])}
            return 200, {"file": name, "format": "memt-memory-v1" if not self.bad_format else "other",
                         "has_memory": True}
        if action == "import":
            if body.get("encoding") != ENCODING:
                return 400, {"error": "Snapshot codec mismatch"}
            snapshot = self.exports.get(body.get("file"))
            if snapshot is None:
                return 400, {"error": "no such snapshot"}
            state["pending"] = list(snapshot["committed"])
            state["replace"] = True
            return 200, {"file": snapshot["name"],
                         "format": "other" if self.bad_import_format else "memt-memory-v1",
                         "has_memory": True}
        if action in ("reset", "close", "release"):
            return 200, {"status": "ok"}
        return 400, {"error": "unknown action"}

    def generate(self, payload):
        self.payloads.append(payload)
        if self.validate is not None:
            self.validate(payload, max_prefill_tokens=64)
        rid = payload["rid"]
        if self.bad_rid:
            rid = rid + "x"
        parts = payload["rid"].split("|")
        if len(parts) != 4 or parts[0] != "memt":
            return 400, {"error": "bad rid"}
        session, action = parts[1], parts[2]
        state = self.sessions.get(session)
        if state is None:
            return 409, {"error": "unknown session"}
        if payload.get("memt_lease") != state["lease"]:
            return 409, {"error": "invalid lease"}
        meta = {"id": rid}
        finish = {"type": "stop"}
        if self.bad_finish:
            finish = {"type": "abort"}
        if action.startswith("commit:"):
            if payload.get("memt_transaction") is None:
                return 409, {"error": "a commit needs a transaction"}
            if state["transaction"] is None:
                return 409, {"error": "no open transaction"}
            state["pending"].extend(payload["input_ids"])
            return 200, {"text": "", "output_ids": [], "meta_info": dict(meta, finish_reason={"type": "length"})}
        if action != "query":
            return 400, {"error": "unknown operation"}
        if not payload.get("stream"):
            return 200, {"text": "hi", "output_ids": [7, 8],
                         "meta_info": dict(meta, finish_reason=finish)}
        return "stream", None, [frame for frame in self.stream_frames]


def make_handler(stub: Stub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status, value):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/memt/identity":
                return self._send(200, stub.identity())
            if path == "/memt/capabilities":
                return self._send(200, stub.capabilities())
            self._send(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw)
            except ValueError:
                return self._send(400, {"error": "not json"})
            path = self.path.split("?")[0]
            if path == "/memt/control":
                status, value = stub.control(body)
                return self._send(status, value)
            if path != "/generate":
                return self._send(404, {"error": "not found"})
            answer = stub.generate(body)
            if answer[0] != "stream":
                status, value = answer
                return self._send(status, value)
            frames = [] if stub.all_in_terminal else answer[2]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            cumulative = [token for frame in stub.stream_frames for token in frame] if stub.all_in_terminal else []
            rid = body["rid"]
            for frame in frames:
                cumulative += frame
                self.wfile.write(("data: " + json.dumps({"text": "", "output_ids": list(cumulative),
                                                         "meta_info": {"id": rid}}) + "\n\n").encode())
                self.wfile.flush()
            self.wfile.write(("data: " + json.dumps({"text": "", "output_ids": list(cumulative),
                                                     "meta_info": {"id": rid,
                                                                   "finish_reason": {"type": "stop"}}})
                              + ("" if stub.stream_no_done else "\n\ndata: [DONE]\n\n")).encode())
            self.wfile.flush()

    return Handler


class Driver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub = Stub()
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.stub))
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        self.stub.payloads.clear()
        self.stub.control_calls.clear()
        self.ref = R.Reference("http://127.0.0.1:%d" % self.port)

    def session(self):
        caps = self.ref.capabilities()
        self.assertEqual(caps["snapshot_identity"], IDENTITY)
        session = self.ref.create()
        return session, self.ref.acquire(session)

    def test_capabilities_sets_epoch_and_identity(self):
        caps = self.ref.capabilities()
        self.assertEqual(self.ref.epoch, "epoch-1")
        self.assertEqual(self.ref.snapshot_identity(), IDENTITY)
        self.assertEqual(caps["memory_encoding"], ENCODING)

    def test_commit_and_query(self):
        session, lease = self.session()
        self.ref.begin(session, lease, "t1")
        self.ref.commit(session, [1, 2, 3], lease=lease, transaction="t1")
        payload = self.stub.payloads[-1]
        self.assertEqual(payload["rid"].split("|")[2], "commit:3")
        self.assertEqual(payload["sampling_params"]["max_new_tokens"], 0)
        self.assertEqual(payload["memt_transaction"], "t1")
        self.assertEqual(self.stub.sessions[session]["pending"], [1, 2, 3])
        self.ref.publish(session, lease, "t1")
        self.assertEqual(self.stub.sessions[session]["published"], [1, 2, 3])
        answer = self.ref.query(session, [4, 5], lease=lease, max_new=2)
        self.assertEqual(answer["ids"], [7, 8])
        self.assertEqual(answer["finish"], "stop")
        self.assertIsNone(answer["first_ms"])
        self.assertGreater(answer["end_ms"], 0.0)

    def test_streamed_query_reports_first_token(self):
        session, lease = self.session()
        answer = self.ref.query(session, [4], lease=lease, max_new=3, stream=True)
        self.assertEqual(answer["ids"], [7, 8, 9])
        self.assertEqual(answer["frames"], 2)
        self.assertIsNotNone(answer["first_ms"])
        self.assertGreaterEqual(answer["end_ms"], answer["first_ms"])

    def test_all_tokens_in_the_terminal_frame_still_have_a_ttft(self):
        session, lease = self.session()
        self.stub.all_in_terminal = True
        try:
            answer = self.ref.query(session, [4], lease=lease, max_new=3, stream=True)
        finally:
            self.stub.all_in_terminal = False
        self.assertEqual(answer["ids"], [7, 8, 9])
        self.assertTrue(answer["first_from_terminal"])
        self.assertEqual(answer["frames"], 1)
        self.assertIsNotNone(answer["first_ms"])

    def test_a_failed_publish_is_refused(self):
        session, lease = self.session()
        self.ref.begin(session, lease, "t1")
        self.ref.commit(session, [1], lease=lease, transaction="t1")
        self.stub.publish_status = "open"
        try:
            with self.assertRaises(R.ReferenceError) as caught:
                self.ref.publish(session, lease, "t1")
        finally:
            self.stub.publish_status = "published"
        self.assertIn("published", str(caught.exception))

    def test_an_import_with_another_format_is_refused(self):
        session, lease = self.session()
        self.ref.begin(session, lease, "t1")
        self.ref.commit(session, [1], lease=lease, transaction="t1")
        self.ref.publish(session, lease, "t1")
        self.ref.export(session, lease, "a.safetensors")
        self.stub.bad_import_format = True
        try:
            with self.assertRaises(R.ReferenceError) as caught:
                self.ref.import_(session, lease, "t2", "a.safetensors")
        finally:
            self.stub.bad_import_format = False
        self.assertIn("format", str(caught.exception))

    def test_stream_without_done_is_not_success(self):
        session, lease = self.session()
        self.stub.stream_no_done = True
        try:
            with self.assertRaises(R.ReferenceError) as caught:
                self.ref.query(session, [4], lease=lease, max_new=2, stream=True)
        finally:
            self.stub.stream_no_done = False
        self.assertIn("[DONE]", str(caught.exception))

    def test_export_import(self):
        session, lease = self.session()
        self.ref.begin(session, lease, "t1")
        self.ref.commit(session, [1], lease=lease, transaction="t1")
        self.ref.publish(session, lease, "t1")
        result = self.ref.export(session, lease, "p6.safetensors")
        self.assertEqual(result["file"], "p6.safetensors")
        self.assertEqual(self.stub.exports["p6.safetensors"]["committed"], [1])
        with self.assertRaises(R.ReferenceError):
            self.ref.export(session, lease, "p6.safetensors")   # no overwrite
        self.assertEqual(self.ref.export(session, lease, "p6.safetensors", overwrite=True)["file"],
                         "p6.safetensors")
        self.ref.begin(session, lease, "t2")
        self.ref.import_(session, lease, "t2", "p6.safetensors")
        self.ref.publish(session, lease, "t2")
        self.assertEqual(self.stub.sessions[session]["published"], [1])

    def test_commit_length_matches_rid_length(self):
        session, lease = self.session()
        self.ref.begin(session, lease, "t1")
        with self.assertRaises(ValueError):
            self.ref.commit(session, list(range(70)), lease=lease, transaction="t1")

    def test_sampling_reserved_keys_are_refused(self):
        session, lease = self.session()
        for reserved in ({"n": 4}, {"ignore_eos": True}, {"max_new_tokens": 99}, {"skip_special_tokens": True}):
            with self.assertRaises(ValueError):
                self.ref.query(session, [1], lease=lease, max_new=3, sampling=reserved)
        self.assertEqual(self.stub.payloads, [])   # nothing reached the server

    def test_a_wrong_rid_is_refused(self):
        session, lease = self.session()
        self.stub.bad_rid = True
        try:
            with self.assertRaises(R.ReferenceError) as caught:
                self.ref.query(session, [1], lease=lease, max_new=1)
        finally:
            self.stub.bad_rid = False
        self.assertIn("rid", str(caught.exception))

    def test_an_aborted_finish_is_refused(self):
        session, lease = self.session()
        self.stub.bad_finish = True
        try:
            with self.assertRaises(R.ReferenceError) as caught:
                self.ref.query(session, [1], lease=lease, max_new=1)
        finally:
            self.stub.bad_finish = False
        self.assertIn("finish_reason", str(caught.exception))

    def test_a_wrong_export_format_is_refused(self):
        session, lease = self.session()
        self.stub.bad_format = True
        try:
            with self.assertRaises(R.ReferenceError) as caught:
                self.ref.export(session, lease, "x.safetensors")
        finally:
            self.stub.bad_format = False
        self.assertIn("format", str(caught.exception))

    def test_a_stale_epoch_is_refused(self):
        self.ref.capabilities()
        self.ref.epoch = "stale"
        with self.assertRaises(R.ReferenceError) as caught:
            self.ref.create()
        self.assertEqual(caught.exception.status, 409)

    def test_greedy_sampling_kept(self):
        session, lease = self.session()
        self.ref.query(session, [1], lease=lease, max_new=1, sampling={"temperature": 0.0})
        self.assertEqual(self.stub.payloads[-1]["sampling_params"]["temperature"], 0.0)

    def test_error_status_is_kept(self):
        self.ref.capabilities()
        with self.assertRaises(R.ReferenceError) as caught:
            self.ref.control("nope" * 1, "acquire", ttl=1)
        self.assertEqual(caught.exception.status, 409)
        self.assertFalse(caught.exception.uncertain)

    def test_bad_identifier_is_refused_locally(self):
        with self.assertRaises(ValueError):
            self.ref.control("bad session", "acquire")

    def test_wait_ready(self):
        self.assertTrue(self.ref.wait_ready(timeout=5.0)["epoch"] == "epoch-1")

    def test_reference_validator_ran(self):
        if self.stub.validate is None:
            self.skipTest("set MEMT_SGLANG_PATH to the memt_sglang checkout to run this check (it also needs Python 3.11+)")
        session, lease = self.session()
        self.ref.begin(session, lease, "t1")
        self.ref.commit(session, [1, 2], lease=lease, transaction="t1")
        self.ref.query(session, [3], lease=lease, max_new=1)
        self.assertEqual(len(self.stub.payloads), 2)


if __name__ == "__main__":
    unittest.main()
