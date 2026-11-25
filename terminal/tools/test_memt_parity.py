"""tools/test_memt_parity.py - the P6 procedure over a stub reference and a scripted engine.

    python tools/test_memt_parity.py [-v]

No GPU and no model: the stub HTTP server implements the documented reference
routes and writes real `memt-memory-v1` files, the scripted peer answers the
engine's control lines and writes its snapshot the same way.  That exercises the
real drivers (`memt_ref.Reference`, `memt_engine.MemtEngine`), the real files and
the comparison, so the procedure itself is known to work before it meets a GPU.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import memt_engine as ME  # noqa: E402
import memt_harness as H  # noqa: E402
import memt_parity as P  # noqa: E402
import memt_ref as R  # noqa: E402
import memt_snapshot as S  # noqa: E402

ENCODING = "qwen-text-raw-v1"
LAYERS = (3, 7)


def checkpoint(root: Path, *, hidden: int = 2048) -> str:
    """A synthetic HF checkpoint directory; returns the `memt-memory-v1` identity it hashes to."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({
        "model_type": "memt_qwen3_5_moe",
        "backbone_configs": {"hidden_size": hidden},
        "memt": {"version": 1, "head_count": 16, "head_dim": 256, "gate_rank": 256},
    }))
    (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ x }}"}))
    (root / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"a": 0}}}))
    return ME.identity_from_model_dir(root)


def memory(base: float) -> dict:
    """The state both sides commit: one [1, H, D, D] tensor per memory layer."""
    return {layer: np.full((1, 2, 2, 2), base + layer, dtype=np.float32) for layer in LAYERS}


def token_of(layers: dict) -> int:
    # the stub's "generation": a number carried by the memory itself, so a
    # different state or a failed round trip changes the token
    return int(np.sum(layers[LAYERS[0]])) % 100000


class Stub:
    def __init__(self, state_dir: Path, identity: str):
        self.state_dir = state_dir
        self.identity = identity
        self.epoch = "epoch-p6"
        self.sessions: dict[str, dict] = {}
        self.exported: list[str] = []
        self.imported: list[str] = []
        self.payloads: list[dict] = []
        self.import_no_memory = False

    def capabilities(self):
        return {"protocol": "memt-http-v1", "epoch": self.epoch, "memory_snapshot": "memt-memory-v1",
                "chunk_tokens": 2048, "max_prefill_tokens": 16384, "snapshot_identity": self.identity,
                "memory_encoding": ENCODING}

    def control(self, body):
        action, session = body.get("action"), body.get("session")
        if action == "create":
            self.sessions[session] = {"lease": None, "memory": None, "pending": None}
            return 200, {"epoch": self.epoch, "memory_read": True}
        state = self.sessions.get(session)
        if state is None:
            return 409, {"error": "unknown session"}
        if action == "acquire":
            state["lease"] = "L-" + session
            return 200, {"lease": state["lease"]}
        if body.get("lease") != state["lease"]:
            return 409, {"error": "invalid lease"}
        if action == "begin":
            state["pending"] = None
            return 200, {"status": "open"}
        if action == "publish":
            if state["pending"] is not None:
                state["memory"] = state["pending"]
            return 200, {"status": "published"}
        if action == "export":
            name = body["file"]
            path = self.state_dir / name
            S.save(path, self.identity, state["memory"] or {}, overwrite=bool(body.get("overwrite")))
            self.exported.append(name)
            return 200, {"file": name, "format": "memt-memory-v1", "has_memory": True}
        if action == "import":
            layers = S.load(self.state_dir / body["file"], self.identity)
            state["pending"] = layers
            self.imported.append(body["file"])
            return 200, {"file": body["file"], "format": "memt-memory-v1",
                         "has_memory": not self.import_no_memory}
        if action in ("release", "close", "reset"):
            return 200, {"status": "ok"}
        return 400, {"error": "unknown action"}

    def generate(self, payload):
        self.payloads.append(payload)
        parts = payload["rid"].split("|")
        session, action = parts[1], parts[2]
        state = self.sessions.get(session)
        if state is None:
            return 409, {"error": "unknown session"}
        if payload.get("memt_lease") != state["lease"]:
            return 409, {"error": "invalid lease"}
        if action.startswith("commit:"):
            state["pending"] = memory(1.0)
            return 200, {"text": "", "output_ids": [],
                         "meta_info": {"id": payload["rid"], "finish_reason": {"type": "length"}}}
        if state["memory"] is None:
            return 500, {"error": "nothing committed"}
        token = token_of(state["memory"])
        return 200, {"text": "x", "output_ids": [token],
                     "meta_info": {"id": payload["rid"], "finish_reason": {"type": "stop"}}}


def handler(stub: Stub):
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
            if path == "/memt/capabilities":
                return self._send(200, stub.capabilities())
            if path == "/memt/identity":
                return self._send(200, {"model": "memt-nano"})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            path = self.path.split("?")[0]
            if path == "/memt/control":
                status, value = stub.control(body)
                return self._send(status, value)
            if path == "/generate":
                status, value = stub.generate(body)
                if status != 200 or not body.get("stream"):
                    return self._send(status, value)
                rid = body["rid"]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                frame = {"text": "x", "output_ids": value["output_ids"], "meta_info": {"id": rid}}
                self.wfile.write(("data: " + json.dumps(frame) + "\n\n").encode())
                self.wfile.write(("data: " + json.dumps(dict(frame, meta_info={
                    "id": rid, "finish_reason": {"type": "stop"}})) + "\n\ndata: [DONE]\n\n").encode())
                self.wfile.flush()
                return None
            self._send(404, {"error": "not found"})

    return Handler


class Peer:
    """A scripted engine peer that keeps a memory and writes it like the real one."""

    def __init__(self, state_dir: Path, identity: str, *, base: float = 1.0, scale: float = 1.0):
        self.state_dir = state_dir
        self.identity = identity
        self.memory = memory(base)
        self.scale = scale
        self.answers: deque[str] = deque()
        self.sent: list[str] = []

    def _state(self) -> dict:
        return {layer: value * np.float32(self.scale) for layer, value in self.memory.items()}

    def write(self, line: str) -> None:
        self.sent.append(line)
        command, _, rest = line.rstrip("\n").partition(" ")
        if command == "MEMCOMMIT":
            self.answers.append("MEMDONE %d %d 3.5\n" % (len(rest.split(",")), len(rest.split(","))))
        elif command == "MEMSNAP":
            path = rest.split()[0]
            S.save(Path(path), self.identity, self._state(), overwrite=True)
            self.answers.append("MEMSNAP OK\n")
        elif command == "MEMQUERY":
            self.answers.append("T %d\n" % token_of(self._state()))
            self.answers.append("DONE 1 2 1.0 2.0 stop 0 0 2 0 0 0 0 0.0 2 0\n")
        elif command in ("MEMRESET", "MEMFORGET", "MEMLOAD"):
            self.answers.append("%s OK\n" % command)
        else:
            self.answers.append("ERR unknown\n")

    def readline(self) -> str:
        return self.answers.popleft() if self.answers else ""

    def engine(self) -> ME.MemtEngine:
        return ME.MemtEngine(self.write, self.readline)


class Procedure(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state_dir = self.root / "state"
        self.state_dir.mkdir()
        self.model_dir = self.root / "model"
        self.identity = checkpoint(self.model_dir)
        self.stub = Stub(self.state_dir, self.identity)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler(self.stub))
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.ref = R.Reference("http://127.0.0.1:%d" % self.port)
        self.peer = Peer(self.root, self.identity)
        self.prompts = [("q1", [11, 12, 13]), ("q2", [21])]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
        self.tmp.cleanup()

    def run_parity(self, **kwargs):
        options = dict(commit_ids=[1, 2, 3], prompts=self.prompts, outdir=self.root / "out",
                       ref_state_dir=self.state_dir, max_new=2)
        options.update(kwargs)
        return P.run_parity(self.ref, self.peer.engine(), **options)

    def test_matching_state_and_tokens_pass(self):
        report = self.run_parity()
        self.assertTrue(report["ok"], P.format_report(report))
        snapshot = report["comparisons"]["snapshot"]
        self.assertTrue(snapshot["ok"])
        self.assertTrue(snapshot["layers"]["3"]["bit_identical"])
        self.assertTrue(report["comparisons"]["tokens"]["ok"])
        self.assertTrue(report["comparisons"]["roundtrip"]["ok"])
        self.assertTrue(report["comparisons"]["roundtrip_state"]["ok"])
        self.assertEqual(self.stub.imported, [report["engine"]["snapshot"]])
        self.assertTrue((self.root / "out" / "parity.json").exists())
        formatted = P.format_report(report)
        self.assertIn("roundtrip", formatted)
        self.assertIn("result: PASS", formatted)

    def test_state_difference_fails(self):
        self.peer.scale = -1.0
        report = self.run_parity()
        self.assertFalse(report["ok"])
        self.assertFalse(report["comparisons"]["snapshot"]["ok"])
        self.assertFalse(report["comparisons"]["tokens"]["ok"])
        self.assertIn("result: FAIL", P.format_report(report))

    def test_token_drift_can_be_allowed(self):
        self.peer.scale = -1.0
        report = self.run_parity(allow_token_drift=True)
        self.assertTrue(report["comparisons"]["tokens"]["ok"])
        self.assertFalse(report["comparisons"]["snapshot"]["ok"])   # the state is still wrong
        self.assertFalse(report["ok"])

    def test_no_state_dir_is_not_a_pass(self):
        report = self.run_parity(ref_state_dir=None)
        self.assertFalse(report["ok"])
        self.assertIn("skipped", report["comparisons"]["snapshot"])
        self.assertIn("NOT COMPARED", P.format_report(report))

    def test_roundtrip_can_be_skipped(self):
        report = self.run_parity(roundtrip=False)
        self.assertIn("skipped", report["comparisons"]["roundtrip"])
        self.assertFalse(report["ok"])

    def test_missing_exported_file_is_an_error(self):
        with self.assertRaises(R.ReferenceError):
            self.run_parity(ref_state_dir=self.root / "elsewhere")

    def test_engine_load_is_used(self):
        report = self.run_parity()
        self.assertTrue(report["ok"])
        self.assertTrue(any(line.startswith("MEMFORGET") for line in self.peer.sent))
        self.assertTrue(any(line.startswith("MEMSNAP") for line in self.peer.sent))
        self.assertTrue(report["engine"]["snapshot"].endswith(".safetensors"))

    def test_both_sides_are_asked_to_be_greedy(self):
        self.run_parity()
        queries = [payload for payload in self.stub.payloads if "|query|" in payload["rid"]]
        self.assertTrue(queries)
        for payload in queries:
            self.assertEqual(payload["sampling_params"]["temperature"], 0.0)
        engine_queries = [line for line in self.peer.sent if line.startswith("MEMQUERY")]
        self.assertTrue(engine_queries)
        for line in engine_queries:
            self.assertIn("temperature=0.0", line)

    def test_a_matching_engine_checkpoint_is_accepted(self):
        report = self.run_parity(engine_model_dir=self.model_dir)
        self.assertTrue(report["ok"], P.format_report(report))
        self.assertTrue(report["engine_identity"]["matches"])
        self.assertEqual(report["engine_identity"]["computed"], self.identity)

    def test_a_different_engine_checkpoint_is_refused(self):
        other = checkpoint(self.root / "other-model", hidden=1024)
        self.assertNotEqual(other, self.identity)
        with self.assertRaises(ValueError) as caught:
            self.run_parity(engine_model_dir=self.root / "other-model")
        self.assertIn("snapshot identity", str(caught.exception))

    def test_an_import_without_memory_is_refused(self):
        self.stub.import_no_memory = True
        try:
            with self.assertRaises(R.ReferenceError):
                self.run_parity()
        finally:
            self.stub.import_no_memory = False

    def test_duplicate_prompt_names_are_refused(self):
        with self.assertRaises(ValueError):
            self.run_parity(prompts=[("q", [1]), ("q", [2])])

    def test_an_engine_error_is_not_a_pass(self):
        class Failing(Peer):
            def write(self, line):
                if line.startswith("MEMQUERY"):
                    self.sent.append(line)
                    self.answers.append("ERR the session's checkpoint is gone\n")
                    return
                super().write(line)

        peer = Failing(self.root, self.identity)
        with self.assertRaises(ME.MemtEngineError):
            P.run_parity(self.ref, peer.engine(), commit_ids=[1], prompts=[("q", [2])],
                         outdir=self.root / "out", ref_state_dir=self.state_dir, max_new=1)
        written = json.loads((self.root / "out" / "parity.json").read_text())
        self.assertFalse(written["ok"])
        self.assertIn("MemtEngineError", written["error"])
        self.assertIn("error:", P.format_report(written))


class Parsing(unittest.TestCase):
    def test_token_verdict(self):
        self.assertTrue(P.token_verdict([1, 2], [1, 2])["identical"])
        verdict = P.token_verdict([1, 2, 3], [1, 9, 3])
        self.assertFalse(verdict["identical"])
        self.assertEqual(verdict["identical_prefix"], 1)

    def test_read_ids_forms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.json").write_text(json.dumps([1, 2, 3]))
            (root / "b.json").write_text(json.dumps({"ids": [4, 5]}))
            (root / "c.txt").write_text("6, 7 8\n")
            self.assertEqual(H.read_ids(root / "a.json"), [1, 2, 3])
            self.assertEqual(H.read_ids(root / "b.json"), [4, 5])
            self.assertEqual(H.read_ids(root / "c.txt"), [6, 7, 8])
            (root / "bad.json").write_text("[]")
            with self.assertRaises(H.HarnessError):
                H.read_ids(root / "bad.json")

    def test_cli_reports_a_dead_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "c.txt").write_text("1,2")
            (root / "q.txt").write_text("3")
            status = P.main(["--ref-url", "http://127.0.0.1:1", "--engine", "/bin/true",
                             "--commit", str(root / "c.txt"), "--prompt", str(root / "q.txt"),
                             "--outdir", str(root / "out"), "--ready-timeout", "0.2"])
            self.assertEqual(status, 2)


if __name__ == "__main__":
    unittest.main()
