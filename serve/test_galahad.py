"""Galahad session adapter and HTTP contract, without a GPU, licence or SDK.

    python -m unittest serve.test_galahad serve.test_slots serve.test_security
"""
import ctypes
import json
import os
import sys
import tempfile
import types
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from serve.galahad import GalahadError, GalahadSessions, validate_config
from serve.server import ByteTokenizer, Service, serve
from serve.frontend import ChatTemplate

ROOT = Path(__file__).resolve().parents[1]


class Lookup(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_size_t), ("tier", ctypes.c_int),
                ("token_count", ctypes.c_int32), ("context_handle", ctypes.c_void_p),
                ("seq_id", ctypes.c_int32), ("block_size", ctypes.c_uint64),
                ("access_count", ctypes.c_uint32)]


class Library:
    def __init__(self):
        self.records = {}
        self.failure = 0
        self.short = False
        self.discard = False
        self.closed = 0
        self.checkpoints = 0

    def merlin_lookup_confirmed(self, key, confirm, tenant, out):
        assert tenant == 0
        if self.failure:
            return self.failure
        record = self.records.get(key)
        if not record or record[0] != confirm:
            return 9
        result = ctypes.cast(out, ctypes.POINTER(Lookup)).contents
        assert result.struct_size == ctypes.sizeof(Lookup)
        result.tier = 2
        result.token_count = record[2]
        result.block_size = len(record[1])
        return 0

    def merlin_deposit_bytes(self, key, tenant, buffer, size, tokens, confirm):
        assert tenant == 0
        if not self.discard:
            self.records[key] = (confirm, ctypes.string_at(buffer, size), tokens)
        return self.failure

    def merlin_checkpoint(self, count):
        self.checkpoints += 1
        ctypes.cast(count, ctypes.POINTER(ctypes.c_size_t)).contents.value = 1
        return self.failure

    def merlin_load_block(self, key, tenant, buffer, size, loaded):
        assert tenant == 0
        data = self.records[key][1]
        ctypes.memmove(buffer, data, size)
        ctypes.cast(loaded, ctypes.POINTER(ctypes.c_size_t)).contents.value = size - int(self.short)
        return self.failure

    def merlin_last_error(self):
        return b"test storage failure"


class Engine:
    batch = 0
    def __init__(self):
        self.paths = []
        self.image = b"STRATA-session\x00\xff" * 1024
        self.restored = None
        self.calls = []

    def session_file(self, action, path):
        self.paths.append(path)
        self.calls.append(action)
        self.private = os.stat(os.path.dirname(path)).st_mode & 0o777
        if action == "save":
            Path(path).write_bytes(self.image)
        else:
            self.restored = Path(path).read_bytes()
        return {"tokens": 1000, "bytes": len(self.image), "ms": 1.0}


class Sessions(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.lib = Library()
        self.engine = Engine()
        def init(fp, **kwargs):
            self.init_args = (fp, kwargs)
            def close():
                self.lib.closed += 1
            return types.SimpleNamespace(lib=self.lib, close=close)
        self.modules = {"galahad": types.SimpleNamespace(init=init),
                        "merlin": types.SimpleNamespace(MerlinLookupResult=Lookup)}
        with patch.dict(sys.modules, self.modules), patch("sys.platform", "linux"):
            self.store = GalahadSessions(self.directory.name, 1234, 1)
        self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.galahad_sessions = self.store

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()
        for path in self.engine.paths:
            self.assertFalse(Path(path).exists(), "plaintext temporary session leaked")

    def test_roundtrip_and_restart(self):
        status, body = self.svc.slot_action("0", "save", "chat.bin")
        self.assertEqual(status, 200)
        self.assertEqual(body["n_saved"], 1000)
        self.assertEqual(body["n_written"], len(self.engine.image))
        self.assertEqual(self.lib.checkpoints, 1)
        self.assertTrue(self.init_args[1]["rehydrate_on_init"])
        self.assertTrue(self.init_args[1]["fsync_payloads"])
        self.store.close()
        with patch.dict(sys.modules, self.modules), patch("sys.platform", "linux"):
            self.store = GalahadSessions(self.directory.name, 1234, 1)
        self.svc.galahad_sessions = self.store
        status, body = self.svc.slot_action("0", "restore", "chat.bin")
        self.assertEqual(status, 200)
        self.assertEqual(body["n_restored"], 1000)
        self.assertEqual(self.engine.restored, self.engine.image)
        self.assertEqual(self.engine.private, 0o700)
        self.assertFalse(self.svc.status["busy"])

    def test_missing_and_immutable_name(self):
        self.assertEqual(self.svc.slot_action("0", "restore", "missing.bin")[0], 404)
        self.svc.slot_action("0", "save", "chat.bin")
        self.assertEqual(self.svc.slot_action("0", "save", "chat.bin")[0], 409)
        self.assertEqual(self.engine.calls, ["save"])

    def test_confirmation_and_model_isolation(self):
        self.svc.slot_action("0", "save", "chat.bin")
        key, confirm = self.store.keys("chat.bin")
        self.assertIsNone(self.store.lookup(key, confirm ^ 1))
        self.store.fingerprint = 1235
        self.assertEqual(self.svc.slot_action("0", "restore", "chat.bin")[0], 404)
        self.assertIsNone(self.engine.restored)

    def test_size_limit_before_engine_restore(self):
        self.svc.slot_action("0", "save", "chat.bin")
        self.store.max_bytes = 1
        self.assertEqual(self.svc.slot_action("0", "restore", "chat.bin")[0], 413)
        self.assertEqual(self.svc.slot_action("0", "save", "other.bin")[0], 413)
        self.assertIsNone(self.engine.restored)

    def test_short_read_never_reaches_engine(self):
        self.svc.slot_action("0", "save", "chat.bin")
        self.lib.short = True
        self.assertEqual(self.svc.slot_action("0", "restore", "chat.bin")[0], 500)
        self.assertIsNone(self.engine.restored)

    def test_storage_failure_and_silent_licence_refusal(self):
        self.lib.failure = 21
        self.assertEqual(self.svc.slot_action("0", "save", "chat.bin")[0], 507)
        self.lib.failure = 0
        self.lib.discard = True
        self.assertEqual(self.svc.slot_action("0", "save", "chat.bin")[0], 500)
        self.assertFalse(self.svc.status["busy"])

    def test_existing_slot_guards_still_apply(self):
        for slot, action, name in [("1", "save", "chat.bin"), ("0", "delete", "chat.bin"),
                                   ("0", "save", "../secret"), ("0", "save", "NUL")]:
            self.assertEqual(self.svc.slot_action(slot, action, name)[0], 400)
        self.engine.batch = 2
        self.assertEqual(self.svc.slot_action("0", "save", "chat.bin")[0], 501)
        self.assertEqual(self.engine.calls, [])

    def test_http_api_requires_key_and_roundtrips(self):
        self.svc.api_key = "galahad-test-key"
        httpd = serve(self.svc, port=0)
        def request(action, key=None):
            headers = {"Content-Type": "application/json"}
            if key:
                headers["Authorization"] = f"Bearer {key}"
            req = urllib.request.Request(
                f"http://127.0.0.1:{httpd.server_address[1]}/slots/0?action={action}",
                data=json.dumps({"filename": "http.bin"}).encode(), headers=headers)
            try:
                with urllib.request.urlopen(req) as response:
                    return response.status, json.load(response)
            except urllib.error.HTTPError as e:
                with e:
                    return e.code, json.load(e)
        try:
            self.assertEqual(request("save")[0], 401)
            self.assertEqual(self.engine.calls, [])
            self.assertEqual(request("save", self.svc.api_key)[0], 200)
            status, body = request("restore", self.svc.api_key)
            self.assertEqual(status, 200)
            self.assertEqual(body["filename"], "http.bin")
            self.assertEqual(self.engine.restored, self.engine.image)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_disabled_path_uses_engine_file_without_galahad(self):
        self.svc.galahad_sessions = None
        self.svc.slot_save_path = self.directory.name
        status, body = self.svc.slot_action("0", "save", "plain.bin")
        self.assertEqual(status, 200)
        self.assertEqual(body["timings"], {"save_ms": 1.0})
        self.assertEqual(Path(self.directory.name, "plain.bin").read_bytes(), self.engine.image)
        self.engine.paths.clear()  # ordinary file sessions are intentionally retained
        self.assertEqual(self.lib.records, {})

    def test_engine_refusal_cleans_plaintext(self):
        from serve.server import SessionRefused
        def refuse(action, path):
            self.engine.paths.append(path)
            Path(path).write_bytes(b"partial")
            raise SessionRefused("invalid", "invalid test session")
        with patch.object(self.engine, "session_file", refuse):
            self.assertEqual(self.svc.slot_action("0", "save", "chat.bin")[0], 400)
        self.assertFalse(self.svc.status["busy"])

    def test_config_rejects_invalid_identity_and_limits(self):
        for fp in [None, True, 0, -1, 2**64, "1234"]:
            with self.assertRaises(ValueError):
                validate_config("cache", fp, 1)
        for cap in [True, 0, -1, 1.5]:
            with self.assertRaises(ValueError):
                validate_config("cache", 1, cap)
        with self.assertRaises(ValueError), patch("sys.platform", "win32"):
            GalahadSessions("cache", 1)


if __name__ == "__main__":
    unittest.main()
