"""Local model switching rejects unsafe or overlapping operations; no GPU required."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
import urllib.error
import urllib.request
from unittest import mock

from serve.server import ByteTokenizer, MockEngine, Service, serve
from serve.frontend import ChatTemplate
from serve.model_switch import ModelSwitcher, MODELS, save_switch

ROOT = Path(__file__).resolve().parents[1]


class ModelSelection(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        tok_dir = self.root / "tokenizer"
        tok_dir.mkdir()
        (tok_dir / "tokenizer.json").write_text("{}")
        (self.root / "serve").mkdir()
        (self.root / "serve" / "model_switch_worker.py").write_text("")
        for name, (tag, _) in MODELS.items():
            cfg = {"exe": str(self.root / "engine"), "args": [], "tokenizer": str(tok_dir),
                   "model_name": "model-" + name, "host": "127.0.0.1", "port": 8080, "model_switch": True}
            (self.root / f"strata-{tag}.json").write_text(json.dumps(cfg))
        (self.root / "engine").write_text("")
        self.selector = ModelSwitcher(self.root)
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), "model-coder")
        self.svc.model_switcher = self.selector
        self.owned = mock.patch.object(self.selector.manager, "tracked_server", return_value={"pid": os.getpid()})
        self.owned.start()
        self.addCleanup(self.owned.stop)

    def test_unknown_unavailable_and_foreign_listener_do_not_spawn(self):
        with mock.patch("serve.model_switch.spawn_detached") as spawn:
            for value in ("../../other", None, [], {"name": "swift"}):
                self.assertEqual(self.selector.begin(self.svc, value)[0], 400)
            path = self.root / "strata-swift-iq2_xs.json"
            cfg = json.loads(path.read_text())
            cfg["host"] = "0.0.0.0"
            path.write_text(json.dumps(cfg))
            self.assertEqual(self.selector.begin(self.svc, "swift")[0], 404)
            self.selector.manager.tracked_server.return_value = {"pid": os.getpid() + 100000}
            self.assertEqual(self.selector.begin(self.svc, "original")[0], 409)
            spawn.assert_not_called()

    def test_running_or_queued_request_is_preserved(self):
        with mock.patch("serve.model_switch.spawn_detached") as spawn:
            for status in ({"busy": True, "queued": 0}, {"busy": False, "queued": 1}):
                self.svc.status.update(status)
                self.assertEqual(self.selector.begin(self.svc, "original")[0], 409)
            self.svc.status.update(busy=False, queued=0)
            with self.svc.fifo:
                self.assertEqual(self.selector.begin(self.svc, "original")[0], 409)
            spawn.assert_not_called()

    def test_current_model_noop_and_overlapping_switch_rejected(self):
        with mock.patch("serve.model_switch.spawn_detached", return_value=mock.Mock(pid=os.getpid())) as spawn:
            self.assertEqual(self.selector.begin(self.svc, "coder")[0], 200)
            spawn.assert_not_called()
            code, first = self.selector.begin(self.svc, "original")
            self.assertEqual(code, 202)
            self.assertEqual(self.selector.begin(self.svc, "swift")[0], 409)
            self.assertEqual(self.selector.snapshot(self.svc)["switch"]["id"], first["id"])
            spawn.assert_called_once()
            self.assertNotIn("pid", self.selector.snapshot(self.svc)["switch"])

    def test_dead_worker_unlocks_selection_and_reports_failure(self):
        save_switch(self.selector.path, {"id": "dead", "status": "starting", "target": "original",
                                        "pid": os.getpid() + 100000, "ident": "missing"})
        self.assertEqual(self.selector.snapshot(self.svc)["switch"]["status"], "failed")

    @unittest.skipUnless(os.name == "nt", "Windows venv redirector")
    def test_managed_venv_parent_is_recognized(self):
        self.selector.manager.tracked_server.return_value = {"pid": os.getppid()}
        self.assertTrue(self.selector.snapshot(self.svc)["can_switch"])

    def test_http_origin_auth_and_body_guards(self):
        httpd = serve(self.svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"

        def post(body, content="application/json", origin=None, key=None):
            headers = {"Content-Type": content}
            if origin:
                headers["Origin"] = origin
            if key:
                headers["Authorization"] = "Bearer " + key
            req = urllib.request.Request(base + "/api/local-models/switch", data=body, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=3) as response:
                    return response.status
            except urllib.error.HTTPError as error:
                with error:
                    return error.code

        with mock.patch("serve.model_switch.spawn_detached") as spawn:
            self.assertEqual(post(b'{"model":"original"}', origin="http://untrusted.example"), 403)
            self.assertEqual(post(b'{"model":"original"}', content="text/plain"), 415)
            for body in (b"[]", b"{}", b'{"model":"coder","args":[]}', b"invalid", b" " * 1025):
                self.assertEqual(post(body), 400)
            self.svc.api_key = "test-key"
            self.assertEqual(post(b'{"model":"original"}'), 401)
            self.assertEqual(post(b'{"model":"coder"}', origin=base, key="test-key"), 200)
            spawn.assert_not_called()


class WorkerRecovery(unittest.TestCase):
    def test_failed_load_restores_previous_model(self):
        spec = importlib.util.spec_from_file_location("local_switch_worker", ROOT / "serve/model_switch_worker.py")
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state_path = root / ".strata-mcp/model-switch.json"
            save_switch(state_path, {"id": "test", "status": "starting", "target": "swift", "previous": "coder"})
            with mock.patch.object(worker, "ROOT", root), mock.patch.object(worker.time, "sleep"), \
                 mock.patch.object(worker.sys, "argv", ["worker", "swift", "coder", "test"]), \
                 mock.patch.object(worker, "Tools"), mock.patch.object(worker, "Strata"), \
                 mock.patch.object(worker, "start_model", side_effect=[RuntimeError("bad model"), {"model": "coder"}]) as start:
                worker.main()
            result = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "failed")
            self.assertTrue(result["restored"])
            self.assertEqual([c.args[1] for c in start.call_args_list], ["swift-iq2_xs", "coder-iq1_m"])


if __name__ == "__main__":
    unittest.main()
