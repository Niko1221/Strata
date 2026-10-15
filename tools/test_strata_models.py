"""Browser model launcher tests: fake setup and servers, no GPU or downloads."""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools import strata_models as W
from tools import strata_mcp as M
import setup as official

FAKE_SETUP = r'''
import json, sys, time
from pathlib import Path
root = Path(__file__).resolve().parent
if __name__ == "__main__":
    args = sys.argv[1:]
    (root / "setup-args.json").write_text(json.dumps(args))
    print("=== Step 5: downloading model ===", flush=True)
    print("model.gguf:  1.00 / 2.00 GB (50%)", flush=True)
    if (root / "slow").exists():
        time.sleep(30)
    if (root / "fail").exists():
        print("  [X]  fake download failed", flush=True)
        sys.exit(1)
    model = args[args.index("--model") + 1].lower()
    cfg = {"exe": sys.executable, "tokenizer": str(root / "tokenizer"), "cwd": str(root),
           "args": ["--pack", str(root / "pack"), "--max-context", "65536"],
           "model_name": "qwen3.8-flash-next-" + model, "port": 8080, "host": "127.0.0.1"}
    (root / ("strata-" + model + ".json")).write_text(json.dumps(cfg))
'''
FAKE_SERVER = r'''
import argparse, json, os, subprocess, sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
ap = argparse.ArgumentParser()
ap.add_argument("--config"); ap.add_argument("--port", type=int); ap.add_argument("--engine")
a = ap.parse_args(); root = Path(__file__).resolve().parent.parent
cfg = json.loads(Path(a.config).read_text())
if cfg.get("fail_start"):
    raise SystemExit(2)
if cfg.get("spawn_child"):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    (root / "server-child.json").write_text(json.dumps({"pid":child.pid}))
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        data = {"status":"ok", "service":"strata", "model":cfg["model_name"], "max_context":65536,
                "loaded":not cfg.get("lazy_load"), "busy":(root / "busy").exists(), "images":False, "api_key":False}
        b = json.dumps(data).encode(); self.send_response(200); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        if (root / "busy").exists():
            self.send_response(409); self.end_headers(); return
        b = b'{"status":"unloaded"}'; self.send_response(200); self.send_header("Content-Length",str(len(b)))
        self.end_headers(); self.wfile.write(b)
class Server(HTTPServer):
    allow_reuse_address = os.name != "nt"
    allow_reuse_port = False
Server(("127.0.0.1", a.port), Handler).serve_forever()
'''
HW = {"ram_gb":64, "cpu":{"avx2":True}, "gpus":[{"index":0, "name":"Fake GPU", "vram_gb":16,
                                                     "vendor":"nvidia", "usable":True}]}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "setup.py").write_text(FAKE_SETUP)
        (self.root / "serve").mkdir()
        (self.root / "serve" / "server.py").write_text(FAKE_SERVER)
        (self.root / "tokenizer").mkdir()
        (self.root / "pack").mkdir()
        self.s = M.Strata(self.root)
        self.s._setup, self.s._setup_tried = official, True
        self.s._hardware = lambda: HW
        self.s.disk_free = lambda _: 1000
        self.s.data_dir = lambda: self.root / "Strata-data"
        self.s.python_override = sys.executable
        self.s.health_timeout = .2
        self.port = free_port()
        self.w = W.ModelManager(self.s, self.port)
        self.w.load_timeout = 8
        self.w.live = self.live
        self.http = W.ModelsHTTPServer(("127.0.0.1", 0), W.make_handler(self.w))
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()

    def live(self):
        tracked = self.s.tracked_server()
        info = self.s.probe(self.port)
        return ([{**info, "managed":bool(tracked and not tracked.get("ended"))}]
                if info and not info.get("other") else []), tracked

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        for name in ("server", "install"):
            state = self.s.state(name)
            if state and M.proc_alive(state.get("pid"), state.get("ident")):
                M.proc_kill_tree(state["pid"], state["ident"])
                M.wait_gone(state["pid"], state["ident"], 10)
        result = self.s.state("install-result")
        if result.get("child_pid"):
            M.proc_kill_tree(result["child_pid"], result.get("child_ident"))
        for child in self.s.children:
            try: child.wait(10)
            except Exception: pass
        if self.w.operation and self.w.operation["state"] == "running":
            self.finish()
        # Windows may report the process as ended a moment before its inherited
        # log handle closes. Allow that close to finish before removing the fixture.
        for attempt in range(10):
            try:
                self.tmp.cleanup()
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(.1)

    def cfg(self, model="iq2_xs", **extra):
        cfg = {"exe":sys.executable, "args":["--pack", str(self.root / "pack"), "--max-context", "32768"],
               "tokenizer":str(self.root / "tokenizer"), "cwd":str(self.root), "host":"127.0.0.1",
               "port":self.port, "model_name":"qwen3.8-flash-next-" + model, **extra}
        path = self.root / f"strata-{model}.json"
        path.write_text(json.dumps(cfg))
        return path

    def finish(self, timeout=15):
        end = time.monotonic() + timeout
        while self.w.operation["state"] == "running" and time.monotonic() < end:
            time.sleep(.05)
        self.assertNotEqual(self.w.operation["state"], "running")
        return self.w.operation

    def call(self, path, body=None, headers=None):
        port = self.http.server_port
        h = {"Content-Type":"application/json", "Origin":f"http://127.0.0.1:{port}",
             "X-Strata-Token":self.w.token, **(headers or {})}
        c = HTTPConnection("127.0.0.1", port, timeout=5)
        c.request("POST" if body is not None else "GET", path,
                  json.dumps(body) if body is not None else None, h)
        r = c.getresponse(); data = r.read(); code = r.status; c.close()
        return code, data

    def test_reopening_finishes_pending_installer(self):
        self.s.save_state("install", {"job":"completed", "pid":0, "ident":None})
        self.s.save_state("install-result", {"job":"completed", "exit_code":0})
        self.s.save_state("web-install", {"job":"completed", "model":"iq3_xxs", "lookup":None})
        self.w = W.ModelManager(self.s, self.port)
        self.assertEqual(self.finish()["state"], "done")
        self.assertFalse(self.s.state("web-install"))

    def test_one_launcher_per_checkout(self):
        with W.launcher_lock(self.root):
            with self.assertRaisesRegex(M.ToolError, "already runs"):
                with W.launcher_lock(self.root):
                    pass
        with W.launcher_lock(self.root):
            pass

    def test_second_launcher_cannot_share_http_port(self):
        with self.assertRaises(OSError):
            W.ModelsHTTPServer(self.http.server_address, W.make_handler(self.w))

    def test_cancel_can_resume(self):
        (self.root / "slow").touch()
        self.w.install(self.w.plan("iq3_xxs")["plan_id"])
        end = time.monotonic() + 5
        while not self.s.state("install-result").get("child_pid") and time.monotonic() < end:
            time.sleep(.05)
        self.w.cancel()
        op = self.finish()
        self.assertEqual(op["state"], "done")
        self.assertIn("resume", op["message"])
        (self.root / "slow").unlink()
        self.w.install(self.w.plan("iq3_xxs")["plan_id"])
        self.assertEqual(self.finish()["state"], "done")

    def test_cancel_before_installer_starts(self):
        preparing, proceed = threading.Event(), threading.Event()
        def preparing_model():
            preparing.set()
            if not proceed.wait(5):
                raise RuntimeError("test did not release startup")
        with mock.patch.object(self.w, "stop_owned", side_effect=preparing_model):
            try:
                self.w.install(self.w.plan("iq3_xxs")["plan_id"])
                self.assertTrue(preparing.wait(5))
                result = self.w.cancel()
                self.assertIn("Cancellation requested", result["summary"])
            finally:
                proceed.set()
            op = self.finish()
        self.assertEqual(op["state"], "done", op)
        self.assertIn("resume", op["message"])
        self.assertFalse((self.root / "setup-args.json").exists())
        self.assertFalse(self.s.state("install"))

    def test_cancel_waits_for_job_and_recovery_record_publication(self):
        (self.root / "slow").touch()
        publishing, proceed, requested = (threading.Event() for _ in range(3))
        original = self.w.tools.launch_install
        errors = []
        def launch(plan, args):
            result = original(plan, args)
            publishing.set()
            if not proceed.wait(5):
                raise RuntimeError("test did not release publication")
            return result
        def cancel():
            requested.set()
            try:
                self.w.cancel()
            except Exception as exc:
                errors.append(exc)
        with mock.patch.object(self.w.tools, "launch_install", side_effect=launch):
            self.w.install(self.w.plan("iq3_xxs")["plan_id"])
            try:
                self.assertTrue(publishing.wait(5))
                canceller = threading.Thread(target=cancel)
                canceller.start()
                self.assertTrue(requested.wait(5))
            finally:
                proceed.set()
            canceller.join(10)
            self.assertFalse(canceller.is_alive())
            self.assertEqual(errors, [])
            op = self.finish()
        self.assertEqual(op["state"], "done", op)
        self.assertIn("resume", op["message"])
        self.assertFalse(self.s.state("web-install"))

    def test_catalog_uses_official_regular_models(self):
        rows, _ = self.w.catalog()
        expected = {(f, m) for f, fd in official.FAMILIES.items() if not fd.get("experimental")
                    for m, md in official.MODELS.items() if f in md.get("families", ("qwen","swift"))
                    and not md.get("experimental")}
        self.assertEqual({(r["family"], r["model"]) for r in rows}, expected)
        self.assertFalse(any(r["experimental"] for r in rows))

    def test_setup_import_failure_blocks_installs(self):
        self.s._setup = None
        with self.assertRaisesRegex(M.ToolError, "setup.py"):
            self.w.catalog()

    def test_unknown_or_experimental_model_rejected(self):
        for name in ("../../x", "orca-iq3_xxs", "unsloth-ud-q4_k_xl", 12):
            with self.assertRaises(M.ToolError): self.w.plan(name)

    def test_plan_does_not_start_setup_or_override_settings(self):
        p = self.w.plan("iq3_xxs")
        args = self.w.plans[p["plan_id"]][2]
        self.assertEqual(args, ["--setup", "--yes", "--no-start", "--family", "qwen", "--model",
                                "IQ3_XXS", "--port", str(self.port), "--data-dir", str(self.s.data_dir())])
        self.assertFalse((self.root / "setup-args.json").exists())

    def test_install_runs_official_setup_and_is_single_use(self):
        p = self.w.plan("iq3_xxs")
        self.w.install(p["plan_id"])
        self.assertEqual(self.finish()["state"], "done")
        args = json.loads((self.root / "setup-args.json").read_text())
        self.assertIn("--no-start", args)
        self.assertNotIn("--context", args)
        with self.assertRaises(M.ToolError): self.w.install(p["plan_id"])
        self.assertTrue((self.root / "strata-iq3_xxs.json").exists())

    def test_install_failure_is_reported(self):
        (self.root / "fail").touch()
        self.w.install(self.w.plan("iq3_xxs")["plan_id"])
        op = self.finish()
        self.assertEqual(op["state"], "failed")
        self.assertIn("fake download failed", op["message"])

    def test_disk_space_is_checked_again(self):
        p = self.w.plan("iq3_xxs")
        self.s.disk_free = lambda _: 1
        with self.assertRaisesRegex(M.ToolError, "disk space"):
            self.w.install(p["plan_id"])
        self.assertFalse((self.root / "setup-args.json").exists())

    def test_plan_expiry(self):
        p = self.w.plan("iq3_xxs")
        _, plan, args, lookup = self.w.plans[p["plan_id"]]
        self.w.plans[p["plan_id"]] = (time.monotonic() - 901, plan, args, lookup)
        with self.assertRaisesRegex(M.ToolError, "expired"): self.w.install(p["plan_id"])

    def test_switch_starts_and_stops_real_fake_servers(self):
        self.cfg(); self.cfg("iq3_xxs")
        self.w.start("iq2_xs")
        self.assertEqual(self.finish()["state"], "done")
        old = self.s.state("server")
        self.w.start("iq3_xxs")
        self.assertEqual(self.finish()["state"], "done")
        self.assertFalse(M.proc_alive(old["pid"], old["ident"]))
        self.assertEqual(self.s.state("server")["model"], "iq3_xxs")
        self.w.stop()
        self.assertEqual(self.finish()["state"], "done")
        self.assertFalse(self.s.tracked_server())

    def test_repeated_switches_reuse_the_same_port(self):
        self.cfg(); self.cfg("iq3_xxs")
        previous = None
        for model in ("iq2_xs", "iq3_xxs") * 3:
            self.w.start(model)
            op = self.finish()
            self.assertEqual(op["state"], "done", op)
            if previous:
                self.assertFalse(M.proc_alive(previous["pid"], previous["ident"]))
            previous = self.s.state("server")
            self.assertEqual(previous["port"], self.port)
            self.assertEqual(previous["model"], model)

    @unittest.skipUnless(M.WIN, "Windows launcher/descendant shutdown")
    def test_windows_stop_ends_owned_descendants(self):
        self.cfg(spawn_child=True)
        self.w.start("iq2_xs")
        self.assertEqual(self.finish()["state"], "done")
        pid = json.loads((self.root / "server-child.json").read_text())["pid"]
        ident = M.proc_identity(pid)
        self.assertIsNotNone(ident)
        self.addCleanup(M.proc_kill_tree, pid, ident)
        self.w.stop()
        op = self.finish()
        self.assertEqual(op["state"], "done", op)
        self.assertFalse(M.proc_alive(pid, ident))
        self.assertIsNone(self.s.probe(self.port))

    def test_busy_model_is_preserved(self):
        self.cfg(); self.cfg("iq3_xxs")
        self.w.start("iq2_xs"); self.finish()
        old = self.s.state("server")
        (self.root / "busy").touch()
        self.w.start("iq3_xxs")
        self.assertEqual(self.finish()["state"], "failed")
        self.assertEqual(self.s.state("server")["pid"], old["pid"])

    def test_unknown_busy_state_does_not_stop_model(self):
        self.cfg(); self.w.start("iq2_xs"); self.finish()
        old = self.s.state("server")
        with mock.patch.object(M, "http_json", return_value=(401, {})):
            with self.assertRaisesRegex(M.ToolError, "Cannot verify"):
                self.w.stop_owned()
        self.assertTrue(M.proc_alive(old["pid"], old["ident"]))

    def test_failed_switch_restores_previous_model(self):
        self.cfg(); self.cfg("iq3_xxs", fail_start=True)
        self.w.start("iq2_xs"); self.finish()
        self.w.start("iq3_xxs")
        op = self.finish()
        self.assertEqual(op["state"], "failed")
        self.assertIn("previous model was restored", op["message"])
        self.assertEqual(self.s.state("server")["model"], "iq2_xs")

    def test_missing_model_checked_before_stopping_current(self):
        self.cfg(); self.w.start("iq2_xs"); self.finish()
        old = self.s.state("server")
        with self.assertRaisesRegex(M.ToolError, "files are missing"): self.w.start("iq3_xxs")
        self.assertEqual(self.s.state("server")["pid"], old["pid"])

    def test_external_server_is_not_stopped(self):
        self.cfg()
        self.w.live = lambda: ([{"managed":False}], None)
        with self.assertRaisesRegex(M.ToolError, "existing server window"): self.w.start("iq2_xs")
        with self.assertRaises(M.ToolError): self.w.stop()
        with self.assertRaises(M.ToolError): self.w.install(self.w.plan("iq3_xxs")["plan_id"])

    def test_lan_config_not_started(self):
        self.cfg(host="0.0.0.0")
        with self.assertRaisesRegex(M.ToolError, "local servers only"): self.w.start("iq2_xs")

    def test_lazy_server_is_ready_before_its_first_request(self):
        self.cfg(lazy_load=True)
        self.w.start("iq2_xs")
        op = self.finish()
        self.assertEqual(op["state"], "done", op)
        self.assertIn("Open Chat", op["message"])
        self.assertFalse(self.s.probe(self.port)["loaded"])

    def test_parallel_mutation_rejected(self):
        self.w.operation = {"state":"running"}
        with self.assertRaises(M.ToolError): self.w.plan("iq3_xxs")
        self.w.operation = None

    def test_shared_lookup_changes_paths_only_and_keeps_backup(self):
        target = self.root / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf"
        source = self.root / "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf"
        for p in (target, source): p.write_bytes(b"fake")
        cfg_path = self.cfg("iq3_xxs")
        cfg = json.loads(cfg_path.read_text())
        cfg["args"] += ["--native", "native.gguf", "--ple-gguf", str(target), "--spec", "4"]
        cfg_path.write_text(json.dumps(cfg)); original = cfg_path.read_bytes()
        fake = SimpleNamespace(metadata={"general.architecture":"qwen4exp"},
                               tensors=[SimpleNamespace(name="per_layer_token_embd.weight", shape=[90,10], type_id=30)])
        with mock.patch.object(W, "GGUFFile", return_value=fake): self.w.reuse_lookup("iq3_xxs", source)
        updated = json.loads(cfg_path.read_text())
        self.assertEqual(W.arg_value(updated["args"], "--ple-gguf"), str(source))
        self.assertEqual(W.arg_value(updated["args"], "--spec"), "4")
        self.assertEqual(updated["args"].count("--native-dense-gguf"), 2)
        self.assertEqual(cfg_path.with_suffix(".json.lookup.bak").read_bytes(), original)
        self.assertEqual({k:v for k,v in cfg.items() if k != "args"}, {k:v for k,v in updated.items() if k != "args"})

    def test_lookup_reuse_preserves_official_setup_backup(self):
        target, source = self.root / "target.gguf", self.root / "ssd.gguf"
        for p in (target, source): p.write_bytes(b"fake")
        cfg_path = self.cfg()
        original_cfg = json.loads(cfg_path.read_text())
        original_cfg["args"] += ["--native", "native.gguf", "--ple-gguf", str(source),
                                 "--vram-reserve-mib", "5120"]
        cfg_path.write_text(json.dumps(original_cfg))
        original = cfg_path.read_bytes()
        generated = {**original_cfg, "args":["--native", "native.gguf", "--ple-gguf", str(target)]}
        with mock.patch.object(official, "say"):
            official.write_setup_config(cfg_path, generated)
        setup_config = cfg_path.read_bytes()
        self.assertEqual(cfg_path.with_suffix(".json.bak").read_bytes(), original)
        fake = SimpleNamespace(metadata={"general.architecture":"qwen4exp"},
                               tensors=[SimpleNamespace(name="per_layer_token_embd.weight",shape=[1],type_id=30)])
        with mock.patch.object(W, "GGUFFile", return_value=fake):
            self.w.reuse_lookup("iq2_xs", source)
            self.w.reuse_lookup("iq2_xs", source)  # Recovery is safe to repeat.
        self.assertEqual(cfg_path.with_suffix(".json.bak").read_bytes(), original)
        self.assertEqual(cfg_path.with_suffix(".json.lookup.bak").read_bytes(), setup_config)
        updated = json.loads(cfg_path.read_text())
        self.assertEqual(W.arg_value(updated["args"], "--ple-gguf"), str(source))

    def test_mismatched_lookup_is_not_substituted(self):
        target, source = self.root / "target.gguf", self.root / "source.gguf"
        target.write_bytes(b"a"); source.write_bytes(b"bb")
        cfg_path = self.cfg("iq3_xxs")
        cfg = json.loads(cfg_path.read_text()); cfg["args"] += ["--native", "native.gguf", "--ple-gguf", str(target)]
        cfg_path.write_text(json.dumps(cfg)); original = cfg_path.read_bytes()
        fake = SimpleNamespace(metadata={}, tensors=[SimpleNamespace(name="per_layer_token_embd.weight",shape=[1],type_id=30)])
        with mock.patch.object(W, "GGUFFile", return_value=fake):
            with self.assertRaisesRegex(M.ToolError, "do not match"): self.w.reuse_lookup("iq3_xxs", source)
        self.assertEqual(cfg_path.read_bytes(), original)

    def test_http_requires_host_origin_and_session_token(self):
        for headers in ({"Host":"evil.test"}, {"Origin":"https://evil.test"}, {"Origin":""}, {"X-Strata-Token":""}):
            with self.subTest(headers=headers):
                self.assertEqual(self.call("/api/models/plan", {"model":"iq3_xxs"}, headers)[0], 403)

    def test_http_rejects_extra_fields_and_arbitrary_paths(self):
        self.assertEqual(self.call("/api/models/plan", {"model":"iq3_xxs", "command":"evil"})[0], 400)
        for path in ("/setup.py", "/web/../setup.py", "/.strata-models/server.json"):
            self.assertEqual(self.call(path)[0], 404)

    def test_http_plan_and_static_page(self):
        self.assertEqual(self.call("/")[0], 200)
        self.assertEqual(self.call("/web/models.js")[0], 200)
        code, data = self.call("/api/models/plan", {"model":"iq3_xxs"})
        self.assertEqual(code, 200)
        self.assertIn("plan_id", json.loads(data))
        self.assertFalse((self.root / "setup-args.json").exists())


if __name__ == "__main__":
    unittest.main()
