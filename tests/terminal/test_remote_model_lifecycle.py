"""Offline regression contracts for Mac-owned remote prepare/switch operations."""
from __future__ import annotations

import copy
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import socket
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("remote_lifecycle_tested", ROOT / "apps/host/python/remote_model_lifecycle.py")
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def synthetic_gguf():
    def string(value):
        raw = value.encode(); return struct.pack("<Q", len(raw)) + raw
    header = (b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + string("general.architecture") + struct.pack("<I", 8)
              + string("qwen4exp") + string("token_embd.weight") + struct.pack("<IQIQ", 1, 32, 0, 0))
    return header + b"\0" * (-len(header) % 32) + b"\0" * 128


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "Strata"
        self.source.mkdir()
        (self.source / "strata_control").mkdir()
        (self.source / "strata_control/acquisition.py").write_text("# fixture\n")
        (self.source / "strata.py").write_text("# fixture\n")
        payload = synthetic_gguf()
        self.payload = payload
        url = "https://huggingface.co/fixture/model/resolve/" + "a" * 40 + "/"
        (self.source / "setup.py").write_text(
            "HF_REVISIONS={}\nMODELS={'Q2_0':{'families':('qwen',)}}\n"
            + "FAMILIES=" + repr({"qwen": {"title": "Fixture", "hf": url, "file": "Fixture-{q}-{i}.gguf", "shards": 1, "tag": ""}})
            + "\nOWN_PATH_FAMILIES={}\nREPO_PINS=" + repr({"fixture/model": {"Fixture-Q2_0-1.gguf": (len(payload), hashlib.sha256(payload).hexdigest())}}) + "\n")
        self.home = self.root / "remote-library"
        self.namespace, self.destination = "a" * 64, "b" * 64
        self.clock = 1_800_000_000.0
        self.lifecycle = module.Lifecycle(ROOT, self.namespace, self.destination, self.source, self.home, now=lambda: self.clock)
        row = self.lifecycle.manager.current_catalog()["native-q2_0"]
        manifest = {**row, "readPictures": False}
        asset_key = module.files.digest(manifest["files"])
        self.model = module.files.owned_directory(self.lifecycle.manager.assets, asset_key) / row["files"][0]["path"]
        self.model.write_bytes(payload)
        self.lifecycle.manager.save({"plans": {}, "imports": {}, "jobs": {"download": {"id": "download", "modelId": row["id"], "name": row["name"],
            "state": "done", "requestId": "receipt", "owner": "desktop", "totalBytes": len(payload), "doneBytes": len(payload),
            "manifest": manifest, "assetKey": asset_key, "verifiedFiles": {row["files"][0]["path"]: module.files.identity(self.model)}}}})
        self.engine = self.source / "engine/strata"
        self.engine.parent.mkdir(); self.engine.write_bytes(b"fixture-engine-do-not-execute")
        self.pack = self.source / "prepared/pack"
        self.pack.mkdir(parents=True); (self.pack / "native_experts.txt").write_text("fixture metadata")
        self.tokenizer = self.pack / "tokenizer"
        self.tokenizer.mkdir(); (self.tokenizer / "vocab.json").write_text('{}')
        self.config = self.source / "strata-fixture.json"
        self.config.write_text(json.dumps({"exe": str(self.engine), "args": ["--native", str(self.model), "--pack", str(self.pack), "--max-context", "131072"],
            "tokenizer": str(self.tokenizer), "model_name": "fixture-strata", "port": 8080, "api_key": "EXISTING-SECRET", "gpu": 0}))
        self.maya = self.root / "Maya"
        self.maya.mkdir(); (self.maya / "config.json").write_text('{"model":"Maya-S24","port":8080}')
        self.active_settings = self.root / "original-settings.json"
        self.active_settings.write_text(json.dumps({"data_dir": "/original/data", "installs": [str(self.source)]}))
        self.signature = mock.patch.object(module, "source_signature", return_value="c" * 64)
        self.signature.start(); self.addCleanup(self.signature.stop)
        self.disk = mock.patch.object(module.shutil, "disk_usage", return_value=types.SimpleNamespace(free=10 ** 15))
        self.disk.start(); self.addCleanup(self.disk.stop)
        self.spawn = mock.patch.object(module.Lifecycle, "spawn_worker", return_value=None)
        self.spawn.start(); self.addCleanup(self.spawn.stop)
        self.original_env = {"STRATA_INSTALL_DIR": str(self.source), "STRATA_SETTINGS_FILE": str(self.active_settings), "STRATA_COMPONENTS_DIR": "/original/components", "STRATA_PROFILE_DIR": "/original/profiles"}
        self.environment = mock.patch.dict(os.environ, self.original_env)
        self.environment.start(); self.addCleanup(self.environment.stop)

    def engine_id(self): return self.lifecycle.engine_candidates()[0]["id"]

    def plan(self, **extra):
        return self.lifecycle.dispatch("desktop", "prepare-plan", {"modelId": "native-q2_0", "engineId": self.engine_id(), **extra})

    def assert_error(self, status, action, body, *, owner="desktop", lifecycle=None):
        with self.assertRaises(module.ApiError) as caught:
            (lifecycle or self.lifecycle).dispatch(owner, action, body)
        self.assertEqual(caught.exception.status, status)
        return caught.exception

    def source_snapshot(self):
        return {str(path): (path.read_bytes(), path.stat().st_mtime_ns) for directory in (self.source, self.maya) for path in directory.rglob("*") if path.is_file()} | {
            str(self.active_settings): (self.active_settings.read_bytes(), self.active_settings.stat().st_mtime_ns)}

    def accepted(self):
        plan = self.plan()
        accepted = self.lifecycle.dispatch("desktop", "prepare", {"planId": plan["planId"]})
        state = self.lifecycle.load()
        return state["jobs"][accepted["job"]["id"]]

    def run_worker(self, job):
        with mock.patch.object(module.files, "process_identity", return_value=("fixture-owner", "fixture worker " + job["token"])):
            module.worker(str(self.home), self.namespace, job["id"], job["token"])
        return self.lifecycle.load()["jobs"][job["id"]]

    def prepared(self):
        job = self.accepted(); finished = self.run_worker(job)
        self.assertEqual(finished["state"], "done", finished)
        return self.lifecycle.load()["profiles"][finished["profileId"]]

    def real_engine_fixture(self, *, valid_tokenizer=True):
        from strata_control import runtime
        from tools.strata_tokenizer import BYTE_TO_UNICODE
        marker = self.root / "unexpected-generation"
        self.engine.write_text("#!" + sys.executable + "\nimport sys\nfrom pathlib import Path\n"
            "print('READY 4096 stop',flush=True)\n"
            "for line in sys.stdin:\n"
            " if line.strip()=='QUIT':break\n"
            " if line.startswith(('GEN ','BGEN ')):\n"
            f"  Path({str(marker)!r}).write_text('generation')\n"
            "  print('T 97',flush=True)\n"
            "  print('DONE 1 1 1.0 1.0 length 0 0 0',flush=True)\n")
        self.engine.chmod(0o700)
        if valid_tokenizer:
            tokens = [BYTE_TO_UNICODE[index] for index in range(256)] + ["ab", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
            (self.tokenizer / "vocab.json").write_text(json.dumps({token: index for index, token in enumerate(tokens)}))
            (self.tokenizer / "merges.txt").write_text("a b")
            (self.tokenizer / "token_type.json").write_text(json.dumps([0] * 257 + [3, 3, 3]))
        def stop_owned():
            with self.lifecycle.environment():
                try:
                    if runtime._read_state(self.lifecycle.runtime_name): runtime.stop(self.lifecycle.runtime_name, timeout=15)
                except Exception: pass
        self.addCleanup(stop_owned)
        return marker

    def test_downloaded_plan_binds_model_engine_and_generates_no_external_action(self):
        before = self.source_snapshot()
        with mock.patch.object(module.subprocess, "Popen", side_effect=AssertionError("no process in preview")):
            plan = self.plan()
        self.assertEqual(plan["context"], 32768)
        self.assertGreater(plan["diskNeededGb"], len(self.payload) * 2 / 1e9)
        self.assertTrue(plan["warnings"])
        self.assertEqual(self.source_snapshot(), before)
        self.assertNotIn("EXISTING-SECRET", json.dumps(plan))

    def test_phone_remote_and_arbitrary_actions_are_rejected_before_any_spawn(self):
        for owner in ("phone:" + "a" * 32, "remote:token", "Desktop", ""):
            self.assert_error(403, "prepare-plan", {"modelId": "native-q2_0"}, owner=owner)
        for field in ({"command": "touch /bad"}, {"data_dir": "/bad"}, {"loadAfter": True}, {"sourcePath": "/bad"}):
            self.assert_error(400, "prepare-plan", {"modelId": "native-q2_0", **field})
        self.assert_error(404, "exec", {"command": "true"})

    def test_namespace_validation_cannot_escape_library(self):
        for namespace in ("../other", "a" * 63, "/tmp/outside", "a" * 63 + "G"):
            with self.assertRaises(module.ApiError): module.Lifecycle(ROOT, namespace, self.destination, self.source, self.home)

    def test_expired_wrong_kind_foreign_destination_and_replayed_plan_do_not_start(self):
        plan = self.plan()
        other = module.Lifecycle(ROOT, self.namespace, "d" * 64, self.source, self.home, now=lambda: self.clock)
        self.assert_error(409, "prepare", {"planId": plan["planId"]}, lifecycle=other)
        self.assert_error(409, "serve", {"planId": plan["planId"]})
        self.clock += 301
        self.assert_error(409, "prepare", {"planId": plan["planId"]})
        fresh = self.plan()
        self.lifecycle.dispatch("desktop", "prepare", {"planId": fresh["planId"]})
        self.assert_error(409, "prepare", {"planId": fresh["planId"]})
        self.assertEqual(len(self.lifecycle.load()["jobs"]), 1)

    def test_same_size_replacement_and_catalog_change_invalidate_preview(self):
        plan = self.plan(); details = self.model.stat()
        self.model.write_bytes(b"X" * len(self.payload)); os.utime(self.model, ns=(details.st_atime_ns, details.st_mtime_ns))
        self.assert_error(409, "prepare", {"planId": plan["planId"]})
        self.assertFalse(self.lifecycle.load()["jobs"])

    def test_engine_or_source_change_invalidates_preview_without_mutating_config(self):
        plan = self.plan(); self.engine.write_bytes(b"changed engine")
        self.assert_error(409, "prepare", {"planId": plan["planId"]})
        self.engine.write_bytes(b"restored engine")
        plan = self.plan()
        with mock.patch.object(module, "source_signature", return_value="f" * 64):
            self.assert_error(409, "prepare", {"planId": plan["planId"]})
        self.assertFalse(self.lifecycle.load()["jobs"])

    def test_maya_and_existing_unload_hook_configs_are_not_engine_candidates(self):
        cfg = json.loads(self.config.read_text())
        cfg["before_load"] = ["unload-maya"]
        self.config.write_text(json.dumps(cfg))
        self.assertEqual(self.lifecycle.engine_candidates(), [])
        del cfg["before_load"]; cfg["model_name"] = "Maya-S24"
        self.config.write_text(json.dumps(cfg))
        self.assertEqual(self.lifecycle.engine_candidates(), [])

    def test_unsupported_import_stays_referenced_but_cannot_prepare(self):
        imported = {"id": "local-" + "d" * 24, "path": str(self.model), "name": "Other model", "files": [], "bytes": len(self.payload)}
        state = self.lifecycle.manager.load(); state["imports"][imported["id"]] = imported; self.lifecycle.manager.save(state)
        classifier = types.SimpleNamespace(plan_import=lambda path: {"candidate_selection": None, "issues": ["unsupported_architecture"], "file_set_status": "invalid_or_incomplete"})
        with mock.patch.object(module.files, "inspect_files", return_value=imported), mock.patch.object(self.lifecycle, "source_modules", return_value=(classifier, None, None)):
            self.assert_error(409, "prepare-plan", {"modelId": imported["id"], "engineId": self.engine_id()})
        self.assertIn(imported["id"], self.lifecycle.manager.load()["imports"])
        self.assertFalse(self.lifecycle.load()["jobs"])

    def test_reusing_prepared_data_is_private_does_not_download_or_change_maya(self):
        before = self.source_snapshot()
        job = self.accepted()
        with mock.patch.object(module.subprocess, "Popen", side_effect=AssertionError("reuse must not run setup or an engine")):
            finished = self.run_worker(job)
        self.assertEqual(finished["state"], "done", finished)
        profile = self.lifecycle.load()["profiles"][finished["profileId"]]
        config = module.checked_json(profile["configPath"])
        self.assertEqual(config["exe"], str(self.engine)); self.assertTrue(config["lazy_load"])
        self.assertNotIn("api_key", config)
        self.assertTrue(config["before_load_strict"])
        self.assertFalse(config["before_load_ignore_failure"])
        self.assertEqual(config["before_load"][0], sys.executable)
        self.assertTrue(Path(config["before_load"][1]).is_relative_to(self.lifecycle.root))
        self.assertEqual(config["args"][config["args"].index("--max-context") + 1], "32768")
        self.assertNotEqual(config["args"][config["args"].index("--native") + 1], str(self.model))
        self.assertEqual(self.source_snapshot(), before)

    def test_wrong_worker_token_cannot_stage_or_modify_job(self):
        job = self.accepted(); before = self.lifecycle.state_file.read_bytes()
        module.worker(str(self.home), self.namespace, job["id"], "e" * 64)
        self.assertEqual(self.lifecycle.state_file.read_bytes(), before)
        self.assertFalse((self.lifecycle.root / "profiles").exists())

    def test_cancelled_prepare_keeps_existing_inputs_and_never_signals_stored_pid(self):
        job = self.accepted(); before = self.source_snapshot()
        self.lifecycle.dispatch("desktop", "jobs/" + job["id"] + "/cancel", {})
        with mock.patch.object(os, "kill", side_effect=AssertionError("no stored PID signals")):
            finished = self.run_worker(job)
        self.assertEqual(finished["state"], "cancelled")
        self.assertEqual(self.source_snapshot(), before)

    def test_private_prepare_and_runtime_environment_replaces_active_storage_and_skips_leftovers(self):
        with self.lifecycle.environment():
            for key, original in self.original_env.items(): self.assertNotEqual(os.environ.get(key), original)
            self.assertEqual(os.environ["STRATA_KEEP_LEFTOVERS"], "1")
            self.assertEqual(os.environ["STRATA_NO_PRESET"], "1")
            self.assertEqual(os.environ["STRATA_NO_ACQUIRE"], "1")
        for key, value in self.original_env.items(): self.assertEqual(os.environ[key], value)

    def test_busy_port_is_rejected_without_stopping_the_listener(self):
        profile = self.prepared()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0)); listener.listen()
            port = listener.getsockname()[1]
            self.assert_error(409, "serve-plan", {"profileId": profile["id"], "host": "127.0.0.1", "port": port})
            self.assertEqual(listener.getsockname()[1], port)

    def test_prepared_input_or_engine_changes_prevent_serving(self):
        profile = self.prepared()
        Path(profile["inputs"][0]["path"]).write_bytes(b"different")
        self.assert_error(409, "serve-plan", {"profileId": profile["id"]})

    def test_busy_owned_server_and_changed_identity_are_rejected_before_switch_job(self):
        profile = self.prepared()
        ready = {"state": "ready", "profileId": profile["id"], "modelId": "original", "host": "127.0.0.1", "port": 19001,
                 "loaded": True, "busy": True, "ownerId": "a" * 32, "instanceId": "b" * 32}
        state = self.lifecycle.load(); state["runtime"] = ready; self.lifecycle.save(state)
        with mock.patch.object(self.lifecycle, "runtime_snapshot", return_value=ready):
            self.assert_error(409, "serve-plan", {"profileId": profile["id"]})
        ready["busy"] = False
        with mock.patch.object(self.lifecycle, "runtime_snapshot", return_value=copy.deepcopy(ready)):
            plan = self.lifecycle.dispatch("desktop", "serve-plan", {"profileId": profile["id"]})
        ready["instanceId"] = "c" * 32
        with mock.patch.object(self.lifecycle, "runtime_snapshot", return_value=ready):
            self.assert_error(409, "serve", {"planId": plan["planId"]})

    def test_snapshot_never_returns_api_keys_or_calls_runtime_status(self):
        profile = self.prepared()
        state = self.lifecycle.load(); state["runtime"] = {"profileId": profile["id"], "modelId": "model", "apiKey": "PRIVATE-MANAGED-KEY", "ownerId": "a" * 32, "instanceId": "b" * 32}; self.lifecycle.save(state)
        runtime = types.SimpleNamespace(inspect=lambda name: {"state": "ready", "loaded": False, "busy": False, "owner_id": "a" * 32, "instance_id": "b" * 32},
                                        status=mock.Mock(side_effect=AssertionError("generation forbidden")))
        with mock.patch.object(self.lifecycle, "source_modules", return_value=(None, None, runtime)):
            snapshot = self.lifecycle.snapshot()
        self.assertFalse(snapshot["runtime"]["loaded"])
        self.assertNotIn("EXISTING-SECRET", json.dumps(snapshot)); self.assertNotIn("PRIVATE-MANAGED-KEY", json.dumps(snapshot))
        runtime.status.assert_not_called()

    def test_connect_receipt_is_private_durable_and_instance_bound(self):
        record = {"destination": self.destination, "expiresAt": (self.clock + 300) * 1000, "receiptId": "receipt", "profileId": "prepared",
                  "host": "0.0.0.0", "port": 19001, "apiKey": "PRIVATE-MANAGED-KEY", "modelId": "model", "instanceId": "a" * 32}
        state = self.lifecycle.load(); state["receipts"]["receipt"] = record; self.lifecycle.save(state)
        with mock.patch.object(self.lifecycle, "profile", return_value={}), mock.patch.object(self.lifecycle, "runtime_snapshot", return_value={"state": "ready", "instanceId": "a" * 32, "profileId": "prepared"}):
            result = self.lifecycle.dispatch("desktop", "connect", {"receiptId": "receipt"})
            self.assertEqual(result["endpoint"]["apiKey"], "PRIVATE-MANAGED-KEY")
            self.clock += 301
            self.assertEqual(self.lifecycle.dispatch("desktop", "connect", {"receiptId": "receipt"})["endpoint"]["apiKey"], "PRIVATE-MANAGED-KEY")
        with mock.patch.object(self.lifecycle, "profile", return_value={}), mock.patch.object(self.lifecycle, "runtime_snapshot", return_value={"state": "ready", "instanceId": "b" * 32, "profileId": "prepared"}):
            self.assert_error(409, "connect", {"receiptId": "receipt"})

    def test_fixed_gpu_guard_only_reads_memory_and_refuses_unknown_or_insufficient_cards(self):
        completed = types.SimpleNamespace(stdout="0, 9000\n1, 12000\n")
        compiled = compile(module.GUARD, "guard", "exec")
        for cards, minimum, expected in (("[0]", "8192", None), ("[0,1]", "10000", 1), ("[2]", "8192", 1)):
            with self.subTest(cards=cards, minimum=minimum), mock.patch.object(sys, "argv", ["guard", cards, minimum]), mock.patch.object(module.subprocess, "run", return_value=completed) as run:
                if expected is None: exec(compiled, {})
                else:
                    with self.assertRaises(SystemExit) as caught: exec(compiled, {})
                    self.assertEqual(caught.exception.code, expected)
                self.assertEqual(run.call_args.args[0], ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"])

    def test_fresh_prepare_runs_control_setup_with_isolated_storage_and_no_start(self):
        cfg = json.loads(self.config.read_text()); cfg["args"][1] = "/other/model.gguf"; self.config.write_text(json.dumps(cfg))
        before = self.source_snapshot(); job = self.accepted()
        calls = []
        def start(command, **options):
            calls.append((command, options))
            install = Path(options["env"]["STRATA_INSTALL_DIR"])
            self.assertTrue(install.is_relative_to(self.lifecycle.root))
            self.assertTrue(Path(options["env"]["STRATA_SETTINGS_FILE"]).is_relative_to(install))
            self.assertNotIn("STRATA_NO_ACQUIRE", options["env"])
            self.assertEqual(options["env"]["STRATA_CONTROL_INSTALL"], "1")
            self.assertEqual(options["env"]["STRATA_ALLOW_NO_ENGINE"], "1")
            self.assertIn("--yes", command); self.assertIn("--no-start", command); self.assertIn("--no-browser", command)
            inputs = command[command.index("--gguf-dir") + 1]
            self.assertNotEqual(inputs, str(self.model.parent))
            module.files.atomic_json(install / "strata-prepared.json", {"exe": "component:strata-engine", "args": ["--native", str(Path(inputs)/self.model.name), "--max-context", "32768"],
                "tokenizer": str(self.tokenizer), "model_name": "fixture-prepared"})
            return types.SimpleNamespace(poll=lambda: 0, returncode=0)
        with mock.patch.object(module.Lifecycle, "check_dependencies"), mock.patch.object(module.subprocess, "Popen", side_effect=start):
            finished = self.run_worker(job)
        self.assertEqual(finished["state"], "done", finished)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.source_snapshot(), before)

    def test_start_creates_lazy_listener_then_busy_switch_restores_saved_profile(self):
        from strata_control import runtime, profiles
        first = self.prepared()
        ready = {"ok": True, "state": "ready", "owner_id": "a" * 32, "instance_id": "b" * 32, "loaded": False, "busy": False}
        running = {}
        def inspect(name):
            return {**ready, **running} if running else {"state": "stopped"}
        def start_runtime(name, **kwargs):
            saved, _ = profiles.load(name)
            path = Path(saved["source_config"])
            cfg = module.checked_json(path)
            running.update(source_config=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), model_id=cfg["model_name"],
                           api_identity_verified=True, launch_attested=True, child_alive=True, owner_host_alive=True)
            return ready
        before = self.source_snapshot()
        plan = self.lifecycle.dispatch("desktop", "serve-plan", {"profileId": first["id"], "host": "127.0.0.1"})
        accepted = self.lifecycle.dispatch("desktop", "serve", {"planId": plan["planId"]})
        job = self.lifecycle.load()["jobs"][accepted["job"]["id"]]
        with mock.patch.object(module.Lifecycle, "check_dependencies"), mock.patch.object(runtime, "start", side_effect=start_runtime) as start, mock.patch.object(runtime, "inspect", side_effect=inspect):
            finished = self.run_worker(job)
        self.assertEqual(finished["state"], "done", finished)
        with self.lifecycle.environment(): old_saved, _ = profiles.load(self.lifecycle.runtime_name)
        private_cfg = module.checked_json(old_saved["source_config"])
        self.assertTrue(private_cfg["lazy_load"]); self.assertTrue(private_cfg["api_key"])
        old_runtime = copy.deepcopy(self.lifecycle.load()["runtime"])
        self.assertEqual(start.call_args.kwargs["expected_source_sha256"], hashlib.sha256(Path(old_saved["source_config"]).read_bytes()).hexdigest())
        second = self.prepared()
        with mock.patch.object(runtime, "inspect", side_effect=inspect):
            plan = self.lifecycle.dispatch("desktop", "serve-plan", {"profileId": second["id"]})
            accepted = self.lifecycle.dispatch("desktop", "serve", {"planId": plan["planId"]})
        job = self.lifecycle.load()["jobs"][accepted["job"]["id"]]
        with mock.patch.object(module.Lifecycle, "check_dependencies"), mock.patch.object(runtime, "inspect", side_effect=inspect), \
                mock.patch.object(runtime, "restart_if_idle", return_value={"ok": False, "state": "ready", "error": "busy"}) as restart:
            finished = self.run_worker(job)
        self.assertEqual(finished["state"], "error")
        self.assertEqual(restart.call_args.kwargs["expected_owner_id"], ready["owner_id"])
        self.assertEqual(restart.call_args.kwargs["expected_instance_id"], ready["instance_id"])
        with self.lifecycle.environment(): current_saved, _ = profiles.load(self.lifecycle.runtime_name)
        self.assertEqual(current_saved, old_saved)
        self.assertEqual(self.lifecycle.load()["runtime"], old_runtime)
        self.assertEqual(self.source_snapshot(), before)

    def test_dependency_failure_does_not_save_runtime_profile_or_launch_server(self):
        from strata_control import runtime
        profile = self.prepared()
        plan = self.lifecycle.dispatch("desktop", "serve-plan", {"profileId": profile["id"], "host": "127.0.0.1"})
        accepted = self.lifecycle.dispatch("desktop", "serve", {"planId": plan["planId"]})
        job = self.lifecycle.load()["jobs"][accepted["job"]["id"]]
        with mock.patch.object(module.Lifecycle, "check_dependencies", side_effect=module.ApiError(503, "Missing dependencies")), mock.patch.object(runtime, "start") as start:
            finished = self.run_worker(job)
        self.assertEqual(finished["state"], "error"); start.assert_not_called()
        self.assertFalse((self.lifecycle.root / "runtime-profiles").exists())

    @unittest.skipUnless(os.name == "posix", "disposable POSIX engine fixture")
    def test_real_private_prepare_start_and_switch_keep_endpoint_lazy_without_generation(self):
        from strata_control import runtime
        from tools.strata_tokenizer import BYTE_TO_UNICODE
        generation_marker = self.root / "unexpected-generation"
        self.engine.write_text("#!" + sys.executable + "\nimport sys\nfrom pathlib import Path\n"
            "print('READY 4096 stop',flush=True)\n"
            "for line in sys.stdin:\n"
            " if line.strip()=='QUIT':break\n"
            " if line.startswith(('GEN ','BGEN ')):\n"
            f"  Path({str(generation_marker)!r}).write_text('generation')\n"
            "  print('T 97',flush=True)\n"
            "  print('DONE 1 1 1.0 1.0 length 0 0 0',flush=True)\n")
        self.engine.chmod(0o700)
        tokens = [BYTE_TO_UNICODE[index] for index in range(256)] + ["ab", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
        (self.tokenizer / "vocab.json").write_text(json.dumps({token: index for index, token in enumerate(tokens)}))
        (self.tokenizer / "merges.txt").write_text("a b")
        (self.tokenizer / "token_type.json").write_text(json.dumps([0] * 257 + [3, 3, 3]))
        before = self.source_snapshot()
        def stop_owned():
            with self.lifecycle.environment():
                try:
                    if runtime._read_state(self.lifecycle.runtime_name): runtime.stop(self.lifecycle.runtime_name, timeout=15)
                except Exception: pass
        self.addCleanup(stop_owned)
        first = self.prepared()
        plan = self.lifecycle.dispatch("desktop", "serve-plan", {"profileId": first["id"], "host": "127.0.0.1"})
        accepted = self.lifecycle.dispatch("desktop", "serve", {"planId": plan["planId"]})
        finished = self.run_worker(self.lifecycle.load()["jobs"][accepted["job"]["id"]])
        self.assertEqual(finished["state"], "done", finished)
        first_runtime = self.lifecycle.runtime_snapshot(self.lifecycle.load())
        self.assertEqual(first_runtime["state"], "ready", first_runtime)
        self.assertFalse(first_runtime["loaded"])
        first_record = copy.deepcopy(self.lifecycle.load()["runtime"])
        endpoint = self.lifecycle.dispatch("desktop", "connect", {"receiptId": finished["receiptId"]})["endpoint"]
        self.assertEqual(endpoint["port"], plan["port"])
        # Prepare a second immutable profile/context while the first API remains
        # listening. This uses real staging, profile files, runtime owner IPC,
        # attested raw telemetry and HTTP, with only a disposable engine binary.
        plan2 = self.plan(context=16384)
        accepted2 = self.lifecycle.dispatch("desktop", "prepare", {"planId": plan2["planId"]})
        finished2 = self.run_worker(self.lifecycle.load()["jobs"][accepted2["job"]["id"]])
        self.assertEqual(finished2["state"], "done", finished2)
        serve2 = self.lifecycle.dispatch("desktop", "serve-plan", {"profileId": finished2["profileId"]})
        self.assertEqual(serve2["action"], "switch")
        accepted3 = self.lifecycle.dispatch("desktop", "serve", {"planId": serve2["planId"]})
        finished3 = self.run_worker(self.lifecycle.load()["jobs"][accepted3["job"]["id"]])
        self.assertEqual(finished3["state"], "done", finished3)
        final = self.lifecycle.runtime_snapshot(self.lifecycle.load())
        self.assertEqual(final["state"], "ready", final)
        self.assertEqual(final["port"], first_runtime["port"])
        self.assertEqual(final["ownerId"], first_runtime["ownerId"])
        self.assertNotEqual(final["instanceId"], first_runtime["instanceId"])
        self.assertFalse(final["loaded"])
        self.assertEqual(self.lifecycle.load()["runtime"]["apiKey"], first_record["apiKey"])
        self.assert_error(409, "connect", {"receiptId": finished["receiptId"]})
        current_endpoint = self.lifecycle.dispatch("desktop", "connect", {"receiptId": finished3["receiptId"]})["endpoint"]
        connection = http.client.HTTPConnection("127.0.0.1", current_endpoint["port"], timeout=5)
        try:
            connection.request("GET", "/v1/status", headers={"Authorization": "Bearer " + current_endpoint["apiKey"]})
            response = connection.getresponse(); status = json.loads(response.read())
        finally: connection.close()
        self.assertEqual(response.status, 200); self.assertFalse(status["loaded"])
        self.assertFalse(generation_marker.exists())
        self.assertEqual(self.source_snapshot(), before)

    def test_failed_unpublished_preparation_cleans_only_its_owned_temporary_copy(self):
        before = self.source_snapshot(); job = self.accepted()
        with mock.patch.object(module.Lifecycle, "check_prepare", side_effect=[None, module.ApiError(409, "Changed during staging")]):
            finished = self.run_worker(job)
        self.assertEqual(finished["state"], "error")
        self.assertFalse((self.lifecycle.root / "profiles" / job["id"]).exists())
        self.assertEqual(self.source_snapshot(), before)
        published = self.prepared()
        self.assertTrue(Path(published["configPath"]).exists())

    def test_reuse_canonicalizes_alias_and_relative_inputs_into_private_copies(self):
        cfg = json.loads(self.config.read_text())
        (self.model.parent / "sub").mkdir()
        alias = self.model.parent / "sub" / ".." / self.model.name
        cfg["cwd"] = str(self.root)
        cfg["args"] = ["--native", str(alias), "--pack", "Strata/prepared/pack", "--max-context", "131072"]
        cfg["tokenizer"] = "Strata/prepared/pack/tokenizer"
        vector = self.source / "vector.gguf"; vector.write_bytes(self.payload)
        cfg["args"] += ["--control-vector-scaled", "Strata/vector.gguf:1.0", "--control-vector-layer-range", "4", "44", "--cvec-mode", "project"]
        self.config.write_text(json.dumps(cfg)); before = self.source_snapshot()
        profile = self.prepared(); prepared = module.checked_json(profile["configPath"])
        for flag in ("--native", "--pack"):
            path = Path(prepared["args"][prepared["args"].index(flag) + 1])
            self.assertTrue(path.is_relative_to(self.lifecycle.root), (flag, path))
            self.assertTrue(path.exists())
        vector_value = prepared["args"][prepared["args"].index("--control-vector-scaled") + 1]
        self.assertTrue(Path(vector_value.rpartition(":")[0]).is_relative_to(self.lifecycle.root))
        self.assertTrue(Path(prepared["tokenizer"]).is_relative_to(self.lifecycle.root))
        self.assertEqual(self.source_snapshot(), before)

    def test_reuse_rejects_output_and_unknown_path_options_before_copying(self):
        original = json.loads(self.config.read_text())
        for extra in (["--expert-profile-save", str(self.source / "must-not-write.bin")], ["--future-file-option", str(self.source / "file")], ["--prefill", "../../outside"]):
            with self.subTest(extra=extra):
                cfg = copy.deepcopy(original); cfg["args"] += extra; self.config.write_text(json.dumps(cfg))
                self.assert_error(409, "prepare-plan", {"modelId": "native-q2_0", "engineId": self.engine_id()})
                self.assertFalse(self.lifecycle.load()["jobs"])
        self.assertFalse((self.source / "must-not-write.bin").exists())

    def test_reused_profile_drops_device_visibility_and_external_output_environment(self):
        cfg = json.loads(self.config.read_text())
        cfg["env"] = {"CUDA_VISIBLE_DEVICES": "1", "STRATA_EXPERT_PROFILE_SAVE": str(self.source / "outside-profile.bin"), "LD_PRELOAD": "/outside/library.so"}
        cfg["draft_vocab"] = "/outside/draft.bin"
        self.config.write_text(json.dumps(cfg)); before = self.source_snapshot()
        profile = self.prepared(); prepared = module.checked_json(profile["configPath"])
        self.assertNotIn("env", prepared); self.assertNotIn("draft_vocab", prepared)
        self.assertEqual(json.loads(prepared["before_load"][2]), [0])
        self.assertEqual(self.source_snapshot(), before)

    def test_reuse_dependency_change_after_review_refuses_job(self):
        plan = self.plan(); (self.pack / "native_experts.txt").write_text("changed original prepared data")
        self.assert_error(409, "prepare", {"planId": plan["planId"]})
        self.assertFalse(self.lifecycle.load()["jobs"])

    def test_standard_kv_resident_arity_and_string_gpu_selection_are_preserved(self):
        cfg = json.loads(self.config.read_text())
        cfg["args"] += ["--kv-resident", "32768"]
        cfg["gpu"] = "0,1"
        self.config.write_text(json.dumps(cfg))
        profile = self.prepared(); prepared = module.checked_json(profile["configPath"])
        self.assertEqual(prepared["args"][prepared["args"].index("--kv-resident") + 1], "32768")
        self.assertEqual(prepared["gpu"], [0, 1]); self.assertEqual(json.loads(prepared["before_load"][2]), [0, 1])

    def test_standard_legacy_gpu_flags_are_qualified_and_vision_is_disabled(self):
        cfg = json.loads(self.config.read_text())
        cfg["args"] += ["--vision", "--remote-expert-opt", "--pcie-frac", "0.75", "--spec-min-p", "0.5"]
        cfg["vision"] = {"exe": "/original/vision", "model": "/original/mmproj.gguf"}
        self.config.write_text(json.dumps(cfg)); before = self.source_snapshot()
        profile = self.prepared(); prepared = module.checked_json(profile["configPath"])
        self.assertNotIn("--vision", prepared["args"]); self.assertNotIn("vision", prepared)
        self.assertIn("--remote-expert-opt", prepared["args"])
        self.assertEqual(prepared["args"][prepared["args"].index("--pcie-frac") + 1], "0.75")
        self.assertEqual(self.source_snapshot(), before)

    def test_old_catalog_cannot_cache_old_control_runtime_in_fresh_python(self):
        (self.source / "strata_control/__init__.py").write_text("# deliberately old package\n")
        (self.source / "strata_control/runtime.py").write_text("raise RuntimeError('old runtime must never execute')\n")
        setup = self.source / "setup.py"
        setup.write_text("from strata_control import acquisition\n" + setup.read_text())
        script = ("import importlib.util,json,sys\nfrom pathlib import Path\n"
            + f"spec=importlib.util.spec_from_file_location('lifecycle', {str(ROOT / 'apps/host/python/remote_model_lifecycle.py')!r})\n"
            + "m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)\n"
            + f"life=m.Lifecycle({str(ROOT)!r},{self.namespace!r},{self.destination!r},{str(self.source)!r},{str(self.home)!r})\n"
            + "life.manager.current_catalog()\nwith life.environment():\n models,profiles,runtime=life.source_modules()\n print(json.dumps({'runtime':str(Path(runtime.__file__).resolve())}))\n")
        completed = module.subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=20, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["runtime"], str(ROOT / "strata_control/runtime.py"))

    @unittest.skipUnless(os.name == "posix", "disposable POSIX owner fixture")
    def test_real_first_start_worker_crash_recovers_durable_intent_and_connection(self):
        marker = self.real_engine_fixture()
        profile = self.prepared()
        plan = self.lifecycle.dispatch("desktop", "serve-plan", {"profileId": profile["id"], "host": "127.0.0.1"})
        accepted = self.lifecycle.dispatch("desktop", "serve", {"planId": plan["planId"]})
        job = self.lifecycle.load()["jobs"][accepted["job"]["id"]]
        with mock.patch.object(module.Lifecycle, "publish_runtime", side_effect=SystemExit("disposable worker crash")):
            with self.assertRaises(SystemExit): self.run_worker(job)
        interrupted = self.lifecycle.load()
        self.assertFalse(interrupted["runtime"])
        self.assertTrue(interrupted["intent"]["apiKey"])
        self.assertEqual(interrupted["intent"]["jobId"], job["id"])
        recovered = self.lifecycle.snapshot()
        self.assertEqual(recovered["runtime"]["state"], "ready", recovered)
        current = self.lifecycle.load()
        self.assertFalse(current["intent"])
        finished = current["jobs"][job["id"]]
        self.assertEqual(finished["state"], "done")
        connected = self.lifecycle.dispatch("desktop", "connect", {"receiptId": finished["receiptId"]})
        self.assertEqual(connected["endpoint"]["port"], plan["port"])
        self.assertNotIn(current["runtime"]["apiKey"], json.dumps(recovered))
        self.assertFalse(marker.exists())

    @unittest.skipUnless(os.name == "posix", "disposable POSIX owner fixture")
    def test_real_failed_first_start_reaps_only_exited_owner_and_can_retry_from_gui(self):
        marker = self.real_engine_fixture(valid_tokenizer=False)
        profile = self.prepared()
        plan = self.lifecycle.dispatch("desktop", "serve-plan", {"profileId": profile["id"], "host": "127.0.0.1"})
        accepted = self.lifecycle.dispatch("desktop", "serve", {"planId": plan["planId"]})
        failed = self.run_worker(self.lifecycle.load()["jobs"][accepted["job"]["id"]])
        self.assertEqual(failed["state"], "error", failed)
        failed_state = self.lifecycle.load()
        self.assertTrue(failed_state["intent"])
        observed = self.lifecycle.runtime_snapshot(failed_state)
        self.assertEqual(observed["state"], "stopped", observed)
        self.assertTrue(observed["reapNeeded"], observed)
        original_key = failed_state["intent"]["apiKey"]
        # Supply a valid new prepared profile; the failed profile's private
        # inputs stay immutable. No original API or model process is present.
        self.real_engine_fixture(valid_tokenizer=True)
        repaired = self.prepared()
        preview = self.lifecycle.dispatch("desktop", "serve-plan", {"profileId": repaired["id"]})
        self.assertEqual(preview["port"], plan["port"])
        accepted = self.lifecycle.dispatch("desktop", "serve", {"planId": preview["planId"]})
        finished = self.run_worker(self.lifecycle.load()["jobs"][accepted["job"]["id"]])
        self.assertEqual(finished["state"], "done", finished)
        self.assertEqual(self.lifecycle.load()["runtime"]["apiKey"], original_key)
        self.assertFalse(marker.exists())


if __name__ == "__main__": unittest.main()
