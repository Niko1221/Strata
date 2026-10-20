"""Offline contracts for the opt-in server model manager's isolated references."""

import hashlib
import importlib.util
import io
import json
import os
import struct
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "apps/host/python/remote_model_manager.py"
SPEC = importlib.util.spec_from_file_location("remote_model_manager_for_test", HELPER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import remote model manager")
manager_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = manager_module
SPEC.loader.exec_module(manager_module)


def gguf_fixture(architecture="llama", model_type=None, split=None):
    """One tiny float32 tensor; its declared payload is complete."""
    def string(value):
        raw = value.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    pairs = [("general.architecture", 8, string(architecture))]
    if model_type is not None:
        pairs.append(("general.type", 8, string(model_type)))
    if split is not None:
        pairs += [(key, 4, struct.pack("<I", value)) for key, value in
                  (("split.count", split[0]), ("split.no", split[1]), ("split.tensors.count", split[2]))]
    header = (b"GGUF" + struct.pack("<IQQ", 3, 1, len(pairs))
              + b"".join(string(key) + struct.pack("<I", kind) + value for key, kind, value in pairs)
              + string("token_embd.weight") + struct.pack("<IQIQ", 1, 32, 0, 0))
    return header + b"\0" * (-len(header) % 32) + b"\0" * 128


class RemoteModelManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source/"strata_control").mkdir()
        (self.source/"strata_control/acquisition.py").write_text("# source identity fixture\n")
        (self.source/"strata.py").write_text("# source identity fixture\n")
        self.home = self.root / "manager"
        self.clock = 1_800_000_000.0
        self.payload = b"synthetic pinned fixture bytes"
        self.write_catalog()
        self.active = self.source / "strata-serving.json"
        self.active.write_text(json.dumps({"api_key": "keep-active-secret", "port": 8080,
                                           "exe": "/serving/engine", "args": ["--native", "/serving/model.gguf"]}))
        self.settings = self.root / "settings.json"
        self.settings.write_text(json.dumps({"data_dir": "/serving/assets", "installs": [str(self.source)]}))
        self.environment = mock.patch.dict(os.environ, {
            "STRATA_INSTALL_DIR": str(self.source), "STRATA_SETTINGS_FILE": str(self.settings),
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.manager = manager_module.Manager(self.home, self.source, now=lambda: self.clock)
        self.worker = mock.patch.object(self.manager, "spawn_worker", return_value=None)
        self.spawn = self.worker.start()
        self.addCleanup(self.worker.stop)
        self.owner = "remote:owner-one"
        self.other = "remote:owner-two"

    def write_catalog(self, *, pinned=True):
        model = {"Q2_0": {"about": "Fixture", "families": ("qwen",),
                            "download_gb": len(self.payload) / 1e9, "vision": True}}
        family = {"qwen": {"title": "Fixture", "hf": "https://huggingface.co/demo/repo/resolve/" + "a" * 40 + "/",
                            "file": "Fixture-{q}-{i}.gguf", "shards": 1, "tag": ""}}
        pins = {"demo/repo": {"Fixture-Q2_0-1.gguf":
                               (len(self.payload), hashlib.sha256(self.payload).hexdigest())}} if pinned else {}
        (self.source / "setup.py").write_text(
            "MODELS = " + repr(model) + "\nFAMILIES = " + repr(family)
            + "\nOWN_PATH_FAMILIES = {}\nREPO_PINS = " + repr(pins)
            + "\nHF_REVISIONS = {'demo/repo': '" + "a" * 40 + "'}\n", encoding="utf-8")

    def call(self, action="", method="GET", body=None, *, owner=None, request_id=None):
        return self.manager.dispatch(owner or self.owner, action, method, body, request_id=request_id)

    def preview(self, **extra):
        return self.call("/plan", "POST", {"modelId": "native-q2_0", "readPictures": False, **extra})

    def assert_api_error(self, status, action, method="POST", body=None, *, owner=None):
        with self.assertRaises(manager_module.ApiError) as caught:
            self.call(action, method, body, owner=owner)
        expected = (status,) if isinstance(status, int) else status
        self.assertIn(caught.exception.status, expected)
        return caught.exception

    @staticmethod
    def snapshot(paths):
        return {str(path): (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}

    def import_preview(self, path):
        return self.call("/imports/inspect", "POST", {"path": str(path)})

    def model_file(self, name="external.gguf"):
        path = self.root / name
        path.write_bytes(gguf_fixture())
        return path

    def test_catalog_and_preview_use_only_pinned_catalog_ids(self):
        catalog = self.call()
        self.assertTrue(catalog["supported"])
        self.assertTrue(catalog["capabilities"]["downloads"])
        row = next(row for row in catalog["models"] if row["id"] == "native-q2_0")
        self.assertNotEqual(row["state"], "loaded")
        preview = self.preview()
        self.assertIsInstance(preview["planId"], str)
        self.assertGreater(preview["expiresAt"], self.clock * 1000)
        self.assertTrue(preview["warnings"])
        self.spawn.assert_not_called()
        self.assertNotIn("keep-active-secret", json.dumps([catalog, preview]))

    def test_catalog_inspection_writes_nothing_into_source_tree(self):
        before = self.snapshot([path for path in self.source.rglob("*") if path.is_file()])
        self.call()
        self.preview()
        after = self.snapshot([path for path in self.source.rglob("*") if path.is_file()])
        self.assertEqual(after, before)

    def test_unrelated_setup_is_rejected_without_executing_it(self):
        unrelated=self.root/"unrelated"
        unrelated.mkdir()
        marker=self.root/"must-not-exist"
        (unrelated/"setup.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('installer ran')\n")
        with self.assertRaises(manager_module.ApiError): manager_module.find_source(unrelated)
        with self.assertRaises(manager_module.ApiError): manager_module.catalog(unrelated)
        self.assertFalse(marker.exists())

    def test_actual_repo_catalog_is_fully_pinned_without_network_or_installer(self):
        before = self.snapshot([ROOT / "setup.py"])
        with mock.patch.object(manager_module.urllib.request, "urlopen", side_effect=AssertionError("network forbidden")), \
                mock.patch.object(manager_module.subprocess, "Popen", side_effect=AssertionError("process forbidden")), \
                mock.patch.object(manager_module.subprocess, "run", side_effect=AssertionError("process forbidden")):
            rows = manager_module.catalog(ROOT)
        self.assertIn("native-q2_0", rows)
        self.assertTrue(rows)
        for row in rows.values():
            self.assertTrue(row["files"])
            for entry in row["files"] + row["visionFiles"]:
                self.assertGreater(entry["bytes"], 0)
                self.assertRegex(entry["sha256"], r"\A[0-9a-f]{64}\Z")
                self.assertRegex(entry["url"], r"\Ahttps://[^/]+/.+/resolve/[0-9a-f]{40}/")
                self.assertFalse(Path(entry["path"]).is_absolute())
                self.assertNotIn("..", Path(entry["path"]).parts)
        self.assertEqual(self.snapshot([ROOT / "setup.py"]), before)

    def test_source_discovery_finds_current_and_packaged_terminal_folder(self):
        with mock.patch.object(manager_module.Path, "cwd", return_value=self.source):
            self.assertEqual(manager_module.find_source(), self.source.resolve())
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        for layout in ("", "python"):
            packaged = self.root / ("terminal-"+layout) / "current"
            source=packaged/layout
            source.mkdir(parents=True)
            (source/"setup.py").write_bytes((self.source/"setup.py").read_bytes())
            (source/"strata.py").write_text("# packaged identity fixture\n")
            (source/"strata_control").mkdir()
            (source/"strata_control/acquisition.py").write_text("# packaged identity fixture\n")
            executable = packaged / "bin/strata"
            executable.parent.mkdir()
            executable.write_text("synthetic launcher; never executed")
            before = self.snapshot([p for p in packaged.rglob("*") if p.is_file()])
            with mock.patch.dict(os.environ, {}, clear=True), \
                    mock.patch.object(manager_module.Path, "cwd", return_value=elsewhere), \
                    mock.patch.object(manager_module.shutil, "which", return_value=str(executable)):
                self.assertEqual(manager_module.find_source(), source.resolve())
            self.assertEqual(self.snapshot([p for p in packaged.rglob("*") if p.is_file()]), before)

    def test_arbitrary_download_options_and_untyped_picture_choice_are_rejected(self):
        for extra in ({"data_dir": "/tmp/elsewhere"}, {"gguf_dir": "/tmp/elsewhere"},
                      {"loadAfter": True}, {"readPictures": "false"}):
            with self.subTest(extra=extra):
                self.assert_api_error(400, "/plan", body={"modelId": "native-q2_0", "readPictures": False, **extra})
        self.assert_api_error((400, 404), "/plan", body={"modelId": "https://example.invalid/model.gguf", "readPictures": False})
        self.spawn.assert_not_called()

    def test_preview_is_owner_bound_and_single_use(self):
        plan = self.preview()
        self.assert_api_error(409, "/downloads", body={"planId": plan["planId"]}, owner=self.other)
        accepted = self.call("/downloads", "POST", {"planId": plan["planId"]})
        self.assertIn("job", accepted)
        self.spawn.assert_called_once()
        self.assert_api_error(409, "/downloads", body={"planId": plan["planId"]})
        self.spawn.assert_called_once()

    def test_expired_preview_does_not_start_worker(self):
        plan = self.preview()
        self.clock += 301
        self.assert_api_error(409, "/downloads", body={"planId": plan["planId"]})
        self.spawn.assert_not_called()

    def test_only_accepting_owner_can_cancel_download_job(self):
        plan = self.preview()
        accepted = self.call("/downloads", "POST", {"planId": plan["planId"]})
        job_id = accepted["job"]["id"]
        cancel_path = f"/downloads/{job_id}/cancel"
        self.assert_api_error((403, 404, 409), cancel_path, body={}, owner=self.other)
        catalog = self.call()
        job = next(job for job in catalog["jobs"] if job["id"] == job_id)
        self.assertNotIn(job["state"], ("cancelled", "canceled", "cancelling"))
        result = self.call(cancel_path, "POST", {})
        self.assertTrue(result)
        self.spawn.assert_called_once()

    def test_desktop_can_stop_a_phone_owned_job_without_signalling_any_pid(self):
        job=self.accepted_job()
        desktop=next(j for j in self.call(owner="desktop")["jobs"] if j["id"]==job["id"])
        self.assertTrue(desktop["canCancel"])
        with mock.patch.object(manager_module.os,"kill",side_effect=AssertionError("Never signal a stored PID")):
            result=self.call(f"/downloads/{job['id']}/cancel","POST",{},owner="desktop")
        self.assertEqual(result["job"]["state"],"cancelled")

    def test_interrupted_admission_without_pid_recovers_after_startup_grace(self):
        job=self.accepted_job()
        self.assertEqual(job["state"],"queued")
        self.clock+=31
        recovered=next(j for j in self.call()["jobs"] if j["id"]==job["id"])
        self.assertEqual(recovered["state"],"error")
        self.accepted_job()

    def test_missing_source_keeps_jobs_visible_and_cancellable(self):
        job=self.accepted_job()
        self.source.rename(self.root/"moved-source")
        manager=manager_module.Manager(self.home,self.source,now=lambda:self.clock)
        snapshot=manager.dispatch(self.owner,"","GET")
        self.assertFalse(snapshot["capabilities"]["downloads"])
        self.assertTrue(snapshot["capabilities"]["imports"])
        self.assertEqual(snapshot["jobs"][0]["id"],job["id"])
        cancelled=manager.dispatch(self.owner,f"/downloads/{job['id']}/cancel","POST",{})
        self.assertEqual(cancelled["job"]["state"],"cancelled")
        with self.assertRaises(manager_module.ApiError):
            manager.dispatch(self.owner,"/plan","POST",{"modelId":"native-q2_0","readPictures":False})

    def test_snapshot_recovers_accepted_job_with_plan_id_and_owner(self):
        plan = self.preview()
        with self.assertRaises(manager_module.ApiError) as caught:
            self.call("/downloads", "POST", {"planId": plan["planId"]}, request_id="mismatched-preview")
        self.assertEqual(caught.exception.status, 400)
        accepted = self.call("/downloads", "POST", {"planId": plan["planId"]}, request_id=plan["planId"])
        recovered = next(job for job in self.call()["jobs"] if job["requestId"] == plan["planId"])
        self.assertEqual(recovered["id"], accepted["job"]["id"])
        self.assertTrue(recovered["canCancel"])
        other = next(job for job in self.call(owner=self.other)["jobs"] if job["id"] == recovered["id"])
        self.assertFalse(other["canCancel"])
        self.spawn.assert_called_once()

    def test_changed_setup_source_invalidates_download_preview(self):
        plan = self.preview()
        with (self.source / "setup.py").open("a") as stream:
            stream.write("\n# source changed after inspection\n")
        self.assert_api_error(409, "/downloads", body={"planId": plan["planId"]})
        self.spawn.assert_not_called()

    def test_missing_repository_pin_fails_closed(self):
        self.write_catalog(pinned=False)
        manager = manager_module.Manager(self.root / "unpinned-manager", self.source, now=lambda: self.clock)
        catalog = manager.dispatch(self.owner, "", "GET", None)
        self.assertFalse(any(row["id"] == "native-q2_0" for row in catalog.get("models", [])))
        with self.assertRaises(manager_module.ApiError):
            manager.dispatch(self.owner, "/plan", "POST", {"modelId": "native-q2_0", "readPictures": False})

    def test_generic_gguf_import_registers_only_an_unverified_reference(self):
        path = self.model_file()
        before = self.snapshot([path, self.active, self.settings])
        plan = self.import_preview(path)
        self.assertTrue(plan["warnings"])
        imported = self.call("/imports", "POST", {"planId": plan["planId"]})
        self.assertIn("model", imported)
        state = json.loads((self.home / "state.json").read_text())
        self.assertTrue(state["imports"])
        self.assertEqual(self.snapshot([path, self.active, self.settings]), before)
        self.assertEqual(list(self.home.rglob("strata-*.json")), [])
        self.assertEqual(list(self.source.glob("strata-*.json")), [self.active])
        self.spawn.assert_not_called()
        snapshot = self.call()
        item = next(row for row in snapshot["models"] if row["id"] == imported["model"]["id"])
        self.assertEqual(item["state"], "imported")
        self.assertNotIn("keep-active-secret", json.dumps([plan, imported, snapshot]))

    def test_import_preview_is_owner_bound_and_single_use(self):
        plan = self.import_preview(self.model_file())
        self.assert_api_error(409, "/imports", body={"planId": plan["planId"]}, owner=self.other)
        self.call("/imports", "POST", {"planId": plan["planId"]})
        self.assert_api_error(409, "/imports", body={"planId": plan["planId"]})

    def test_expired_import_preview_cannot_register_reference(self):
        plan = self.import_preview(self.model_file())
        self.clock += 301
        self.assert_api_error(409, "/imports", body={"planId": plan["planId"]})
        self.assertFalse(json.loads((self.home / "state.json").read_text()).get("imports"))

    def test_import_rejects_changed_source_after_preview(self):
        path = self.model_file()
        plan = self.import_preview(path)
        path.write_bytes(path.read_bytes() + b"changed")
        self.assert_api_error(409, "/imports", body={"planId": plan["planId"]})
        self.assertFalse(json.loads((self.home / "state.json").read_text()).get("imports"))

    def test_import_rejects_same_size_replacement_even_with_original_mtime(self):
        path = self.model_file()
        plan = self.import_preview(path)
        original = path.stat()
        replacement = self.root / "replacement.gguf"
        replacement.write_bytes(path.read_bytes())
        os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
        os.replace(replacement, path)
        self.assert_api_error(409, "/imports", body={"planId": plan["planId"]})

    def test_import_requires_complete_matching_shards_and_rechecks_every_member(self):
        first = self.root / "Shard-00001-of-00002.gguf"
        second = self.root / "Shard-00002-of-00002.gguf"
        first.write_bytes(gguf_fixture(split=(2, 0, 2)))
        with self.assertRaises((manager_module.ApiError, OSError)):
            self.import_preview(first)
        second.write_bytes(gguf_fixture(split=(2, 1, 2)))
        before = self.snapshot([first, second])
        plan = self.import_preview(second)
        self.assertEqual(plan["files"], [first.name, second.name])
        self.assertEqual(self.snapshot([first, second]), before)
        second.write_bytes(gguf_fixture(architecture="mistral", split=(2, 1, 2)))
        self.assert_api_error((400, 409), "/imports", body={"planId": plan["planId"]})
        self.assertFalse(json.loads((self.home / "state.json").read_text()).get("imports"))

    def test_import_rejects_symlink_and_truncated_payload(self):
        path = self.model_file()
        if hasattr(os, "symlink"):
            link = self.root / "link.gguf"
            link.symlink_to(path)
            self.assert_api_error((400, 409), "/imports/inspect", body={"path": str(link)})
        path.write_bytes(gguf_fixture()[:-4])
        self.assert_api_error((400, 409), "/imports/inspect", body={"path": str(path)})
        self.spawn.assert_not_called()

    def test_import_metadata_count_is_bounded_before_payload_reads(self):
        path = self.root / "malicious.gguf"
        path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1_000_000_000))
        self.assert_api_error(400, "/imports/inspect", body={"path": str(path)})
        self.spawn.assert_not_called()

    def test_import_rejects_adapter_marked_as_a_complete_model(self):
        path = self.root / "adapter.gguf"
        path.write_bytes(gguf_fixture(model_type="adapter"))
        self.assert_api_error(400, "/imports/inspect", body={"path": str(path)})

    def test_download_acceptance_preserves_active_config_and_settings(self):
        before = self.snapshot([self.active, self.settings])
        plan = self.preview()
        self.call("/downloads", "POST", {"planId": plan["planId"]})
        self.assertEqual(self.snapshot([self.active, self.settings]), before)
        self.assertEqual(list(self.source.glob("strata-*.json")), [self.active])
        self.assertEqual(list(self.home.rglob("strata-*.json")), [])

    def accepted_job(self):
        plan = self.preview()
        accepted = self.call("/downloads", "POST", {"planId": plan["planId"]})
        state = json.loads((self.home / "state.json").read_text())
        return state["jobs"][accepted["job"]["id"]]

    def asset_path(self, job):
        return self.manager.assets / manager_module.digest(job["manifest"]["files"]) / "Fixture-Q2_0-1.gguf"

    def run_worker(self, job, payload, on_read=None, *, token=None, status=200, headers=None, response_url=None):
        class Response(io.BytesIO):
            def read(response, size=-1):
                if on_read is not None:
                    on_read()
                return super().read(size)

        response = Response(payload)
        response.url = response_url or "https://huggingface.co/demo/repo/resolve/" + "a" * 40 + "/Fixture-Q2_0-1.gguf"
        response.status, response.headers = status, headers or {}
        with mock.patch.object(manager_module.urllib.request, "urlopen", return_value=response) as fetch, \
                mock.patch.object(manager_module, "process_identity", return_value=("offline-fixture", "offline worker")), \
                mock.patch.object(manager_module.signal, "signal"):
            manager_module.worker(self.home, job["id"], token or job["workerToken"])
        return fetch

    def test_worker_verifies_pinned_bytes_into_isolated_assets(self):
        job = self.accepted_job()
        before = self.snapshot([self.active, self.settings, self.source / "setup.py"])
        fetch = self.run_worker(job, self.payload)
        fetch.assert_called_once()
        request = fetch.call_args.args[0]
        self.assertTrue(request.full_url.startswith("https://huggingface.co/demo/repo/resolve/" + "a" * 40 + "/"))
        state = json.loads((self.home / "state.json").read_text())
        self.assertEqual(state["jobs"][job["id"]]["state"], "done")
        self.assertEqual(state["jobs"][job["id"]]["doneBytes"], len(self.payload))
        destination = self.asset_path(job)
        self.assertEqual(destination.read_bytes(), self.payload)
        self.assertFalse(destination.with_name(destination.name + ".part").exists())
        self.assertEqual(self.snapshot([self.active, self.settings, self.source / "setup.py"]), before)
        self.assertEqual(list(self.source.glob("strata-*.json")), [self.active])
        self.assertEqual(list(self.home.rglob("strata-*.json")), [])

    def test_downloaded_state_requires_current_manifest_and_unchanged_verified_assets(self):
        job=self.accepted_job()
        self.run_worker(job,self.payload)
        def state(): return next(r["state"] for r in self.call()["models"] if r["id"]==job["modelId"])
        self.assertEqual(state(),"downloaded")
        destination=self.asset_path(job)
        destination.write_bytes(b"X"*len(self.payload))
        self.assertEqual(state(),"available","A changed same-size asset cannot remain downloaded")
        destination.write_bytes(self.payload)
        self.run_worker(job,self.payload)  # Terminal jobs cannot be silently restarted.
        self.assertEqual(json.loads((self.home/"state.json").read_text())["jobs"][job["id"]]["state"],"done")
        self.payload=b"new pinned bytes"
        self.write_catalog()
        self.assertEqual(state(),"available","A newer pinned revision with the same model ID must be downloadable")

    def test_slow_blocked_read_is_cancelled_by_worker_self_exit_without_pid_signals(self):
        job=self.accepted_job()
        entered,released=threading.Event(),threading.Event()
        exits,errors=[],[]
        class SlowResponse(io.BytesIO):
            url="https://huggingface.co/demo/repo/resolve/"+"a"*40+"/Fixture-Q2_0-1.gguf"
            status,headers=200,{}
            def read(response,size=-1):
                entered.set()
                if not released.wait(5): raise AssertionError("Cancellation failed to bound the blocked read")
                return b""
        def cancel():
            try:
                if not entered.wait(2): raise AssertionError("Worker never entered the synthetic read")
                self.call(f"/downloads/{job['id']}/cancel","POST",{})
            except BaseException as error: errors.append(error); released.set()
        def self_exit(code):
            exits.append(code); released.set()
        canceller=threading.Thread(target=cancel,daemon=True); canceller.start()
        started=time.monotonic()
        with mock.patch.object(manager_module.urllib.request,"urlopen",return_value=SlowResponse()), \
                mock.patch.object(manager_module,"process_identity",return_value=("offline","offline worker")), \
                mock.patch.object(manager_module.signal,"signal"), \
                mock.patch.object(manager_module.os,"kill",side_effect=AssertionError("Never signal a stored PID")), \
                mock.patch.object(manager_module,"worker_exit",side_effect=self_exit):
            manager_module.worker(self.home,job["id"],job["workerToken"])
        canceller.join(1)
        self.assertEqual(errors,[])
        self.assertEqual(exits,[130])
        self.assertLess(time.monotonic()-started,4)
        self.assertEqual(json.loads((self.home/"state.json").read_text())["jobs"][job["id"]]["state"],"cancelled")

    def test_mac_process_identity_requests_untruncated_command(self):
        completed=mock.Mock(stdout="Tue Oct 20 10:04:13 2026 python worker --token "+"f"*64)
        with mock.patch.object(manager_module.sys,"platform","darwin"), \
                mock.patch.object(manager_module.subprocess,"run",return_value=completed) as run:
            identity=manager_module.process_identity(1234)
        self.assertIn("-ww",run.call_args.args[0])
        self.assertIn("f"*64,identity[1])

    def test_worker_hash_mismatch_discards_corrupt_complete_partial_without_publishing_asset(self):
        job = self.accepted_job()
        self.run_worker(job, b"X" * len(self.payload))
        state = json.loads((self.home / "state.json").read_text())
        self.assertEqual(state["jobs"][job["id"]]["state"], "error")
        destination = self.asset_path(job)
        self.assertFalse(destination.exists())
        self.assertFalse(destination.with_name(destination.name + ".part").exists())

    def test_worker_observes_owned_cancellation_and_preserves_partial(self):
        job = self.accepted_job()

        def cancel_during_read():
            with self.manager.locked():
                state = self.manager.load()
                state["jobs"][job["id"]]["cancelRequested"] = True
                self.manager.save(state)

        self.run_worker(job, self.payload[:8], on_read=cancel_during_read)
        state = json.loads((self.home / "state.json").read_text())
        self.assertEqual(state["jobs"][job["id"]]["state"], "cancelled")
        destination = self.asset_path(job)
        self.assertFalse(destination.exists())
        self.assertEqual(destination.with_name(destination.name + ".part").read_bytes(), self.payload[:8])

    def test_wrong_worker_token_cannot_download_or_change_owned_job(self):
        job = self.accepted_job()
        before = (self.home / "state.json").read_bytes()
        fetch = self.run_worker(job, self.payload, token="0" * 64)
        fetch.assert_not_called()
        self.assertEqual((self.home / "state.json").read_bytes(), before)
        self.assertFalse(self.asset_path(job).parent.exists())

    def test_retry_reuses_verified_completed_asset_without_fetch(self):
        first = self.accepted_job()
        self.run_worker(first, self.payload)
        before = self.snapshot([self.asset_path(first)])
        second = self.accepted_job()
        fetch = self.run_worker(second, b"must not be downloaded")
        fetch.assert_not_called()
        self.assertEqual(self.asset_path(second), self.asset_path(first))
        self.assertEqual(self.snapshot([self.asset_path(second)]), before)
        state = json.loads((self.home / "state.json").read_text())
        self.assertEqual(state["jobs"][second["id"]]["state"], "done")

    def test_worker_resumes_partial_only_with_matching_content_range(self):
        job = self.accepted_job()
        destination = self.asset_path(job)
        destination.parent.mkdir()
        partial = destination.with_name(destination.name + ".part")
        partial.write_bytes(self.payload[:8])
        fetch = self.run_worker(job, self.payload[8:], status=206,
                                headers={"Content-Range": f"bytes 8-{len(self.payload) - 1}/{len(self.payload)}"})
        self.assertEqual(fetch.call_args.args[0].get_header("Range"), "bytes=8-")
        self.assertEqual(destination.read_bytes(), self.payload)
        self.assertFalse(partial.exists())

    def test_invalid_range_response_cannot_publish_or_append_partial(self):
        job = self.accepted_job()
        destination = self.asset_path(job)
        destination.parent.mkdir()
        partial = destination.with_name(destination.name + ".part")
        partial.write_bytes(self.payload[:8])
        self.run_worker(job, self.payload[8:], status=206,
                        headers={"Content-Range": f"bytes 9-{len(self.payload) - 1}/{len(self.payload)}"})
        self.assertFalse(destination.exists())
        self.assertEqual(partial.read_bytes(), self.payload[:8])
        state = json.loads((self.home / "state.json").read_text())
        self.assertEqual(state["jobs"][job["id"]]["state"], "error")

    def test_insecure_final_redirect_cannot_publish_asset(self):
        job = self.accepted_job()
        self.run_worker(job, self.payload, response_url="http://example.invalid/model.gguf")
        self.assertFalse(self.asset_path(job).exists())
        state = json.loads((self.home / "state.json").read_text())
        self.assertEqual(state["jobs"][job["id"]]["state"], "error")


if __name__ == "__main__":
    unittest.main()
