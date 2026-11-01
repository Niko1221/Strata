"""The DFlash2 option: where the engine can run it, the window it is started with, and the download when there is no internet or the files are already here.
(tests/terminal/test_drafting.py has the table, the Terminal flag and the pinned download.)"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from strata_control import drafting, drafting_fetch
from tests.terminal.launch_support import run_cli

ROOT = Path(__file__).resolve().parents[2]
GYRO = "qwen3.8-flash-next-gyro-gyro-s"


def serving_dflash2():
    return mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", True)


class WhereTheEngineCanRunIt(unittest.TestCase):
    """DFlash2 exists in the CUDA and HIP engines only (CMakeLists.txt STRATA_HAVE_DFLASH2): the choice is offered, and applied, only there."""

    def test_the_matrix(self):
        h = drafting.hardware_support
        for backend in ("metal", "sycl", "", None, "cpu"):
            self.assertEqual(h(backend, ["sm_120"]), drafting.Support(False, "none", ""), backend)
        self.assertEqual((h("cuda", ["sm_120"]).level, h("cuda", ["sm_121"]).level, h("hip", ["gfx1151"]).level), ("tested",) * 3)
        self.assertEqual((h("cuda", ["sm_86"]).level, h("hip", ["gfx1201"]).level), ("slower",) * 2)
        for backend, arch in (("cuda", "sm_75"), ("cuda", "sm_89"), ("cuda", "sm_90"), ("cuda", "sm_61"), ("hip", "gfx1100"), ("hip", "gfx1030"), ("cuda", None)):
            support = h(backend, [arch] if arch else [])
            self.assertEqual((support.can_run, support.level, support.line), (True, "untested", drafting.LINE_UNTESTED), (backend, arch))
        for backend, arch in (("cuda", "sm_120"), ("cuda", "sm_86"), ("cuda", "sm_75")):
            self.assertTrue(h(backend, [arch]).line, "every card that is offered it gets one plain line")

    def test_no_verdicts_in_the_lines(self):
        for line in (drafting.LINE_TESTED, drafting.LINE_SLOWER, drafting.LINE_UNTESTED):
            for word in ("recommend", "best", "better", "should", "avoid"):
                self.assertNotIn(word, line.lower())

    def test_a_build_without_the_code_keeps_the_mtp_and_downloads_nothing(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with serving_dflash2():
            for backend in ("metal", "sycl"):
                d = drafting.decide("dflash2", klass="other", model_name=GYRO, data_dir=tmp.name, backend=backend, archs=[])
                self.assertEqual((d.drafter, d.note), ("mtp", drafting.NOT_THIS_CARD), backend)
                self.assertFalse(drafting.needs_pack("dflash2", klass="other", model_name=GYRO, data_dir=tmp.name, backend=backend, archs=[]), backend)
            self.assertTrue(drafting.needs_pack("dflash2", klass="blackwell", model_name=GYRO, data_dir=tmp.name, backend="cuda", archs=["sm_120"]))
            self.assertEqual(drafting.decide("auto", klass="other", model_name=GYRO, data_dir=tmp.name, backend="metal", archs=[]).note, "", "Auto never nags")

    def test_a_window_or_confidence_the_person_set_by_hand_stays(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pack = drafting.pack_path(tmp.name)
        pack.parent.mkdir(parents=True)
        pack.write_bytes(b"x" * (2 << 20))
        base = ["--pack", "/d/packs/g", "--mtp", "/d/mtp/rt", "--spec", "5", "--draft-conf", "0.5"]
        with serving_dflash2():
            c = {"model_name": GYRO, "drafting": "dflash2", "args": list(base), "user_args": ["--spec", "5", "--draft-conf", "0.5"]}
            drafting.apply(c, klass="blackwell", data_dir=tmp.name)
            self.assertEqual(c["args"][c["args"].index("--spec") + 1], "5")
            self.assertEqual(c["args"][c["args"].index("--draft-conf") + 1], "0.5")
            self.assertEqual((c["args"].count("--spec"), c["args"].count("--draft-conf")), (1, 1))
            c = {"model_name": GYRO, "drafting": "mtp", "args": [*base, "--draft", "dflash2", "--draft-model", "/p"]}
            drafting.apply(c, klass="blackwell", data_dir=tmp.name)
            self.assertNotIn("--draft-conf", c["args"], "back to MTP: DFlash2's confidence goes with it")
            self.assertEqual(c["args"][c["args"].index("--spec") + 1], "5", "and the MTP window is untouched")

    def test_the_plan_reports_whether_this_computer_can_run_it(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for hardware, serves, level in ((("cuda", ["sm_120"]), True, "tested"), (("metal", []), False, "none"), (("hip", ["gfx1201"]), True, "slower")):
            with mock.patch.object(drafting, "detect_hardware", return_value=hardware):
                code, out, err = run_cli(["drafting", "plan", "--data-dir", tmp.name, "--mode", "auto", "--output", "json"])
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertEqual((payload["serves"], payload["support"]), (serves, level), hardware)


class OfflineAndReuse(unittest.TestCase):
    """No internet is a plain sentence in seconds, never a hang; files that are already here are used, not fetched."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "tools"))
        spec = importlib.util.spec_from_file_location("t_pack2", ROOT / "tools" / "test_dflash2_pack.py")
        cls.synth = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.synth)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.synth.write_safetensors(self.dir / "m.safetensors", self.synth.tensors())
        (self.dir / "c.json").write_text(json.dumps(self.synth.CFG))
        self.blobs = {"config.json": (self.dir / "c.json").read_bytes(), "model.safetensors": (self.dir / "m.safetensors").read_bytes()}
        self.files = tuple((n, len(b), hashlib.sha256(b).hexdigest()) for n, b in self.blobs.items())
        self.data = self.dir / "data"
        patcher = mock.patch.object(drafting, "pack_ready", lambda d: bool(d) and drafting.pack_path(d).is_file())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_no_internet_is_a_plain_sentence_and_nothing_is_left_behind(self):
        with mock.patch.object(drafting_fetch, "online", return_value=False), mock.patch.object(drafting_fetch.transfer, "download") as download:
            with self.assertRaises(drafting_fetch.DraftingError) as ctx:
                drafting_fetch.ensure(self.data, files=self.files)
        self.assertEqual(str(ctx.exception), "There is no internet connection, so the DFlash2 drafter can't be downloaded now.")
        download.assert_not_called()
        self.assertFalse(drafting.pack_path(self.data).exists())

    def test_the_reachability_check_gives_up_fast_on_a_connection_that_hangs(self):
        hung = socket.socket()
        hung.bind(("127.0.0.1", 0))
        hung.listen(0)                                        # the kernel accepts the connection; nobody ever answers
        self.addCleanup(hung.close)
        started = time.monotonic()
        self.assertFalse(drafting_fetch.online(f"http://127.0.0.1:{hung.getsockname()[1]}/x", timeout=1.0))
        self.assertLess(time.monotonic() - started, 5)
        self.assertFalse(drafting_fetch.online("http://127.0.0.1:9/x", timeout=1.0), "nothing listens: refused at once")

    def test_verified_source_files_that_are_already_here_are_used_without_the_network(self):
        src = drafting.pack_path(self.data).parent / "src"
        src.mkdir(parents=True)
        for name, blob in self.blobs.items():
            (src / name).write_bytes(blob)
        with mock.patch.object(drafting_fetch, "online", side_effect=AssertionError("no network needed")), \
                mock.patch.object(drafting_fetch.transfer, "download", side_effect=AssertionError("nothing to download")):
            out = drafting_fetch.ensure(self.data, files=self.files)
        self.assertEqual(out.read_bytes()[:4], b"DF2P")
        self.assertFalse(src.exists())

    def test_a_copy_kept_by_the_hugging_face_tools_is_found_checked_and_never_deleted(self):
        hub = self.dir / "hub"
        hub.mkdir()
        for name, blob in self.blobs.items():
            (hub / name).write_bytes(blob)
        self.assertEqual(set(drafting_fetch._present([hub], self.files)), {"config.json", "model.safetensors"})
        (hub / "model.safetensors").write_bytes(b"corrupt")
        self.assertNotIn("model.safetensors", drafting_fetch._present([hub], self.files), "a copy whose hash differs is never used")

    def test_the_hub_folders_are_the_pinned_revision_only(self):
        with mock.patch.dict("os.environ", {"HF_HOME": str(self.dir / "hf")}):
            folders = drafting_fetch._hub_snapshots()
        self.assertIn(self.dir / "hf" / "hub" / "models--incoai--Qwen3.8-Flash-Next-DFlash2" / "snapshots" / drafting.REVISION, folders)


class OneRequestAtATime(unittest.TestCase):
    """DFlash2 keeps the MTP when the engine runs request slots (--batch); Strata fills slots in by itself, so choosing DFlash2 means one request."""

    def test_only_a_named_choice_on_a_computer_that_can_run_it(self):
        want = drafting.wants_one_request
        with serving_dflash2():
            self.assertTrue(want("dflash2", model_name=GYRO, backend="cuda", archs=["sm_121"]))
            self.assertTrue(want("DFlash2", family="qwen"))
            for mode in ("auto", "mtp", "off", None):
                self.assertFalse(want(mode, model_name=GYRO, backend="cuda", archs=["sm_121"]), mode)
            self.assertFalse(want("dflash2", model_name="qwen3.8-flash-next-coder-iq1_m", backend="cuda", archs=["sm_121"]))
            self.assertFalse(want("dflash2", model_name=GYRO, backend="metal", archs=[]))
        with mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", False):
            self.assertFalse(want("dflash2", model_name=GYRO, backend="cuda", archs=["sm_121"]), "while the engine does not serve it")


class TheQ4Pack(unittest.TestCase):
    """The owner chose the Q4_0 pack (522 MB, same speed as Q8, less memory): made on the person's computer from the pinned download, checked, and an old Q8 pack keeps working."""

    def test_the_pins(self):
        self.assertEqual(drafting.PACK_NAME, "qwen3.8-flash-next-dflash2.q4_0.pack")
        self.assertEqual(drafting.PACK_BYTES, 522458112)
        self.assertEqual(len(drafting.PACK_SHA256), 64)
        self.assertEqual(drafting.PACK_GB, 0.52)
        self.assertEqual(drafting.LEGACY_PACK_NAME, "qwen3.8-flash-next-dflash2.q8.pack")

    def test_an_old_q8_pack_is_used_and_never_fetched_again(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        folder = Path(tmp.name) / drafting.FOLDER
        folder.mkdir()
        old = folder / drafting.LEGACY_PACK_NAME
        old.write_bytes(b"x" * (2 << 20))
        self.assertEqual(drafting.pack_path(tmp.name), old)
        self.assertTrue(drafting.pack_ready(tmp.name))
        with serving_dflash2():
            self.assertFalse(drafting.needs_pack("dflash2", klass="blackwell", model_name=GYRO, data_dir=tmp.name))
            self.assertEqual(drafting.decide("dflash2", klass="blackwell", model_name=GYRO, data_dir=tmp.name).flags[3], str(old))
        with mock.patch.object(drafting_fetch.transfer, "download", side_effect=AssertionError("no download for a pack that is here")):
            self.assertEqual(drafting_fetch.ensure(tmp.name), old)
        self.assertEqual(drafting.target_path(tmp.name), folder / drafting.PACK_NAME, "a new fetch would write the Q4_0 name")
        (folder / drafting.PACK_NAME).write_bytes(b"y" * (2 << 20))
        self.assertEqual(drafting.pack_path(tmp.name), folder / drafting.PACK_NAME, "when both are there, the Q4_0 pack is the one used")

    def test_the_pack_is_made_q4_and_checked_against_its_pins(self):
        sys.path.insert(0, str(ROOT / "tools"))
        spec = importlib.util.spec_from_file_location("t_pack3", ROOT / "tools" / "test_dflash2_pack.py")
        synth = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(synth)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name)
        synth.write_safetensors(d / "m.safetensors", synth.tensors())
        (d / "c.json").write_text(json.dumps(synth.CFG))
        files = tuple((n, (d / f).stat().st_size, hashlib.sha256((d / f).read_bytes()).hexdigest()) for n, f in (("config.json", "c.json"), ("model.safetensors", "m.safetensors")))
        src = d / "data" / drafting.FOLDER / "src"
        src.mkdir(parents=True)
        (src / "config.json").write_bytes((d / "c.json").read_bytes())
        (src / "model.safetensors").write_bytes((d / "m.safetensors").read_bytes())
        # not the real pinned files (files=...): the pack is made Q4_0 but the real pins do not apply
        with mock.patch.object(drafting, "pack_ready", lambda x: False):
            out = drafting_fetch.ensure(d / "data", files=files)
        blob = out.read_bytes()
        self.assertEqual(out.name, drafting.PACK_NAME)
        self.assertIn(b"q4_0", blob[:4096])
        # with the real list (files=None) a pack whose bytes are not the pinned ones is deleted and the words are plain
        (d / "data2").mkdir()
        with mock.patch.object(drafting_fetch, "_present", lambda folders, spec: {"config.json": d / "c.json", "model.safetensors": d / "m.safetensors"}),                 mock.patch.object(drafting, "pack_ready", lambda x: False):
            with self.assertRaises(drafting_fetch.DraftingError) as ctx:
                drafting_fetch.ensure(d / "data2")
        self.assertIn("did not pass the safety check", str(ctx.exception))
        self.assertFalse(drafting.target_path(d / "data2").exists())
        self.assertFalse(drafting.target_path(d / "data2").with_suffix(".tmp").exists())
        # and with the pins set to what this pack really is, it is kept
        pins = {"PACK_BYTES": len(blob), "PACK_SHA256": hashlib.sha256(blob).hexdigest()}
        with mock.patch.object(drafting_fetch, "_present", lambda folders, spec: {"config.json": d / "c.json", "model.safetensors": d / "m.safetensors"}),                 mock.patch.object(drafting, "pack_ready", lambda x: False), mock.patch.multiple(drafting, **pins):
            self.assertEqual(drafting_fetch.ensure(d / "data2").read_bytes(), blob)


class TheTerminalStep(unittest.TestCase):
    """strata launch: the Drafting step exists only where there is something to choose, like the DeepSeek-only Speed step."""

    def flow(self, family, hardware):
        from types import SimpleNamespace
        from strata_cli.launch.screens import flow_for
        ctx = SimpleNamespace(computer=SimpleNamespace(usable_gpus=[0]), settings=SimpleNamespace(family=family))
        with mock.patch.object(drafting, "detect_hardware", return_value=hardware), serving_dflash2(),                 mock.patch("strata_cli.launch.screens.speed_choice_available", return_value=False):
            return flow_for(ctx)

    def test_only_flash_next_on_a_cuda_or_hip_engine_gets_the_step(self):
        flow = self.flow("qwen", ("cuda", ["sm_120"]))
        self.assertIn("drafting", flow)
        self.assertEqual(flow.index("drafting"), flow.index("pictures") + 1)
        self.assertIn("drafting", self.flow("gyro", ("hip", ["gfx1151"])))
        for family in ("deepseek41", "swift", "coder", "unsloth", "nvfp4", "qwen36", None):
            self.assertNotIn("drafting", self.flow(family, ("cuda", ["sm_120"])), family)
        for hardware in (("metal", []), ("sycl", []), (None, [])):
            self.assertNotIn("drafting", self.flow("qwen", hardware), hardware)

    def test_the_choice_is_saved_as_the_flag_would_and_auto_writes_nothing(self):
        from types import SimpleNamespace
        from strata_cli.launch import settings as S
        from strata_cli.launch.screens import DraftingScreen
        self.assertEqual([k for k, _ in DraftingScreen.CHOICES], list(S.DRAFTING_MODES))
        for key, saved in (("dflash2", "dflash2"), ("auto", None), ("off", "off")):
            fake = SimpleNamespace(ctx=SimpleNamespace(settings=S.LaunchSettings()), menu=lambda key=key: SimpleNamespace(current=SimpleNamespace(key=key)))
            DraftingScreen.commit(fake)
            self.assertEqual(fake.ctx.settings.drafting, saved)

    def test_the_words_are_the_apps_words(self):
        folder = ROOT / "apps" / "host" / "src" / "locales"
        en = json.loads((folder / "df2-en.json").read_text(encoding="utf-8"))
        self.assertEqual(en["df01"], drafting.EXPLAIN_NUMBERS)
        self.assertEqual(en["df02"], drafting.EXPLAIN_LIMITS)
        self.assertEqual((en["df03"], en["df04"], en["df05"]), (drafting.LINE_TESTED, drafting.LINE_SLOWER, drafting.LINE_UNTESTED))
        self.assertEqual(en["df06"], drafting.mtp_line(drafting.NOT_THIS_CARD))
        self.assertEqual(en["df07"], "There is no internet connection, so the DFlash2 drafter can't be downloaded now.")
        self.assertEqual(en["df09"], drafting.ONE_REQUEST_NOTE)


if __name__ == "__main__":
    unittest.main()
