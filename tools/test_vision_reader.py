"""The picture reader choice for an AMD engine folder (strata_control/vision_reader.py) and the server's fallback from the GPU
reader to the CPU reader (serve/server.py, Vision._start): hip present, hip missing, hip start failure, then cpu.

    python -m unittest tools.test_vision_reader
"""
from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from strata_control import vision_reader as VR
import serve.server as server


def folder(*names: str) -> Path:
    d = Path(tempfile.mkdtemp(prefix="vr-"))
    for n in names:
        (d / n).write_bytes(b"x")
    return d


class Choose(unittest.TestCase):
    def test_the_default_reader_by_card_memory_is_one_rule_for_every_make(self):
        self.assertEqual(VR.default_mode(12), "cpu")
        self.assertEqual(VR.default_mode(16), "cpu")
        self.assertEqual(VR.default_mode(23.5), "gpu")
        self.assertEqual(VR.default_mode(24), "gpu")
        self.assertEqual(VR.default_mode(32), "gpu")
        self.assertEqual(VR.default_mode(None), "gpu")                  # memory unknown: the GPU reader falls back by itself
        for asked, vram, want in ((None, 16, "cpu"), ("yes", 12, "cpu"), ("on", 32, "gpu"), ("gpu", 12, "gpu"), ("cpu", 32, "cpu"),
                                  ("no", 32, "none"), ("off", 12, "none")):
            self.assertEqual(VR.wanted(asked, vram), want, (asked, vram))

    def test_hip_present_gives_the_gpu_reader_with_the_cpu_one_to_fall_back_on(self):
        d = folder("strata-vision", "strata-vision-hip")
        for ask in ("yes", "on", "gpu"):
            r = VR.choose(d, ask, windows=False, gpu_arch="gfx1201", archs=["gfx1200", "gfx1201"])
            self.assertEqual((r.mode, r.exe.name, r.cpu_exe.name), ("gpu", "strata-vision-hip", "strata-vision"), ask)
            self.assertEqual(r.note, "")

    def test_hip_missing_gives_the_cpu_reader(self):
        d = folder("strata-vision")
        r = VR.choose(d, "yes", windows=False, gpu_arch="gfx1201")
        self.assertEqual((r.mode, r.exe.name, r.cpu_exe), ("cpu", "strata-vision", None))
        self.assertEqual(r.note, "")                               # "yes" never asked for the GPU in particular
        self.assertIn("no GPU picture reader", VR.choose(d, "gpu", windows=False).note)   # --vision gpu: said once

    def test_card_not_in_the_build_gives_the_cpu_reader(self):
        d = folder("strata-vision", "strata-vision-hip")
        r = VR.choose(d, "yes", windows=False, gpu_arch="gfx1100", archs=["gfx1200", "gfx1201"])
        self.assertEqual((r.mode, r.exe.name), ("cpu", "strata-vision"))
        self.assertIn("gfx1100", r.note)

    def test_unknown_card_tries_the_gpu_reader(self):
        d = folder("strata-vision", "strata-vision-hip")
        self.assertEqual(VR.choose(d, "yes", windows=False, gpu_arch="", archs=["gfx1201"]).mode, "gpu")
        self.assertEqual(VR.choose(d, "yes", windows=False, gpu_arch="gfx1100", archs=None).mode, "gpu")

    def test_gcn_arch_names_with_features_match(self):
        self.assertEqual(VR.gfx_of("gfx1201:sramecc-:xnack-"), "gfx1201")
        d = folder("strata-vision", "strata-vision-hip")
        self.assertEqual(VR.choose(d, "yes", windows=False, gpu_arch="gfx1151:sramecc+:xnack-", archs=["gfx1151"]).mode, "gpu")

    def test_cpu_asked_and_off(self):
        d = folder("strata-vision", "strata-vision-hip")
        self.assertEqual(VR.choose(d, "cpu", windows=False).exe.name, "strata-vision")
        for ask in ("no", "none", "off", None):
            self.assertEqual(VR.choose(d, ask, windows=False).mode, "none")

    def test_no_reader_at_all(self):
        d = folder("strata")
        for ask in ("yes", "cpu", "gpu"):
            r = VR.choose(d, ask, windows=False)
            self.assertEqual(r.mode, "none", ask)
            self.assertIn("no picture reader", r.note)

    def test_windows_names(self):
        d = folder("strata-vision.exe", "strata-vision-hip.exe")
        r = VR.choose(d, "yes", windows=True, gpu_arch="gfx1201", archs=["gfx1201"])
        self.assertEqual((r.mode, r.exe.name, r.cpu_exe.name), ("gpu", "strata-vision-hip.exe", "strata-vision.exe"))
        self.assertEqual(VR.choose(folder("strata-vision-hip"), "yes", windows=True).mode, "none")   # a Linux name on Windows

    def test_run_config_section(self):
        d = folder("strata-vision", "strata-vision-hip")
        base = {"mmproj": "m.gguf", "model": "t.gguf", "max_tokens": 1024}
        sec = VR.vision_section(VR.choose(d, "yes", windows=False), base)
        self.assertTrue(sec["gpu"])
        self.assertEqual(Path(sec["exe"]).name, "strata-vision-hip")
        self.assertEqual(Path(sec["cpu_exe"]).name, "strata-vision")
        cpu = VR.cpu_fallback(sec)
        self.assertFalse(cpu["gpu"])
        self.assertEqual(Path(cpu["exe"]).name, "strata-vision")
        self.assertNotIn("cpu_exe", cpu)
        self.assertEqual(cpu["mmproj"], "m.gguf")
        cpu_only = VR.vision_section(VR.choose(folder("strata-vision"), "yes", windows=False), base)
        self.assertFalse(cpu_only["gpu"])
        self.assertNotIn("cpu_exe", cpu_only)
        self.assertIsNone(VR.cpu_fallback(cpu_only))
        self.assertIsNone(VR.cpu_fallback({**sec, "cpu_exe": ""}))
        self.assertIsNone(VR.cpu_fallback({"exe": "strata-vision", "gpu": True}))   # an NVIDIA config has no fallback


class FakeProc:
    def __init__(self, first_line: str):
        self.stdin, self.stdout = io.StringIO(), io.StringIO(first_line)
        self.killed = False

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return None


def make_vision(cfg: dict) -> "server.Vision":
    with mock.patch.object(server.Vision, "work_dir", staticmethod(lambda: Path(tempfile.gettempdir()))):
        return server.Vision(cfg, lazy=True)


class Fallback(unittest.TestCase):
    CFG = {"exe": "/e/strata-vision-hip", "cpu_exe": "/e/strata-vision", "gpu": True, "mmproj": "/m/mm.gguf",
           "model": "/m/t.gguf", "max_tokens": 1024, "threads": 6}

    def run_start(self, answers: dict, cfg: dict | None = None):
        """answers: exe name -> its first stdout line, or an exception to raise from popen."""
        started = []

        def popen(what, args, **kw):
            name = Path(args[0]).name
            started.append(args)
            a = answers[name]
            if isinstance(a, Exception):
                raise a
            return FakeProc(a)

        v = make_vision(cfg or self.CFG)
        out = io.StringIO()
        with mock.patch.object(server, "popen", popen), mock.patch("builtins.print", lambda *a, **k: out.write(" ".join(map(str, a)) + "\n")):
            err = None
            try:
                v._start()
            except RuntimeError as e:
                err = e
        return v, started, out.getvalue(), err

    def test_hip_present_and_starts(self):
        v, started, log, err = self.run_start({"strata-vision-hip": "READY 2048\n"})
        self.assertIsNone(err)
        self.assertEqual([Path(a[0]).name for a in started], ["strata-vision-hip"])
        self.assertIn("--gpu", started[0])
        self.assertEqual(log, "")
        self.assertFalse(v.stopped)

    def test_hip_start_failure_falls_back_to_cpu_with_one_line(self):
        v, started, log, err = self.run_start({"strata-vision-hip": "ERR this GPU (gfx1100) is not in the picture reader's build (gfx1200,gfx1201)\n",
                                               "strata-vision": "READY 2048\n"})
        self.assertIsNone(err)
        self.assertEqual([Path(a[0]).name for a in started], ["strata-vision-hip", "strata-vision"])
        self.assertNotIn("--gpu", started[1])
        self.assertIn("--threads", started[1])
        self.assertEqual(log.count("\n"), 1)
        self.assertIn("gfx1100", log)
        self.assertIn("CPU", log)
        self.assertFalse(v.stopped)
        self.assertIsNone(v.fallback)

    def test_hip_process_that_dies_silently_falls_back(self):                  # a GPU runtime that aborts: stdout closes empty
        v, started, log, err = self.run_start({"strata-vision-hip": "", "strata-vision": "READY 2048\n"})
        self.assertIsNone(err)
        self.assertEqual(len(started), 2)
        self.assertEqual(log.count("\n"), 1)

    def test_hip_missing_file_falls_back(self):                               # popen raises (exe gone, not executable)
        v, started, log, err = self.run_start({"strata-vision-hip": FileNotFoundError("strata-vision-hip"),
                                               "strata-vision": "READY 2048\n"})
        self.assertIsNone(err)
        self.assertEqual(len(started), 2)

    def test_restart_after_a_fallback_stays_on_the_cpu_reader(self):
        v, started, log, err = self.run_start({"strata-vision-hip": "ERR nope\n", "strata-vision": "READY 2048\n"})
        started.clear()
        with mock.patch.object(server, "popen", lambda what, args, **kw: started.append(args) or FakeProc("READY 2048\n")):
            v.restart()
        self.assertEqual([Path(a[0]).name for a in started], ["strata-vision"])

    def test_cpu_failure_after_the_fallback_is_an_error(self):
        v, started, log, err = self.run_start({"strata-vision-hip": "ERR nope\n", "strata-vision": "ERR cannot load the vision encoder\n"})
        self.assertIsNotNone(err)
        self.assertIn("cannot load", str(err))

    def test_a_config_without_cpu_exe_has_no_fallback(self):                  # NVIDIA: the error is shown as before
        cfg = {k: v for k, v in self.CFG.items() if k != "cpu_exe"}
        v, started, log, err = self.run_start({"strata-vision-hip": "ERR nope\n"}, cfg)
        self.assertIsNotNone(err)
        self.assertEqual(len(started), 1)
        self.assertEqual(log, "")

    def test_cpu_config_is_unchanged(self):
        cfg = {"exe": "/e/strata-vision", "gpu": False, "mmproj": "/m/mm.gguf", "model": "/m/t.gguf"}
        v, started, log, err = self.run_start({"strata-vision": "READY 2048\n"}, cfg)
        self.assertIsNone(err)
        self.assertNotIn("--gpu", started[0])


if __name__ == "__main__":
    unittest.main()
