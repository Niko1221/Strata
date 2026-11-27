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


REPO = Path(__file__).resolve().parents[2]


class InklingRule(unittest.TestCase):
    """Inkling-2-Mini follows the Qwen rule for pictures (owner 2026-11-26): vision_reader.inkling_reader / inkling_lines / set_reader_arg, and the numbers
    the engine keeps in engine/src/inkling2/ink_vision_policy.h."""

    def test_nvidia_and_amd_by_card_memory(self):
        for gb, want in ((8, "cpu"), (12, "cpu"), (16, "cpu"), (20, "cpu"), (23, "cpu"), (24, "gpu"), (32, "gpu"), (48, "gpu")):
            for make in ("nvidia", "amd", ""):
                self.assertEqual(VR.inkling_reader(None, gb, make), want, (gb, make))
                self.assertEqual(VR.inkling_reader("yes", gb, make), want, (gb, make))
        self.assertEqual(VR.inkling_reader(None, 23.5), "gpu")                 # a "24 GB" card reports 23.5 to 24 GiB
        self.assertEqual(VR.inkling_reader(None, 23.49), "cpu")

    def test_the_card_is_unknown(self):
        """Nothing is written then (None): the engine's own rule decides at start."""
        for v in (None, 0, 0.0):
            self.assertIsNone(VR.inkling_reader(None, v))
            self.assertIsNone(VR.inkling_reader("yes", v, "amd"))
        self.assertEqual(VR.inkling_reader("gpu", None), "gpu")               # asked: kept without a reading
        self.assertEqual(VR.inkling_reader("no", None), "none")

    def test_forced_modes_win_on_every_card(self):
        for gb in (8, 12, 16, 24, 32):
            self.assertEqual(VR.inkling_reader("gpu", gb), "gpu")
            self.assertEqual(VR.inkling_reader("cpu", gb), "cpu")
            self.assertEqual(VR.inkling_reader("no", gb), "none")
            self.assertEqual(VR.inkling_reader("off", gb, "amd"), "none")
        self.assertEqual(VR.inkling_reader("gpu", 8, mac=True), "gpu")
        self.assertEqual(VR.inkling_reader("cpu", 64, mac=True), "cpu")

    def test_a_mac_reads_on_the_gpu_whatever_its_memory(self):
        for gb in (None, 0, 8, 16, 32, 128):
            self.assertEqual(VR.inkling_reader(None, gb, mac=True), "gpu", gb)
            self.assertEqual(VR.inkling_reader("yes", gb, "apple", mac=True), "gpu", gb)

    def test_an_intel_arc_defaults_to_the_cpu_and_gpu_is_asked_for(self):
        for gb in (None, 0, 8, 16, 24, 32):
            self.assertEqual(VR.inkling_reader(None, gb, "intel"), "cpu", gb)
            self.assertEqual(VR.inkling_reader("gpu", gb, "intel"), "gpu", gb)
        with mock.patch.object(VR, "INTEL_GPU_READER_BY_DEFAULT", True):
            self.assertEqual(VR.inkling_reader(None, 32, "intel"), "gpu")
            self.assertEqual(VR.inkling_reader(None, 8, "intel"), "cpu")

    def test_machines_whose_gpu_shares_the_ram_use_the_pool_they_report(self):
        """Strix Halo and DGX Spark: setup passes the shared pool as the card's memory, as it does for the Qwen path: far above 24 GB, so the GPU."""
        self.assertEqual(VR.inkling_reader(None, 96, "amd"), "gpu")
        self.assertEqual(VR.inkling_reader(None, 121, "nvidia"), "gpu")
        self.assertEqual(VR.inkling_reader(None, 16, "amd"), "cpu")           # a small BIOS carve-out is read like any small card

    def test_the_screen_lines(self):
        def lines(asked, mode, gb, make="", mac=False, backend="cuda"):
            return VR.inkling_lines(asked, mode, gb, make, mac=mac, backend=backend)
        self.assertEqual(lines(None, "none", 12), [(False, "images: off (no picture reader is installed)")])
        (warn, text), = lines(None, "cpu", 12)
        self.assertFalse(warn)
        self.assertIn("on the CPU", text)
        self.assertIn("12 GB card", text)
        self.assertIn("--vision gpu", text)
        (warn, text), = lines(None, "gpu", 32)
        self.assertFalse(warn)
        self.assertIn("on the GPU", text)
        self.assertIn("2 minutes without one", text)                          # NVIDIA: the on-demand reader
        (warn, text), = lines(None, "gpu", 32, backend="hip")
        self.assertNotIn("without one", text)
        (warn, text), = lines("gpu", "gpu", 12)                               # asked for on a small card: kept, with a warning
        self.assertTrue(warn)
        self.assertIn("kept as you chose", text)
        self.assertIn("--vision cpu", text)
        (warn, text), = lines("gpu", "gpu", 32)
        self.assertFalse(warn)
        (warn, text), = lines("cpu", "cpu", 32)
        self.assertIn("as you asked", text)
        (warn, text), = lines(None, "cpu", None, "intel", backend="sycl")
        self.assertIn("Intel Arc", text)
        (warn, text), = lines(None, "gpu", None, mac=True, backend="metal")
        self.assertIn("Mac's GPU", text)
        (warn, text), = lines("cpu", "cpu", None, mac=True, backend="metal")
        self.assertEqual(text, VR.MAC_CPU_NOTE)
        (warn, text), = lines(None, None, None)
        self.assertIn("by the card's memory", text)

    def test_the_flag_in_an_argument_list(self):
        a = ["--native", "m", "--vision", "v"]
        self.assertIsNone(VR.reader_arg(a))
        self.assertEqual(VR.set_reader_arg(a, "cpu"), a + ["--vision-reader", "cpu"])
        self.assertEqual(VR.reader_arg(VR.set_reader_arg(a, "cpu")), "cpu")
        self.assertEqual(VR.set_reader_arg(a + ["--vision-reader", "cpu", "--kv", "fp8"], "gpu"), a + ["--kv", "fp8", "--vision-reader", "gpu"])
        self.assertEqual(VR.set_reader_arg(a + ["--vision-reader", "cpu"], None), a)
        self.assertEqual(VR.set_reader_arg(a + ["--vision-reader", "cpu"], "none"), a)
        self.assertEqual(VR.reader_arg(a + ["--vision-reader", "AUTO"]), "auto")

    @unittest.skipUnless((REPO / "engine" / "src" / "inkling2" / "ink_vision_policy.h").is_file(), "ink_vision_policy.h is in the engine branch (f-ink-vision)")
    def test_the_numbers_are_the_engines(self):
        """engine/src/inkling2/ink_vision_policy.h holds the same rule: this keeps the two files from drifting apart."""
        import re
        text = (REPO / "engine" / "src" / "inkling2" / "ink_vision_policy.h").read_text(encoding="utf-8")
        gib = float(re.search(r"kVisionGpuMinGiB\s*=\s*([0-9.]+)", text).group(1))
        idle = int(re.search(r"kVisionIdleSecondsDefault\s*=\s*(\d+)", text).group(1))
        intel = re.search(r"kVisionIntelGpuByDefault\s*=\s*(true|false)", text).group(1) == "true"
        self.assertEqual(gib, VR.GPU_MIN_VRAM_GB)
        self.assertEqual(idle, VR.ELASTIC_IDLE_S_DEFAULT)
        self.assertEqual(intel, VR.INTEL_GPU_READER_BY_DEFAULT)


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

    def test_an_intel_arc_keeps_the_cpu_reader_by_default_and_gpu_is_asked_for(self):
        """The one exception to the 24 GB rule: the SYCL reader has run on a 8 GB A750 only (INTEL_GPU_READER_BY_DEFAULT)."""
        self.assertFalse(VR.INTEL_GPU_READER_BY_DEFAULT)
        for vram in (8, 16, 24, 32):
            self.assertEqual(VR.default_mode(vram, "intel"), "cpu")
            self.assertEqual(VR.wanted("yes", vram, "intel"), "cpu")
            self.assertEqual(VR.wanted("gpu", vram, "intel"), "gpu")
        self.assertEqual(VR.wanted("yes", 32), "gpu")                        # other makes: unchanged
        with mock.patch.object(VR, "INTEL_GPU_READER_BY_DEFAULT", True):
            self.assertEqual(VR.default_mode(32, "intel"), "gpu")
            self.assertEqual(VR.default_mode(8, "intel"), "cpu")              # with the exception off, the 24 GB rule is the same as for every make

    def test_the_sycl_reader_is_the_gpu_reader_of_an_intel_folder(self):
        d = folder("strata-vision", "strata-vision-sycl")
        r = VR.choose(d, "gpu", windows=False)
        self.assertEqual((r.mode, r.exe.name, r.cpu_exe.name), ("gpu", "strata-vision-sycl", "strata-vision"))
        self.assertEqual(VR.vision_section(r, {"mmproj": "m"})["gpu"], True)
        r = VR.choose(folder("strata-vision"), "gpu", windows=False)         # no SYCL reader built: the CPU one, said
        self.assertEqual((r.mode, r.exe.name), ("cpu", "strata-vision"))
        self.assertIn("no GPU picture reader", r.note)
        self.assertEqual(VR.choose(d, "cpu", windows=False).exe.name, "strata-vision")

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
        # an on-demand reader (vision.elastic) always has one: the NVIDIA program without --gpu, with the same picture tokens
        nv = VR.cpu_fallback({"exe": "/e/strata-vision", "gpu": True, "max_tokens": 1024, "elastic": True}, same_program=True)
        self.assertEqual((nv["exe"], nv["gpu"], nv["max_tokens"], "elastic" in nv), ("/e/strata-vision", False, 1024, False))
        self.assertIsNotNone(VR.cpu_fallback({"exe": "C:/e/strata-vision.exe", "gpu": True}, same_program=True))
        self.assertIsNone(VR.cpu_fallback({"exe": "/e/strata-vision-hip", "gpu": True}, same_program=True))   # AMD names its own
        self.assertIsNone(VR.cpu_fallback({"exe": "/e/strata-vision", "gpu": False}, same_program=True))


class MacReader(unittest.TestCase):
    """A Mac has one picture program, strata-vision with Metal built in: --gpu reads on the GPU, without it on the CPU."""

    def test_a_mac_reads_on_the_gpu_by_default_whatever_its_memory(self):
        for ram in (16, 32, 64, 128, None):
            self.assertEqual(VR.default_mode(ram, unified=True), "gpu", ram)
            self.assertEqual(VR.wanted(None, ram, unified=True), "gpu", ram)
            self.assertEqual(VR.wanted("yes", ram, unified=True), "gpu", ram)
        self.assertEqual(VR.wanted("cpu", 32, unified=True), "cpu")             # asked for: kept
        self.assertEqual(VR.wanted("off", 32, unified=True), "none")
        self.assertEqual(VR.default_mode(16), "cpu")                           # a 16 GB card of a PC still follows the 24 GB rule

    def test_the_one_program_is_both_readers_and_its_own_fallback(self):
        d = folder("strata", "strata-vision")
        for ask in ("yes", "on", "gpu"):
            r = VR.choose(d, ask, windows=False, mac=True)
            self.assertEqual((r.mode, r.exe.name, r.cpu_exe.name, r.note), ("gpu", "strata-vision", "strata-vision", ""), ask)
        sec = VR.vision_section(VR.choose(d, "yes", windows=False, mac=True), {"mmproj": "m.gguf", "model": "t.gguf", "max_tokens": 1024})
        self.assertTrue(sec["gpu"])
        self.assertEqual(sec["exe"], sec["cpu_exe"])
        fallback = VR.cpu_fallback(sec)
        self.assertFalse(fallback["gpu"])
        self.assertEqual(fallback["exe"], sec["exe"])

    def test_cpu_asked_is_kept_with_the_time_it_takes(self):
        r = VR.choose(folder("strata-vision"), "cpu", windows=False, mac=True)
        self.assertEqual((r.mode, r.exe.name, r.cpu_exe), ("cpu", "strata-vision", None))
        self.assertIn("50 seconds", r.note)
        self.assertFalse(VR.vision_section(r, {})["gpu"])

    def test_an_engine_without_the_reader_gets_pictures_off(self):
        for ask in ("yes", "gpu", "cpu"):
            r = VR.choose(folder("strata"), ask, windows=False, mac=True)
            self.assertEqual(r.mode, "none", ask)
            self.assertIn("no picture reader", r.note)
        self.assertEqual(VR.choose(folder("strata-vision"), "off", windows=False, mac=True).mode, "none")

    def test_a_hip_program_is_not_a_mac_reader(self):
        r = VR.choose(folder("strata-vision-hip"), "yes", windows=False, mac=True)
        self.assertEqual(r.mode, "none")


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

    def test_the_ready_line_names_the_reader_and_older_readers_still_start(self):
        """strata-vision says which reader runs ("READY <n> own" / "READY <n> mtmd"); an older one says "READY <n>" and starts as before."""
        cpu = {"exe": "/e/strata-vision", "gpu": False, "mmproj": "/m/mm.gguf", "model": "/m/t.gguf"}
        for line, reader in (("READY 2560 own\n", "own"), ("READY 2560 mtmd\n", "mtmd"), ("READY 2048\n", None)):
            v, started, log, err = self.run_start({"strata-vision": line}, cpu)
            self.assertIsNone(err, line)
            self.assertEqual(v.reader, reader, line)
            self.assertEqual(log, "", line)                              # nothing said when the chosen reader started

    def test_an_older_readers_fallback_word_starts_without_a_line(self):
        """0.2.16: strata-vision has no llama.cpp reader to fall back on, and the server no longer prints a line for the "fallback" word an older
        reader (0.2.13-0.2.15) put after READY; the reader still starts."""
        cpu = {"exe": "/e/strata-vision", "gpu": False, "mmproj": "/m/mm.gguf", "model": "/m/t.gguf"}
        v, started, log, err = self.run_start({"strata-vision": "READY 2560 mtmd fallback\n"}, cpu)
        self.assertIsNone(err)
        self.assertEqual((v.reader, len(started)), ("mtmd", 1))           # the same process: no second start
        self.assertNotIn("llama.cpp", log)

    def test_an_old_run_configs_reader_entry_still_starts(self):
        """The removed "Picture reader" setting wrote STRATA_VISION_TOWER=mtmd into a model's run config "env".  A config that still carries it keeps
        working: the server hands the env to the reader as before (no reader argument changes), and strata-vision 0.2.16 only logs that it no longer
        reads the name."""
        cfg = {"env": {"STRATA_VISION_TOWER": "mtmd"}, "vision": {"exe": "/e/strata-vision", "mmproj": "/m/mm.gguf", "model": "/m/t.gguf"}}
        env = server.vision_env(cfg, server.child_env(cfg))
        self.assertEqual(env.get("STRATA_VISION_TOWER"), "mtmd")
        self.assertNotIn("--tower", server.Vision.reader_args(cfg["vision"]))
        with mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("STRATA_VISION_TOWER", None)
            self.assertNotIn("STRATA_VISION_TOWER", server.vision_env({}, server.child_env({})))   # a default file carries nothing

    def test_cpu_config_is_unchanged(self):
        cfg = {"exe": "/e/strata-vision", "gpu": False, "mmproj": "/m/mm.gguf", "model": "/m/t.gguf"}
        v, started, log, err = self.run_start({"strata-vision": "READY 2048\n"}, cfg)
        self.assertIsNone(err)
        self.assertNotIn("--gpu", started[0])


if __name__ == "__main__":
    unittest.main()
