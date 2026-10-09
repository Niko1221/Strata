"""Tests for the remote-stage node agent (tools/stage_node.py: the file store, the model names, the clocks), the
shipper's node paths (tools/stage_ship.py) and the tuner's /start body and notes (tools/stage_tune.py): no GPU, no
network, no engine.

    python -m unittest tools.test_stage_node
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import stage_node as sn  # noqa: E402
import stage_ship as ss  # noqa: E402
import stage_tune as st  # noqa: E402


def on_disk(p: Path) -> int:
    """The bytes a file occupies on the drive (a sparse file's holes take none)."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        f = ctypes.windll.kernel32.GetCompressedFileSizeW
        f.restype = wintypes.DWORD
        hi = wintypes.DWORD()
        lo = f(str(p), ctypes.byref(hi))
        return (hi.value << 32) | lo
    return os.stat(p).st_blocks * 512


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.st = sn.Store({"data_dir": str(self.d), "drop_cache": False})

    def tearDown(self):
        self.tmp.cleanup()

    def put(self, rel, off, total, data, fp, mode=None, st=None):
        return (st or self.st).put(rel, off, total, io.BytesIO(data), len(data), mode, fp)

    def test_ranges_merge_and_fingerprint(self):
        self.assertTrue(self.put("models/m/a.gguf", 0, 100, b"x" * 10, "fp1")["ok"])
        self.put("models/m/a.gguf", 50, 100, b"y" * 10, "fp1")
        rg = self.st.ranges("models/m/a.gguf")
        self.assertEqual(rg["ranges"], [[0, 10], [50, 10]])
        self.assertEqual(rg["fp"], "fp1")
        self.put("models/m/a.gguf", 10, 100, b"z" * 5, "fp1")
        self.assertEqual(self.st.ranges("models/m/a.gguf")["ranges"], [[0, 15], [50, 10]])

    def test_another_source_voids_what_was_written(self):
        self.put("models/m/a.gguf", 0, 100, b"x" * 10, "fp1")
        self.put("models/m/a.gguf", 20, 100, b"z" * 5, "fp2")   # same size, another fingerprint
        rg = self.st.ranges("models/m/a.gguf")
        self.assertEqual((rg["ranges"], rg["fp"]), ([[20, 5]], "fp2"))
        data = (self.d / "models/m/a.gguf").read_bytes()
        self.assertEqual(data[:10], b"\0" * 10)
        self.assertEqual(data[20:25], b"z" * 5)
        self.put("models/m/a.gguf", 0, 200, b"w" * 4, "fp2")    # another size
        self.assertEqual(self.st.ranges("models/m/a.gguf")["ranges"], [[0, 4]])
        self.assertEqual((self.d / "models/m/a.gguf").stat().st_size, 200)

    def test_legacy_sidecar_is_kept(self):
        (self.d / "models/m").mkdir(parents=True)
        (self.d / "models/m/b.gguf").write_bytes(b"\0" * 64)
        (self.d / "models/m/b.gguf.ranges.json").write_text(json.dumps([[0, 32]]))
        self.put("models/m/b.gguf", 32, 64, b"q" * 8, "fpb")
        rg = self.st.ranges("models/m/b.gguf")
        self.assertEqual((rg["ranges"], rg["fp"]), ([[0, 40]], "fpb"))

    def test_executables_need_allow_put_bin(self):
        self.assertFalse(self.put("bin/strata", 0, 4, b"abcd", None, mode="755")["ok"])
        self.assertFalse(self.put("bin/strata", 0, 4, b"abcd", None)["ok"])
        st2 = sn.Store({"data_dir": str(self.d), "drop_cache": False, "allow_put_bin": True})
        self.assertTrue(self.put("bin/strata", 0, 4, b"abcd", None, mode="755", st=st2)["ok"])

    def test_path_outside_data_dir(self):
        with self.assertRaises(ValueError):
            self.put("../evil", 0, 1, b"e", None)

    def test_new_file_is_sparse(self):
        total, n = 2 << 30, 1 << 20
        self.assertTrue(self.put("models/m/big.gguf", total - n, total, b"x" * n, "fp")["ok"])
        p = self.d / "models/m/big.gguf"
        self.assertEqual(p.stat().st_size, total)
        self.assertLess(on_disk(p), 64 << 20, "the gap before the written range was allocated")
        with open(p, "rb") as f:
            f.seek(total - n - 4)
            self.assertEqual(f.read(8), b"\0\0\0\0xxxx")


class StartCheckTest(unittest.TestCase):
    def test_refused_before_a_process_starts(self):
        w = sn.Workers({"exe": "false", "pack": "p", "native": "n", "ple_gguf": "q", "expert_profile": "e", "bind": "x"})
        self.assertIn("not allowed", w.start({"begin": 40, "extra": ["--expert-profile-save", "x"]})["error"])
        self.assertIn("both end and next", w.start({"begin": 24, "end": 40})["error"])


class ModelTest(unittest.TestCase):
    """/start's "model": a model shipped under data_dir, by plain names."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        m = self.d / "models" / "q4"
        m.mkdir(parents=True)
        for i in (1, 2, 3):
            (m / f"M-0000{i}-of-00003.gguf").write_bytes(b"")
        (self.d / "packs" / "pk").mkdir(parents=True)
        (self.d / "data").mkdir()
        (self.d / "data" / "prof.bin").write_bytes(b"")
        self.w = sn.Workers({"exe": "x", "bind": "x", "data_dir": str(self.d)})

    def tearDown(self):
        self.tmp.cleanup()

    def test_shipped_model(self):
        d = self.d
        m = self.w.model({"model": {"dir": "q4", "native": "M-00001-of-00003.gguf", "pack": "pk",
                                    "profile": "prof.bin"}})
        self.assertEqual(Path(m["native"]), d / "models" / "q4" / "M-00001-of-00003.gguf")
        self.assertEqual(m["ple_gguf"], "")   # not named: no --ple-gguf (a worker never runs the PLE layer)
        self.assertEqual(Path(m["pack"]), d / "packs" / "pk")
        self.assertEqual(Path(m["expert_profile"]), d / "data" / "prof.bin")
        m = self.w.model({"model": {"dir": "q4", "native": "M-00001-of-00003.gguf", "pack": "pk",
                                    "profile": "prof.bin", "ple": "M-00002-of-00003.gguf"}})
        self.assertEqual(Path(m["ple_gguf"]), d / "models" / "q4" / "M-00002-of-00003.gguf")

    def test_names_only(self):
        # "C:.." and "C:x": a drive on Windows would replace data_dir as the root; "..." is "." to Windows; NUL and
        # COM1 are devices in any directory.  The message: refused by name, not because the file is missing.
        for bad in ("../q4", "q4/x", "..", ".", "...", "", "a\\b", "C:..", "C:x", "q4 ", "~", "%TEMP%", "NUL",
                    "com1.txt", "a:b"):
            with self.assertRaisesRegex(ValueError, "plain file or directory name", msg=bad):
                self.w.model({"model": {"dir": bad, "native": "M-00001-of-00003.gguf", "pack": "pk"}})
            with self.assertRaisesRegex(ValueError, "plain file or directory name", msg=bad):
                self.w.model({"model": {"dir": "q4", "native": "M-00001-of-00003.gguf", "pack": "pk",
                                        "profile": bad}})
        for good in ("q4.1", "UD-Q4_K_XL", "nul2", "console"):   # names that only look like the above
            with self.assertRaisesRegex(ValueError, "not shipped here", msg=good):
                self.w.model({"model": {"dir": good, "native": "M-00001-of-00003.gguf", "pack": "pk"}})

    def test_not_shipped(self):
        with self.assertRaisesRegex(ValueError, "not shipped here"):
            self.w.model({"model": {"dir": "q4", "native": "M-00001-of-00003.gguf", "pack": "other",
                                    "profile": "prof.bin"}})

    def test_no_model_anywhere(self):
        with self.assertRaisesRegex(ValueError, "must name a model"):
            self.w.model({})
        self.assertIn("must name a model", self.w.start({"begin": 40})["error"])

    def test_node_json_model(self):
        w = sn.Workers({"exe": "x", "bind": "x", "pack": "p", "native": "n", "expert_profile": "e"})
        self.assertEqual(w.model({}), {"pack": "p", "native": "n", "ple_gguf": "", "expert_profile": "e"})

    def test_no_clocks_asked(self):
        self.assertEqual(self.w.lock_clocks(), "")


class FakeProc:
    def __init__(self):
        self.code = None

    def poll(self):
        return self.code


class ClocksTest(unittest.TestCase):
    """gpu_clocks: locked while a worker runs, reset when none does (nvidia-smi faked)."""

    def setUp(self):
        self.calls = []
        self.rc = 0
        run = lambda args, **kw: (self.calls.append(args[1:]),   # noqa: E731
                                  subprocess.CompletedProcess(args, self.rc, "", "Insufficient Permissions"))[1]
        self.patches = [mock.patch.object(sn.shutil, "which", return_value="nvidia-smi"),
                        mock.patch.object(sn.subprocess, "run", side_effect=run)]
        for p in self.patches:
            p.start()
        self.w = sn.Workers({"exe": "x", "bind": "x",
                             "gpu_clocks": {"graphics": [1500, 1905], "memory": [7001, 7001], "gpu": 0}})

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_lock_then_reset_when_none_runs(self):
        a, b = FakeProc(), FakeProc()
        self.w.procs.update({7841: a, 7842: b})
        note = self.w.lock_clocks()
        self.assertEqual(note, "graphics 1500-1905 MHz: locked; memory 7001-7001 MHz: locked")
        self.assertEqual(self.calls, [["-i", "0", "-lgc", "1500,1905"], ["-i", "0", "-lmc", "7001,7001"]])
        self.calls.clear()
        self.w.reap()                      # both run: nothing reset
        self.assertEqual(self.calls, [])
        a.code = 0
        self.w.reap()                      # one exited on its own, the other runs: still nothing
        self.assertEqual(self.calls, [])
        b.code = 0
        self.w.reap()                      # none runs: reset once
        self.assertEqual(self.calls, [["-i", "0", "-rgc"], ["-i", "0", "-rmc"]])
        self.w.reap()
        self.assertEqual(len(self.calls), 2)

    def test_not_locked_says_so(self):
        self.rc = 1
        self.w.procs[7841] = FakeProc()
        self.assertIn("NOT locked (Insufficient Permissions; the agent needs administrator / root)",
                      self.w.lock_clocks())

    def test_worker_gone_before_the_lock(self):
        p = FakeProc()
        p.code = 1                         # exited while the agent was about to lock: no lock left behind
        self.w.procs[7841] = p
        self.assertEqual(self.w.lock_clocks(), "")
        self.assertEqual(self.calls, [])
        self.assertFalse(self.w.clocks_locked)

    def test_close_stops_a_start_under_way(self):
        self.w.close()
        killed = []
        fake = mock.Mock(**{"kill.side_effect": lambda: killed.append(1), "wait.return_value": 0})
        with tempfile.TemporaryDirectory() as logs, mock.patch.object(sn.subprocess, "Popen", return_value=fake), \
                mock.patch.object(sn.Workers, "model", return_value={"pack": "p", "native": "n", "ple_gguf": "",
                                                                      "expert_profile": "e"}):
            self.w.cfg.update({"_token": "t", "log_dir": logs})
            r = self.w.start({"begin": 40, "port": 7899})
        self.assertEqual(r["error"], "the agent is stopping")
        self.assertEqual(killed, [1])
        self.assertEqual(self.w.procs, {})
        self.assertEqual(self.calls, [])   # no clock lock for it


class TunerPiecesTest(unittest.TestCase):
    """stage_tune's /start body and its notes, on a Tuner built without a cluster."""

    def tuner(self, nodes):
        t = st.Tuner.__new__(st.Tuner)
        t.nodes = [st.Node(n, "tok") for n in nodes]
        t.max_context = 131072
        t.ship_cfg = {"model_dir": "/data/models/q4", "pack_dir": "/data/packs/pk", "profile": "/opt/prof.bin"}
        t.main = mock.Mock(base={"args": ["--native", "/data/models/q4/M-00001-of-00003.gguf"]})
        t.clock_notes = {}
        return t

    def test_start_req(self):
        t = self.tuner([{"name": "b", "agent": "http://192.0.2.11:7840", "host": "198.51.100.11", "port": 7841,
                         "ship": True, "extra": ["--stage-bind", "198.51.100.11"]},
                        {"name": "c", "agent": "http://192.0.2.12:7840", "host": "192.0.2.12", "cache": 1000}])
        r = t.start_req(0, 24, 40, 5888)
        self.assertEqual(r["model"], {"dir": "q4", "pack": "pk", "native": "M-00001-of-00003.gguf",
                                      "profile": "prof.bin"})
        self.assertEqual((r["end"], r["next"]), (40, "192.0.2.12:7841"))
        self.assertEqual(r["extra"], ["--stage-bind", "198.51.100.11", "--max-context", "131072"])
        r = t.start_req(1, 40, 48, 5888)
        self.assertNotIn("model", r)       # not shipped: the node's node.json names the model
        self.assertNotIn("next", r)
        self.assertEqual(r["cache"], 1000)
        self.assertEqual(t.nodes[0].stage_agent, "http://198.51.100.11:7840")

    def test_notes(self):
        t = self.tuner([{"name": "w", "agent": "http://192.0.2.12:7840", "host": "192.0.2.12"},
                        {"name": "l", "agent": "http://192.0.2.11:7840", "host": "192.0.2.11"}])
        t.nodes[0].info = {"system": "Windows-11"}
        t.nodes[1].info = {"system": "Linux-6.14"}
        self.assertEqual(len(t.node_notes()), 1)
        t.nodes[0].info["gpu_clocks"] = {"graphics": [1500, 1905]}
        self.assertEqual(t.node_notes(), [])
        t.clock_notes["w"] = "graphics 1500-1905 MHz: NOT locked (...)"
        self.assertIn("not applied", t.node_notes()[0])


class NodePathsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.m = Path(self.tmp.name, "rvn")
        self.m.mkdir()
        for i in (1, 2):
            (self.m / f"M-0000{i}-of-00002.gguf").write_bytes(b"")

    def tearDown(self):
        self.tmp.cleanup()

    def test_posix_node(self):
        p = ss.node_paths("/srv/stage/", self.m, Path("packs/pk"), Path("data/prof.bin"))
        self.assertEqual(p["exe"], "/srv/stage/bin/strata")
        self.assertEqual(p["native"], "/srv/stage/models/rvn/M-00001-of-00002.gguf")
        self.assertEqual(p["ple_gguf"], "/srv/stage/models/rvn/M-00002-of-00002.gguf")

    def test_windows_node(self):
        p = ss.node_paths("D:\\stage", self.m, Path("packs/pk"), Path("data/prof.bin"))
        self.assertEqual(p["exe"], "D:\\stage\\bin\\strata.exe")
        self.assertEqual(p["pack"], "D:\\stage\\packs\\pk")
        self.assertEqual(p["expert_profile"], "D:\\stage\\data\\prof.bin")


if __name__ == "__main__":
    unittest.main()
