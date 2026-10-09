"""Tests for the remote-stage node agent (tools/stage_node.py: the file store, the model names, the clocks), the
shipper's node paths (tools/stage_ship.py) and the tuner's /start body and notes (tools/stage_tune.py): no GPU, no
network, no engine.

    python -m unittest tools.test_stage_node
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
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

    def test_concurrent_puts_to_one_file(self):
        # the shipper's pieces in flight: one file, other offsets, at once
        n, k = 256 << 10, 8
        total = n * k + 1000
        datas = [os.urandom(n) for _ in range(k)]
        go = threading.Barrier(k)
        out = [None] * k

        def put(i):
            go.wait()
            out[i] = self.put("models/m/c.gguf", i * n, total, datas[i], "fpc")

        ts = [threading.Thread(target=put, args=(i,)) for i in reversed(range(k))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        for i in range(k):
            self.assertEqual(out[i]["sha256"], hashlib.sha256(datas[i]).hexdigest())
        self.assertEqual((self.d / "models/m/c.gguf").read_bytes(), b"".join(datas) + b"\0" * 1000)
        rg = self.st.ranges("models/m/c.gguf")
        self.assertEqual((rg["ranges"], rg["fp"], rg["size"]), ([[0, n * k]], "fpc", total))

    def test_put_streams_the_body(self):
        # read in pieces of at most 4 MiB (not the whole body at once), hashed on the way; a short body is not
        # recorded (what arrived of it is written, the next ship sends it again)
        class Trickle(io.BytesIO):
            def __init__(self, data):
                super().__init__(data)
                self.most = 0

            def read(self, size=-1):
                self.most = max(self.most, size)
                return super().read(min(size, 100_000))

        data = os.urandom((9 << 20) + 123)
        body = Trickle(data)
        r = self.st.put("models/m/s.gguf", 5, len(data) + 5, body, len(data), None, "fps")
        self.assertEqual(r["sha256"], hashlib.sha256(data).hexdigest())
        self.assertLessEqual(body.most, 4 << 20)
        self.assertEqual((self.d / "models/m/s.gguf").read_bytes()[5:], data)
        r = self.st.put("models/m/s.gguf", 5, len(data) + 5, io.BytesIO(data[:10]), 20, None, "fps")
        self.assertIn("10 bytes short", r["error"])
        # the 10 bytes it wrote are no longer counted as shipped (what was there is gone); the rest stays recorded
        self.assertEqual(self.st.ranges("models/m/s.gguf")["ranges"], [[15, len(data) - 10]])

    def test_damaged_retry_drops_the_recorded_range(self):
        # a piece recorded, its answer lost, the retry damaged on the way: the damaged bytes are on disk now, so the
        # range is no longer recorded and a resume sends it again
        good = os.urandom(1000)
        self.assertTrue(self.st.put("models/m/r.gguf", 100, 3000, io.BytesIO(good), 1000, None, "fpr",
                                    hashlib.sha256(good).hexdigest())["ok"])
        self.put("models/m/r.gguf", 2000, 3000, b"k" * 500, "fpr")
        bad = bytes([good[0] ^ 1]) + good[1:]
        r = self.st.put("models/m/r.gguf", 100, 3000, io.BytesIO(bad), 1000, None, "fpr",
                        hashlib.sha256(good).hexdigest())
        self.assertIn("damaged", r["error"])
        self.assertEqual(self.st.ranges("models/m/r.gguf")["ranges"], [[2000, 500]])

    def test_started_anew_meanwhile_is_not_recorded(self):
        # a put of one source still writing when a ship of another source starts the file anew: its bytes are not
        # recorded in the new source's sidecar
        st = self.st

        class Late(io.BytesIO):
            def __init__(self, data):
                super().__init__(data)
                self.once = True

            def read(self, size=-1):
                if self.once:
                    self.once = False
                    assert st.put("models/m/n.gguf", 0, 4000, io.BytesIO(b"n" * 10), 10, None, "fp-new")["ok"]
                return super().read(size)

        self.put("models/m/n.gguf", 0, 4000, b"o" * 10, "fp-old")
        r = st.put("models/m/n.gguf", 1000, 4000, Late(b"o" * 100), 100, None, "fp-old")
        self.assertIn("started anew", r["error"])
        rg = st.ranges("models/m/n.gguf")
        self.assertEqual((rg["ranges"], rg["fp"]), ([[0, 10]], "fp-new"))

    def test_cut_range(self):
        rs = [[0, 10], [20, 10], [40, 10]]
        self.assertEqual(sn.cut_range(rs, 5, 20), [[0, 5], [25, 5], [40, 10]])
        self.assertEqual(sn.cut_range(rs, 20, 10), [[0, 10], [40, 10]])
        self.assertEqual(sn.cut_range(rs, 42, 2), [[0, 10], [20, 10], [40, 2], [44, 6]])
        self.assertEqual(sn.cut_range(rs, 100, 5), rs)

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


class ShipTest(unittest.TestCase):
    """stage_ship's pipeline against an agent in this process on 127.0.0.1 (small pieces, so many are in flight)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = Path(self.tmp.name, "src")
        self.dst = Path(self.tmp.name, "dst")
        self.src.mkdir()
        self.piece = mock.patch.object(ss, "PIECE", 64 << 10)
        self.piece.start()
        self.servers = []

    def tearDown(self):
        for srv in self.servers:
            srv.shutdown()
            srv.server_close()
        self.piece.stop()
        self.tmp.cleanup()

    def agent(self, http10=False, idle=600) -> tuple[ss.Agent, set]:
        cfg = {"_token": "tok", "bind": "127.0.0.1", "data_dir": str(self.dst), "drop_cache": False}
        base = sn.make_handler(cfg, sn.Workers(cfg), sn.Store(cfg))
        conns = set()   # the client ends the /puts came from

        class H(base):
            protocol_version = "HTTP/1.0" if http10 else base.protocol_version   # an agent from before keep-alive
            timeout = idle

            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                conns.add(self.client_address)
                super().do_POST()

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.servers.append(srv)
        return ss.Agent(f"http://127.0.0.1:{srv.server_address[1]}", "tok"), conns

    def items(self) -> list[ss.Item]:
        a = self.src / "a.gguf"
        a.write_bytes(os.urandom((64 << 10) * 20 + 777))
        b = self.src / "b.bin"
        b.write_bytes(os.urandom(300_000))
        sparse = ss.Item(a, "models/m/a.gguf", [[0, 1000], [200_000, 500_000], [(64 << 10) * 20, 777]])
        sparse.fp = "fpa"
        return [sparse, ss.Item(b, "packs/p/b.bin")]

    def check(self, items):
        for it in items:
            got = (self.dst / it.rel).read_bytes()
            want = it.local.read_bytes()
            self.assertEqual(len(got), len(want))
            for off, n in it.ranges:
                self.assertEqual(got[off:off + n], want[off:off + n], it.rel)
            self.assertEqual(sn.Store({"data_dir": str(self.dst)}).ranges(it.rel)["ranges"], it.ranges)

    def test_pipeline_and_resume(self):
        agent, conns = self.agent()
        items = self.items()
        lines = []
        sent = ss.ship(agent, items, say=lines.append, parallel=3)
        self.assertEqual(sent, sum(it.nbytes for it in items))
        self.check(items)
        self.assertLessEqual(len(conns), 3, "the /puts did not keep their connections")
        self.assertEqual(len([s for s in lines if s.startswith("  ")]), 4)   # one line a range, as before
        self.assertEqual(ss.ship(agent, items, say=lines.append), 0)        # all there: nothing again

    def test_empty_file(self):
        agent, _ = self.agent()
        e = self.src / "empty.txt"
        e.write_bytes(b"")
        it = ss.Item(e, "packs/p/empty.txt")
        self.assertEqual(ss.ship(agent, [it], say=lambda m: None), 0)
        self.assertEqual((self.dst / "packs/p/empty.txt").stat().st_size, 0)   # created (it never was before)
        puts = []
        put = ss.Agent.put
        with mock.patch.object(ss.Agent, "put", lambda s, *a: (puts.append(1), put(s, *a))[1]):
            ss.ship(agent, [it], say=lambda m: None)
        self.assertEqual(puts, [])   # there now: not sent again

    def test_no_proxy(self):
        # an HTTP proxy in the environment (unreachable here) is not used for the agent: urllib takes the proxies
        # when an opener is built, so a default opener built now would use it - the agent's opener has none
        agent, _ = self.agent()
        items = self.items()
        def proxies(op):
            return [h.proxies for h in op.handlers if isinstance(h, urllib.request.ProxyHandler)]
        with mock.patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9",
                                          "no_proxy": "", "NO_PROXY": ""}):
            self.assertIn("http", proxies(urllib.request.build_opener())[0])
            self.assertFalse(any(proxies(ss.Agent.direct)))   # (an empty ProxyHandler is not even installed)
            self.assertEqual(ss.ship(agent, items, say=lambda m: None), sum(it.nbytes for it in items))
        self.check(items)

    def test_old_agent(self):
        agent, conns = self.agent(http10=True)
        items = self.items()
        self.assertEqual(ss.ship(agent, items, say=lambda m: None, parallel=4), sum(it.nbytes for it in items))
        self.check(items)
        self.assertGreater(len(conns), 4)   # a connection a request

    def test_old_shipper(self):
        # urllib, one request a connection, as the shipper before keep-alive put
        agent, _ = self.agent()
        req = urllib.request.Request(agent.url + "/put?path=data/x.bin&offset=2&total=6", data=b"abcd", method="POST",
                                     headers={"X-Stage-Token": "tok"})
        with urllib.request.urlopen(req, timeout=30) as r:
            self.assertEqual(json.load(r)["sha256"], hashlib.sha256(b"abcd").hexdigest())
        self.assertEqual((self.dst / "data/x.bin").read_bytes(), b"\0\0abcd")

    def test_kept_connection_closed_meanwhile(self):
        # the agent closed an idle kept-alive connection: the next piece goes once more on a new one
        agent, conns = self.agent(idle=0.3)
        it = self.items()[1]
        conn = agent.connect()
        data = it.local.read_bytes()
        self.assertTrue(agent.put(conn, it, 0, data[:1000])["ok"])
        time.sleep(1)
        r = agent.put(conn, it, 1000, data[1000:2000])
        conn.close()
        self.assertEqual(r.get("sha256"), hashlib.sha256(data[1000:2000]).hexdigest())
        self.assertEqual(len(conns), 2)

    def test_failing_piece_stops_the_run(self):
        agent, _ = self.agent()
        items = self.items()
        before = threading.active_count()
        real = sn.Store.put
        bad = 200_000 + 3 * (64 << 10)

        def put(store, rel, offset, *a):
            r = real(store, rel, offset, *a)
            return dict(r, sha256="0" * 64) if offset == bad else r

        with mock.patch.object(sn.Store, "put", put):
            with self.assertRaisesRegex(SystemExit, f"models/m/a.gguf at {bad}: sha256 differs"):
                ss.ship(agent, items, say=lambda m: None, parallel=3)
        time.sleep(0.2)   # (the agent's handler threads see their connections closed)
        self.assertLessEqual(threading.active_count(), before, "the shipper's threads did not end")
        with mock.patch.object(sn.Store, "put", lambda *a: {"ok": False, "error": "disk full"}):
            with self.assertRaisesRegex(SystemExit, "HTTP 400: .*disk full"):
                ss.ship(agent, items, say=lambda m: None, parallel=2)
        sent = ss.ship(agent, items, say=lambda m: None)   # resumed: the rest only
        self.assertLess(sent, sum(it.nbytes for it in items))
        self.check(items)

    def test_damaged_piece_is_not_recorded(self):
        # a piece whose bytes change on the way: the agent (given its sha256) writes but does not record it, the run
        # stops, and the next ship sends that piece again - a damaged piece is never taken for one that arrived
        agent, _ = self.agent()
        items = self.items()
        real = sn.Store.put
        bad = 200_000 + 3 * (64 << 10)

        class Flip:
            def __init__(self, body):
                self.body, self.done = body, False

            def read(self, n):
                b = self.body.read(n)
                if b and not self.done:
                    self.done = True
                    return bytes([b[0] ^ 1]) + b[1:]
                return b

        def put(store, rel, offset, total, body, *a):
            return real(store, rel, offset, total, Flip(body) if offset == bad else body, *a)

        with mock.patch.object(sn.Store, "put", put):
            with self.assertRaisesRegex(SystemExit, f"models/m/a.gguf at {bad}: .*damaged"):
                ss.ship(agent, items, say=lambda m: None, parallel=3)
        recorded = sn.Store({"data_dir": str(self.dst)}).ranges("models/m/a.gguf")["ranges"]
        self.assertFalse(any(o <= bad < o + n for o, n in recorded), "the damaged piece was recorded")
        ss.ship(agent, items, say=lambda m: None)   # resumed: the damaged piece again
        self.check(items)

    def test_pieces_in_flight_at_once(self):
        agent, _ = self.agent()
        items = self.items()
        real = sn.Store.put
        lock = threading.Lock()
        live = [0, 0]   # in flight now, most at once

        def put(*a):
            with lock:
                live[0] += 1
                live[1] = max(live[1], live[0])
            time.sleep(0.05)
            try:
                return real(*a)
            finally:
                with lock:
                    live[0] -= 1

        with mock.patch.object(sn.Store, "put", put):
            ss.ship(agent, items, say=lambda m: None, parallel=4)
        self.check(items)
        self.assertGreaterEqual(live[1], 3, "the pieces did not overlap")


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
