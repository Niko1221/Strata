"""serve/test_park_ramlike.py - STRATA_PARK_RAMLIKE=1: the keeper (serve/park_keeper.py) holding named sections after
the process that made them has gone, and the server's side of it over a mock engine.  No GPU, no engine.

    python -m unittest serve.test_park_ramlike -v
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import park_keeper  # noqa: E402
from serve.server import StrataEngine  # noqa: E402
from serve.test_park_files import FakeEngine  # noqa: E402

MAKER = r"""
import sys
sys.path.insert(0, sys.argv[1])
from serve import park_keeper as K
names = sys.argv[3:]
held = []
for i, n in enumerate(names):
    s = K.Section(n, 3 << 20)
    s.write(bytes([i + 1]) * (3 << 20))
    held.append(s)
print("MADE", flush=True)
sys.stdin.readline()          # the test starts the keeper meanwhile; then this process exits, unmapping its views
"""


def write_manifest(folder: str, convs: list[str], stages: int = 2) -> None:
    with open(os.path.join(folder, park_keeper.MANIFEST), "w", encoding="ascii") as f:
        f.write(f"strata-park 2\ntag 0123456789abcdef\nstages {stages}\n")
        for c in convs:
            f.write(f"conv {c} 60000 {int(time.time())}\n")
        f.write("end\n")


class Manifest(unittest.TestCase):
    def test_names(self):
        with tempfile.TemporaryDirectory() as d:
            write_manifest(d, ["ca", "cb"])
            names, head = park_keeper.manifest_sections(d)
        self.assertEqual(head, {"tag": "0123456789abcdef", "stages": 2})
        self.assertEqual(names, [f"Local\\strata-park-{c}-s{s}" for c in ("ca", "cb") for s in (0, 1)])

    def test_malformed_or_a_path_lists_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            write_manifest(d, ["..\\x"])
            self.assertEqual(park_keeper.manifest_sections(d), ([], {}))
            Path(d, park_keeper.MANIFEST).write_text("strata-park 2\ntag x\nstages 1\nconv a 1 1\n")   # no end
            self.assertEqual(park_keeper.manifest_sections(d), ([], {}))


@unittest.skipUnless(os.name == "nt", "named sections are Windows only")
class Keeper(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="strata-park-ramlike-test-")
        stamp = f"{os.getpid()}-{time.time_ns()}"
        self.convs = [f"ctest{stamp}-{k}" for k in ("a", "b")]
        write_manifest(self.dir, self.convs)

    def tearDown(self):
        park_keeper.release(self.dir)

    def test_sections_outlive_their_maker(self):
        names, _ = park_keeper.manifest_sections(self.dir)
        root = str(Path(__file__).resolve().parents[1])
        maker = subprocess.Popen([sys.executable, "-c", MAKER, root, "-"] + names, stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, text=True)
        self.assertEqual(maker.stdout.readline().strip(), "MADE")
        doc = park_keeper.spawn(self.dir, 600, timeout_s=60)
        maker.stdin.write("\n")
        maker.stdin.flush()
        maker.wait(30)
        self.assertEqual((doc["sections"], doc["missing"]), (4, []))
        self.assertGreaterEqual(doc["bytes"], 4 * (3 << 20))
        for i, n in enumerate(names):                      # the maker is gone; the keeper still holds every section
            s = park_keeper.Section(n)
            self.assertEqual(s.read(), bytes([i + 1]) * (3 << 20))
            s.close()
        self.assertIsNotNone(park_keeper.status(self.dir))
        park_keeper.release(self.dir)
        end = time.monotonic() + 15
        while time.monotonic() < end and park_keeper.pid_alive(doc["pid"]):
            time.sleep(0.2)
        self.assertFalse(park_keeper.pid_alive(doc["pid"]), "released: the keeper ends")
        with self.assertRaises(OSError):
            park_keeper.Section(names[0])                  # gone with the last handle

    def test_newer_manifest_ends_the_keeper(self):
        names, _ = park_keeper.manifest_sections(self.dir)
        made = [park_keeper.Section(n, 1 << 20) for n in names]
        doc = park_keeper.spawn(self.dir, 600, timeout_s=60)
        for s in made:
            s.close()
        os.utime(os.path.join(self.dir, park_keeper.MANIFEST), ns=(time.time_ns(), time.time_ns() + 10**9))
        end = time.monotonic() + 15
        while time.monotonic() < end and park_keeper.pid_alive(doc["pid"]):
            time.sleep(0.2)
        self.assertFalse(park_keeper.pid_alive(doc["pid"]))

    def test_spawn_gives_up_within_its_whole_wait(self):
        names, _ = park_keeper.manifest_sections(self.dir)
        made = [park_keeper.Section(n, 1 << 20) for n in names]
        t0 = time.monotonic()
        with self.assertRaises(RuntimeError) as cm:
            park_keeper.spawn(self.dir, 600, timeout_s=4, python=os.environ.get("COMSPEC", "cmd.exe"))   # never a keeper
        for s in made:
            s.close()
        self.assertLess(time.monotonic() - t0, 12)
        self.assertIn("no keeper ready", str(cm.exception))
        self.assertIn("keeper.json", str(cm.exception))                 # and says what the folder shows

    def test_touch_stops_at_its_deadline_and_goes_on_later(self):
        s = park_keeper.Section(f"Local\\strata-park-touch-{os.getpid()}-{time.time_ns()}", 3 << 20)
        s.write(b"x" * (3 << 20))
        seen = []
        with mock.patch.object(park_keeper, "TOUCH_CHUNK", 1 << 20), \
                mock.patch.object(s, "_touch_range", lambda a, b: seen.append((a, b))):
            self.assertFalse(s.touch(until=time.monotonic() - 1))       # no time: nothing touched, nowhere to resume
            self.assertEqual(seen, [])
            calls = {"n": 0}
            real = time.monotonic

            def clock():                                                # time runs out after the second step
                calls["n"] += 1
                return real() + (100 if calls["n"] > 2 else 0)
            with mock.patch.object(park_keeper.time, "monotonic", clock):
                self.assertFalse(s.touch(until=real() + 50))
            self.assertEqual(len(seen), 2)
            self.assertTrue(s.touch())                                  # the rest from where it stopped
            self.assertEqual(seen[2][0], seen[1][1])
            self.assertEqual(seen[-1][1], 64 + (3 << 20))               # up to the end of the data (head included)
        s.close()

    def test_no_keeper_no_status(self):
        self.assertIsNone(park_keeper.status(self.dir))


class Server(unittest.TestCase):
    """The server's side: which engine is asked, and what a start without a keeper does."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        p1 = mock.patch.object(StrataEngine, "PARK_DIR", self.dir.name)
        p2 = mock.patch.object(StrataEngine, "PARK_RAMLIKE", True)
        p1.start()
        p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)

    def test_an_engine_without_sections_is_not_asked(self):
        eng = FakeEngine()                                 # park_files=1: files only
        eng.unload()
        self.assertEqual([x.split()[0] for x in eng.proc.sent], ["CLOSE"])

    def test_no_keeper_at_start_drops_the_manifest_and_loads_nothing(self):
        eng = FakeEngine()
        eng.info["park_files"] = 2
        write_manifest(self.dir.name, ["ca"])
        with mock.patch.object(park_keeper, "status", return_value=None):
            eng.park_from_disk()
        self.assertEqual(eng.proc.sent, [])
        self.assertFalse(os.path.exists(os.path.join(self.dir.name, park_keeper.MANIFEST)))

    def test_a_live_keeper_is_loaded_then_released(self):
        eng = FakeEngine()
        eng.info["park_files"] = 2
        with mock.patch.object(park_keeper, "status", return_value={"pid": 1, "ready": True}), \
                mock.patch.object(park_keeper, "release") as release:
            eng.park_from_disk()
        self.assertEqual(eng.proc.sent, [f"PARKLOAD {self.dir.name}\n"])
        release.assert_called_once_with(self.dir.name)

    def test_the_save_hands_the_sections_to_a_keeper_before_the_close(self):
        eng = FakeEngine()
        eng.info["park_files"] = 2
        order = []
        with mock.patch.object(park_keeper, "spawn", side_effect=lambda *a, **k: order.append("KEEPER") or
                               {"pid": 1, "sections": 1, "bytes": 1, "missing": []}):
            real = eng.close
            eng.close = lambda: (order.append("CLOSE"), real())
            eng.unload()
        self.assertEqual(eng.proc.sent[0].split()[0], "PARKSAVE")
        self.assertEqual(order, ["KEEPER", "CLOSE"])


if __name__ == "__main__":
    unittest.main()
