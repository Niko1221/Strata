"""serve/test_park_files.py - STRATA_PARK_DIR: the parked conversations saved at a graceful stop / unload (the
engine's PARKSAVE) and loaded at the next start (PARKLOAD), over a mock engine: no GPU, no pack.

    python -m unittest serve.test_park_files -v
"""
from __future__ import annotations

import os
import queue
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Service, StrataEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class FakeProc:
    """Answers PARKSAVE / PARKLOAD like `strata --serve` does; records every line it was sent."""

    def __init__(self, engine):
        self.engine = engine
        self.stdin = self
        self.sent: list[str] = []
        self.killed = False

    def write(self, s):
        self.sent.append(s)
        cmd, _, path = s.strip().partition(" ")
        if cmd == "PARKSAVE":
            Path(path, "park.manifest").write_text("strata-park 2\ntag 0\nstages 1\nend\n")
            self.engine.lines.put("SESSION 0 900\n")
            self.engine.lines.put("SAVED 2 900 12.5\n")
        elif cmd == "PARKLOAD":
            self.engine.lines.put("RESTORED 2 900 7.5\n")

    def flush(self):
        pass

    def close(self):
        pass

    def poll(self):
        return -9 if self.killed else None

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


class FakeEngine(StrataEngine):
    def __init__(self, park_files: bool = True):
        super().__init__("strata", ["--max-context", "65536"], lazy=True)
        self.lines = queue.Queue()
        self.ended = False
        self.proc = FakeProc(self)
        if park_files:
            self.info["park_files"] = 1

    def close(self):                    # the real close() sends STOP / QUIT and waits for the process
        self.proc.sent.append("CLOSE\n")


class ParkFiles(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def commands(self, eng):
        return [line.split()[0] for line in eng.proc.sent]

    def test_unset_nothing_is_sent(self):
        eng = FakeEngine()
        with mock.patch.object(StrataEngine, "PARK_DIR", ""):
            eng.unload()
            eng.park_from_disk()
        self.assertEqual(self.commands(eng), ["CLOSE"])

    def test_unload_saves_before_the_engine_closes(self):
        eng = FakeEngine()
        with mock.patch.object(StrataEngine, "PARK_DIR", self.dir.name):
            eng.unload()
        self.assertEqual(eng.proc.sent[0], f"PARKSAVE {self.dir.name}\n")
        self.assertEqual(self.commands(eng), ["PARKSAVE", "CLOSE"])

    def test_load_only_with_a_manifest(self):
        eng = FakeEngine()
        with mock.patch.object(StrataEngine, "PARK_DIR", self.dir.name):
            eng.park_from_disk()
            self.assertEqual(eng.proc.sent, [])                 # nothing saved yet
            eng.park_to_disk()
            eng.park_from_disk()
        self.assertEqual(self.commands(eng), ["PARKSAVE", "PARKLOAD"])

    def test_an_engine_without_the_commands_is_not_asked(self):
        eng = FakeEngine(park_files=False)
        with mock.patch.object(StrataEngine, "PARK_DIR", self.dir.name):
            Path(self.dir.name, "park.manifest").write_text("x")
            eng.park_from_disk()
            eng.unload()
        self.assertEqual(self.commands(eng), ["CLOSE"])

    def test_batch_slots_are_not_parked(self):
        eng = FakeEngine()
        eng.batch = 2
        with mock.patch.object(StrataEngine, "PARK_DIR", self.dir.name):
            eng.unload()
        self.assertEqual(self.commands(eng), ["CLOSE"])

    def test_a_refusal_does_not_stop_the_unload(self):
        eng = FakeEngine()

        def refuse(s):
            eng.proc.sent.append(s)
            eng.lines.put("SERR storage 0 not enough disk space\n")
        eng.proc.write = refuse
        with mock.patch.object(StrataEngine, "PARK_DIR", self.dir.name):
            eng.unload()
        self.assertEqual(self.commands(eng), ["PARKSAVE", "CLOSE"])
        self.assertTrue(eng.alive())                            # SERR: still in step

    def test_exit_saves_between_requests_only(self):
        eng = FakeEngine()
        svc = Service(eng, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
        with mock.patch.object(StrataEngine, "PARK_DIR", self.dir.name):
            held = threading.Event()
            done = threading.Event()

            def request():                                      # a request still holds the engine
                with svc.fifo:
                    held.set()
                    done.wait(10)
            t = threading.Thread(target=request)
            t.start()
            held.wait(5)
            svc.park_at_exit()
            self.assertEqual(eng.proc.sent, [])
            done.set()
            t.join(5)
            svc.park_at_exit()
        self.assertEqual(self.commands(eng), ["PARKSAVE"])


if __name__ == "__main__":
    unittest.main()
