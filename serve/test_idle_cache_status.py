import io
import queue
import unittest
from types import SimpleNamespace

from serve.server import StrataEngine


class IdleStatus(unittest.TestCase):
    def test_idle_events_do_not_enter_the_request_queue(self):
        engine = StrataEngine.__new__(StrataEngine)
        engine.proc = SimpleNamespace(stdout=io.StringIO(
            "CACHE state=disk released_vram_bytes=1048576 operation_ms=12.5\n"
            "DONE 1 1 0 0 stop\n"))
        engine.lines, engine.slot_q = queue.Queue(), []
        engine._pump()
        self.assertEqual(engine.cache, {"state": "disk", "released_vram_bytes": 1048576,
                                        "operation_ms": 12.5})
        self.assertTrue(engine.lines.get_nowait().startswith("DONE "))
        self.assertIsNone(engine.lines.get_nowait())
        self.assertTrue(engine.lines.empty())


if __name__ == "__main__":
    unittest.main()
