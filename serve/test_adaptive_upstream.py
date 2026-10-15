"""Compatibility of adaptive controls with the v0.1.41 server changes."""
import threading
import time
import unittest
from unittest import mock

from serve import server


class AdaptiveUpstreamTests(unittest.TestCase):
    def test_background_heartbeats_are_not_a_frozen_engine(self):
        from serve.test_server import SilentEngine
        engine = SilentEngine().bare(300, can_stop=True)
        engine._activity = lambda: (12.0, 4096)
        engine.gpu_busy = lambda: False
        def feed():
            for _ in range(6):
                time.sleep(.06)
                engine.lines.put("BACKGROUND status=waiting lease_remaining_ms=5000")
            engine.lines.put("T 7")
            engine.lines.put("DONE 1 1 1 1 length")
        worker = threading.Thread(target=feed)
        with mock.patch.object(server, "ENGINE_STALL_S", .1):
            worker.start()
            try:
                result = list(engine.generate([1], 8, {}, threading.Event()))
            finally:
                worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual([token for token in result if token is not None], [7])
        self.assertGreaterEqual(result.count(None), 6)
        engine.proc.kill.assert_not_called()

    def test_live_memory_and_automatic_vision_flags_coexist(self):
        cfg = {"args": ["--resident-budget-gib", "20", "--vram-reserve-mib", "448"],
               "memory_policy": {"enabled": True, "mode": "live"},
               "vision": {"gpu": True}}
        args = server.engine_args(cfg)
        self.assertEqual(args.count("--live-memory"), 1)
        self.assertEqual(args.count("--vision"), 1)
        self.assertEqual(args[args.index("--vram-reserve-mib") + 1], "448")

    def test_live_memory_rejects_queued_foresight_host_copies(self):
        cfg = {"args": ["--live-memory"]}
        with mock.patch.dict(server.os.environ, {"STRATA_FS_SLOTS": "2"}):
            with self.assertRaisesRegex(ValueError, "ForesightSwap"):
                server.engine_args(cfg)
            cfg["env"] = {"STRATA_FS_SLOTS": "0"}
            self.assertIn("--live-memory", server.engine_args(cfg))
        cfg["env"] = {"STRATA_FS_SLOTS": "  +2"}
        with self.assertRaisesRegex(ValueError, "STRATA_FS_SLOTS=0"):
            server.engine_args(cfg)
        cfg["args"] = []
        self.assertEqual(server.engine_args(cfg), [])


if __name__ == "__main__":
    unittest.main()
