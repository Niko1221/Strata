"""Smaller serve-side robustness fixes of round 0213 (each test fails on the base a3a10b90a):

#3443 ConvCacheLog.poll without a lock, #4792 a failed engine start when another thread closed the process,
#4738 lost wake-ups of the control-line queue, #5044 a byte that is not UTF-8 on the engine's stdout.
The fake engine and helpers are those of test_stream_ownership.py.
"""
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve.server import StrataEngine
from serve.test_stream_ownership import FakeEngines, canary_script, collect, expected_reply, prompt_ids


class ConvCachePoll(unittest.TestCase):
    """ConvCacheLog.poll ran outside any lock: a second poll that moved `pos` between the first one's size check and
    its read left a negative read length (ValueError).  Found by HWSTALL while triaging #3443."""
    PARK = ("strata serve: conversation cache: parked 5000 tokens in 12.0 ms; parked=1 bytes=104857600 evictions=0 "
            "snapshot_bytes=104857600 reused_kv_bytes=0\n")

    def test_a_poll_that_overlaps_another_one_does_not_raise(self):
        import os
        from serve.server import ConvCacheLog
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text(self.PARK, encoding="utf-8")
            c = ConvCacheLog()
            c.poll(str(log), 0)
            with open(log, "a", encoding="utf-8") as f:
                f.write(self.PARK)
            first_sized, first_may_go = threading.Event(), threading.Event()
            real_size = os.path.getsize
            calls = []

            def slow_first(path):
                size = real_size(path)
                if not calls and path == str(log):
                    calls.append(1)
                    first_sized.set()
                    first_may_go.wait(2.0)             # the first poll has its size; a second poll runs meanwhile
                return size
            errors = []

            def first():
                try:
                    c.poll(str(log), 0)
                except BaseException as e:             # noqa: BLE001
                    errors.append(e)
            with mock.patch("serve.server.os.path.getsize", slow_first):
                t = threading.Thread(target=first, daemon=True)
                t.start()
                self.assertTrue(first_sized.wait(5))
                with open(log, "a", encoding="utf-8") as f:
                    f.write(self.PARK * 2)
                second = threading.Thread(target=lambda: c.poll(str(log), 0), daemon=True)
                second.start()
                second.join(0.5)                       # (with the lock it waits for the first one)
                first_may_go.set()
                t.join(5)
                second.join(5)
            self.assertEqual(errors, [])
            self.assertEqual(c.poll(str(log), 0)["parks"], 4, "every parked line counted once")


if __name__ == "__main__":
    unittest.main()
