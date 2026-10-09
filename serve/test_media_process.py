"""Owned CPU media process tests; no model or GPU."""
import subprocess
import sys
import threading
import time
import unittest

from serve.media_process import OwnedMediaProcess
from serve.video import VideoCancelled, VideoError, VideoLimitError, VideoRequestBudget
from serve.test_video_policy import policy


class OwnedProcessTests(unittest.TestCase):
    def test_output_and_cpu_only_environment(self):
        script="import os; print(repr(os.environ.get('CUDA_VISIBLE_DEVICES'))); print(repr(os.environ.get('HIP_VISIBLE_DEVICES')))"
        with OwnedMediaProcess([sys.executable,"-c",script],VideoRequestBudget(policy()),max_stdout=100) as p:
            self.assertEqual(p.read(), b"''\n''\n")
        self.assertIsNotNone(p.proc.returncode)
        self.assertFalse(any(t.is_alive() for t in p.threads))

    def test_stdout_stderr_and_nonzero_are_bounded(self):
        scripts=[("import sys;sys.stdout.buffer.write(b'x'*100000)","output"),
                 ("import sys;sys.stderr.buffer.write(b'x'*100000)","diagnostics"),
                 ("import sys;sys.exit(2)","failed")]
        for script, message in scripts:
            with self.subTest(message=message):
                with self.assertRaisesRegex(VideoError,message):
                    with OwnedMediaProcess([sys.executable,"-c",script],VideoRequestBudget(policy()),max_stdout=100) as p:
                        p.read()
                self.assertIsNotNone(p.proc.returncode)
                self.assertLessEqual(len(p.stderr),16384)
                self.assertFalse(any(t.is_alive() for t in p.threads))

    def test_cancel_reaps_only_owned_child(self):
        unrelated = subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"])
        try:
            cancel=threading.Event()
            timer=threading.Timer(.15,cancel.set);timer.start()
            with self.assertRaises(VideoCancelled):
                with OwnedMediaProcess([sys.executable,"-c","import time;time.sleep(30)"],
                                       VideoRequestBudget(policy(),cancel),max_stdout=0) as p:
                    p.read()
            timer.join()
            self.assertIsNotNone(p.proc.returncode)
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.kill();unrelated.wait(timeout=5)

    def test_deadline_without_sleep_polling_in_caller(self):
        start=time.monotonic()
        with self.assertRaises(VideoLimitError):
            with OwnedMediaProcess([sys.executable,"-c","import time;time.sleep(30)"],
                                   VideoRequestBudget(policy(deadline_s=.15)),max_stdout=0) as p:
                p.read()
        self.assertLess(time.monotonic()-start,4)
        self.assertIsNotNone(p.proc.returncode)



if __name__ == "__main__":
    unittest.main()
