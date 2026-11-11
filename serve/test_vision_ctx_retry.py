"""A GPU picture reader whose start failed in its CUDA context ("CUDA error: out of memory ... in function ggml_cuda_init") on a Linux Arm machine with shared
CPU / GPU memory (DGX Spark) is started again, up to 8 times a second apart, before the CPU reader takes over.  Every other failure goes the way it went.

    python -m serve.test_vision_ctx_retry
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import serve.server as server
from strata_control import vision_reader as VR

CTX = ("CUDA error: out of memory\n  current device: 0, in function ggml_cuda_init at ggml/src/ggml-cuda/ggml-cuda.cu:367\n"
       "ggml/src/ggml-cuda/ggml-cuda.cu:108: CUDA error\n")
ALLOC = "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 865.48 MiB on device 0: cudaMalloc failed: out of memory\n"


def rule(text=CTX, retries_done=0, **kw):
    args = dict(linux=True, arm=True, hip=False)
    args.update(kw)
    return VR.retry_gpu_start(text=text, retries_done=retries_done, **args)


class Rule(unittest.TestCase):
    def test_the_context_failure_on_linux_arm_is_retried_up_to_eight_times(self):
        self.assertEqual((VR.CONTEXT_RETRIES, VR.CONTEXT_RETRY_WAIT_S), (8, 1.0))
        for done in range(8):
            self.assertTrue(rule(retries_done=done))
        self.assertFalse(rule(retries_done=8))
        self.assertFalse(rule(retries_done=20))

    def test_the_own_tower_wording_counts_too(self):
        self.assertTrue(rule("strata-vision: Strata's picture reader did not start (cannot use GPU 0: out of memory)"))

    def test_other_platforms_never_retry(self):
        self.assertFalse(rule(linux=False))           # Windows, Mac
        self.assertFalse(rule(arm=False))             # a PC with a card of its own
        self.assertFalse(rule(hip=True))              # the AMD reader

    def test_other_errors_never_retry(self):
        for text in (ALLOC,                                               # a card another program filled: no retry helps (reader_room reports it)
                     CTX + ALLOC,
                     "failed to allocate CUDA0 buffer of size 1048576 out of memory",
                     "CUDA error: device kernel image is invalid\n  in function ggml_cuda_init",
                     "CUDA error: CUDA-capable device(s) is/are busy or unavailable\n  in function ggml_cuda_init",
                     "ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected",
                     "the vision encoder did not start: it said nothing for 300 s",
                     "out of memory"):                                     # no sign of the context
            with self.subTest(text=text[:50]):
                self.assertFalse(rule(text))


def make_vision(args=("/engine/strata-vision", "--gpu"), log_text=""):
    f = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
    f.write(log_text)
    f.flush()
    v = server.Vision.__new__(server.Vision)
    v.spawn = (list(args), f, None)
    return v, f


class Spawn(unittest.TestCase):
    def run_start(self, outcomes, args=("/engine/strata-vision", "--gpu"), system="linux", machine="aarch64", logged=CTX):
        """_start_gpu_reader with `_start_reader` giving `outcomes` in turn (an exception is raised, anything else returned); returns (calls, sleeps, result or error)."""
        v, f = make_vision(args)
        self.addCleanup(lambda: (f.close(), Path(f.name).unlink()))
        calls, sleeps = [], []

        def start_reader():
            calls.append(1)
            o = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
            if isinstance(o, Exception):
                f.write(logged)
                f.flush()
                raise o
            return o

        v._start_reader = start_reader
        with mock.patch.object(server.sys, "platform", system), mock.patch.object(server.platform, "machine", return_value=machine), \
                mock.patch.object(server.time, "sleep", lambda s: sleeps.append(s)):
            try:
                res = v._start_gpu_reader()
            except RuntimeError as e:
                res = e
        return len(calls), sleeps, res

    def test_a_start_that_works_the_first_time_is_not_delayed(self):
        calls, sleeps, res = self.run_start([None])
        self.assertEqual((calls, sleeps), (1, []))

    def test_a_context_failure_is_tried_again_until_it_works(self):
        calls, sleeps, res = self.run_start([RuntimeError("the vision encoder did not start (killed by signal 6)")] * 3 + [None])
        self.assertEqual((calls, sleeps, res), (4, [1.0, 1.0, 1.0], None))

    def test_a_context_that_never_works_gives_up_after_eight_retries_and_raises_as_before(self):
        err = RuntimeError("the vision encoder did not start (killed by signal 6)")
        calls, sleeps, res = self.run_start([err])
        self.assertEqual((calls, len(sleeps)), (9, 8))
        self.assertIs(res, err)

    def test_no_room_is_not_retried(self):
        err = server.VisionNoRoom("the vision encoder did not start: the graphics card has no room for it", None)
        calls, sleeps, res = self.run_start([err], logged=ALLOC)
        self.assertEqual((calls, sleeps), (1, []))
        self.assertIs(res, err)

    def test_other_systems_are_not_retried(self):
        err = RuntimeError("the vision encoder did not start")
        for kw in ({"system": "win32"}, {"system": "darwin"}, {"machine": "x86_64"}, {"machine": "AMD64"},
                   {"args": ("/engine/strata-vision-hip", "--gpu")}):
            with self.subTest(kw=kw):
                calls, sleeps, res = self.run_start([err], **kw)
                self.assertEqual((calls, sleeps), (1, []))

    def test_an_old_failure_in_a_shared_log_does_not_count(self):
        v, f = make_vision(log_text=CTX)       # an earlier start's lines are in the file already
        self.addCleanup(lambda: (f.close(), Path(f.name).unlink()))
        v._start_reader = mock.Mock(side_effect=RuntimeError("the vision encoder did not start: ERR cannot open the model"))
        with mock.patch.object(server.sys, "platform", "linux"), mock.patch.object(server.platform, "machine", return_value="aarch64"), \
                mock.patch.object(server.time, "sleep", lambda s: self.fail("slept")):
            with self.assertRaises(RuntimeError):
                v._start_gpu_reader()
        self.assertEqual(v._start_reader.call_count, 1)

    def test_start_falls_back_to_the_cpu_reader_after_the_last_retry(self):
        v, f = make_vision()
        self.addCleanup(lambda: (f.close(), Path(f.name).unlink()))
        v.fallback = (["/engine/strata-vision"], f, None)
        v.on_cpu, v.elastic = False, False
        starts = []

        def start_reader():
            starts.append(v.on_cpu)
            if not v.on_cpu:
                f.write(CTX)
                f.flush()
                raise RuntimeError("the vision encoder did not start (killed by signal 6)")

        v._start_reader = start_reader
        v.not_started_note = lambda: "note"
        with mock.patch.object(server.sys, "platform", "linux"), mock.patch.object(server.platform, "machine", return_value="aarch64"), \
                mock.patch.object(server.time, "sleep", lambda s: None), mock.patch("builtins.print"):
            v._start()
        self.assertEqual(starts, [False] * 9 + [True])
        self.assertTrue(v.on_cpu)


if __name__ == "__main__":
    unittest.main()
