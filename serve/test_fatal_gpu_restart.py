"""serve/test_fatal_gpu_restart.py: #4350.  After a fatal GPU error the DeepSeek engine ends itself (ERR fatal GPU error: ...) and the
server starts it again on the next request; three such deaths within ten minutes stop the restarts with a plain 503 text.
CPU only, no engine: the server's bookkeeping is called on a stand-in."""
import sys
import time
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import server as S  # noqa: E402


def stand_in(pid, last_err):
    return types.SimpleNamespace(engine=types.SimpleNamespace(last_err=last_err, proc=types.SimpleNamespace(pid=pid)))


class FatalGpuRestart(unittest.TestCase):
    def note(self, svc, code=3):
        S.Service._note_fatal_gpu_exit.__get__(svc)(code)

    def make(self, pid, err="fatal GPU error: cudaStreamSynchronize(s): an illegal memory access was encountered"):
        svc = stand_in(pid, err)
        svc.FATAL_GPU_MAX = S.Service.FATAL_GPU_MAX
        svc.FATAL_GPU_WINDOW_S = S.Service.FATAL_GPU_WINDOW_S
        return svc

    def test_one_fatal_death_is_restarted_with_one_plain_line(self):
        svc = self.make(100)
        self.note(svc)                      # no exception: the caller goes on to restart the engine

    def test_an_ordinary_death_is_not_counted(self):
        svc = self.make(100, err="the engine ran out of memory")
        for pid in range(10):
            svc.engine.proc.pid = pid
            self.note(svc)
        self.assertFalse(svc.__dict__.get("_fatal_gpu_times"))

    def test_the_same_death_seen_by_many_requests_counts_once(self):
        svc = self.make(100)
        for _ in range(8):
            self.note(svc)
        self.assertEqual(len(svc._fatal_gpu_times), 1)

    def test_three_fatal_deaths_in_ten_minutes_stop_the_restarts_and_say_so(self):
        svc = self.make(1)
        self.note(svc)
        svc.engine.proc.pid = 2
        self.note(svc)
        svc.engine.proc.pid = 3
        with self.assertRaises(S.EngineDied) as cm:
            self.note(svc)
        self.assertIn("3 times in 10 minutes", str(cm.exception))
        self.assertIn("not started again", str(cm.exception))
        with self.assertRaises(S.EngineDied):          # the same dead engine asked again: still refused
            self.note(svc)

    def test_after_the_window_it_tries_again(self):
        svc = self.make(1)
        self.note(svc)
        svc.engine.proc.pid = 2
        self.note(svc)
        svc._fatal_gpu_times[:] = [t - S.Service.FATAL_GPU_WINDOW_S - 1 for t in svc._fatal_gpu_times]
        svc.engine.proc.pid = 3
        self.note(svc)                      # the two old ones aged out: only this one counts

    def test_the_request_in_flight_fails_as_an_engine_death_not_a_bad_request(self):
        eng = types.SimpleNamespace(last_err=None, ended=False)
        with self.assertRaises(S.EngineDied) as cm:
            S.StrataEngine._raise_if_fatal_gpu(eng, "ERR fatal GPU error: cudaStreamSynchronize(s): an illegal memory access was encountered")
        self.assertIn("illegal memory access", str(cm.exception))
        self.assertTrue(eng.ended)                                  # the next request starts the engine at once
        self.assertTrue(eng.last_err.startswith("fatal GPU error"))

    def test_an_ordinary_engine_error_stays_a_bad_request(self):
        eng = types.SimpleNamespace(last_err=None, ended=False)
        S.StrataEngine._raise_if_fatal_gpu(eng, "ERR the prompt (9 tokens) and max_new 5 are more than the context 8")   # no exception
        self.assertFalse(eng.ended)


# #6054: "prefill copy_i32: out of memory".  The engine used to call std::exit(1) there, whose teardown could hang, so the request
# waited for the 300 s silence watchdog.  Now, under --serve, the engine prints "ERR fatal GPU error: prefill copy_i32: out of memory"
# and ends at once (src/prefill/kernels.cu, include/strata/core/device_error_exit.hpp).  A stand-in process plays that engine here.
OOM_ENGINE = (
    "import sys\n"
    "sys.stdin.readline()\n"                                   # the GEN line: the prompt read starts
    "print('prefill copy_i32: out of memory', file=sys.stderr, flush=True)\n"
    "print('ERR fatal GPU error: prefill copy_i32: out of memory', flush=True)\n"
    "sys.exit(3)\n")
OOM_LINE = "ERR fatal GPU error: prefill copy_i32: out of memory"


class PrefillOutOfMemory(unittest.TestCase):
    def engine_on(self, proc, silence_s=300.0):
        import queue
        import threading
        eng = S.StrataEngine.__new__(S.StrataEngine)
        eng.proc, eng.lines, eng.slot_q, eng.slot_busy = proc, queue.Queue(), [], []
        eng.can_stop, eng.max_context, eng.silence_s, eng.log_path = False, 4096, silence_s, None
        eng.last_err = None
        threading.Thread(target=eng._pump, daemon=True).start()
        return eng

    def test_the_request_fails_at_once_not_after_the_silence_watchdog(self):
        import subprocess
        import threading
        proc = subprocess.Popen([sys.executable, "-c", OOM_ENGINE], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True)
        try:
            eng = self.engine_on(proc)                        # the watchdog would wait 300 s
            t0 = time.monotonic()
            with self.assertRaises(S.EngineDied) as cm:
                list(eng.generate([1] * 64, 8, {}, threading.Event()))
            self.assertLess(time.monotonic() - t0, 10)
            self.assertNotIsInstance(cm.exception, S.EngineSilent)   # the engine's own words, not the watchdog's
            self.assertIn("ran out of memory", str(cm.exception))
            self.assertIn("prefill copy_i32: out of memory", str(cm.exception))
            self.assertIn("--conversation-cache-slots", str(cm.exception))
            self.assertTrue(cm.exception.restarts)            # the client is told the next request starts it again
            self.assertEqual(proc.wait(10), 3)
            self.assertFalse(eng.alive())                     # the next request starts the engine again
            self.assertIn("ran out of memory", eng.death_note())
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_an_oom_line_is_an_engine_death_with_memory_words(self):
        eng = types.SimpleNamespace(last_err=None, ended=False)
        with self.assertRaises(S.EngineDied) as cm:
            S.StrataEngine._raise_if_fatal_gpu(eng, OOM_LINE)
        self.assertIn("ran out of memory", str(cm.exception))
        self.assertNotIn("GPU reported a fatal error", str(cm.exception))
        self.assertTrue(eng.ended)

    def test_oom_deaths_do_not_stop_the_restarts(self):
        svc = stand_in(1, OOM_LINE[4:])
        svc.FATAL_GPU_MAX, svc.FATAL_GPU_WINDOW_S = S.Service.FATAL_GPU_MAX, S.Service.FATAL_GPU_WINDOW_S
        for pid in range(1, 6):                               # a restart gives the memory back: never "not started again"
            svc.engine.proc.pid = pid
            S.Service._note_fatal_gpu_exit.__get__(svc)(3)
        self.assertFalse(svc.__dict__.get("_fatal_gpu_times"))

    def http(self, stream):
        import json
        import urllib.error
        import urllib.request

        class OomEngine(S.MockEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                S.StrataEngine._raise_if_fatal_gpu(types.SimpleNamespace(last_err=None, ended=False), OOM_LINE)
                yield from ()

        from serve.frontend import ChatTemplate
        tok = S.ByteTokenizer()
        svc = S.Service(OomEngine(tok, "x", max_context=4096), tok,
                        ChatTemplate(Path(__file__).resolve().parents[1] / "serve/chat_template.jinja"))
        httpd = S.serve(svc, port=0)
        try:
            body = json.dumps({"model": "m", "max_tokens": 8, "stream": stream,
                               "messages": [{"role": "user", "content": "hi"}]}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    return r.status, r.read().decode()
            except urllib.error.HTTPError as e:
                return e.code, e.read().decode()
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_a_plain_request_gets_a_503_that_names_the_memory(self):
        code, text = self.http(stream=False)
        self.assertEqual(code, 503)
        self.assertIn("ran out of memory", text)

    def test_a_streamed_request_gets_an_error_event_that_names_the_memory(self):
        code, text = self.http(stream=True)
        self.assertIn('"error"', text)
        self.assertIn("ran out of memory", text)


if __name__ == "__main__":
    unittest.main()
