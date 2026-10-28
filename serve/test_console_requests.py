"""Blocked request diagnostics do not own inference resources (#4479)."""
import contextlib
import json
import threading
import time
import sys
import subprocess
import tempfile
import unittest
import urllib.request
from unittest import mock
from pathlib import Path

from serve import console
from serve.test_console_support import blocked_console
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, GpuBusy, MockEngine, Server, Service, serve

ROOT = Path(__file__).parent


class RequestOutputIsolation(unittest.TestCase):
    def test_server_error_preserves_peer_in_one_record_and_suppresses_disconnect(self):
        peer = ("127.0.0.1", 43210)
        server = object.__new__(Server)
        real_write = console._Sink.write
        for error in (ValueError("injected handler failure"), ConnectionResetError("peer left")):
            with self.subTest(error=type(error).__name__), \
                    tempfile.TemporaryFile(mode="w+", encoding="utf-8") as destination:
                records = []
                writer = console.Console()

                def write(sink, text):
                    records.append(text)
                    return real_write(sink, text)

                with contextlib.redirect_stderr(destination), \
                        mock.patch.object(console, "_console", writer), \
                        mock.patch.object(console._Sink, "write", autospec=True, side_effect=write):
                    try:
                        try:
                            raise error
                        except Exception:
                            server.handle_error(None, peer)
                    finally:
                        self.assertTrue(writer.close(2), "error output did not finish")
                destination.seek(0)
                output = destination.read()
                if isinstance(error, ConnectionError):
                    self.assertEqual(records, [])
                    self.assertEqual(output, "")
                else:
                    self.assertEqual(len(records), 1)
                    self.assertEqual(output, records[0])
                    self.assertIn(f"Exception occurred during processing of request from {peer}\n", output)
                    self.assertIn("Traceback (most recent call last):", output)
                    self.assertIn("ValueError: injected handler failure", output)
                    self.assertTrue(output.startswith("-" * 40 + "\n"))
                    self.assertTrue(output.endswith("-" * 40 + "\n"))

    def test_cancel_during_prompt_and_decode_progress(self):
        for phase in ("prompt", "decode"):
            with self.subTest(phase=phase):
                blocked, resume, closed, entered = [threading.Event() for _ in range(4)]
                cancel = threading.Event()
                errors = []

                def write(text):
                    blocked.set()
                    resume.wait(5)

                tok = ByteTokenizer()

                class Engine(MockEngine):
                    calls = 0

                    def generate(self, *args, **kwargs):
                        self.calls += 1
                        first = self.calls == 1
                        if not first:
                            entered.set()
                        try:
                            if first:
                                time.sleep(1.05)
                                if phase == "prompt":
                                    yield None
                            yield from super().generate(*args, **kwargs)
                        finally:
                            if first:
                                closed.set()

                engine = Engine(tok, "</think>\n\n" + "answer " * 100, max_context=4096)
                engine.delay = 0.005
                svc = Service(engine, tok, ChatTemplate(ROOT / "chat_template.jinja"))
                output = console.Console()

                def run(event):
                    try:
                        list(svc.run(tok.encode("hi"), True, None, 64, {}, event))
                    except BaseException as error:
                        errors.append(error)

                first = threading.Thread(target=run, args=(cancel,), daemon=True)
                second = threading.Thread(target=run, args=(threading.Event(),), daemon=True)
                with blocked_console(output, write):
                    first.start()
                    try:
                        self.assertTrue(blocked.wait(3))
                        cancel.set()
                        self.assertTrue(closed.wait(2), "console delayed cancellation cleanup")
                        first.join(2)
                        self.assertFalse(first.is_alive(), "completion still depends on console")
                        second.start()
                        self.assertTrue(entered.wait(2), "console retained the engine turn")
                        second.join(2)
                        self.assertFalse(second.is_alive())
                        self.assertFalse(svc.terminal_requests)
                        self.assertEqual(svc.status["queued"], 0)
                        self.assertEqual(errors, [])
                    finally:
                        cancel.set()
                        resume.set()
                        first.join(2)
                        if second.ident is not None:
                            second.join(2)
                        output.close()

    def test_completion_releases_http_admission_while_console_is_blocked(self):
        blocked, resume = threading.Event(), threading.Event()

        def write(text):
            if "[strata] done:" in text:
                blocked.set()
                resume.wait(5)

        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "</think>\n\nok", max_context=4096), tok,
                      ChatTemplate(ROOT / "chat_template.jinja"))
        svc.max_queue = 0
        output = console.Console()
        httpd = serve(svc, port=0)

        def request():
            req = urllib.request.Request(
                f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                data=json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                 "max_tokens": 32}).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=2) as response:
                body = json.loads(response.read())
                self.assertTrue(body["choices"])
                return response.status

        try:
            with blocked_console(output, write):
                self.assertEqual(request(), 200)
                self.assertTrue(blocked.wait(2))
                self.assertEqual(svc.wait_for_drain(1), 0)
                self.assertEqual(request(), 200)
                self.assertEqual(svc.wait_for_drain(1), 0)
        finally:
            resume.set()
            output.close()
            httpd.shutdown()
            httpd.server_close()


    def test_verbose_before_load_does_not_inherit_console(self):
        from serve.test_server import UnloadableEngine
        tok = ByteTokenizer()
        engine = UnloadableEngine(tok, "</think>\n\nok", max_context=4096)
        svc = Service(engine, tok, ChatTemplate(ROOT / "chat_template.jinja"))
        blocked, resume = threading.Event(), threading.Event()

        def write(text):
            blocked.set()
            resume.wait(5)

        output = console.Console()
        svc.before_load = [sys.executable, "-c", "import sys; sys.stdout.write('hook output ' * 100000)"]
        svc.before_load_timeout_s = 2
        svc.before_load_strict = True
        with blocked_console(output, write):
            try:
                svc.unload()
                self.assertTrue(blocked.wait(2))
                # The hook must get a file, not inherit terminal descriptors.
                import subprocess
                real_run = subprocess.run
                def run(*args, **kwargs):
                    self.assertIn("stdout", kwargs)
                    self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
                    return real_run(*args, **kwargs)
                with mock.patch("serve.server.subprocess.run", side_effect=run):
                    svc.load()
                self.assertTrue(engine.alive())
            finally:
                resume.set()
                output.close()


    def test_http_and_process_exit_with_unread_stdout_pipe(self):
        code = r"""
import json,sys,time,threading,urllib.request
from pathlib import Path
from serve.console import console_print
from serve.frontend import ChatTemplate
from serve.server import Service,MockEngine,ByteTokenizer,serve
from serve.telemetry import Telemetry
sampled=threading.Event()
def fail_sample(self):
    sampled.set()
    raise RuntimeError('injected sampler failure')
Telemetry.sample=fail_sample
for _ in range(8):
    for _ in range(64): console_print('x'*4090)
    time.sleep(.02)
tok=ByteTokenizer()
svc=Service(MockEngine(tok,'</think>\n\nok',max_context=4096),tok,ChatTemplate('serve/chat_template.jinja'))
svc.max_queue=0
server=serve(svc,port=0)
assert sampled.wait(2), 'actual sampler did not attempt a sample'
results=[]
try:
    for _ in range(2):
        req=urllib.request.Request(f'http://127.0.0.1:{server.server_address[1]}/v1/chat/completions',
            data=json.dumps({'model':'m','messages':[{'role':'user','content':'hi'}],'max_tokens':32}).encode(),
            headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=2) as r:
            r.read();results.append(r.status)
        assert svc.wait_for_drain(1)==0
finally:
    server.shutdown();server.server_close()
    svc.telemetry._thread.join(1)
    assert not svc.telemetry._thread.is_alive(), 'sampler failure diagnostic blocked shutdown'
Path(sys.argv[1]).write_text(json.dumps(results))
"""
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            child = subprocess.Popen([sys.executable, "-c", code, str(result)], cwd=ROOT.parent,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                # Do not consume stdout until after exit: the real pipe stays full.
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
                self.fail("blocked console prevented HTTP completion or process exit")
            finally:
                stdout, stderr = child.communicate(timeout=2)
            self.assertEqual(child.returncode, 0, stderr.decode(errors="replace"))
            self.assertTrue(stdout)
            self.assertEqual(json.loads(result.read_text()), [200, 200])


    def test_hook_capture_failure_does_not_change_hook_result(self):
        from serve.test_server import UnloadableEngine
        real_file = tempfile.TemporaryFile
        for failure, outcome in ((f, o) for f in ("create", "read", "close")
                                 for o in ("success", "failure", "timeout")):
            with self.subTest(failure=failure, outcome=outcome):
                tok = ByteTokenizer()
                engine = UnloadableEngine(tok, "</think>\n\nok", max_context=4096)
                svc = Service(engine, tok, ChatTemplate(ROOT / "chat_template.jinja"))
                script = {"success": "pass", "failure": "import sys; sys.exit(7)",
                          "timeout": "import time; time.sleep(2)"}[outcome]
                svc.before_load = [sys.executable, "-c", script]
                svc.before_load_timeout_s = 0.1 if outcome == "timeout" else 2
                svc.before_load_strict = True
                svc.unload()

                def temporary():
                    if failure == "create":
                        raise OSError("injected spool failure")
                    file = real_file()
                    class ClosingFile:
                        def __getattr__(self, name):
                            return getattr(file, name)
                        def read(self, *args):
                            if failure == "read":
                                raise OSError("injected read failure")
                            return file.read(*args)
                        def close(self):
                            file.close()
                            if failure == "close":
                                raise OSError("injected close failure")
                    return ClosingFile()

                with mock.patch("serve.server.tempfile.TemporaryFile", side_effect=temporary):
                    if outcome == "success":
                        svc.load()
                    else:
                        reason = "exited with code 7" if outcome == "failure" else "did not finish"
                        with self.assertRaisesRegex(GpuBusy, reason):
                            svc.load()
                self.assertEqual(engine.alive(), outcome == "success")


if __name__ == "__main__":
    unittest.main()
