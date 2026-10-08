"""#1527: a READY timeout must release the request even when a wrapper's child inherited stdout.

Real subprocess pipes and HTTP requests; no GPU, model weights or tokenizer download.
    python -m unittest serve.test_startup_timeout -v
"""
import io
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve import server


READY = """import sys
print('READY 4096 stop', flush=True)
for line in sys.stdin:
    if line.startswith('QUIT'):
        break
    if line.startswith('GEN '):
        print('T 90\\nDONE 1 20 1 1 stop', flush=True)
"""


class StartupTimeout(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.script = self.root / 'engine.py'
        self.script.write_text('import time; time.sleep(600)', encoding='utf-8')
        self.command = [sys.executable, str(self.script)]
        self.procs, self.engines, self.sessions = [], [], []
        real_popen = subprocess.Popen

        def popen(what, cmd, **kw):
            self.sessions.append(kw.get('start_new_session', False))
            # Also isolate the test's cleanup when this regression runs against the old server.
            kw['start_new_session'] = os.name != 'nt'
            proc = real_popen(self.command, **kw)
            self.procs.append(proc)
            return proc

        self.patches = [mock.patch.object(server, 'popen', popen),
                        mock.patch.object(server, 'ENGINE_READY_S', .2),
                        mock.patch.object(server.StrataEngine, 'RESTART_RETRY_S', 0),
                        mock.patch.object(server, 'narrate_start', lambda *a: None)]
        for patch in self.patches:
            patch.start()

    def kill_owned(self):
        # Independent of engine.close(): a broken cleanup must not hang the suite or leave its children running.
        for proc in self.procs:
            if os.name != 'nt':
                try:
                    os.killpg(proc.pid, signal.SIGKILL)  # only the session this test created for this process
                except ProcessLookupError:
                    pass
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=3)

    def tearDown(self):
        self.kill_owned()
        for engine in self.engines:
            engine.close()
        for patch in reversed(self.patches):
            patch.stop()
        self.tmp.cleanup()

    def bounded(self, fn, limit=5):
        result = []

        def call():
            try:
                result.append(fn())
            except BaseException as error:
                result.append(error)

        thread = threading.Thread(target=call, daemon=True)
        before = time.monotonic()
        thread.start()
        thread.join(limit)
        if thread.is_alive():
            self.kill_owned()
            thread.join(3)
            self.fail('engine cleanup held the caller past its deadline')
        self.assertLess(time.monotonic() - before, limit)
        return result[0]

    def engine(self, lazy=False):
        engine = server.StrataEngine.__new__(server.StrataEngine)
        self.engines.append(engine)
        engine.__init__('fake', [], log=str(self.root / 'engine.log'), lazy=lazy)
        return engine

    def assert_failed(self, error, reason='did not report READY'):
        self.assertIsInstance(error, server.EngineDied, repr(error))
        self.assertIn(reason, str(error))
        engine = self.engines[-1]
        self.assertFalse(engine.alive())
        self.assertIsNotNone(engine.proc.poll())
        self.assertFalse(engine.pump.is_alive())
        self.assertTrue(engine.proc.stdin.closed)
        self.assertTrue(engine.proc.stdout.closed)
        self.assertTrue(engine.log.closed)

    def test_silent_single_process_times_out(self):
        self.assert_failed(self.bounded(self.engine))

    @unittest.skipUnless(os.name == 'posix' and Path('/bin/bash').exists(), 'POSIX shell wrapper')
    def test_exact_reported_shell_child_times_out(self):
        script = self.root / 'engine.sh'
        script.write_text("#!/bin/bash\ntrap 'exit 0' TERM INT\nsleep 3600\n", encoding='utf-8')
        script.chmod(0o755)
        self.command = [str(script)]
        self.assert_failed(self.bounded(self.engine))

    @unittest.skipUnless(os.name == 'posix' and shutil.which('sleep'), 'POSIX child process')
    def test_exited_wrapper_with_inherited_stdout_times_out(self):
        self.script.write_text("import subprocess\nsubprocess.Popen(['sleep', '600'])\n", encoding='utf-8')
        self.assert_failed(self.bounded(self.engine))

    def test_eof_before_ready_reports_reason_and_closes_pipes(self):
        self.script.write_text("import sys\nprint('strata: fake startup refused', file=sys.stderr)\n", encoding='utf-8')
        result = self.bounded(self.engine)
        self.assert_failed(result, 'exited before it was ready')
        self.assertIn('fake startup refused', str(result))

    def test_eof_before_ready_ends_process_even_if_it_stays_alive(self):
        self.script.write_text('import os, time\nos.close(1)\ntime.sleep(600)\n', encoding='utf-8')
        self.assert_failed(self.bounded(self.engine), 'exited before it was ready')

    def test_interrupted_startup_closes_detached_process(self):
        with mock.patch.object(server.queue.Queue, 'get', side_effect=KeyboardInterrupt):
            result = self.bounded(self.engine)
        self.assertIsInstance(result, KeyboardInterrupt)
        engine = self.engines[-1]
        self.assertIsNotNone(engine.proc.poll())
        self.assertFalse(engine.pump.is_alive())
        self.assertTrue(engine.proc.stdout.closed)
        self.assertTrue(engine.log.closed)

    def test_ready_then_silent_does_not_time_out_startup(self):
        self.script.write_text(READY, encoding='utf-8')
        engine = self.engine()
        time.sleep(.3)
        self.assertTrue(engine.alive())
        self.assertEqual(engine.max_context, 4096)
        self.assertEqual(self.sessions, [os.name != 'nt'])
        self.assertEqual(list(engine.generate([1], 1, {}, threading.Event())), [90])
        self.assertIsNone(self.bounded(engine.close))

    @unittest.skipUnless(os.name == 'posix' and shutil.which('sleep'), 'POSIX child process')
    def test_close_after_ready_is_bounded_with_inherited_writer(self):
        self.script.write_text("import subprocess\nsubprocess.Popen(['sleep', '600'])\n" + READY, encoding='utf-8')
        engine = self.engine()
        proc, reader = engine.proc, engine.pump
        self.assertIsNone(self.bounded(engine.close))
        self.assertIsNone(engine.proc)
        # QUIT+wait reaped the leader. Its numeric group must not be signalled again, so closure is deferred
        # until the inherited writer exits. The test ends its own child independently.
        self.kill_owned()
        reader.join(2)
        deadline = time.monotonic() + 2
        while not proc.stdout.closed and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertFalse(reader.is_alive())
        self.assertTrue(proc.stdout.closed)

    def request(self, http, path='/v1/chat/completions'):
        body = {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 1, 'reasoning_effort': 'none'}
        req = urllib.request.Request(f'http://127.0.0.1:{http.server_address[1]}' + path,
                                     data=json.dumps(body if 'completions' in path else {}).encode(),
                                     headers={'Content-Type': 'application/json'})
        try:
            response = urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read())

    def http_failure_then_retry(self, lazy, path='/v1/chat/completions', with_child=False):
        if not lazy:
            self.script.write_text(READY, encoding='utf-8')
        engine = self.engine(lazy=lazy)
        cv, ctl = engine.slot_cv, engine.ctl
        if not lazy:
            engine.proc.kill()
            engine.proc.wait(timeout=3)
        failure = 'import time; time.sleep(600)'
        if with_child:
            failure = "import subprocess, time\nsubprocess.Popen(['sleep', '600'])\ntime.sleep(600)"
        self.script.write_text(failure, encoding='utf-8')
        svc = server.Service(engine, server.ByteTokenizer(), server.ChatTemplate(server.ROOT / 'serve/chat_template.jinja'))
        with mock.patch.object(svc, 'start_telemetry'):
            http = server.serve(svc, port=0)
        try:
            status, body = self.bounded(lambda: self.request(http, path))
            self.assertEqual(status, 503, body)
            self.assertIn('did not report READY', body['error']['message'])
            self.assertNotIn('malformed', body['error']['message'])
            self.assertFalse(engine.starting)
            self.assertFalse(svc.fifo.locked())
            self.assertIs(engine.slot_cv, cv)
            self.assertIs(engine.ctl, ctl)
            self.script.write_text(READY, encoding='utf-8')
            status, body = self.bounded(lambda: self.request(http))
            self.assertEqual(status, 200, body)
            self.assertEqual(body['choices'][0]['message']['content'], 'Z')
            self.assertTrue(engine.alive())
        finally:
            http.shutdown()
            http.server_close()

    def test_lazy_http_failure_has_real_reason_and_next_request_works(self):
        self.http_failure_then_retry(lazy=True)

    @unittest.skipUnless(os.name == 'posix' and shutil.which('sleep'), 'POSIX child process')
    def test_inherited_writer_http_failure_and_next_request_works(self):
        self.http_failure_then_retry(lazy=True, with_child=True)

    @unittest.skipUnless(os.name == 'posix', 'POSIX process groups')
    def test_process_group_is_signalled_only_once(self):
        engine = self.engine(lazy=True)
        engine.proc = mock.Mock(pid=123, returncode=None, _waitpid_lock=threading.Lock())
        engine._process_group = True
        try:
            def signal_group(pid, sig):
                self.assertFalse(engine.proc._waitpid_lock.acquire(blocking=False))

            with mock.patch.object(server.os, 'killpg', create=True, side_effect=signal_group) as killpg:
                engine._kill_process_group()
                engine._kill_process_group()
            killpg.assert_called_once_with(123, signal.SIGKILL)
        finally:
            engine.proc = None

    def test_missing_wait_lock_never_signals_process_group(self):
        engine = self.engine(lazy=True)
        engine.proc = mock.Mock(pid=123, returncode=None, _waitpid_lock=None)
        engine._process_group = True
        try:
            with mock.patch.object(server.os, 'killpg', create=True) as killpg:
                engine._kill_process_group()
            killpg.assert_not_called()
        finally:
            engine.proc = None

    def test_reaped_process_group_is_never_signalled(self):
        engine = self.engine(lazy=True)
        engine.proc = mock.Mock(pid=123, returncode=0, _waitpid_lock=threading.Lock())
        engine._process_group = True
        try:
            with mock.patch.object(server.os, 'killpg', create=True) as killpg:
                engine._kill_process_group()
            killpg.assert_not_called()
            self.assertFalse(engine._process_group)
        finally:
            engine.proc = None

    def test_explicit_load_failure_has_real_reason_and_next_request_works(self):
        self.http_failure_then_retry(lazy=True, path='/v1/load')

    def test_restart_after_ready_has_real_reason_and_next_request_works(self):
        self.http_failure_then_retry(lazy=False)

    def test_deferred_close_only_closes_old_stdout_after_reader_returns(self):
        # No process-group support, or a child outside it: close still must not wait for an inherited writer.
        # Cover EOF and READY arriving just after a timeout. The latter reader deliberately leaves stdout open.
        for late_ready in (False, True):
            with self.subTest(late_ready=late_ready):
                engine = self.engine(lazy=True)
                rfd, wfd = os.pipe()
                stdout, writer = os.fdopen(rfd), os.fdopen(wfd, 'w')
                proc = mock.Mock(stdin=io.StringIO(), stdout=stdout)
                proc.poll.return_value = 0
                reader = threading.Thread(target=engine._ready_pump, args=(proc, queue.Queue()), daemon=True)
                reader.start()
                engine.proc, engine.pump = proc, reader
                try:
                    self.assertIsNone(self.bounded(engine.close, limit=4))
                    self.assertFalse(stdout.closed)
                    successor = mock.Mock(stdout=io.StringIO())
                    engine.proc = successor
                    if late_ready:
                        writer.write('READY 4096\n')
                        writer.flush()
                    else:
                        writer.close()
                    reader.join(2)
                    deadline = time.monotonic() + 2
                    while not stdout.closed and time.monotonic() < deadline:
                        time.sleep(.01)
                    self.assertTrue(stdout.closed)
                    self.assertFalse(successor.stdout.closed)
                    successor.stdout.close()
                finally:
                    writer.close()
                    reader.join(2)
                    engine.proc = None


if __name__ == '__main__':
    unittest.main()
