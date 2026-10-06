"""#1059: a native ERR ends a command without DONE / BADM, including with batch slots enabled.

Real pipes and HTTP, a fake native process, no GPU.  The rejection is gated until a second request is queued,
so returning the first error is not enough: the control lock, reserved slot and the next request must recover.
The production drain timeout is unchanged.  A regression is bounded by the test's joins and engine cleanup.

    python -m unittest serve.test_engine_errors -v
"""
import contextlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

from serve.server import ByteTokenizer, ChatTemplate, Service, StrataEngine, serve

ERROR = "a token id is outside the vocabulary"
FAKE = r'''
import sys, threading, time
from pathlib import Path
args = sys.argv[1:]
root = Path(args[args.index('--root') + 1])
slots = int(args[args.index('--batch') + 1])
output_lock = threading.Lock()
def out(text):
    with output_lock: print(text, flush=True)
def active(slot):
    while not (root / 'rejected').exists(): time.sleep(.005)
    for _ in range(4):
        out(f'BT {slot} 65')
        time.sleep(.005)
    out(f'BDONE {slot} 5 stop 1')
out('INFO batch_slots=' + str(slots) + '\nREADY 4096 stop')
for raw in sys.stdin:
    f = raw.strip().split()
    if not f: continue
    with (root / 'commands').open('a') as log:
        log.write(f[0] + '\n')
    if f[0] == 'QUIT': break
    if f[0] == 'STOP': continue  # native stdin reader sets stop_req; idle STOP has no response
    if f[0] not in ('GEN', 'BGEN'): continue
    ids = [int(t) for t in f[-1].split(',')]
    text = bytes(t for t in ids if 0 <= t < 256).decode('utf-8', 'replace')
    if 'REJECT_REQUEST' in text:
        (root / 'waiting').touch()
        while not (root / 'release').exists(): time.sleep(.005)
        out('ERR a token id is outside the vocabulary')
        (root / 'rejected').touch()
        continue  # generate.cpp's validation rejection: no DONE, and no BADM for a BGEN
    if 'KEEP_SLOT' in text:
        out(f'T 65\nDONE 1 1 0 1 length\nBADM {f[1]} 1')
        threading.Thread(target=active, args=(f[1],), daemon=True).start()
        continue
    if 'FATAL_REQUEST' in text:
        out('ERR verify: timed out at layer 18\nDONE 0 1 0 1 error')
    else:
        malformed = {'MALFORMED_TOKEN': 'T malformed', 'MALFORMED_PP': 'PP 1 2 invalid 3',
                     'MALFORMED_DONE': 'DONE invalid 1 0 1 stop'}
        out(malformed.get(text, 'T 90') + '\nDONE 1 1 0 1 stop')
    if f[0] == 'BGEN': out('BADM ' + f[1] + ' 0')
'''


class EngineErrors(unittest.TestCase):
    @contextlib.contextmanager
    def running(self, slots):
        with tempfile.TemporaryDirectory() as tmp:
            self.root = Path(tmp)
            script = self.root / 'native.py'
            script.write_text(FAKE, encoding='utf-8')
            real = subprocess.Popen
            with mock.patch('serve.server.subprocess.Popen', lambda cmd, **kw:
                            real([sys.executable, str(script), *cmd[1:]], **kw)):
                self.engine = StrataEngine('fake-native', ['--batch', str(slots), '--root', tmp])
            self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))
            self.http = serve(self.svc, port=0)
            self.base = f'http://127.0.0.1:{self.http.server_address[1]}'
            self.threads = []
            self.drains = mock.Mock(wraps=self.engine._drain_control)
            self.engine._drain_control = self.drains
            try:
                yield
            finally:
                self.engine.close()  # also releases a regression's 300 s wait before shutting down the HTTP server
                for th in self.threads:
                    th.join(3)
                self.http.shutdown()
                self.http.server_close()

    def commands(self):
        path = self.root / 'commands'
        return path.read_text().splitlines() if path.exists() else []

    def wait_for(self, predicate):
        end = time.monotonic() + 3
        while not predicate() and time.monotonic() < end:
            time.sleep(.005)
        self.assertTrue(predicate())

    def clean(self):
        e = self.engine
        self.assertFalse(e.ctl.locked())
        self.assertEqual((e.waiting, e.wait_lens), (0, []))
        self.assertFalse(any(e.slot_busy))
        self.assertTrue(all(x is None for x in e.slot_live))
        self.assertTrue(e.lines.empty())

    def request(self, api, text, stream, limit, out, key):
        data = {'model': 'test', 'stream': stream}
        if api == 'responses':
            data.update(input=text, max_output_tokens=limit, reasoning={'effort': 'none'})
            path = '/v1/responses'
        else:
            data.update(messages=[{'role': 'user', 'content': text}], max_tokens=limit,
                        reasoning_effort='none', thinking={'type': 'disabled'})
            path = '/v1/messages' if api == 'anthropic' else '/v1/chat/completions'
        req = urllib.request.Request(self.base + path, json.dumps(data).encode(), {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                out[key] = (response.status, response.read().decode())
        except urllib.error.HTTPError as exc:
            out[key] = (exc.code, exc.read().decode())
        except Exception as exc:  # reported by the assertions, including socket timeouts
            out[key] = ('exception', repr(exc))

    def start_request(self, *args):
        th = threading.Thread(target=self.request, args=args, daemon=True)
        self.threads.append(th)
        th.start()

    def http_matrix(self, api):
        for slots, limit in ((0, 8), (2, 8), (2, 1)):  # serial, batch solo, BGEN admission
            for stream in (False, True):
                with self.subTest(slots=slots, limit=limit, stream=stream), self.running(slots):
                    out = {}
                    self.start_request(api, 'REJECT_REQUEST', stream, limit, out, 'bad')
                    self.wait_for(lambda: (self.root / 'waiting').exists())
                    self.start_request(api, 'valid request', stream, limit, out, 'next')
                    if slots:
                        self.wait_for(lambda: self.engine.waiting == 1)
                    start = time.monotonic()
                    (self.root / 'release').touch()
                    deadline = start + 2
                    for th in self.threads:
                        th.join(max(0, deadline - time.monotonic()))
                    self.assertFalse(any(th.is_alive() for th in self.threads), 'ERR waited for nonexistent DONE/BADM')
                    self.assertLess(time.monotonic() - start, 2)
                    self.assertIn(ERROR, out['bad'][1])
                    self.assertEqual(out['bad'][0], 200 if stream else 500 if api == 'responses' else 400)
                    self.assertEqual(out['next'][0], 200, out['next'])
                    self.assertNotIn(ERROR, out['next'][1])
                    self.assertIn('Z', out['next'][1])
                    self.drains.assert_not_called()
                    command = 'BGEN' if slots and limit == 1 else 'GEN'
                    self.assertEqual(self.commands(), [command, command])  # no stale STOP
                    self.assertTrue(self.engine.alive())
                    self.clean()

    def test_chat_http(self):
        self.http_matrix('chat')

    def test_anthropic_http(self):
        self.http_matrix('anthropic')

    def test_responses_http(self):
        self.http_matrix('responses')

    def test_malformed_protocol_still_drains(self):
        for limit in (8, 1):
            for malformed in (b'MALFORMED_TOKEN', b'MALFORMED_PP', b'MALFORMED_DONE'):
                with self.subTest(limit=limit, malformed=malformed), self.running(2):
                    with self.assertRaises(ValueError):
                        list(self.engine.generate(list(malformed), limit, {}, threading.Event()))
                    self.drains.assert_called_once_with('DONE' if limit > 1 else 'BADM', born=self.engine.gen)
                    self.assertEqual(list(self.engine.generate([1], limit, {}, threading.Event())), [90])
                    self.clean()

    def test_consumer_close_still_drains(self):
        for limit in (8, 1):
            with self.subTest(limit=limit), self.running(2):
                gen = self.engine.generate([1], limit, {}, threading.Event())
                self.assertEqual(next(gen), 90)
                gen.close()
                self.drains.assert_called_once_with('DONE' if limit > 1 else 'BADM', born=self.engine.gen)
                self.assertEqual(list(self.engine.generate([1], limit, {}, threading.Event())), [90])
                self.clean()

    def test_fatal_error_still_invalidates_engine_and_suppresses_trailing_lines(self):
        from serve.server import EngineDied
        for limit in (8, 1):
            with self.subTest(limit=limit), self.running(2):
                with self.assertRaises(EngineDied):
                    list(self.engine.generate(list(b'FATAL_REQUEST'), limit, {}, threading.Event()))
                self.assertFalse(self.engine.alive())
                self.drains.assert_not_called()
                self.clean()
                for q in self.engine.slot_q:
                    self.assertIsNone(q.get_nowait())
                    self.assertTrue(q.empty())

    def test_rejected_admission_preserves_another_active_slot(self):
        with self.running(2):
            e, result = self.engine, {}
            (self.root / 'release').touch()
            e.slot_busy[1] = True  # put the first request in BGEN rather than the alone/GEN path
            first = e.generate(list(b'KEEP_SLOT'), 8, {}, threading.Event())
            self.assertEqual(next(first), 65)
            e.slot_busy[1] = False

            def consume(key, gen):
                try:
                    result[key] = list(gen)
                except Exception as exc:
                    result[key] = exc

            th = threading.Thread(target=consume, args=('first', first), daemon=True)
            self.threads.append(th)
            th.start()
            self.wait_for(lambda: not e.ctl.locked())
            bad = e.generate(list(b'REJECT_REQUEST'), 8, {}, threading.Event())
            th = threading.Thread(target=consume, args=('bad', bad), daemon=True)
            self.threads.append(th)
            th.start()
            deadline = time.monotonic() + 2
            for th in self.threads:
                th.join(max(0, deadline - time.monotonic()))
            self.assertFalse(any(th.is_alive() for th in self.threads))
            self.assertIsInstance(result['bad'], ValueError)
            self.assertIn(ERROR, str(result['bad']))
            self.assertEqual(result['first'], [65] * 4)
            self.assertEqual(list(e.generate([1], 8, {}, threading.Event())), [90])
            self.assertEqual(self.commands(), ['BGEN', 'BGEN', 'GEN'])
            self.drains.assert_not_called()
            self.clean()


if __name__ == '__main__':
    unittest.main()
