"""Fast CPU regressions for the rebased preemption driver (no GPU or HTTP socket)."""
import queue
import threading
import time
import unittest
from unittest.mock import Mock
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineRequest, EngineSilent, Service

ROOT = Path(__file__).resolve().parents[1]


class Pipe:
    can_stop = True
    silence_s = .01
    max_context = 10000
    can_preempt = True
    def __init__(self):
        self.proc = object()
        self.lines = queue.Queue()
        self.commands = []
        self.last = None
        self.info = {'preempt': 1}
    def sampling_keys(self, sampling):
        return ''
    def write_line(self, line):
        self.commands.append(line)
    def _silent(self, reason):
        return EngineSilent(reason)
    def _parse_done(self, line):
        f = line.split()
        self.last = dict(generated=int(f[1]), prompt_tokens=int(f[2]), prompt_ms=float(f[3]),
                         decode_ms=float(f[4]), finish=f[5])


def service(engine):
    return Service(engine, ByteTokenizer(), ChatTemplate(ROOT / 'serve/chat_template.jinja'), preempt=True)


class ProtocolRegression(unittest.TestCase):
    def test_silence_deadline_is_enforced(self):
        eng = Pipe()
        req = EngineRequest(eng, [1], 5, {}, 1)
        req.allow = .01
        req.heard = time.monotonic() - 1
        eng.lines.put('PP 1 2 10 100 id=1')   # baseline returns a heartbeat instead of enforcing deadline
        with self.assertRaises(EngineSilent):
            next(req)
        self.assertTrue(req._closed)

    def test_stop_drain_is_bounded(self):
        eng = Pipe()
        req = EngineRequest(eng, [1], 5, {}, 1)
        req.allow = .01
        class TimedQueue:
            def get(self, timeout=None):
                if timeout is None:
                    raise AssertionError('STOP drain used an unlimited wait')
                raise queue.Empty
        eng.lines = TimedQueue()
        with self.assertRaises(EngineSilent):
            req.close()
        self.assertEqual(eng.commands[-1], 'STOP')

    def test_resume_rejects_replacement_process(self):
        eng = Pipe()
        req = EngineRequest(eng, [1], 5, {}, 1)
        req.parked = True
        eng.proc = object()
        with self.assertRaises(ValueError):
            req.resume()
        self.assertFalse(any(x.startswith('RESUME') for x in eng.commands))

    def test_cancel_does_not_talk_to_replacement_process(self):
        eng = Pipe()
        req = EngineRequest(eng, [1], 5, {}, 1)
        req.parked = True
        eng.proc = object()
        eng.lines.put('DONE 0 1 0 0 cancel id=1')
        req.cancel_parked()
        self.assertFalse(any(x.startswith('CANCEL') for x in eng.commands))

    def test_done_is_owned_by_request(self):
        eng = Pipe()
        req = EngineRequest(eng, [1], 5, {}, 1)
        eng.lines.put('DONE 3 1 4 5 length id=1')
        self.assertEqual(list(req), [])
        eng.last = {'generated': 999}
        self.assertEqual(req.last['generated'], 3)


class ServiceRegression(unittest.TestCase):
    def test_cancelled_queue_never_submits_gen(self):
        eng = Pipe()
        eng.open_request = Mock(side_effect=AssertionError('cancelled request was submitted'))
        svc = service(eng)
        cancel = threading.Event()
        cancel.set()
        done = list(svc.run([1], False, None, 5, {}, cancel))[-1][1]
        self.assertEqual(done['finish'], 'cancel')
        eng.open_request.assert_not_called()
        self.assertEqual(svc.status['queued'], 0)

    def test_cancel_on_prefill_heartbeat_stops_without_reading_token(self):
        eng = Pipe()
        cancel = threading.Event()
        class Request:
            parked = False
            last = None
            calls = 0
            closed = False
            def __iter__(self): return self
            def __next__(self):
                self.calls += 1
                if self.calls == 1:
                    cancel.set()
                    return None
                return 42
            def close(self): self.closed = True
        req = Request()
        eng.open_request = lambda *args: req
        svc = service(eng)
        done = list(svc.run([1], False, None, 5, {}, cancel))[-1][1]
        self.assertEqual(req.calls, 1)
        self.assertTrue(req.closed)
        self.assertEqual(done['completion_tokens'], 0)

    def test_loading_uses_existing_admission_hooks(self):
        eng = Pipe()
        svc = service(eng)
        class AdmissionBlocked(Exception): pass
        svc.ensure_loaded = Mock(side_effect=AdmissionBlocked)
        eng.open_request = Mock(side_effect=AssertionError('bypassed ensure_loaded'))
        with self.assertRaises(AdmissionBlocked):
            list(svc.run([1], False, None, 5, {}, threading.Event()))
        svc.ensure_loaded.assert_called_once()
        eng.open_request.assert_not_called()

    def test_reasoning_budget_uses_budget_aware_path(self):
        eng = Pipe()
        svc = service(eng)
        svc.reasoning_budget_tokens = 20
        svc.run_preemptable = Mock(side_effect=AssertionError('budget would be ignored'))
        eng.generate = Mock(return_value=(x for x in []))
        list(svc.run([1], True, None, 100, {}, threading.Event()))
        svc.run_preemptable.assert_not_called()
        eng.generate.assert_called_once()

    def test_unload_refuses_a_parked_request(self):
        eng = Pipe()
        eng.alive = lambda: True
        eng.unload = Mock()
        svc = service(eng)
        svc.status.update(busy=False, queued=0, parked_requests=1)
        self.assertEqual(svc.unload(), 'busy')
        eng.unload.assert_not_called()

    def test_late_bookkeeping_does_not_clear_other_request(self):
        svc = service(Pipe())
        svc.status.update(busy=True, request_id=2, phase='reading B', tail='B', tool='tool B')
        svc._record_done(10, 0, 'cancel', {}, [], None, None, threading.Event(),
                         record=False, rid=1, request_last=None)
        self.assertTrue(svc.status['busy'])
        self.assertEqual(svc.status['tail'], 'B')


if __name__ == '__main__':
    unittest.main()
