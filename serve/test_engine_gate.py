"""CPU scheduling regressions: overdue resumes own the next period, never the active one."""
import math
import threading
import time
import unittest
from serve.engine_gate import EngineGate
from serve.test_preempt import make_service, drive


def wait_for_waiters(gate, n):
    end = time.monotonic() + 2
    with gate._cv:
        while len(gate._waiters) < n:
            if time.monotonic() >= end:
                raise AssertionError("waiter did not enter the gate")
            gate._cv.wait(.005)


class GateTests(unittest.TestCase):
    def test_fifo_and_overdue_priority_without_interrupting_active_owner(self):
        gate, order = EngineGate(), []
        gate.acquire()
        def work(name, priority=False):
            with gate.period(priority=priority): order.append(name)
        threads = []
        for i, (name, priority) in enumerate([('B', False), ('C', False), ('A', True)]):
            t = threading.Thread(target=work, args=(name, priority))
            t.start(); threads.append(t)
            wait_for_waiters(gate, i + 1)
        self.assertEqual(order, [])
        self.assertFalse(gate.acquire(blocking=False))
        gate.release()
        for t in threads: t.join(2); self.assertFalse(t.is_alive())
        self.assertEqual(order, ['A', 'B', 'C'])

    def test_timed_out_waiter_does_not_block_following_owner(self):
        gate = EngineGate()
        with gate:
            result = []
            t = threading.Thread(target=lambda: result.append(gate.acquire(timeout=.01)))
            t.start(); t.join(1)
            self.assertEqual(result, [False])
        self.assertTrue(gate.acquire(blocking=False))
        gate.release()

    def test_exception_releases_period(self):
        gate = EngineGate()
        with self.assertRaises(ValueError):
            with gate.period(): raise ValueError('test')
        self.assertTrue(gate.acquire(blocking=False))
        gate.release()

    def test_service_resumes_before_backlog_after_deadline(self):
        svc, eng = make_service()
        svc.preempt_max_wait_s = .03
        commands, errors = [], []
        original = eng.write_line
        def write(line):
            commands.append(line)
            original(line)
        eng.write_line = write
        def work(ids):
            try: drive(svc, ids)
            except Exception as e: errors.append(e)
        svc.fifo.acquire()
        threads = []
        try:
            for i, ids in enumerate([[20] * 80, [0] * 4, [0] * 4]):
                t = threading.Thread(target=work, args=(ids,))
                t.start(); threads.append(t)
                wait_for_waiters(svc.fifo, i + 1)
        finally: svc.fifo.release()
        try:
            for t in threads: t.join(5); self.assertFalse(t.is_alive())
            self.assertEqual(errors, [])
            resume = commands.index('RESUME id=1')
            b = next(i for i, line in enumerate(commands) if line.startswith('GEN ') and 'id=2 ' in line)
            c = next(i for i, line in enumerate(commands) if line.startswith('GEN ') and 'id=3 ' in line)
            self.assertLess(b, resume)
            self.assertLess(resume, c)
            self.assertEqual(svc.status['parked_requests'], 0)
        finally:
            eng.closed = True

    def test_disabled_mode_keeps_existing_lock(self):
        svc, eng = make_service(preempt=False)
        self.assertNotIsInstance(svc.fifo, EngineGate)
        eng.closed = True

    def test_invalid_wait_limits_rejected(self):
        from serve.server import Service
        from serve.test_preempt_regressions import Pipe
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, ROOT
        for value in [-1, math.inf, math.nan]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                Service(Pipe(), ByteTokenizer(), ChatTemplate(ROOT / 'serve/chat_template.jinja'),
                        preempt=True, preempt_max_wait_s=value)


if __name__ == '__main__': unittest.main()
