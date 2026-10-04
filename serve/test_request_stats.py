"""Deterministic CPU-only concurrency regressions: no model, GPU, or production port."""
import json
from pathlib import Path
import queue
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch
import urllib.request

from serve.frontend import ChatTemplate
from serve.request_stats import GenerationStats, ObservedGeneration, current_stats
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve


def done(n=1, prompt=100, reused=80, pms=20, dms=2, finish='length'):
    return (f'DONE {n} {prompt} {pms} {dms} {finish} 0 0 {reused} '
            f'9 10 2 1 1.5 {prompt-reused}')


def parsed(line):
    e = SimpleNamespace()
    StrataEngine._parse_done(e, line)
    return e.last


class ProtocolCounters(unittest.TestCase):
    def test_live_queue_wait_is_counted_before_admission_finishes(self):
        stats = GenerationStats(100)
        with patch('serve.request_stats.time.monotonic') as clock:
            clock.return_value = 10.0
            start = stats.wait('waiting_control')
            clock.return_value = 12.0
            self.assertEqual(stats.view()['control_wait_ms'], 2000)
            stats.waited('control_wait_ms', start)
            clock.return_value = 13.0
            self.assertEqual(stats.view()['control_wait_ms'], 2000)

    def test_admission_is_not_a_complete_batch_request(self):
        stats = GenerationStats(100)
        stats.start('admit', 100)
        stats.assign_slot(3, 7)
        stats.done(parsed(done()))
        self.assertEqual(stats.result(), {})
        stats.admitted(True)
        self.assertEqual(stats.result(), {})
        stats.batch_done('BDONE 3 12 length 200')
        r = stats.result()
        self.assertEqual((r['generated'], r['decode_ms'], r['prompt_ms'], r['reused']), (12, 202, 20, 80))
        self.assertNotIn('hits', r)  # GEN-1 expert statistics are not full batch statistics
        self.assertEqual(r['decode_scope'], 'engine_segments_including_batch_stalls')
        self.assertEqual(stats.view()['slot_generation'], 7)

    def test_promotion_counts_each_segment_without_double_counting_input(self):
        stats = GenerationStats(100)
        stats.start('solo', 100)
        stats.done(parsed(done(n=3, reused=50, pms=10, dms=6, finish='cancel')))
        stats.start('admit', 103)
        stats.assign_slot(0, 1)
        stats.done(parsed(done(prompt=103, reused=103, pms=2, dms=1)))
        stats.admitted(True)
        stats.batch_done('BDONE 0 4 stop 30')
        r = stats.result()
        self.assertEqual((r['generated'], r['decode_ms'], r['prompt_ms']), (7, 37, 12))
        self.assertEqual((r['prompt_tokens'], r['reused'], r['prompt_read']), (100, 50, 50))
        self.assertEqual(r['engine_prompt_read_total'], 50)
        self.assertEqual(r['finish'], 'stop')

    def test_error_after_a_completed_segment_is_not_a_complete_request(self):
        stats = GenerationStats(100)
        stats.start('solo', 100)
        stats.done(parsed(done(n=3, finish='cancel')))
        stats.start('admit', 103)  # ERR; no DONE for this admission
        self.assertEqual(stats.result(), {})
        self.assertFalse(stats.view()['engine_complete'])

    def test_late_drain_is_bound_to_its_original_part(self):
        stats = GenerationStats(100)
        stats.start('admit', 100)
        stats.assign_slot(0, 1)
        stats.done(parsed(done()))
        stats.admitted(True)
        old_drain = stats.drain_callback()
        stats.start('admit', 104)
        stats.assign_slot(1, 1)
        stats.done(parsed(done(prompt=104, reused=104)))
        stats.admitted(True)
        old_drain('BDONE 0 4 cancel 40')
        self.assertEqual(stats.view()['engine_phase'], 'batch')
        self.assertEqual(stats.result(), {})
        stats.batch_done('BDONE 1 3 length 30')
        snapshot = stats.result()
        self.assertEqual(snapshot['generated'], 7)
        old_drain('BDONE 0 4 cancel 40')
        self.assertEqual(snapshot, stats.result())

    def test_two_generators_interleaved_on_one_thread_do_not_share_context(self):
        e = SimpleNamespace(last={}, batch=2)
        def gen(prompt):
            try:
                yield prompt
            finally:
                StrataEngine._parse_done(e, done(prompt=prompt, reused=0))
        a, b = GenerationStats(10), GenerationStats(20)
        ga = ObservedGeneration(e, gen(10), a)
        gb = ObservedGeneration(e, gen(20), b)
        self.assertEqual(next(ga), 10)
        self.assertIsNone(current_stats(e))
        self.assertEqual(next(gb), 20)
        gb.close()
        ga.close()
        self.assertEqual((a.result()['prompt_tokens'], b.result()['prompt_tokens']), (10, 20))
        self.assertIsNone(current_stats(e))

    def test_unknown_batch_backend_does_not_use_global_last(self):
        e = SimpleNamespace(last={}, batch=2)
        def gen():
            yield 1
            e.last = parsed(done(prompt=999, reused=900))
        stats = GenerationStats(10)
        self.assertEqual(list(ObservedGeneration(e, gen(), stats)), [1])
        self.assertEqual(stats.result(), {})


def batch_engine():
    e = StrataEngine.__new__(StrataEngine)
    e.batch, e.max_context = 2, 4096
    e.slot_order, e.slot_busy, e.slot_generation = [0, 1], [True, False], [0, 0]
    e.slot_q = [queue.Queue(), queue.Queue()]
    e.slot_cv, e.ctl = threading.Condition(), threading.Lock()
    e.waiting, e.lines, e.last = 0, queue.Queue(), {}
    e.alive = lambda: True
    return e


class NativeBatchProtocol(unittest.TestCase):
    def test_interleaved_native_admissions_keep_own_done_and_slot_generation(self):
        e = batch_engine()
        def send(command):
            if not command.startswith('BGEN '):
                return
            _, slot, max_new, ids = command.split()
            slot, count = int(slot), int(max_new)
            prompt = len(ids.split(','))
            for line in [f'T {prompt}', done(prompt=prompt, reused=prompt//2), f'BADM {slot} {int(count > 1)}']:
                e.lines.put(line)
            if count > 1:
                for i in range(count-1): e.slot_q[slot].put(f'BT {slot} {prompt+i+1}')
                e.slot_q[slot].put(f'BDONE {slot} {count} length {count*10}')
        e._send = send
        def request(n, count):
            stats = GenerationStats(n)
            return ObservedGeneration(e, e.generate_batched([7]*n, count, {}, threading.Event()), stats), stats
        a, sa = request(10, 3)
        self.assertEqual((next(a), next(a)), (10, 11))
        e.slot_busy[0] = False  # the preexisting request finishes
        b, sb = request(20, 4)
        self.assertEqual((next(b), next(b)), (20, 21))
        self.assertEqual(list(a), [12])
        self.assertEqual(sa.result()['generated'], 3)
        self.assertEqual(sa.result()['reused'], 5)
        self.assertEqual(list(b), [22, 23])
        self.assertEqual(sb.result()['generated'], 4)
        self.assertEqual(sb.result()['reused'], 10)
        self.assertEqual(e.slot_busy, [False, False])
        c, sc = request(30, 1)
        self.assertEqual(list(c), [30])
        self.assertEqual(sc.view()['slot_id'], 0)
        self.assertEqual(sc.view()['slot_generation'], 2)
        self.assertEqual(sc.result()['generated'], 1)

    def test_eos_close_consumes_buffered_bdone_without_losing_counters(self):
        e = batch_engine()
        def send(command):
            if command.startswith('BGEN '):
                for line in ['T 10', done(prompt=10, reused=5), 'BADM 1 1']:
                    e.lines.put(line)
                e.slot_q[1].put('BT 1 42')
                e.slot_q[1].put('BDONE 1 2 stop 30')
        e._send = send
        stats = GenerationStats(10)
        gen = ObservedGeneration(e, e.generate_batched([7]*10, 9, {}, threading.Event()), stats)
        self.assertEqual((next(gen), next(gen)), (10, 42))
        gen.close()
        self.assertEqual(stats.result()['generated'], 2)
        self.assertEqual(stats.result()['finish'], 'stop')
        self.assertFalse(e.slot_busy[1])

    def test_copy_failure_after_done_drains_badm_but_not_next_request(self):
        e = batch_engine()
        e._send = Mock()
        for line in ['T 10', done(prompt=10, reused=5), 'ERR slot copy failed', 'BADM 1 0']:
            e.lines.put(line)
        stats = GenerationStats(10)
        gen = ObservedGeneration(e, e.generate_batched([7]*10, 9, {}, threading.Event()), stats)
        self.assertEqual(next(gen), 10)
        with self.assertRaisesRegex(ValueError, 'copy failed'):
            next(gen)
        self.assertTrue(e.lines.empty())
        self.assertFalse(e.ctl.locked())
        self.assertEqual(e.slot_busy, [True, False])
        for line in ['T 20', done(prompt=20, reused=10), 'BADM 1 0']:
            e.lines.put(line)
        other = GenerationStats(20)
        self.assertEqual(list(ObservedGeneration(e, e.generate_batched([7]*20, 1, {}, threading.Event()), other)), [20])
        self.assertEqual(other.result()['reused'], 10)

    def test_rejected_admission_has_no_previous_counters(self):
        e = batch_engine()
        e.last = parsed(done(prompt=999, reused=900))
        e._send = lambda command: e.lines.put('ERR invalid request')
        stats = GenerationStats(10)
        with self.assertRaises(ValueError):
            list(ObservedGeneration(e, e.generate_batched([7]*10, 9, {}, threading.Event()), stats))
        self.assertEqual(stats.result(), {})
        self.assertFalse(e.ctl.locked())
        self.assertEqual(e.slot_busy, [True, False])


class GatedEngine:
    """Independent workers; protocol DONE has distinct input/clock values per request."""
    batch, max_context = 8, 4096
    def __init__(self, tok, sizes, fail=None):
        self.token = tok.encode('x')[0]
        self.entered = {n: threading.Event() for n in sizes}
        self.release = {n: threading.Event() for n in sizes}
        self.last, self.fail = {}, fail
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        n = len(ids)
        yield self.token
        self.entered[n].set()
        if not self.release[n].wait(5): raise TimeoutError('test gate was not released')
        if n == self.fail: raise ValueError('injected rejection')
        if cancel.is_set(): return
        yield self.token
        StrataEngine._parse_done(self, done(n=2, prompt=n, reused=n//2, pms=n, dms=n*2))


class ConcurrentService(unittest.TestCase):
    def setup_service(self, sizes=(10, 20), fail=None):
        tok = ByteTokenizer()
        e = GatedEngine(tok, sizes, fail)
        svc = Service(e, tok, ChatTemplate(Path(__file__).parent/'chat_template.jinja'))
        return svc, e

    def test_finishing_one_request_does_not_clear_other_status_or_counters(self):
        svc, e = self.setup_service()
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(list, svc.run([1]*10, False, None, 2, {}, threading.Event()))
            b = pool.submit(list, svc.run([1]*20, False, None, 2, {}, threading.Event()))
            try:
                for n in (10, 20): self.assertTrue(e.entered[n].wait(3))
                self.assertEqual(svc.v1_status()['activity']['in_flight'], 2)
                self.assertEqual(len(svc.metrics()['active_requests']), 2)
                self.assertEqual({r['engine_phase'] for r in svc.metrics()['active_requests']}, {'decode'})
                self.assertIsNone(svc._prefill_tok_s_mean())
                e.release[10].set()
                a.result(3)
                m = svc.metrics()
                self.assertEqual(m['live']['state'], 'generating')
                self.assertEqual(m['live']['in_flight'], 1)
                self.assertEqual(m['live']['generated'], 1)
                self.assertEqual(m['requests'][0]['prompt_tokens'], 10)
                e.release[20].set()
                b.result(3)
            finally:
                for gate in e.release.values(): gate.set()
        rows = {r['prompt_tokens']: r for r in svc.metrics()['requests']}
        self.assertEqual(set(rows), {10, 20})
        self.assertEqual((rows[10]['reused'], rows[20]['reused']), (5, 10))
        self.assertEqual((rows[10]['decode_ms'], rows[20]['decode_ms']), (20, 40))
        self.assertEqual(svc.totals['output_tokens'], 4)
        self.assertEqual(svc.totals['prompt_tokens'], 30)
        self.assertFalse(svc.status['busy'])

    def test_seven_way_reverse_completion_keeps_all_requests(self):
        sizes = [10*n for n in range(1, 8)]
        svc, e = self.setup_service(sizes)
        with ThreadPoolExecutor(max_workers=7) as pool:
            jobs = {n: pool.submit(list, svc.run([1]*n, False, None, 2, {}, threading.Event())) for n in sizes}
            try:
                for n in sizes: self.assertTrue(e.entered[n].wait(3))
                self.assertEqual(svc.v1_status()['activity']['in_flight'], 7)
                for i, n in enumerate(reversed(sizes)):
                    e.release[n].set()
                    jobs[n].result(3)
                    self.assertEqual(svc.v1_status()['activity']['in_flight'], 6-i)
            finally:
                for gate in e.release.values(): gate.set()
        rows = svc.metrics()['requests']
        self.assertEqual(len(rows), 7)
        self.assertEqual(len({r['id'] for r in rows}), 7)
        self.assertEqual(svc.totals['prompt_tokens'], sum(sizes))
        self.assertEqual(svc.totals['reused'], sum(sizes)//2)
        self.assertEqual(svc.totals['output_tokens'], 14)
        for r in rows:
            self.assertEqual(r['decode_ms'], 2*r['prompt_tokens'])
            self.assertNotIn('tail', r)

    def test_error_does_not_clear_another_live_request(self):
        svc, e = self.setup_service(fail=10)
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(list, svc.run([1]*10, False, None, 2, {}, threading.Event()))
            b = pool.submit(list, svc.run([1]*20, False, None, 2, {}, threading.Event()))
            try:
                for n in (10, 20): self.assertTrue(e.entered[n].wait(3))
                e.release[10].set()
                with self.assertRaises(ValueError): a.result(3)
                row = svc.metrics()['requests'][0]
                self.assertEqual(row['finish'], 'error')
                self.assertIsNone(row['engine_generated'])
                self.assertTrue(svc.status['busy'])
                e.release[20].set()
                b.result(3)
            finally:
                for gate in e.release.values(): gate.set()
        self.assertEqual(svc.totals['requests'], 2)
        self.assertEqual(svc.v1_status()['activity']['in_flight'], 0)

    def test_cancellation_does_not_steal_another_requests_done(self):
        svc, e = self.setup_service()
        cancel = threading.Event()
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(list, svc.run([1]*10, False, None, 2, {}, cancel))
            b = pool.submit(list, svc.run([1]*20, False, None, 2, {}, threading.Event()))
            try:
                for n in (10, 20): self.assertTrue(e.entered[n].wait(3))
                cancel.set()
                e.release[10].set()
                a.result(3)
                self.assertEqual(svc.metrics()['requests'][0]['finish'], 'cancel')
                self.assertTrue(svc.status['busy'])
                e.release[20].set()
                b.result(3)
            finally:
                for gate in e.release.values(): gate.set()
        self.assertEqual(svc.totals['output_tokens'], 3)


class RateAndLifecycle(unittest.TestCase):
    def test_rate_has_multiple_buckets_and_decays_to_zero_during_a_stall(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, 'x'), tok, ChatTemplate(Path(__file__).parent/'chat_template.jinja'))
        stats = GenerationStats(10)
        stats.phase = 'prefill'
        state = {'id': 'a', 'request_id': 'a', 'started': time.time(), 'clock': 100.0, 'queued': False,
                 'first_token': None, 'generated': 0, 'max_token_gap_s': 0.0, 'stats': stats,
                 'phase': 'answering', 'prompt_tokens': 10, 'max_tokens': 100}
        svc.active_requests['a'] = state
        with patch('serve.server.time.monotonic') as clock:
            for i in range(80):
                clock.return_value = 100.0 + i*0.02
                svc._note(state, i+1, [])
            self.assertGreater(len(svc.rate), 20)
            self.assertLess(len(svc.rate), 40)
            self.assertAlmostEqual(svc._tok_s(), 50, delta=3)
            clock.return_value = 105.0
            self.assertEqual(svc._tok_s(), 0)

    def test_unload_refuses_a_slot_still_draining_after_disconnect(self):
        tok = ByteTokenizer()
        e = batch_engine()
        e.unload = Mock()
        svc = Service(e, tok, ChatTemplate(Path(__file__).parent/'chat_template.jinja'))
        self.assertFalse(svc.status['busy'])
        self.assertEqual(svc.metrics()['live']['state'], 'draining')
        self.assertEqual(svc.unload(), 'busy')
        e.unload.assert_not_called()

    def test_old_drain_does_not_free_restarted_engines_slot(self):
        e = batch_engine()
        e._send = Mock()
        e.slot_busy[1] = True
        old_queue, old_busy = e.slot_q[1], e.slot_busy
        e._release_slot_when_done(1)
        e.slot_q = [queue.Queue(), queue.Queue()]
        e.slot_busy = [False, True]
        e.slot_cv = threading.Condition()
        old_queue.put('BDONE 1 2 cancel 10')
        deadline = time.monotonic() + 2
        while old_busy[1] and time.monotonic() < deadline: time.sleep(0.001)
        self.assertFalse(old_busy[1])
        self.assertTrue(e.slot_busy[1])


class HttpCorrelation(unittest.TestCase):
    def test_id_is_correlated_without_capturing_content_in_both_dialects(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, 'ok', max_context=4096), tok,
                      ChatTemplate(Path(__file__).parent/'chat_template.jinja'))
        httpd = serve(svc, port=0)
        try:
            ids = []
            for path in ('/v1/chat/completions', '/v1/messages'):
                for stream in (False, True):
                    body = {'messages': [{'role': 'user', 'content': 'DO_NOT_RETAIN_THIS_INPUT'}],
                            'max_tokens': 10, 'stream': stream, 'reasoning_effort': 'none'}
                    req = urllib.request.Request(f'http://127.0.0.1:{httpd.server_address[1]}{path}',
                        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(req, timeout=5) as r:
                        rid = r.headers.get('X-Strata-Request-ID')
                        self.assertIsNotNone(rid)
                        self.assertEqual(len(rid), 32)
                        r.read()
                    ids.append(rid)
                    self.assertEqual(svc.metrics()['requests'][0]['request_id'], rid)
                    self.assertNotIn('DO_NOT_RETAIN_THIS_INPUT', json.dumps(svc.metrics()))
            self.assertEqual(len(set(ids)), 4)
            self.assertEqual(len(svc.api_requests), 0)
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == '__main__':
    unittest.main()
