"""Replica pipe ownership and caller-owned DONE statistics, without a model or GPU.

Use the repository's real StrataEngine reader/pump and ReplicaEngine. Only the
native producer is a deterministic protocol fixture. Events force a legal
handoff; the tests do not rely on hoping to win a scheduler race.
"""
import concurrent.futures
import queue
import subprocess
import sys
import threading
import types
import unittest
from unittest import mock
from serve.replicas import ReplicaEngine
from serve.server import StrataEngine, EngineNonFinite, EngineSilent
PROTOCOL_WORKER = """
import sys
for line in sys.stdin:
    if not line.startswith(("GEN ", "GENI ")):
        continue
    ids = [int(x) for x in line.split()[-1].split(",")]
    marker = ids[-1]
    count = 13 if marker % 2 else 12
    for token in list(range(marker * 100, marker * 100 + count - 1)) + [99]:
        print("T", token)
    print("DONE", count, len(ids), 2, 1, "stop", 0, 0, 0, 0, 0, 0, 0, 0, len(ids), flush=True)
"""

class Frame(str):

    def __new__(cls, text, owner):
        x = super().__new__(cls, text)
        x.owner = owner
        return x

class Probe:

    def __init__(self, enabled=True):
        self.events = []
        self.mu = threading.Lock()
        self.tl = threading.local()
        self.enabled = enabled
        self.specs = {}
        self.engines = []
        self.pending = {}
        self.overlaps = []
        self.wrong_readers = []

    def log(self, event, **kw):
        if self.enabled:
            with self.mu:
                self.events.append({'seq': len(self.events), 'thread': threading.current_thread().name, 'event': event, **kw})

    def add(self, name, marker, count=13, shared=True, mode='normal', delay=0, barrier=None, gate=None):
        ids = (list(range(4096)) if shared else [marker] * 64) + [marker]
        spec = {'name': name, 'ids': ids, 'tokens': list(range(marker * 100, marker * 100 + count - 1)) + [99], 'mode': mode, 'delay': delay, 'barrier': barrier, 'gate': gate}
        self.specs[marker] = spec
        return spec

    def engine(self):
        e = StrataEngine.__new__(StrataEngine)
        e.batch = 0
        e.last = {'generated': 77, 'prompt_ms': 7, 'decode_ms': 7}
        e.lines = Lines(self, len(self.engines))
        e.slot_q = []
        e.ended = False
        e.unloaded = False
        e.can_stop = True
        e.silence_s = 2
        e.alive = lambda: not e.ended
        e.sampling_keys = lambda s: ''
        e.exit_code = lambda: None
        e.info = {}

        def silent(message):
            e.ended = True
            return EngineSilent(message)
        e._silent = silent
        e.proc = types.SimpleNamespace(stdin=Writer(e, self, len(self.engines)))
        self.engines.append(e)
        return e

    def close(self):
        for e in self.engines:
            if hasattr(e, '_child'):
                e.proc.stdin.close()
                e.proc.wait(timeout=3)
                e.pump.join(1)
                e.proc.stdout.close()

class Lines(queue.Queue):

    def __init__(self, p, k):
        super().__init__()
        self.probe = p
        self.k = k

    def get(self, *args, **kw):
        value = super().get(*args, **kw)
        if isinstance(value, Frame):
            owner = getattr(self.probe.tl, 'request', None)
            if owner != value.owner:
                self.probe.wrong_readers.append((owner, value.owner, value.strip()))
            if value.startswith(('DONE', 'ERR')):
                self.probe.pending.pop(self.k, None)
            self.probe.log('read', engine=self.k, reader=owner, producer=value.owner, line=value.strip())
        return value

class Writer:

    def __init__(self, e, p, k):
        self.e = e
        self.p = p
        self.k = k

    def write(self, text):
        line = text.strip()
        self.p.log('write', engine=self.k, command=line.split()[0])
        if line.startswith(('GEN ', 'GENI ')):
            ids = [int(x) for x in line.split()[-1].split(',')]
            s = self.p.specs[ids[-1]]
            self.p.tl.request = s['name']
            if self.k in self.p.pending:
                self.p.overlaps.append((self.p.pending[self.k], s['name']))
            self.p.pending[self.k] = s['name']
            self.p.log('gen', engine=self.k, request=s['name'])
            self.e._last_spec = s
            if s['barrier']:
                s['barrier'].wait(timeout=2)
            if s['gate']:
                s['gate'][0].set()
                assert s['gate'][1].wait(3), 'fixture gate not released'
            if s['mode'] == 'error':
                self.e.lines.put(Frame('ERR fixture rejection\n', s['name']))
                return len(text)
            for token in s['tokens']:
                self.e.lines.put(Frame(f'T {token}\n', s['name']))
            if s['mode'] != 'missing-done':
                done = f"DONE {len(s['tokens'])} {len(ids)} 2 1 {('nonfinite' if s['mode'] == 'nonfinite' else 'stop')} 0 0 0 0 0 0 0 0 {len(ids)}\n"
                if s['delay']:
                    timer = threading.Timer(s['delay'], lambda: self.e.lines.put(Frame(done, s['name'])))
                    timer.daemon = True
                    timer.start()
                else:
                    self.e.lines.put(Frame(done, s['name']))
        return len(text)

    def flush(self):
        pass

class YieldAfterUnlock:
    """Force a legal scheduler interleaving; do not alter native reader logic."""

    def __init__(self):
        self.lock = threading.Lock()
        self.released = threading.Event()
        self.b_done = threading.Event()
        self.once = False

    def acquire(self, *a, **kw):
        return self.lock.acquire(*a, **kw)

    def release(self):
        self.lock.release()
        if threading.current_thread().name == 'A' and (not self.once):
            self.once = True
            self.released.set()
            assert self.b_done.wait(3), 'B did not finish in forced schedule'

def consume(rep, spec, stop_after=None, cancel=None):
    before = rep.last
    g = rep.generate(spec['ids'], 128, {}, cancel or threading.Event())
    out = []
    try:
        for token in g:
            if token is None:
                continue
            out.append(token)
            if token == 99 or (stop_after is not None and len(out) >= stop_after):
                break
    finally:
        g.close()
    return {'tokens': out, 'last': dict(rep.last), 'new_done': rep.last is not before}

def pipe_engine(p):
    """Repository _pump/generate, real OS pipes and an explicit CPU-only producer."""
    e = p.engine()
    e.lines = queue.Queue()
    e.proc = subprocess.Popen([sys.executable, '-u', '-c', PROTOCOL_WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    e._child = True
    e.pump = threading.Thread(target=e._pump, daemon=True)
    e.pump.start()
    return e

class ReplicaContracts(unittest.TestCase):

    def setUp(self):
        self.stall = mock.patch('serve.server.ENGINE_STALL_S', 0)
        self.stall.start()
        self.addCleanup(self.stall.stop)
        self.p = Probe()
        self.rep = ReplicaEngine([self.p.engine(), self.p.engine()])
        self.rep.heal = lambda: None

    def tearDown(self):
        shutdown = getattr(type(self.rep), 'shutdown', None)
        if shutdown:
            shutdown(self.rep)
        self.p.close()

    def pair(self, stop_after=None, delay=0):
        a = self.p.add('A', 101, 13, delay=delay)
        b = self.p.add('B', 202, 12)
        switch = YieldAfterUnlock()
        self.rep.run_locks[0] = switch
        values = {}
        errors = []

        def first():
            try:
                before = self.rep.last
                g = self.rep.generate(a['ids'], 128, {}, threading.Event())
                out = []
                for token in g:
                    if token is None:
                        continue
                    out.append(token)
                    if token == 99 or (stop_after is not None and len(out) >= stop_after):
                        break
                g.close()
                values['A'] = {'tokens': out, 'last': dict(self.rep.last), 'new_done': self.rep.last is not before}
            except BaseException as err:
                errors.append(err)
                switch.released.set()
                switch.b_done.set()

        def second():
            try:
                self.assertTrue(switch.released.wait(3))
                values['B'] = consume(self.rep, b)
            except BaseException as err:
                errors.append(err)
            finally:
                switch.b_done.set()
        threads = [threading.Thread(target=first, name='A', daemon=True), threading.Thread(target=second, name='B', daemon=True)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
            self.assertFalse(t.is_alive(), 'contract replay hung')
        if errors:
            raise errors[0]
        return (a, b, values)

    def test_successor_receives_its_own_tokens_after_eos(self):
        a, b, r = self.pair()
        self.assertEqual(r['A']['tokens'], a['tokens'])
        self.assertEqual(r['B']['tokens'], b['tokens'])

    def test_done_statistics_remain_request_owned_after_unlock(self):
        a, b, r = self.pair()
        self.assertEqual(r['A']['last']['generated'], len(a['tokens']))
        self.assertEqual(r['B']['last']['generated'], len(b['tokens']))

    def test_no_generation_starts_before_previous_ack(self):
        self.pair()
        self.assertEqual(self.p.overlaps, [])

    def test_no_control_or_token_frame_crosses_requests(self):
        self.pair()
        self.assertEqual(self.p.wrong_readers, [])

    def test_client_disconnect_drains_before_handoff(self):
        a, b, r = self.pair(stop_after=1)
        self.assertEqual(len(r['A']['tokens']), 1)
        self.assertEqual(r['B']['tokens'], b['tokens'])
        self.assertEqual(self.p.overlaps, [])

    def test_stop_string_like_early_close_drains_before_handoff(self):
        a, b, r = self.pair(stop_after=3)
        self.assertEqual(len(r['A']['tokens']), 3)
        self.assertEqual(r['B']['tokens'], b['tokens'])
        self.assertEqual(self.p.wrong_readers, [])

    def test_slow_consumer_can_close_a_large_buffered_reply(self):
        a = self.p.add('slow', 101, 512)
        g = self.rep.generate(a['ids'], 512, {}, threading.Event())
        self.assertEqual(next(g), a['tokens'][0])
        g.close()
        b = self.p.add('next', 202, 12)
        self.assertEqual(consume(self.rep, b)['tokens'], b['tokens'])
        self.assertEqual(self.rep.inflight, [0, 0])
        self.assertEqual(self.p.overlaps, [])

    def test_caller_exception_closes_child_before_reusing_replica(self):
        a = self.p.add('throw', 101, 13)
        g = self.rep.generate(a['ids'], 128, {}, threading.Event())
        next(g)
        with self.assertRaisesRegex(RuntimeError, 'caller left'):
            g.throw(RuntimeError('caller left'))
        b = self.p.add('next', 202, 12)
        self.assertEqual(consume(self.rep, b)['tokens'], b['tokens'])
        self.assertEqual(self.rep.inflight, [0, 0])
        self.assertEqual(self.p.overlaps, [])

    def test_delayed_done_is_drained_without_fixed_sleep_workaround(self):
        a, b, r = self.pair(delay=0.02)
        self.assertEqual(r['B']['tokens'], b['tokens'])
        self.assertEqual(self.p.overlaps, [])

    def test_natural_exhaustion_accounts_for_exact_tokens(self):
        a = self.p.add('A', 101, 13)
        before = self.rep.last
        out = list(self.rep.generate(a['ids'], 128, {}, threading.Event()))
        self.assertEqual(out, a['tokens'])
        self.assertEqual(self.rep.last['generated'], 13)
        self.assertIsNot(self.rep.last, before)
        self.assertEqual(self.rep.inflight, [0, 0])

    def test_rejection_does_not_inherit_previous_done(self):
        a = self.p.add('bad', 101, 13, mode='error')
        before = self.rep.last
        with self.assertRaises(ValueError):
            list(self.rep.generate(a['ids'], 128, {}, threading.Event()))
        self.assertIs(self.rep.last, before)
        self.assertEqual(self.rep.inflight, [0, 0])
        b = self.p.add('good', 202, 12)
        self.assertEqual(consume(self.rep, b)['tokens'], b['tokens'])

    def test_cancel_while_queued_sends_no_new_gen_or_stale_done(self):
        started = threading.Event()
        release = threading.Event()
        a = self.p.add('A', 101, 13, gate=(started, release))
        b = self.p.add('B', 202, 12)
        errors = []

        def first():
            try:
                consume(self.rep, a)
            except BaseException as err:
                errors.append(err)
        t = threading.Thread(target=first, daemon=True)
        t.start()
        try:
            self.assertTrue(started.wait(2))
            cancel = threading.Event()
            cancel.set()
            before = self.rep.last
            out = list(self.rep.generate(b['ids'], 128, {}, cancel))
            self.assertEqual(out, [])
            self.assertIs(self.rep.last, before)
            self.assertFalse(any((x.get('event') == 'gen' and x.get('request') == 'B' for x in self.p.events)))
        finally:
            release.set()
            t.join(3)
        self.assertFalse(t.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(self.rep.inflight, [0, 0])

    def test_drain_failure_is_error_and_bad_engine_is_not_reused(self):
        a = self.p.add('A', 101, 13, mode='missing-done')
        self.p.engines[0].silence_s = 0.02
        with self.assertRaises(EngineSilent):
            consume(self.rep, a)
        self.assertFalse(self.p.engines[0].alive())
        self.assertEqual(self.rep.inflight, [0, 0])
        b = self.p.add('B', 202, 12, shared=False)
        self.assertEqual(consume(self.rep, b)['tokens'], b['tokens'])
        self.assertEqual(sum((x.get('event') == 'gen' and x.get('engine') == 0 for x in self.p.events)), 1)

    def test_nonfinite_forgets_advertised_prefix(self):
        a = self.p.add('A', 101, 13, mode='nonfinite')
        with self.assertRaises(EngineNonFinite):
            list(self.rep.generate(a['ids'], 128, {}, threading.Event()))
        self.assertEqual(self.rep.router.matched(a['ids'])[0], 0)
        self.assertEqual(self.rep.inflight, [0, 0])

    def test_two_replicas_make_progress_independently(self):
        barrier = threading.Barrier(2)
        a = self.p.add('A', 101, 13, shared=False, barrier=barrier)
        b = self.p.add('B', 202, 12, shared=False, barrier=barrier)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            rows = list(pool.map(lambda s: consume(self.rep, s), (a, b)))
        self.assertEqual([x['tokens'] for x in rows], [a['tokens'], b['tokens']])
        self.assertEqual(self.p.overlaps, [])

    def test_four_callers_many_turns_no_cross_output_or_stats(self):
        specs = [self.p.add('R' + str(i), 1000 + i, 9 + i % 7) for i in range(80)]
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            rows = list(pool.map(lambda s: consume(self.rep, s), specs))
        self.assertEqual([r['tokens'] for r in rows], [s['tokens'] for s in specs])
        self.assertEqual([r['last']['generated'] for r in rows], [len(s['tokens']) for s in specs])
        self.assertEqual(self.rep.inflight, [0, 0])
        self.assertEqual(self.p.overlaps, [])
        self.assertEqual(self.p.wrong_readers, [])

class RealPipeContracts(unittest.TestCase):

    def test_real_pipe_queued_successor_keeps_output_and_statistics(self):
        self.p = Probe()
        self.rep = ReplicaEngine([pipe_engine(self.p), pipe_engine(self.p)])
        self.rep.heal = lambda: None
        try:
            a, b, r = ReplicaContracts.pair(self)
            self.assertEqual(r['A']['tokens'], a['tokens'])
            self.assertEqual(r['B']['tokens'], b['tokens'])
            self.assertEqual(r['A']['last']['generated'], 13)
            self.assertEqual(r['B']['last']['generated'], 12)
        finally:
            shutdown = getattr(type(self.rep), 'shutdown', None)
            if shutdown:
                shutdown(self.rep)
            self.p.close()

    def test_native_pump_and_reader_with_real_os_pipes(self):
        p = Probe()
        e0 = pipe_engine(p)
        e1 = pipe_engine(p)
        rep = ReplicaEngine([e0, e1])
        rep.heal = lambda: None
        try:
            for i in range(12):
                s = p.add('P' + str(i), 101 if i % 2 == 0 else 202, 13 if i % 2 == 0 else 12, shared=False)
                r = consume(rep, s)
                self.assertEqual(r['tokens'], s['tokens'])
                self.assertEqual(r['last']['generated'], len(s['tokens']))
            self.assertEqual(rep.inflight, [0, 0])
        finally:
            shutdown = getattr(type(rep), 'shutdown', None)
            if shutdown:
                shutdown(rep)
            p.close()
if __name__ == '__main__':
    unittest.main(verbosity=2)
