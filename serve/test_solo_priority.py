"""Single-engine preemption tests against the real pipe reader and streaming parser.

python -m unittest serve.test_solo_priority -v
"""
import concurrent.futures
import json
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from pathlib import Path

from serve.server import ByteTokenizer, Service, StrataEngine
from serve.frontend import ChatTemplate
from serve.solo_priority import SoloScheduler, enable

FAKE = r'''
import queue, sys, threading, time
lines, stop = queue.Queue(), threading.Event()
def reader():
    for line in sys.stdin:
        if line.strip() == 'STOP': stop.set()
        else: lines.put(line.strip())
    lines.put('QUIT')
threading.Thread(target=reader, daemon=True).start()
print('INFO engine=0.1.44.1', flush=True)
print('READY 262144 stop', flush=True)
cache = []
while True:
    line = lines.get()
    if line == 'QUIT': break
    if not line.startswith('GEN '): continue
    f = line.split(); limit = int(f[1]); ids = list(map(int, f[-1].split(',')))
    stop.clear()
    if ids[0] == 3:
        print('ERR injected engine failure', flush=True)
        sys.exit(3)
    marker = ids.index(999)
    base = ids[:marker + 1]; used = len(ids) - len(base)
    if ids[0] == 1:
        reply = list(('中文🙂' * 10 + '\n<tool_call>\n<function=edit>\n<parameter=text>\n' +
                      '海岛' * 80 + '\n</parameter>\n<parameter=enabled>\nfalse\n</parameter>\n</function>\n</tool_call>').encode()) + [257]
    else: reply = list(b'short OK') + [257]
    reused = max([0] + [len(x) for x in cache if ids[:len(x)] == x])
    if len(ids) > 300 and ids[1] == 78:
        reused = max(128, reused)  # A native cache hit with an unfinished suffix.
    print(f'RESUME {reused}', flush=True)
    read = reused
    while read < len(ids) - 1:
        read = min(read + 32, len(ids) - 1)
        time.sleep(.001)
        print(f'PP {read} {len(ids)} 1 1000', flush=True)
        if stop.is_set(): break
    n = 0; finish = 'cancel'
    if not stop.is_set():
        for token in reply[used:used+limit]:
            time.sleep(.001)
            print(f'T {token}', flush=True); n += 1
            if token == 257: finish = 'stop'; break
            if stop.is_set():
                # Real speculative windows can leave accepted tokens after STOP.
                for extra in reply[used+n:used+n+2]:
                    print(f'T {extra}', flush=True); n += 1
                break
        else: finish = 'length'
    cache.append((ids + reply[used:used+n])[:-1] if n else ids[:read])
    print(f'DONE {n} {len(ids)} 1 {max(n, 1)} {finish} 0 0 {reused} 0 0 0 0 0 {max(0, read-reused)}', flush=True)
'''


class Scheduling(unittest.TestCase):
    def test_config_off_does_not_replace_lock_or_engine(self):
        class S: pass
        svc = S(); svc.fifo = threading.Lock(); svc.engine = object()
        original = svc.fifo
        enable(svc, {})
        self.assertIs(svc.fifo, original)
        self.assertFalse(hasattr(svc, 'solo_scheduler'))

    def test_invalid_config_is_rejected(self):
        class S: pass
        svc = S(); svc.engine = object()
        for cfg in ({'solo_priority': 'yes'}, {'solo_priority': True}):
            with self.assertRaises(ValueError): enable(svc, cfg)

    def test_waiting_age_and_minimum_run_time(self):
        s = SoloScheduler(quantum_s=0, age_s=30)
        with s.request([1], 10000, {}, threading.Event()):
            s.acquire()
            owner = s.owner
            from serve.solo_priority import Ticket
            short = Ticket(priority=100)
            s.requests.append(short); s.waiters.append(short)
            self.assertTrue(s.should_pause())
            s.quantum_s = 5
            self.assertFalse(s.should_pause())
            owner.since -= 6
            owner.pauses = 10                  # no per-request cap that strands the next small request
            owner.arrived -= 100              # running time must not age the owner above every waiter
            self.assertTrue(s.should_pause())
            short.priority = -10
            self.assertFalse(s.should_pause())
            short.queued_at -= 31             # a low-priority request cannot wait forever either
            self.assertTrue(s.should_pause())
            s.requests.remove(short); s.waiters.remove(short)


class Streaming(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        p = Path(self.tmp.name) / 'fake.py'; p.write_text(FAKE, encoding='utf-8')
        import subprocess
        with mock.patch('serve.server.popen', side_effect=lambda what, args, **kw:
                        subprocess.Popen([sys.executable, '-u', str(p)], **kw)):
            self.eng = StrataEngine('fake', [])
        self.addCleanup(self.eng.close)
        self.tok = ByteTokenizer()
        self.svc = Service(self.eng, self.tok, ChatTemplate(Path(__file__).with_name('chat_template.jinja')))
        enable(self.svc, {'solo_priority': True, 'args': ['--conversation-cache-mib','32768']})
        self.sched = self.svc.solo_scheduler
        self.sched.quantum_s = .01
        self.pool = concurrent.futures.ThreadPoolExecutor(3)
        self.addCleanup(self.pool.shutdown, True)
        self.tools = [{'name':'edit','parameters':{'type':'object','properties':{'text':{'type':'string'},'enabled':{'type':'boolean'}}}}]

    def run_request(self, marker=1, cancel=None, length=2, priority=0, on_event=None, fill=77):
        evs = []
        ids = [marker] + [fill] * (length-2) + [999]
        for ev in self.svc.run(ids, False, self.tools if marker==1 else None,
                               2000 if marker==1 else 20, {'priority':priority}, cancel or threading.Event()):
            evs.append(ev)
            if on_event: on_event(ev)
        return evs

    def wait_for(self, predicate):
        end = time.monotonic()+8
        while not predicate():
            if time.monotonic()>end: self.fail('timed out')
            time.sleep(.002)

    def test_restored_partial_prefill_reaches_a_token_before_yielding(self):
        stops = []
        send = self.eng._send
        def observe(line):
            if line == 'STOP':
                stops.append(self.sched.owner.state.get('generated', 0))
            return send(line)
        with mock.patch.object(self.eng, '_send', side_effect=observe):
            long = self.pool.submit(self.run_request, length=4096, fill=78)
            self.wait_for(lambda: self.eng.reused == 128 and self.eng.progress is not None)
            short = self.pool.submit(self.run_request, 2, None, 2, 5)
            short.result(8)
            long.result(8)
        self.assertGreater(self.sched.preemptions, 0)
        self.assertTrue(stops)
        self.assertTrue(all(count > 0 for count in stops), stops)

    def test_preemption_keeps_utf8_tool_arguments_and_usage(self):
        baseline = self.run_request()
        long = self.pool.submit(self.run_request)
        self.wait_for(lambda: self.eng.progress is not None and self.sched.locked())
        short = self.pool.submit(self.run_request, 2, None, 2, 5)
        short.result(8)
        self.assertFalse(long.done())
        resumed = long.result(8)
        def content(evs):
            return ''.join(e.text or '' for k,e in evs if k=='event' and e.kind=='content')
        def calls(evs):
            return [e.call for k,e in evs if k=='event' and e.kind=='tool_call']
        self.assertEqual(content(resumed), content(baseline))
        self.assertEqual(len(calls(resumed)), 1)
        self.assertEqual(calls(resumed)[0].arguments, calls(baseline)[0].arguments)
        self.assertEqual(resumed[-1][1]['completion_tokens'], baseline[-1][1]['completion_tokens'])
        self.assertEqual(self.sched.preemptions, 1)
        self.assertLess(self.sched.resume_read, 4)
        self.assertEqual(self.svc.status['queued'], 0)
        self.assertFalse(self.svc.status['busy'])
        self.assertFalse(self.sched.locked())
        self.assertEqual(len(self.svc.history), 3)

    def test_cancel_while_paused_does_not_stop_the_other_request(self):
        cancel = threading.Event()
        long = self.pool.submit(self.run_request, 1, cancel)
        self.wait_for(lambda: self.sched.locked())
        short = self.pool.submit(self.run_request, 2, None, 6000, 5)
        self.wait_for(lambda: self.sched.preemptions == 1)
        cancel.set()
        long.result(8); result = short.result(8)
        self.assertEqual(result[-1][1]['finish'], 'stop')
        self.assertFalse(self.sched.locked())
        self.assertEqual(self.sched.waiters, [])
        self.assertEqual(self.svc.status['queued'], 0)

    def test_cold_prefill_yields_and_resumes(self):
        long = self.pool.submit(self.run_request, 1, None, 6000)
        self.wait_for(lambda: self.eng.progress is not None)
        short = self.pool.submit(self.run_request, 2, None, 2, 5)
        self.assertEqual(short.result(8)[-1][1]['finish'], 'stop')
        self.assertFalse(long.done())
        self.assertEqual(long.result(8)[-1][1]['finish'], 'stop')
        self.assertEqual(self.sched.preemptions, 1)

    def test_equal_priority_does_not_preempt(self):
        long = self.pool.submit(self.run_request)
        self.wait_for(lambda: self.sched.locked())
        short = self.pool.submit(self.run_request, 2)
        long.result(8); short.result(8)
        self.assertEqual(self.sched.preemptions, 0)

    def test_third_short_request_does_not_wait_for_long_request_to_finish(self):
        long = self.pool.submit(self.run_request)
        self.wait_for(lambda: self.sched.locked())
        for _ in range(3):
            # A short queued before the long thread reacquires may run immediately,
            # correctly requiring no additional pause. Exercise three actual resumes.
            self.wait_for(lambda: getattr(self.sched.owner, 'priority', None) == 0)
            self.pool.submit(self.run_request, 2, None, 2, 5).result(8)
            self.assertFalse(long.done())
        self.assertEqual(self.sched.preemptions, 3)
        self.assertEqual(long.result(8)[-1][1]['finish'], 'stop')

    def test_preempt_inside_a_tool_argument_emits_one_call(self):
        started = threading.Event()
        def observe(ev):
            if ev[0]=='event' and ev[1].kind=='tool_args': started.set()
        long = self.pool.submit(self.run_request, on_event=observe)
        self.assertTrue(started.wait(8))
        self.pool.submit(self.run_request, 2, None, 2, 5).result(8)
        events = long.result(8)
        calls = [e.call for k,e in events if k=='event' and e.kind=='tool_call']
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].arguments, {'text':'海岛'*80, 'enabled':False})
        args = ''.join(e.text or '' for k,e in events if k=='event' and e.kind=='tool_args')
        self.assertEqual(json.loads(args), calls[0].arguments)
        self.assertEqual(self.sched.preemptions, 1)

    def test_engine_death_while_paused_releases_all_waiters(self):
        long = self.pool.submit(self.run_request)
        self.wait_for(lambda: self.sched.locked())
        bad = self.pool.submit(self.run_request, 3, None, 2, 5)
        with self.assertRaises(ValueError): bad.result(8)
        from serve.server import EngineDied
        with self.assertRaises(EngineDied): long.result(8)
        self.assertFalse(self.sched.locked())
        self.assertEqual(self.sched.requests, [])
        self.assertEqual(self.sched.waiters, [])

    def test_parked_failure_uses_the_launching_server_exception_class(self):
        # `python serve/server.py` and `import serve.server` create distinct
        # EngineDied classes. Loading a second module reproduces that identity
        # split without starting an HTTP listener or changing this test runner.
        import importlib.util
        import subprocess
        source = Path(__file__).with_name('server.py')
        spec = importlib.util.spec_from_file_location('solo_server_launch_test', source)
        launched = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {spec.name: launched}):
            spec.loader.exec_module(launched)
            from serve.server import EngineDied
            self.assertIsNot(launched.EngineDied, EngineDied)
            with mock.patch.object(launched, 'popen', side_effect=lambda what, args, **kw:
                                   subprocess.Popen([sys.executable, '-u', str(Path(self.tmp.name) / 'fake.py')], **kw)):
                engine = launched.StrataEngine('fake', [])
            self.addCleanup(engine.close)
            single = engine._generate_single
            def finish_exit(ids, *args, **kwargs):
                try:
                    yield from single(ids, *args, **kwargs)
                finally:
                    if ids[0] == 3:
                        # Finish the process exit before releasing this turn,
                        # so the parked request takes the scheduler's error path
                        # rather than racing into a broken native pipe.
                        engine.proc.wait(timeout=5)
            engine._generate_single = finish_exit
            service = launched.Service(engine, launched.ByteTokenizer(),
                                       ChatTemplate(Path(__file__).with_name('chat_template.jinja')))
            enable(service, {'solo_priority': True, 'args': ['--conversation-cache-mib', '32768']})
            service.solo_scheduler.quantum_s = .01
            def run(marker, priority):
                return list(service.run([marker, 999], False, self.tools if marker == 1 else None,
                                        2000, {'priority': priority}, threading.Event()))
            parked = self.pool.submit(run, 1, 0)
            self.wait_for(lambda: service.solo_scheduler.locked())
            failed = self.pool.submit(run, 3, 5)
            with self.assertRaises(ValueError):
                failed.result(8)
            with self.assertRaises(launched.EngineDied):
                parked.result(8)
            self.assertEqual(service.solo_scheduler.requests, [])
            self.assertEqual(service.solo_scheduler.waiters, [])

    def test_close_generator_while_paused_does_not_release_other_owner(self):
        closed = threading.Event()
        def run():
            g = self.svc.run([1,999],False,self.tools,2000,{'priority':0},threading.Event())
            try:
                for k,_ in g:
                    if k=='ping' and self.sched.preemptions:
                        g.close();closed.set();break
            finally: g.close()
        long = self.pool.submit(run)
        self.wait_for(lambda: self.sched.locked())
        short = self.pool.submit(self.run_request, 2, None, 60000, 5)
        self.assertTrue(closed.wait(8))
        long.result(8)
        self.assertEqual(short.result(8)[-1][1]['finish'], 'stop')
        self.assertFalse(self.sched.locked())
        self.assertEqual(self.sched.waiters, [])


if __name__ == '__main__': unittest.main()
