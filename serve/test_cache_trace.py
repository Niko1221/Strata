import unittest
from .cache_trace import snapshot

class CacheTrace(unittest.TestCase):
    def test_prefix_boundary_busy_and_expired(self):
        result=snapshot([1,2,3,4], [[1,2],[1,7],[1,2,3,4,5],[]],
                        [False,True,False,False],[True,True,False,True],[0,99,99,0],100)
        self.assertEqual([r['common_prefix_tokens'] for r in result['slots']], [2,1,4,0])
        self.assertEqual([r['full_prefix_match'] for r in result['slots']], [True,False,False,False])
        self.assertTrue(result['slots'][0]['expired'])
        self.assertTrue(result['slots'][1]['busy'])
        self.assertFalse(result['slots'][2]['valid_hint'])
        self.assertNotIn('prompt', result)
    def test_matches_equal_prefixes_at_every_boundary(self):
        held=list(range(300))
        for i in range(301):
            prompt=held[:i]+[-1]
            result=snapshot(prompt,[held],[False],[True],[0],100)
            self.assertEqual(result['slots'][0]['common_prefix_tokens'],i)

    def test_resume_after_main_source_is_actual_reuse(self):
        import queue, threading
        from .server import StrataEngine
        engine=object.__new__(StrataEngine)
        engine.lines=queue.Queue()
        engine._ctl_mode='solo'
        engine._tl=threading.local()
        engine._tl.request_key='request-a'
        engine._parse_done=lambda line: None
        for line in ['CACHE_SOURCE -1 900', 'RESUME 700', 'DONE']:
            engine.lines.put(line)
        list(engine._control(threading.Event(), lambda token: None))
        self.assertEqual(engine.solo_cache_reused_tokens,700)
        self.assertIsNone(engine.solo_cache_source_slot)
        self.assertEqual(engine.cache_events_view()[0]['request_key'],'request-a')
        self.assertEqual(engine.cache_events_view()[0]['reused_tokens'],700)
    def test_global_expiry_does_not_inherit_request(self):
        import threading
        from .server import StrataEngine
        engine=object.__new__(StrataEngine)
        engine._tl=threading.local()
        engine._tl.request_key='request-a'
        engine.cache_event('expired',slot=0)
        self.assertNotIn('request_key',engine.cache_events_view()[0])
