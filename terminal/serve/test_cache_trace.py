import unittest
from .cache_trace import snapshot

class CacheTrace(unittest.TestCase):

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
