"""Explicit engine phase must not confuse queued requests with prompt work."""
import io
import queue
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine
from serve.frontend import ChatTemplate


class PrefillPhase(unittest.TestCase):
    def test_pump_consumes_phase_and_keeps_protocol(self):
        engine = StrataEngine.__new__(StrataEngine)
        engine.proc = SimpleNamespace(stdout=io.StringIO('PFSTATE 1\nPP 10 20\nPFSTATE 0\nDONE 1\n'))
        engine.lines = queue.Queue()
        engine.slot_q = []
        engine.prefill_active = None
        engine._pump()
        self.assertFalse(engine.prefill_active)
        self.assertEqual(engine.prefill_epoch, 1)
        self.assertEqual(engine.lines.get_nowait(), 'PP 10 20\n')
        self.assertEqual(engine.lines.get_nowait(), 'DONE 1\n')
        self.assertIsNone(engine.lines.get_nowait())

    def test_queued_request_does_not_make_four_decoders_prefill(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, 'ok')
        engine.batch = 4
        engine.prefill_active = False
        engine.prefill_tok_s_mean = 2541
        engine.slots_view = lambda: [dict(slot=i, state='decoding') for i in range(4)]
        svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))
        svc.live_reqs = {i: (dict(busy=True, started=time.time(), first_token=None),) for i in range(5)}
        live = svc.metrics()['live']
        self.assertEqual(live['outside_slots'], 1)
        self.assertEqual(live['state'], 'generating')
        self.assertFalse(live['engine_prefill_active'])
        self.assertIsNone(live['prefill_tok_s_mean'])
        engine.prefill_active = True
        live = svc.metrics()['live']
        self.assertEqual(live['state'], 'reading')
        self.assertEqual(live['prefill_tok_s_mean'], 2541)
        engine.prefill_active = None
        self.assertIsNone(svc.metrics()['live']['engine_prefill_active'])


if __name__ == '__main__':
    unittest.main()
