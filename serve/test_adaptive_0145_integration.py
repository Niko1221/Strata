"""CPU-only checks for upstream .45 prompt/budget behavior under adaptive ownership.

These use a deterministic fake engine; they do not claim a model, GPU, or desktop run.
"""
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve import server
from serve.frontend import ChatTemplate
from serve.test_server import ThinkingEngine

ROOT = Path(__file__).resolve().parents[1]


class Adaptive0145Integration(unittest.TestCase):
    def service(self, max_context=65536):
        tok = server.ByteTokenizer()
        engine = ThinkingEngine(tok)
        engine.max_context = max_context
        service = server.Service(engine, tok, ChatTemplate(ROOT / 'serve/chat_template.jinja'))
        service.tool_leases.enabled = True
        return service

    def test_refused_context_forgets_prompt_and_releases_adaptive_reservation(self):
        service = self.service(max_context=4096)
        messages = [{'role': 'user', 'content': 'ordinary question'}]
        heartbeat = mock.Mock()
        service.request_owner.preparation_heartbeat = heartbeat
        # A real admission generator may yield while waiting. The reservation must
        # cover that wait, then be relinquished even if upstream context fitting refuses.
        def admission(cancel):
            self.assertEqual(service.preparing_requests, 1)
            yield 'ping', None
        with mock.patch.object(service, '_preparation_admission_enabled', return_value=True), \
                mock.patch.object(service, 'load_events', side_effect=admission):
            with self.assertRaisesRegex(ValueError, 'exceeds the context'):
                service.prepare(messages, None, {}, 65536)
        heartbeat.assert_called_once_with()
        self.assertEqual(service.prompts.entries, [])
        self.assertEqual(service.preparing_requests, 0)
        self.assertEqual(service.request_owner.depth, 0)
        # A refused request must not prevent a subsequent admitted turn or its cache entry.
        ids, _, _ = service.prepare(messages, None, {}, 16)
        self.assertIn(ids, [entry[2] for entry in service.prompts.entries])
        self.assertEqual(service.preparing_requests, 0)

    def budget_run(self, sampling, answer_room=True):
        service = self.service()
        service.answer_room = answer_room
        service.engine.THOUGHT = 'reason carefully ' * 600
        generate = service.engine.generate
        def owned_generation(*args, **kwargs):
            self.assertEqual(service.preparing_requests, 1)
            self.assertEqual(service.request_owner.depth, 1)
            yield from generate(*args, **kwargs)
        with mock.patch.object(service.engine, 'generate', side_effect=owned_generation):
            events = list(service.run([1], True, None, 8192, sampling, threading.Event()))
        self.assertEqual(service.preparing_requests, 0)
        self.assertEqual(service.request_owner.depth, 0)
        return service, events

    def test_answer_room_wraps_once_inside_existing_adaptive_owner(self):
        service, events = self.budget_run({})
        self.assertEqual(len(service.engine.prompts), 2)
        wrap = service.tok.encode(server.REASONING_WRAP_UP, parse_special=True)
        self.assertEqual(service.engine.prompts[1], [1] + service.tok.encode(service.engine.THOUGHT[:6144]) + wrap)
        self.assertEqual(events[-1][0], 'done')
        self.assertEqual(events[-1][1]['finish'], 'stop')

    def test_explicit_summary_budget_stays_1024_with_adaptive_ownership(self):
        service, events = self.budget_run({'reasoning_budget_tokens': 1024})
        self.assertEqual(len(service.engine.prompts), 2)
        wrap = service.tok.encode(server.REASONING_WRAP_UP, parse_special=True)
        self.assertEqual(service.engine.prompts[1], [1] + service.tok.encode(service.engine.THOUGHT[:1024]) + wrap)
        self.assertEqual(events[-1][1]['finish'], 'stop')

    def test_answer_room_opt_out_preserves_length_finish_and_releases_owner(self):
        service, events = self.budget_run({}, answer_room=False)
        self.assertEqual(len(service.engine.prompts), 1)
        self.assertEqual(events[-1][1]['finish'], 'length')


if __name__ == '__main__':
    unittest.main()
