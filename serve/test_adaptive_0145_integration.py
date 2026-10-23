"""CPU-only checks for upstream .45 prompt/budget behavior under adaptive ownership.

These use a deterministic fake engine; they do not claim a model, GPU, or desktop run.
"""
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
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
        service.fit_max_tokens = False  # Explicit rejection case; current upstream otherwise clamps.
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


class DeclaredOnlyReservation(unittest.TestCase):
    def service(self, outcome="complete"):
        svc = object.__new__(server.Service)
        svc.keep_awake = mock.Mock()
        svc.tool_leases = SimpleNamespace(enabled=True, demand=lambda: None, blocked=lambda: False)
        svc.request_owner = threading.local()
        svc.shutdown_event = threading.Event()
        svc.status_lock = threading.Lock()
        svc.preparing_requests = 0
        seen = []

        def inner(ids, thinking, tools, max_new, sampling, cancel, force, declared_only):
            seen.append((declared_only, max_new))
            self.assertEqual(svc.preparing_requests, 1)
            self.assertEqual(svc.request_owner.depth, 1)
            yield "piece", "fixture"
            if outcome == "error":
                raise RuntimeError("fixture inner failure")
            if outcome == "cancel":
                self.assertTrue(cancel.is_set())
                raise server.RequestParkCancelled()

        svc._run = inner
        return svc, seen

    def stream(self, svc, cancel):
        return svc.run([1], True, [{"name": "declared_fixture"}], 8192, {},
                       cancel, declared_only=True)

    def released(self, svc):
        self.assertEqual(svc.preparing_requests, 0)
        self.assertEqual(getattr(svc.request_owner, "depth", 0), 0)
        svc.keep_awake.acquire.assert_called_once_with()
        svc.keep_awake.release.assert_called_once_with()

    def test_completion_forwards_declared_only_and_releases(self):
        svc, seen = self.service()
        self.assertEqual(list(self.stream(svc, threading.Event())), [("piece", "fixture")])
        self.assertEqual(seen, [(True, 8192)])
        self.released(svc)

    def test_inner_error_releases(self):
        svc, seen = self.service("error")
        with self.assertRaisesRegex(RuntimeError, "fixture inner failure"):
            list(self.stream(svc, threading.Event()))
        self.assertEqual(seen, [(True, 8192)])
        self.released(svc)

    def test_cancel_after_admission_releases(self):
        svc, seen = self.service("cancel")
        cancel = threading.Event()
        stream = self.stream(svc, cancel)
        self.assertEqual(next(stream), ("piece", "fixture"))
        cancel.set()
        with self.assertRaises(server.RequestParkCancelled):
            next(stream)
        self.assertEqual(seen, [(True, 8192)])
        self.released(svc)

    def test_cancel_before_admission_does_not_release_unowned_reservation(self):
        svc, seen = self.service()
        cancel = threading.Event()
        cancel.set()
        with mock.patch.object(svc, "_release_preparation", wraps=svc._release_preparation) as release:
            with self.assertRaises(server.RequestParkCancelled):
                list(self.stream(svc, cancel))
            release.assert_not_called()
        self.assertEqual(seen, [])
        self.released(svc)

    def test_generator_close_releases(self):
        svc, seen = self.service()
        stream = self.stream(svc, threading.Event())
        next(stream)
        stream.close()
        self.assertEqual(seen, [(True, 8192)])
        self.released(svc)

    def test_nested_reservation_returns_only_its_own_depth(self):
        svc, _ = self.service()
        svc.request_owner.depth = 1
        svc.preparing_requests = 1

        def inner(*args):
            self.assertTrue(args[-1])
            self.assertEqual(svc.request_owner.depth, 2)
            self.assertEqual(svc.preparing_requests, 1)
            yield "piece", "nested"

        svc._run = inner
        self.assertEqual(list(self.stream(svc, threading.Event())), [("piece", "nested")])
        self.assertEqual(svc.request_owner.depth, 1)
        self.assertEqual(svc.preparing_requests, 1)
        svc.keep_awake.release.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
