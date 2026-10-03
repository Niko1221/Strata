"""Production routing boundaries with optional providers enabled, without a GPU."""
import http.client
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from serve.frontend import ChatTemplate
from serve.providers import ProviderManager
from serve.server import ByteTokenizer, MockEngine, Service, make_handler

ROOT = Path(__file__).resolve().parents[1]


class RoutingBoundaries(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tokenizer = ByteTokenizer()
        self.svc = Service(MockEngine(tokenizer, 'fixture', 4096), tokenizer,
                           ChatTemplate(ROOT / 'serve/chat_template.jinja'))
        self.svc.history_directory = Path(self.tmp.name) / 'history'
        self.manager = self.svc.providers = ProviderManager(Path(self.tmp.name) / 'providers.json')
        self.manager.profiles['fixture'] = {'id': 'fixture', 'name': 'Fixture',
            'base_url': 'http://127.0.0.1:9/v1', 'model': 'fixture-model',
            'context': 4096, 'images': False, 'backend': 'generic', 'reasoning_map': {}}
        # Inherit production dispatch unchanged; a dispatching test wrapper can hide admission leaks.
        self.finished = threading.Event()
        fixture = self
        class Handler(make_handler(self.svc)):
            def finish(self):
                try:
                    super().finish()
                finally:
                    fixture.finished.set()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.tmp.cleanup()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        data = json.dumps(body).encode() if body is not None else b'' if method == 'POST' else None
        try:
            connection.request(method, path, data, {'Content-Type': 'application/json', **(headers or {})})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_rejected_native_load_releases_provider_admission(self):
        for headers, expected in (({'Content-Type': 'text/plain'}, 415),
                                  ({'Origin': 'https://foreign.example'}, 403)):
            with self.subTest(headers=headers):
                self.assertEqual(self.request('POST', '/load', headers=headers)[0], expected)
                self.assertEqual(self.manager.native_inflight, 0)
        with mock.patch.object(self.manager, '_probe'):
            code, result = self.request('POST', '/api/providers/select', {'id': 'fixture'})
        self.assertEqual(code, 200)
        self.assertEqual(result['current'], 'fixture')

    def test_foreign_page_cannot_reach_selected_provider(self):
        self.manager.current = 'fixture'
        with mock.patch('serve.providers._chat') as chat, mock.patch('serve.providers._compact') as compact:
            for path in ('/v1/chat/completions', '/v1/chat/completions/count_tokens', '/v1/chat/compact'):
                self.assertEqual(self.request('POST', path, headers={'Origin': 'https://foreign.example'})[0], 403)
            chat.assert_not_called()
            compact.assert_not_called()

    def test_cors_chat_permission_does_not_grant_history_or_model_controls(self):
        self.svc.cors_origins = ['https://client.example']
        self.manager.current = 'fixture'
        headers = {'Origin': 'https://client.example'}
        def reply(handler, *args):
            handler._json(200, {'choices': [{'message': {'content': 'fixture response'}}]})
        with mock.patch('serve.providers._chat', side_effect=reply) as chat:
            self.assertEqual(self.request('POST', '/v1/chat/completions', {'messages': []}, headers)[0], 200)
            chat.assert_called_once()
        self.assertEqual(self.request('GET', '/api/history', headers=headers)[0], 403)
        self.assertEqual(self.request('POST', '/api/providers/select', {'id': None}, headers)[0], 403)
        self.assertEqual(self.request('POST', '/v1/chat/compact', {'messages': []}, headers)[0], 403)
        self.assertEqual(self.manager.current, 'fixture')

    def test_host_guard_precedes_new_routes(self):
        headers = {'Host': 'rebound.example'}
        for method, path in (('GET', '/api/history'), ('GET', '/api/providers'),
                             ('POST', '/api/providers/select')):
            self.assertEqual(self.request(method, path, headers=headers)[0], 403)
        self.assertIsNone(self.svc.chat_history)
        self.assertIsNone(self.manager.current)

    def test_authentication_precedes_history_provider_and_control_routes(self):
        self.svc.api_key = 'unit-test-fixture-key'
        for method, path in (('GET', '/api/history'), ('GET', '/api/providers'),
                             ('POST', '/api/providers/select')):
            self.assertEqual(self.request(method, path)[0], 401)
        self.assertIsNone(self.svc.chat_history)
        self.assertIsNone(self.manager.current)
        self.assertEqual(self.manager.native_inflight, 0)

    def test_incomplete_unauthenticated_body_does_not_hold_handler_open(self):
        self.svc.api_key = 'unit-test-fixture-key'
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=2)
        try:
            connection.putrequest('POST', '/api/providers/select')
            connection.putheader('Content-Type', 'application/json')
            connection.putheader('Content-Length', '1024')
            connection.endheaders()  # Deliberately never send the declared body.
            response = connection.getresponse()
            self.assertEqual(response.status, 401)
            response.read()
            self.assertTrue(self.finished.wait(1), 'Unauthorized body drain did not finish within its bound')
        finally:
            connection.close()


if __name__ == '__main__':
    unittest.main()
