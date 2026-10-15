"""Request inspection through HTTP; no native model or GPU is required."""
import json
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, MockEngine, Service, serve


class ReloadableEngine(MockEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.running, self.unloaded = True, False

    def alive(self):
        return self.running

    def unload(self):
        self.running, self.unloaded = False, True

    def restart(self):
        self.running, self.unloaded = True, False


class MonitorOff(unittest.TestCase):
    """#332 is opt-in: by default no prompt or answer is kept, and the monitor's endpoints do not exist."""

    def test_off_by_default(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "Hello.", max_context=4096), tok,
                      ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            req = urllib.request.Request(base + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"messages": [{"role": "user", "content": "secret"}]}).encode())
            with urllib.request.urlopen(req, timeout=10) as r:
                self.assertEqual(r.status, 200)
            self.assertEqual(len(svc.api_requests), 0)
            for path in ("/api/requests", "/api-monitor", "/api/history"):
                with self.assertRaises(urllib.error.HTTPError) as e:
                    urllib.request.urlopen(base + path, timeout=10)
                self.assertEqual(e.exception.code, 404, path)
                e.exception.close()
        finally:
            httpd.shutdown()
            httpd.server_close()


class Monitor(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = ReloadableEngine(self.tok, "Thought.</think>\nHello <script>alert(1)</script>.", max_context=16384)
        self.svc = Service(self.engine, self.tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc.api_monitor = True                       # opt-in ("api_monitor": true / --api-monitor)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            raw = response.read().decode()
            return response.status, json.loads(raw) if "application/json" in response.headers.get("Content-Type", "") else raw

    def chat(self, api="openai", **kwargs):
        # Anthropic thinks only when asked (#278): "thinking" there, reasoning_effort for OpenAI
        think = {"thinking": {"type": "enabled", "budget_tokens": 1024}} if api == "anthropic" else             {"reasoning_effort": "low"}
        return self.request("/v1/messages" if api == "anthropic" else "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "Say hello."}],
                             "max_tokens": 256, **think, **kwargs})

    def latest(self):
        summary = self.request("/api/requests")[1]["requests"][0]
        return self.request("/api/requests?id=" + summary["id"])[1]

    def test_output_and_reasoning_for_both_dialects_and_stream_modes(self):
        for api in ("openai", "anthropic"):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream):
                    code, reply = self.chat(api, stream=stream)
                    self.assertEqual(code, 200)
                    record = self.latest()
                    self.assertEqual(record["state"], "completed")
                    self.assertEqual(record["http_status"], 200)
                    self.assertIn("Say hello.", record["input"])
                    self.assertIn("<script>alert(1)</script>", record["output"])
                    self.assertIn("Thought.", record["reasoning"])
                    self.assertGreater(record["usage"].get("completion_tokens", record["usage"].get("output_tokens", 0)), 0)
                    self.assertGreaterEqual(record["wallclock_s"], record["first_token_s"])
                    self.assertNotIn("_clock", record)
                    if not stream:
                        self.assertEqual(json.loads(record["response"]), reply)
                    else:
                        self.assertNotIn("response", record)

    def test_error_and_engine_death_are_not_successes(self):
        self.assertEqual(self.chat(max_tokens=20000)[0], 400)
        self.assertEqual(self.latest()["state"], "error")
        for api in ("openai", "anthropic"):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream), mock.patch.object(self.engine, "generate", side_effect=EngineDied("stopped")):
                    self.chat(api, stream=stream)
                    record = self.latest()
                    self.assertEqual(record["state"], "error")
                    self.assertIn("stopped", record["error"]["message"])
                    self.assertEqual(record["output"], "")
                    self.assertNotIn("timings", record)

    def test_history_is_authorized_and_does_not_capture_headers(self):
        self.svc.api_key = "secret"
        auth = {"Authorization": "Bearer secret"}
        self.assertEqual(self.request("/api/requests")[0], 401)
        self.assertEqual(self.request("/api/requests?id=missing")[0], 401)
        self.assertEqual(len(self.svc.api_requests), 0)
        self.assertEqual(self.request("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}, auth)[0], 200)
        summary = self.request("/api/requests", headers=auth)[1]["requests"][0]
        self.assertNotIn("input", summary)
        self.assertNotIn("output", summary)
        self.assertEqual(self.request("/api/requests?id=" + summary["id"])[0], 401)
        detail = self.request("/api/requests?id=" + summary["id"], headers=auth)[1]
        self.assertNotIn("secret", json.dumps(detail))

    def test_capture_is_bounded_and_eviction_returns_404(self):
        huge = "x" * 262200
        first = self.svc.begin_request("/v1/chat/completions", {"messages": [{"content": huge}]})
        self.assertEqual(len(first["input"]), 262144)
        self.assertTrue(first["input_truncated"])
        for _ in range(100):
            self.svc.begin_request("/v1/chat/completions", {"messages": []})
        self.assertEqual(len(self.request("/api/requests")[1]["requests"]), 100)
        self.assertEqual(self.request("/api/requests?id=" + first["id"])[0], 404)
        # Exercise output capture without making a mock engine emit 262K tokens.
        handler = object.__new__(self.httpd.RequestHandlerClass)
        handler.record = self.svc.api_requests[-1]
        chunk = {"choices": [{"delta": {"content": huge, "reasoning_content": huge}}]}
        list(handler._capture((item for item in [chunk]), "openai"))
        for field in ("output", "reasoning"):
            self.assertEqual(len(handler.record[field]), 262144)
            self.assertTrue(handler.record[field + "_truncated"])

    def test_queue_wait_and_loading_are_attributed_separately(self):
        self.engine.unload()
        entered, release = threading.Event(), threading.Event()
        original = self.engine.restart
        def restart():
            entered.set()
            self.assertTrue(release.wait(5))
            original()
        with mock.patch.object(self.engine, "restart", side_effect=restart), ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.chat)
            self.assertTrue(entered.wait(5))
            second = pool.submit(self.chat)
            deadline = time.monotonic() + 5
            while len(self.svc.api_requests) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(len(self.svc.api_requests), 2)
            history = self.request("/api/requests")[1]
            self.assertEqual({r["state"] for r in history["requests"]}, {"queued", "loading"})
            self.assertFalse(history["loaded"])
            self.assertTrue(history["auto_load"])
            time.sleep(0.03)
            release.set()
            self.assertEqual(first.result()[0], 200)
            self.assertEqual(second.result()[0], 200)
        records = list(self.svc.api_requests)
        self.assertGreater(records[0]["load_s"], 0.02)
        self.assertGreater(records[1]["queue_s"], 0.02)
        self.assertLess(records[1]["load_s"], records[0]["load_s"])
        self.assertTrue(all(r["state"] == "completed" for r in records))

    def test_page_and_relative_assets_are_served(self):
        code, html = self.request("/api-monitor")
        self.assertEqual(code, 200)
        self.assertIn('src="/web/monitor.js"', html)
        code, script = self.request("/web/monitor.js")
        self.assertEqual(code, 200)
        self.assertIn("textContent", script)
        self.assertNotIn("innerHTML", script)
        self.assertEqual(self.request("/web/../server.py")[0], 404)


class PersistentHistory(unittest.TestCase):
    request = Monitor.request
    chat = Monitor.chat

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tok = ByteTokenizer()
        self.engine = ReloadableEngine(self.tok, 'Thought.</think>\nHello <script>alert(1)</script>.', max_context=16384)
        self.svc = Service(self.engine, self.tok, ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))
        self.assertTrue(hasattr(self.svc, 'output_history'), 'Persistent output history is not implemented')
        from serve.history import HistoryStore
        self.path = Path(self.tmp.name) / 'history.sqlite3'
        self.svc.output_history = HistoryStore(self.path)
        self.addCleanup(lambda: self.svc.output_history.close())
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f'http://127.0.0.1:{self.httpd.server_address[1]}'

    def records(self, query=''):
        self.svc.output_history.flush()
        status, body = self.request('/api/history' + query)
        self.assertEqual(status, 200, body)
        return body

    def test_all_apis_record_thinking_and_output_without_enabling_memory_monitor(self):
        for api in ('openai', 'anthropic', 'responses'):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream):
                    if api == 'responses':
                        status, _ = self.request('/v1/responses', {'input': 'Say hello.', 'stream': stream,
                                                                   'max_output_tokens': 256})
                    else:
                        status, _ = self.chat(api, stream=stream)
                    self.assertEqual(status, 200)
                    record = self.records()['requests'][0]
                    self.assertEqual(record['state'], 'completed')
                    for field, expected in (('reasoning', 'Thought.'), ('output', '<script>alert(1)</script>')):
                        code, detail = self.request('/api/history?id=' + record['id'] + '&field=' + field)
                        self.assertEqual(code, 200)
                        self.assertIn(expected, detail['text'])
        self.assertFalse(self.svc.api_monitor)
        self.assertEqual(len(self.svc.api_requests), 0)

    def test_app_label_is_recorded_for_all_api_routes_without_saving_raw_headers(self):
        for path, body in (
                ('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'Hello'}], 'max_tokens': 256}),
                ('/v1/messages', {'messages': [{'role': 'user', 'content': 'Hello'}], 'max_tokens': 256}),
                ('/v1/responses', {'input': 'Hello', 'max_output_tokens': 256})):
            with self.subTest(path=path):
                code, reply = self.request(path, body, {'X-Strata-App': 'My Editor', 'User-Agent': 'Trae/1.0 private-device',
                                                       'Cookie': 'private-cookie', 'X-Api-Key': 'private-key'})
                self.assertEqual(code, 200, reply)
                entry = self.records()['requests'][0]
                self.assertEqual(entry.get('client_app'), 'My Editor')
                self.assertEqual(entry.get('client_source'), 'X-Strata-App')
                detail = self.request('/api/history?id=' + entry['id'])[1]
                for private in ('private-device', 'private-cookie', 'private-key'):
                    self.assertNotIn(private, json.dumps(detail))

    def test_history_shows_actual_model_without_changing_requested_alias_or_api_response(self):
        self.svc.model = 'qwen-local-test'
        self.svc.aliases = ['claude-opus-4-6']
        self.svc.api_monitor = True
        for stream in (False, True):
            for path, body in (
                    ('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'Hello'}], 'max_tokens': 256}),
                    ('/v1/messages', {'messages': [{'role': 'user', 'content': 'Hello'}], 'max_tokens': 256}),
                    ('/v1/responses', {'input': 'Hello', 'max_output_tokens': 256})):
                with self.subTest(path=path, stream=stream):
                    status, response = self.request(path, {**body, 'model': 'claude-opus-4-6', 'stream': stream})
                    self.assertEqual(status, 200, response)
                    entry = self.records()['requests'][0]
                    self.assertEqual(entry['model'], 'qwen-local-test')
                    self.assertEqual(entry.get('requested_model'), 'claude-opus-4-6')
                    detail = self.request('/api/history?id=' + entry['id'])[1]['entry']
                    self.assertEqual(detail['model'], 'qwen-local-test')
                    self.assertEqual(detail['requested_model'], 'claude-opus-4-6')
                    self.assertEqual(self.svc.api_requests[-1]['model'], 'claude-opus-4-6')
                    if not stream:
                        self.assertEqual(response['model'], 'claude-opus-4-6')

    def test_recognized_user_agents_are_labels_but_generic_sdks_are_not_editors(self):
        for agent, expected in (('Trae/1.0.0', 'Trae'), ('claude-code/2.0.0', 'Claude Code'),
                                ('codex_cli_rs/0.50.0', 'Codex'), ('ZCode/1.0', 'Z Code'),
                                ('Cursor/1.2.3', 'Cursor'), ('OpenAI/Python 1.0', 'Unknown'),
                                ('Mozilla/5.0 Chrome/123', 'Unknown'), ('not-trae-client/1', 'Unknown')):
            with self.subTest(agent=agent):
                self.assertEqual(self.request('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'Hello'}],
                    'max_tokens': 256}, {'User-Agent': agent})[0], 200)
                entry = self.records()['requests'][0]
                self.assertEqual(entry.get('client_app'), expected)

    def test_app_label_survives_restart_and_old_records_are_honestly_unknown(self):
        from serve.history import HistoryStore
        self.assertEqual(self.request('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'Hello'}],
            'max_tokens': 256}, {'X-Strata-App': 'Strata Web'})[0], 200)
        new_id = self.records()['requests'][0]['id']
        old = self.svc.output_history.begin({'id': 'f' * 12, 'state': 'queued'})
        old.finish({'state': 'completed'})
        self.svc.output_history.flush()
        self.svc.output_history.close()
        self.svc.output_history = HistoryStore(self.path)
        self.assertEqual(self.svc.output_history.detail(new_id)['entry'].get('client_app'), 'Strata Web')
        self.assertEqual(self.svc.output_history.detail('f' * 12)['entry'].get('client_app'), 'Unknown')
        self.assertEqual(len(self.records('?q=Strata%20Web')['requests']), 1)

    def test_explicit_app_label_is_bounded_and_does_not_accept_control_characters(self):
        from email.message import Message
        from serve import history
        identify = getattr(history, 'client_identity', None)
        self.assertIsNotNone(identify, 'Client identity extraction is missing')
        headers = Message()
        headers['X-Strata-App'] = 'a' * 200
        self.assertEqual(len(identify(headers)['client_app']), 80)
        headers.replace_header('X-Strata-App', 'bad\tlabel')
        self.assertEqual(identify(headers)['client_app'], 'Unknown')

    def test_full_text_survives_preview_pagination_and_restart(self):
        from serve.history import HistoryStore
        record = self.svc.output_history.begin({'id': 'a' * 12, 'started_at': time.time(), 'state': 'queued'})
        text = 'Reasoning line.\n' * 25000 + 'END OF THINKING'
        record.append('reasoning', text)
        record.append('output', 'Final answer.')
        record.finish({'state': 'completed'})
        self.svc.output_history.flush()
        self.svc.output_history.close()
        self.svc.output_history = HistoryStore(self.path)
        self.assertEqual(self.records()['requests'][0]['state'], 'completed')
        got, offset = '', 0
        while offset is not None:
            code, detail = self.request('/api/history?id=' + 'a' * 12 + '&field=reasoning&offset=' + str(offset))
            self.assertEqual(code, 200)
            self.assertLessEqual(len(detail['text']), 65536)
            got += detail['text']
            offset = detail['next_offset']
        self.assertEqual(got, text)
        code, exported = self.request('/api/history?id=' + 'a' * 12 + '&download=1')
        self.assertEqual(code, 200)
        self.assertIn(text, exported)
        self.assertIn('Final answer.', exported)

    def test_more_than_100_requests_remain_searchable_and_paginated(self):
        for i in range(105):
            capture = self.svc.output_history.begin({'id': f'{i:012x}', 'started_at': i, 'state': 'queued'})
            capture.append('output', 'Unique result ' + str(i))
            capture.finish({'state': 'completed'})
            if i % 20 == 0:
                self.svc.output_history.flush()
        page = self.records()
        self.assertEqual(len(page['requests']), 50)
        ids = [r['id'] for r in page['requests']]
        while page['next_cursor']:
            page = self.records('?before=' + page['next_cursor'])
            ids.extend(r['id'] for r in page['requests'])
        self.assertEqual(len(set(ids)), 105)
        self.assertEqual(len(self.records('?q=Unique%20result%20104')['requests']), 1)

    def test_search_matches_text_across_flushed_chunks_but_not_across_fields(self):
        capture = self.svc.output_history.begin({'id': 'c' * 12, 'state': 'generating'})
        for text in ('prefix nee', 'd', 'le suffix'):
            capture.append('output', text)
            capture.flush()
        capture.append('reasoning', 'Another field')
        capture.finish({'state': 'completed'})
        self.assertEqual(len(self.records('?q=NEEDLE')['requests']), 1)
        self.assertEqual(len(self.records('?q=suffixAnother')['requests']), 0)

    def test_web_chat_mcp_calls_and_results_are_recorded(self):
        from serve.history import event_parts
        record = self.svc.output_history.begin({'id': 'd' * 12, 'state': 'generating'})
        for activity in ({'event': 'call', 'id': 'call123', 'name': 'edit_file', 'arguments': {'path': 'a.txt'}},
                         {'event': 'result', 'id': 'call123', 'ok': True, 'text': 'File updated.'}):
            fields, _ = event_parts('openai', {'choices': [{'delta': {}}], 'strata_mcp': activity})
            record.append('tools', fields['tools'])
        record.finish({'state': 'completed'})
        self.svc.output_history.flush()
        text = self.svc.output_history.detail('d' * 12, 'tools')['text']
        for expected in ('edit_file', 'a.txt', 'File updated.', 'call123'):
            self.assertIn(expected, text)

    def test_partial_history_is_visible_and_unfinished_work_becomes_interrupted(self):
        from serve.history import HistoryStore
        capture = self.svc.output_history.begin({'id': 'b' * 12, 'started_at': time.time(), 'state': 'generating'})
        capture.append('reasoning', 'Partial thought.')
        capture.flush()
        self.svc.output_history.flush()
        self.assertIn('Partial thought.', self.request('/api/history?id=' + 'b' * 12 + '&field=reasoning')[1]['text'])
        self.svc.output_history.close()
        self.svc.output_history = HistoryStore(self.path)
        self.assertEqual(self.records()['requests'][0]['state'], 'interrupted')

    def test_recording_failure_does_not_fail_or_change_the_model_response(self):
        with mock.patch.object(self.svc.output_history, '_write', side_effect=OSError('disk full')):
            status, reply = self.chat()
            self.svc.output_history.flush()
        self.assertEqual(status, 200)
        self.assertIn('Hello', reply['choices'][0]['message']['content'])
        self.assertIn('disk full', self.svc.output_history.error)

    def test_worker_connection_failure_is_reported_and_does_not_deadlock(self):
        from serve.history import HistoryStore
        self.svc.output_history.close()
        original = HistoryStore._connect
        failed = threading.Event()

        def connect(store):
            if threading.current_thread().name == 'strata-history-writer' and not failed.is_set():
                failed.set()
                raise OSError('connection unavailable')
            return original(store)

        with mock.patch.object(HistoryStore, '_connect', connect):
            self.svc.output_history = HistoryStore(self.path)
            self.assertEqual(self.chat()[0], 200)
            deadline = time.monotonic() + 3
            while self.svc.output_history.jobs.unfinished_tasks and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(self.svc.output_history.jobs.unfinished_tasks, 0)
            self.assertTrue(self.svc.output_history.worker.is_alive())
            self.assertIn('connection unavailable', self.svc.output_history.error)

    def test_recording_warning_survives_restart_after_a_transient_write_failure(self):
        from serve.history import HistoryStore
        store = self.svc.output_history
        capture = store.begin({'id': 'e' * 12, 'state': 'generating'})
        store.flush()
        with mock.patch.object(store, '_write', side_effect=OSError('temporary disk error')):
            capture.append('output', 'lost' * 4096)
            store.flush()
        capture.append('output', 'Saved after recovery.')
        capture.finish({'state': 'completed'})
        store.flush()
        store.close()
        self.svc.output_history = HistoryStore(self.path)
        entry = self.records()['requests'][0]
        self.assertIn('temporary disk error', entry['recording_error'])
        self.assertEqual(self.svc.output_history.detail(entry['id'])['text'], 'Saved after recovery.')

    def test_tool_calls_are_captured_for_all_protocols_and_stream_modes(self):
        self.engine.script = self.tok.encode('Thought.</think>\n<tool_call>\n<function=edit_file>\n'
            '<parameter=path>\nexample.txt\n</parameter>\n<parameter=text>\nupdated\n</parameter>\n'
            '</function>\n</tool_call><|im_end|>', parse_special=True)
        function = {'name': 'edit_file', 'description': 'Edit a file', 'parameters': {'type': 'object', 'properties': {
            'path': {'type': 'string'}, 'text': {'type': 'string'}}}}
        for api in ('openai', 'anthropic', 'responses'):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream):
                    tools = [{'type': 'function', 'function': function}]
                    if api == 'anthropic':
                        tools = [{**{k: v for k, v in function.items() if k != 'parameters'},
                                  'input_schema': function['parameters']}]
                    if api == 'responses':
                        status, reply = self.request('/v1/responses', {'input': 'Edit a file.', 'stream': stream,
                            'max_output_tokens': 512, 'tools': [{'type': 'function', **function}]})
                    else:
                        status, reply = self.chat(api, stream=stream, tools=tools, max_tokens=512)
                    self.assertEqual(status, 200, reply)
                    record = self.records()['requests'][0]
                    text = self.request('/api/history?id=' + record['id'] + '&field=tools')[1]['text']
                    self.assertIn('Tool: edit_file', text)
                    self.assertIn('example.txt', text)
                    self.assertIn('updated', text)

    def test_history_capture_yields_identical_events_without_memory_preview_copies(self):
        record = self.svc.begin_request('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'private input'}]})
        handler = object.__new__(self.httpd.RequestHandlerClass)
        handler.record = record
        events = [{'choices': [{'delta': {'content': 'x' * 17000, 'reasoning_content': 'Thinking.'}}]},
                  {'choices': [], 'usage': {'completion_tokens': 5}}]
        got = list(handler._capture((item for item in events), 'openai'))
        self.assertTrue(all(a is b for a, b in zip(got, events)))
        self.assertEqual(record['output'], '')
        record['_history'].finish({'state': 'completed'})
        self.svc.output_history.flush()
        detail = self.svc.output_history.detail(record['id'])
        self.assertEqual(detail['text'], 'x' * 17000)
        self.assertNotIn('private input', ''.join(self.svc.output_history.export(record['id'])))

    def test_history_requires_existing_api_auth_and_never_saves_headers(self):
        self.svc.api_key = 'private-api-key'
        auth = {'Authorization': 'Bearer private-api-key'}
        self.assertEqual(self.request('/api/history')[0], 401)
        self.assertEqual(self.request('/api/history?id=' + 'a' * 12 + '&download=1')[0], 401)
        self.assertEqual(self.request('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'Hello'}]}, auth)[0], 200)
        self.svc.output_history.flush()
        listing = self.request('/api/history', headers=auth)[1]
        detail = self.request('/api/history?id=' + listing['requests'][0]['id'], headers=auth)[1]
        self.assertNotIn('private-api-key', json.dumps(detail))
        self.assertEqual(self.request('/api/history?id=..%2Fsecret', headers=auth)[0], 400)


class HistoryUi(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node is needed for the deferred-fetch UI regression')
    def test_refresh_during_selection_never_shows_another_requests_text(self):
        script = r'''
const assert = require('node:assert/strict'), fs = require('node:fs'), vm = require('node:vm');
const node = () => ({textContent:'',hidden:false,disabled:false,value:'',dataset:{},children:[],
  setAttribute(k,v){this[k]=v},append(...c){this.children.push(...c)},replaceChildren(){this.children=[]},focus(){}});
const nodes = {}, pending = [], intervals = [];
const panes = ['output','reasoning','tools','details'].map(value => Object.assign(node(), {dataset:{historyPane:value}}));
const root = {querySelector: id => nodes[id] ||= node(), querySelectorAll: () => panes};
const sandbox = {window:{},URLSearchParams,document:{activeElement:null,hidden:false,createElement:node},
  setInterval:fn => intervals.push(fn),setTimeout:()=>1,clearTimeout(){},
  fetch:url => new Promise(resolve => pending.push({url,resolve}))};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), sandbox);
const tick = async () => {for(let i=0;i<12;i++) await Promise.resolve()};
const respond = async (req,body) => {assert.ok(req,'expected pending fetch');req.resolve({ok:true,json:async()=>body});await tick()};
const a = {id:'aaaaaaaaaaaa',state:'completed',started_at:1,path:'/v1/chat/completions',client_app:'Trae'};
const b = {...a,id:'bbbbbbbbbbbb',client_app:undefined};
const listing = {requests:[a,b],next_cursor:null};
const detail = entry => ({entry,field:'output',text:entry.id[0]+' output',next_offset:null});
const click = id => nodes['#history-list'].children.map(li=>li.children[0]).find(button=>button.dataset.historyId===id).onclick();
(async () => {
  sandbox.window.StrataHistory.mount(root,()=>({}),()=>{});
  sandbox.window.StrataHistory.visible(true);
  await respond(pending.shift(), listing);
  const appLabels = nodes['#history-list'].children.map(li=>li.children[0].children[1].textContent);
  assert.deepEqual(appLabels, ['From: Trae', 'From: Unknown']);
  click(b.id); await respond(pending.shift(), detail(b));
  assert.equal(nodes['#history-app'].textContent, 'From: Unknown');
  click(a.id); const delayed = pending.shift();
  nodes['#history-refresh'].onclick();
  await respond(pending.shift(), listing);
  await respond(delayed, detail(a));
  if(pending.length) await respond(pending.shift(), detail(a));
  intervals[0](); await respond(pending.shift(), listing);
  if(pending.length) await respond(pending.shift(), detail(a));
  assert.equal(nodes['#history-text'].textContent, 'a output');
  assert.ok(nodes['#history-title'].textContent.includes(a.id));
  assert.equal(nodes['#history-selected'].hidden, false);
  assert.equal(nodes['#history-app'].textContent, 'From: Trae');
})().catch(error=>{console.error(error);process.exitCode=1});
'''
        result = subprocess.run([shutil.which('node'), '-e', script, str(Path(__file__).parent / 'web/history.js')],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class ConversationCacheCard(unittest.TestCase):
    """#596: /metrics' "conversation_cache": the parked conversations from the engine's log lines, the budget and
    slots from its INFO line, and how much of the prompts the cache gave back."""

    PARK = ("strata serve: conversation cache: parked 5000 tokens in 12.0 ms; parked=1 bytes=104857600 evictions=0 "
            "snapshot_bytes=104857600 reused_kv_bytes=0\n")
    RESTORE = "strata serve: conversation cache: restored 4000 tokens (exact) in 30.0 ms; parked=2 bytes=209715200\n"

    def test_the_log(self):
        from serve.server import ConvCacheLog
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text("an earlier run\n" + self.PARK, encoding="utf-8")
            start = log.stat().st_size                 # this run starts here: the earlier run's park is not counted
            c = ConvCacheLog()
            self.assertEqual(c.poll(str(log), start)["parked"], 0)
            with open(log, "a", encoding="utf-8") as f:
                f.write("strata serve: something else\n" + self.PARK + self.RESTORE + "strata serve: conversation ca")
            st = c.poll(str(log), start)
            self.assertEqual((st["parked"], st["bytes"], st["parks"], st["restores"]), (2, 209715200, 1, 1))
            self.assertEqual((st["last_event"], st["last_tokens"]), ("restored", 4000))
            with open(log, "a", encoding="utf-8") as f:   # the cut line is read once it is whole
                f.write("che: dropped 1 superseded copy of this conversation; parked=1\n")
            self.assertEqual(c.poll(str(log), start)["parked"], 1)
            self.assertEqual(c.poll(str(log), log.stat().st_size)["parked"], 0)   # the engine started again
            self.assertEqual(c.poll(None, None)["parked"], 0)

    def test_metrics(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, "Thought.</think>\nHello.", max_context=4096)
        engine.info = {"conversation_cache_mib": 8192, "conversation_cache_slots": 4}
        svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                c = json.loads(r.read())["conversation_cache"]
            self.assertEqual((c["enabled"], c["budget_mib"], c["slots"], c["parked"], c["requests"]),
                             (True, 8192, 4, 0, 0))
            req = urllib.request.Request(base + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode())
            with urllib.request.urlopen(req, timeout=10) as r:
                r.read()
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                c = json.loads(r.read())["conversation_cache"]
            self.assertEqual(c["requests"], 1)
            self.assertGreater(c["last_prompt"], 0)
            engine.info = {}
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                self.assertFalse(json.loads(r.read())["conversation_cache"]["enabled"])   # off: the reuse lines only
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_the_card_is_on_the_page(self):
        web = Path(__file__).parent / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        for el in ("cc-card", "cc-slots-text", "cc-mem-text", "cc-facts", "cc-note"):
            self.assertIn(f'id="{el}"', html)
            self.assertIn(f'"{el}"', js)


if __name__ == "__main__":
    unittest.main()
