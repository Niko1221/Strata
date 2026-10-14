"""Raster MCP delivery: cache bounds, authenticated downloads and vision forwarding; no GPU."""
import base64
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch
from serve.mcp import cache_images
from serve.frontend import ChatTemplate, images_of
from serve.server import ByteTokenizer, Service, serve
from serve.test_mcp import ScriptedEngine, call_script
from serve import test_mcp as fixtures
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jB0kAAAAASUVORK5CYII=')

def image(data=PNG, mime='image/png'):
    return {'type': 'image', 'mimeType': mime, 'data': base64.b64encode(data).decode()}

class Cache(unittest.TestCase):

    def test_image_text_obeys_result_cap(self):
        from serve.mcp import McpHub
        hub = McpHub({'fake': {'command': 'unused'}}, {'max_result_chars': 1000, 'image_results': True})
        server = hub.servers['fake']
        server.status = 'ready'
        server.tools = [{'name': 'image', 'inputSchema': {}}]
        with tempfile.TemporaryDirectory() as tmp:
            hub.media_dir = Path(tmp)
            with patch.object(server, 'call', return_value={'content': [{'type': 'text', 'text': 'x' * 100000}, image()]}):
                result = hub.call('fake__image', {})
            self.assertTrue(result['truncated'])
            self.assertLessEqual(len(result['model_content'][0]['text']), len(result['text']))
            self.assertNotIn('x' * 1001, result['model_content'][0]['text'])
            self.assertEqual(len(result['images']), 1)

    def test_disabled_is_identical_to_text_only_result(self):
        from serve.mcp import McpHub, result_text
        hub = McpHub({'fake': {'command': 'unused'}})
        server = hub.servers['fake']
        server.status = 'ready'
        server.tools = [{'name': 'image', 'inputSchema': {}}]
        result = {'content': [{'type': 'text', 'text': 'hello'}, image()]}
        with tempfile.TemporaryDirectory() as tmp:
            hub.media_dir = Path(tmp) / 'unused'
            with patch.object(server, 'call', return_value=result), patch('serve.mcp.cache_images') as cache:
                output = hub.call('fake__image', {})
            cache.assert_not_called()
            self.assertEqual(output['text'], result_text(result))
            self.assertNotIn('images', output)
            self.assertNotIn('model_content', output)
            self.assertFalse(hub.media_dir.exists())

    def test_cache_failure_keeps_text_result(self):
        from serve.mcp import McpHub
        hub = McpHub({'fake': {'command': 'unused'}}, {'image_results': True})
        server = hub.servers['fake']
        server.status = 'ready'
        server.tools = [{'name': 'image', 'inputSchema': {}}]
        hub.media_dir = Path('unused')
        with patch.object(server, 'call', return_value={'content': [{'type': 'text', 'text': 'hello'}, image()]}), patch('serve.mcp.cache_images', side_effect=OSError('disk full')):
            output = hub.call('fake__image', {})
        self.assertTrue(output['ok'])
        self.assertIn('hello', output['text'])
        self.assertNotIn('model_content', output)

    def test_null_and_non_object_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(cache_images({'content': None}, tmp), [])
            self.assertEqual(cache_images({'content': [None, False, 'bad', image()]}, tmp), cache_images({'content': [image()]}, tmp))

    def test_cache_deduplicates_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = cache_images({'content': [image()]}, tmp)
            b = cache_images({'content': [image()]}, tmp)
            self.assertEqual(a, b)
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)
            self.assertEqual((Path(tmp) / a[0]['url'].split('/')[-1]).read_bytes(), PNG)

    def test_rejects_invalid_and_active_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocks = [image(b'not a PNG'), image(b'<svg/>', 'image/svg+xml'), {'type': 'image', 'mimeType': 'image/png', 'data': '***'}]
            self.assertEqual(cache_images({'content': blocks}, tmp), [])

    def test_caps_images_per_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(len(cache_images({'content': [image()] * 8}, tmp)), 4)

    def test_authenticated_download_and_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            media = cache_images({'content': [image()]}, tmp)[0]
            tok = ByteTokenizer()
            svc = Service(ScriptedEngine(tok, ['OK']), tok, ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))

            class Hub:
                media_dir = tmp
            svc.mcp = Hub()
            svc.api_key = 'test-key'
            httpd = serve(svc, port=0)
            base = f'http://127.0.0.1:{httpd.server_address[1]}/'
            try:
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(base + media['url'])
                self.assertEqual(error.exception.code, 401)
                req = urllib.request.Request(base + media['url'], headers={'Authorization': 'Bearer test-key'})
                with urllib.request.urlopen(req) as response:
                    self.assertEqual(response.read(), PNG)
                    self.assertEqual(response.headers['Content-Type'], 'image/png')
                req = urllib.request.Request(base + 'tool-media/../../secret.png', headers={'Authorization': 'Bearer test-key'})
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(req)
                self.assertEqual(error.exception.code, 404)
            finally:
                httpd.shutdown()
                httpd.server_close()

class HistoryReferences(unittest.TestCase):

    def test_reference_resolves_only_to_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            im = cache_images({'content': [image()]}, tmp)[0]
            name = im['url'].split('/')[-1]
            tok = ByteTokenizer()
            svc = Service(ScriptedEngine(tok, ['OK']), tok, ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))

            class Hub:
                media_dir = tmp
            svc.mcp = Hub()
            message = [{'role': 'tool', 'content': [{'type': 'image', 'source': 'strata-tool-media:' + name}]}]
            with patch.object(svc, '_note_unreadable_tool_images', return_value={}):
                with self.assertRaisesRegex(ValueError, 'without the vision encoder'):
                    svc.prepare(message, [], {}, 16)
            self.assertEqual(images_of(message), [str(Path(tmp) / name)])

    def test_bad_reference_cannot_read_other_files(self):
        tok = ByteTokenizer()
        svc = Service(ScriptedEngine(tok, ['OK']), tok, ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))
        with self.assertRaisesRegex(ValueError, 'invalid tool image reference'):
            svc.prepare([{'role': 'tool', 'content': [{'type': 'image', 'source': 'strata-tool-media:../../secret.png'}]}], [], {}, 16)

class ImageLoop(unittest.TestCase):
    setUp = fixtures.ToolLoop.setUp
    tearDown = fixtures.ToolLoop.tearDown
    start = fixtures.ToolLoop.start
    post = fixtures.ToolLoop.post
    chunks = staticmethod(fixtures.ToolLoop.chunks)

    def test_default_stream_has_no_image_fields(self):
        self.start(call_script('fake__echo', text='hello'), '</think>\n\nDone.')
        code, text = self.post({'strata_mcp': True})
        self.assertEqual(code, 200)
        event = [c['strata_mcp'] for c in self.chunks(text) if c.get('strata_mcp', {}).get('event') == 'result'][0]
        self.assertEqual(set(event), {'event', 'id', 'ok', 'text', 'chars', 'truncated', 'ms'})

    def test_image_result_streams_and_reaches_vision_prepare(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.hub.settings['image_results'] = True
            self.hub.media_dir = Path(tmp)
            im = cache_images({'content': [image()]}, tmp)
            source = str(Path(tmp) / im[0]['url'].split('/')[-1])
            self.start(call_script('fake__echo', text='picture'), '</think>\n\nImage received.')
            self.svc.vision = object()
            native = self.svc.prepare
            seen = []

            def prepare(messages, *args, **kwargs):
                seen.extend(images_of(messages))
                clean = [{**m, 'content': 'image delivered'} if isinstance(m.get('content'), list) else m for m in messages]
                return native(clean, *args, **kwargs)
            result = {'ok': True, 'text': 'image delivered', 'chars': 15, 'truncated': False, 'ms': 1, 'images': im, 'model_content': [{'type': 'text', 'text': 'image delivered'}, {'type': 'image', 'source': source}]}
            with patch.object(self.hub, 'call', return_value=result), patch.object(self.svc, 'prepare', side_effect=prepare):
                code, text = self.post({'strata_mcp': True})
            self.assertEqual(code, 200)
            self.assertIn(source, seen)
            events = [c['strata_mcp'] for c in self.chunks(text) if 'strata_mcp' in c]
            self.assertEqual(events[-1]['images'], im)
            answer = ''.join((c['choices'][0]['delta'].get('content', '') for c in self.chunks(text)))
            self.assertIn('Image received.', answer)
