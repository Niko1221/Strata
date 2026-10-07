"""Extraction HTTP admission over a mock service; no GPU or persisted uploads."""
import http.client
import json
import unittest
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve
from serve.file_extract import MAX_BODY
from serve.test_file_extract import request


class ExtractionHttp(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, '</think>\n\nok'), tok,
                           ChatTemplate(Path(__file__).resolve().parents[1] / 'serve/chat_template.jinja'))
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]
        self.origin = f'http://127.0.0.1:{self.port}'

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        try:
            data = json.dumps(body or request('a.py', '中文'.encode())).encode()
            connection.request('POST', '/v1/files/extract', data,
                               {'Content-Type': 'application/json', 'Origin': self.origin, **(headers or {})})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_real_extract_without_model(self):
        with mock.patch.object(self.svc, 'begin_request', side_effect=AssertionError('model called')):
            status, data = self.post()
        self.assertEqual(status, 200)
        self.assertEqual(data['text'], '中文')

    def test_auth_origin_and_cross_site_before_extract(self):
        with mock.patch('serve.server.extract_file', side_effect=AssertionError('parser called')):
            self.assertEqual(self.post(headers={'Origin': 'https://foreign.invalid'})[0], 403)
            self.assertEqual(self.post(headers={'Sec-Fetch-Site': 'cross-site'})[0], 403)
            self.assertEqual(self.post(headers={'Content-Type': 'text/plain'})[0], 415)
            self.svc.api_key = 'fixture-key'
            self.assertEqual(self.post()[0], 401)
            self.assertEqual(self.post(headers={'Authorization': 'Bearer wrong'})[0], 401)
        self.assertEqual(self.post(headers={'Authorization': 'Bearer fixture-key'})[0], 200)

    def test_oversize_declared_body_rejected_before_read(self):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=5)
        try:
            connection.putrequest('POST', '/v1/files/extract')
            connection.putheader('Origin', self.origin)
            connection.putheader('Content-Type', 'application/json')
            connection.putheader('Content-Length', str(MAX_BODY + 1))
            connection.endheaders()  # deliberately send no body
            response = connection.getresponse()
            self.assertEqual(response.status, 413)
            response.read()
        finally:
            connection.close()

    def test_schema_and_format_failures(self):
        self.assertEqual(self.post({'name': 'a.txt', 'data': '!!!'})[0], 400)
        self.assertEqual(self.post(request('a.exe', b'abc'))[0], 415)
        self.assertEqual(self.post(request('a.pdf', b'abc'))[0], 422)


if __name__ == '__main__':
    unittest.main()
