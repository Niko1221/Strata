"""Run the real gateway against an isolated loopback fixture; never touch the model or Tailscale."""
import base64
import http.client
import json
import shutil
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


@unittest.skipUnless(shutil.which('node'), 'Gateway checks require Node.js')
class Gateway(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.seen = []
        seen = self.seen

        class Upstream(BaseHTTPRequestHandler):
            def handle_request(self):
                body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
                seen.append((self.path, dict(self.headers), body))
                raw = json.dumps({'ok': True, 'data': [{'id': 'qwen-demo'}]}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            do_GET = do_POST = handle_request

            def log_message(self, *_):
                pass

        upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        self.addCleanup(upstream.server_close)
        self.addCleanup(upstream.shutdown)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        self.key = 'example-only-' + 'k' * 40
        self.config = {'apiKey': self.key, 'port': 0, 'upstreamApiKey': 'fixture-upstream-key'}
        self.config_path = self.folder / 'gateway.json'
        self.save()
        source = (Path(__file__).resolve().parents[1] / 'remote-access/gateway.mjs').read_text()
        target = "hostname: '127.0.0.1', port: 8080,"
        self.assertEqual(source.count(target), 1)
        source = source.replace(target, f"hostname: '127.0.0.1', port: {upstream.server_port},")
        script = self.folder / 'gateway.mjs'
        script.write_text(source)
        self.proc = subprocess.Popen([shutil.which('node'), str(script), str(self.config_path)],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(self.stop)
        # Production port zero asks the OS for a port; the startup log reports the assigned value.
        line = self.proc.stdout.readline()
        self.assertIn('127.0.0.1:', line, line)
        self.port = int(line.rsplit(':', 1)[1].strip())

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
        self.proc.communicate(timeout=10)

    def save(self):
        self.config_path.write_text(json.dumps(self.config))

    def request(self, path='/v1/models', method='GET', headers=None, body=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def bearer(self, key=None):
        return {'Authorization': 'Bearer ' + (key or self.key)}

    def basic(self):
        return {'Authorization': 'Basic ' + base64.b64encode(('strata:' + self.key).encode()).decode()}

    def dashboard(self):
        self.config['exposeDashboard'] = True
        self.save()

    def test_default_requires_key_and_does_not_publish_dashboard_or_history(self):
        self.assertEqual(self.request()[0], 401)
        self.assertEqual(self.request(headers=self.bearer())[0], 200)
        for path in ('/', '/metrics', '/api/history', '/config'):
            self.assertEqual(self.request(path, headers=self.bearer())[0], 404)
        code, headers, _ = self.request(method='OPTIONS', headers=self.bearer())
        self.assertEqual(code, 405)
        self.assertEqual(headers['x-strata-gateway-version'], '2')

    def test_key_rotation_takes_effect_without_process_restart(self):
        self.config['apiKey'] = 'new-example-' + 'n' * 40
        self.save()
        self.assertEqual(self.request(headers=self.bearer())[0], 401)
        self.assertEqual(self.request(headers=self.bearer(self.config['apiKey']))[0], 200)
        self.config_path.write_text('{broken')
        self.assertEqual(self.request(headers=self.bearer(self.config['apiKey']))[0], 401)

    def test_dashboard_requires_login_and_history_is_not_a_model_route(self):
        self.dashboard()
        for path in ('/', '/web/app.js', '/metrics', '/api/history'):
            code, headers, _ = self.request(path)
            self.assertEqual(code, 401)
            self.assertIn('Basic realm=', headers['www-authenticate'])
            self.assertEqual(self.request(path, headers=self.basic())[0], 200)

    def test_browser_basic_post_rejects_missing_null_foreign_or_cross_site_origin(self):
        self.dashboard()
        for extra in ({}, {'Origin': 'null'}, {'Origin': 'https://evil.example'},
                      {'Origin': f'http://user@127.0.0.1:{self.port}'},
                      {'Origin': f'http://127.0.0.1:{self.port}/extra'},
                      {'Origin': f'http://127.0.0.1:{self.port}', 'Sec-Fetch-Site': 'cross-site'}):
            code, _, _ = self.request('/v1/chat/completions', 'POST', {**self.basic(), **extra}, '{}')
            self.assertEqual(code, 403, extra)
        self.assertEqual(self.seen, [])

    def test_same_origin_browser_post_and_native_keys_retain_client_labels_not_credentials(self):
        self.dashboard()
        code, _, _ = self.request('/v1/chat/completions', 'POST', {**self.basic(),
            'Origin': f'http://127.0.0.1:{self.port}', 'Sec-Fetch-Site': 'same-origin',
            'Cookie': 'private-browser-cookie', 'X-Strata-App': 'Strata Web'}, '{}')
        self.assertEqual(code, 200)
        path, headers, _ = self.seen[-1]
        lowered = {key.lower(): value for key, value in headers.items()}
        self.assertEqual(path, '/v1/chat/completions')
        self.assertEqual(lowered.get('authorization'), 'Bearer fixture-upstream-key')
        self.assertEqual(lowered.get('x-strata-app'), 'Strata Web')
        self.assertNotIn('origin', lowered)
        self.assertNotIn('cookie', lowered)
        self.assertEqual(self.request('/v1/messages', 'POST', {'x-api-key': self.key}, '{}')[0], 200)

    def test_unknown_open_access_settings_do_not_disable_contribution_auth(self):
        self.config.update(allowUnauthenticated=True, dashboardRequireAuth=False, exposeDashboard=True)
        self.save()
        self.assertEqual(self.request()[0], 401)
        self.assertEqual(self.request('/api/history')[0], 401)


if __name__ == '__main__':
    unittest.main()
