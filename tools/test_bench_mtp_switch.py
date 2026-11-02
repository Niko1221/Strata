"""Protocol fixtures for the measurement tool. No GPU or engine validation."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest


class ProtocolTest(unittest.TestCase):
    def run_fixture(self, mode, scenario='solo'):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()

                def packet(data):
                    self.wfile.write(('data: ' + json.dumps(data) + '\n\n').encode())
                    self.wfile.flush()

                n = body['max_tokens']
                for _ in range(n):
                    packet({'choices': [{'delta': {'content': 'x'}}]})
                    time.sleep(0.012)
                if mode == 'truncated':
                    return
                packet({'choices': [{'delta': {}, 'finish_reason': 'length'}]})
                if mode != 'no-usage':
                    packet({'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': n}})
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()

        with ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    prompt = root / 'prompt.txt'
                    prompt.write_text('public protocol fixture', encoding='utf-8')
                    output = root / 'result.json'
                    command = [sys.executable, str(Path(__file__).with_name('bench_mtp_switch.py')),
                               '--endpoint', f'http://127.0.0.1:{server.server_port}', '--model', 'fixture',
                               '--build-id', 'protocol-fixture-no-gpu', '--topology', 'dual-gpu',
                               '--scenario', scenario, '--tokens', '16', '--short-tokens', '4',
                               '--repeats', '1', '--timeout', '5', '--prompt-file', str(prompt),
                               '--output', str(output)]
                    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
                    self.assertTrue(output.exists(), result.stderr)
                    report = json.loads(output.read_text(encoding='utf-8'))
                    # Never replace archived results, even after a failed request.
                    again = subprocess.run(command, capture_output=True, text=True, timeout=5)
                    self.assertNotEqual(again.returncode, 0)
                    self.assertEqual(report, json.loads(output.read_text(encoding='utf-8')))
                    return result.returncode, report
            finally:
                server.shutdown()

    def test_switch_and_shared_interval(self):
        code, report = self.run_fixture('complete', 'switch')
        self.assertEqual(code, 0)
        phase = next(p for p in report['phases'] if p['name'] == 'staggered')
        self.assertTrue(phase['observed_1_2_1'])
        pair = next(p for p in report['phases'] if p['name'] == 'simultaneous')
        self.assertTrue(pair['http_intervals_overlap'])
        self.assertAlmostEqual(pair['aggregate_e2e_output_tps'], 32 / pair['wall_s'])
        self.assertEqual(len(report['requests']), 6)

    def test_incomplete_stream_retained_as_failure(self):
        code, report = self.run_fixture('truncated')
        self.assertNotEqual(code, 0)
        self.assertFalse(report['requests'][0]['done'])
        self.assertIsNotNone(report['requests'][0]['error'])
        self.assertIsNone(report['phases'][0]['aggregate_e2e_output_tps'])

    def test_no_usage_never_estimates_tokens(self):
        _, report = self.run_fixture('no-usage')
        self.assertIsNone(report['requests'][0]['usage'])
        self.assertIsNone(report['requests'][0]['e2e_output_tps'])
        self.assertIsNone(report['phases'][0]['aggregate_e2e_output_tps'])


if __name__ == '__main__':
    unittest.main()
