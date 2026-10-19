"""No-GPU beta HTTP accounting and disabled-path comparison.
Run from the checkout root: python docs/usage-v020/check_http.py
"""
import json
from pathlib import Path
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from serve.server import ByteTokenizer, MockEngine, Service, serve
from serve.frontend import ChatTemplate
from serve.usage import Usage


def request(base, route, body=None):
    req = urllib.request.Request(base + route, data=json.dumps(body).encode() if body else None,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=10) as response:
        raw = response.read().decode()
        if 'text/event-stream' not in response.headers.get('Content-Type', ''):
            return [json.loads(raw)]
        return [json.loads(line[6:]) for line in raw.splitlines()
                if line.startswith('data: ') and line != 'data: [DONE]']


def normalize(value):
    # Volatile protocol identity and timestamps differ across independent servers.
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()
                if k not in ('id', 'created', 'created_at', 'completed_at', 'response_id', 'item_id')}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    return value


def run(enabled, folder):
    tok = ByteTokenizer()
    svc = Service(MockEngine(tok, '</think>\nCounted.', max_context=8192), tok,
                  ChatTemplate(ROOT / 'serve/chat_template.jinja'))
    ledger = folder / 'usage.json'
    svc.usage = Usage(ledger if enabled else None)
    httpd = serve(svc, port=0)
    base = 'http://127.0.0.1:' + str(httpd.server_address[1])
    replies = []
    try:
        for stream in (False, True):
            replies.append(request(base, '/v1/chat/completions', {
                'messages': [{'role': 'user', 'content': 'Hello'}], 'max_tokens': 64, 'stream': stream}))
            replies.append(request(base, '/v1/responses', {
                'input': 'Hello', 'max_output_tokens': 64, 'stream': stream}))
        metrics = request(base, '/metrics')[0]
        if enabled:
            saved = json.loads(ledger.read_text())['lifetime']
            assert saved == metrics['usage']['lifetime']
            for key in saved:
                assert saved[key] == svc.totals[key], key
            assert saved['requests'] == 4
            ranged = request(base, '/metrics?from_ts=0&to_ts=9999999999')[0]
            assert ranged['usage']['views']['custom']['total']['output_tokens'] == saved['output_tokens']
        else:
            assert 'usage' not in metrics
            assert not ledger.exists()
            assert not ledger.with_suffix('.json.lock').exists()
    finally:
        httpd.shutdown()
        httpd.server_close()
        svc.usage.close()
    if enabled:
        again = Usage(ledger)
        try:
            assert again.lifetime == saved
        finally:
            again.close()
    return normalize(replies), svc.totals['output_tokens']


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        off, off_tokens = run(False, folder)
        on, on_tokens = run(True, folder)
        assert off == on, 'enabled/disabled API payloads differ after identity/timestamp normalization'
        assert off_tokens == on_tokens
        print(json.dumps({'requests': 8, 'enabled_output_tokens': on_tokens,
                          'disabled_output_tokens': off_tokens, 'restart_preserved': True,
                          'custom_range_correct': True, 'normalized_payloads_equal': True,
                          'disabled_no_ledger_or_metrics_field': True}))
