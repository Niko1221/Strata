import json
import time
import urllib.error
import urllib.request

BASE = 'http://127.0.0.1:18761'
MODEL = 'deepseek-v4.1-flash-ud-iq1-m-sycl'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(path, body=None, headers=None, timeout=600):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json', **(headers or {})})
    start = time.monotonic()
    try:
        with OPENER.open(req, timeout=timeout) as response:
            raw = response.read()
            return {'status': response.status, 'json': json.loads(raw), 'duration_s': time.monotonic() - start}
    except urllib.error.HTTPError as error:
        raw = error.read().decode()
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = {'raw': raw}
        return {'status': error.code, 'json': payload, 'duration_s': time.monotonic() - start}


def chat_body(prompt, cap=256, stream=False, seed=1):
    return {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
            'stream': stream, 'max_tokens': cap, 'temperature': 0, 'top_p': 1,
            'top_k': 0, 'min_p': 0, 'seed': seed, 'reasoning_effort': 'none',
            'chat_template_kwargs': {'enable_thinking': False},
            **({'stream_options': {'include_usage': True}} if stream else {})}


def stream_chat(body, cancel_after_first=False, timeout=600):
    req = urllib.request.Request(BASE + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    result = {'status': None, 'content': '', 'reasoning_content': '', 'ttft_s': None,
              'duration_s': None, 'usage': None, 'timings': None, 'finish_reason': None,
              'done': False, 'cancelled_by_client': False, 'events': [], 'error': None}
    try:
        with OPENER.open(req, timeout=timeout) as response:
            result['status'] = response.status
            for raw in response:
                line = raw.decode('utf-8').strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    result['done'] = True
                    break
                event = json.loads(data)
                result['events'].append(event)
                if event.get('error'):
                    result['error'] = event['error']
                if event.get('usage'):
                    result['usage'] = event['usage']
                if event.get('timings'):
                    result['timings'] = event['timings']
                generated_piece = False
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    for field in ['content', 'reasoning_content']:
                        text = delta.get(field) or ''
                        if text:
                            result[field] += text
                            generated_piece = True
                    if choice.get('finish_reason') is not None:
                        result['finish_reason'] = choice['finish_reason']
                if generated_piece and result['ttft_s'] is None:
                    result['ttft_s'] = time.monotonic() - start
                    if cancel_after_first:
                        result['cancelled_by_client'] = True
                        break
    except urllib.error.HTTPError as error:
        result['status'] = error.code
        result['error'] = error.read().decode('utf-8', 'replace')
    except Exception as error:
        result['error'] = repr(error)
    result['duration_s'] = time.monotonic() - start
    return result
