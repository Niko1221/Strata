"""Text-only release smoke test. Run in a fresh output directory, with one idle GPU visible.
Args: /path/to/strata-inkling /path/to/terminal /path/to/model.gguf /path/to/exported-tokenizer /path/to/dflash2.safetensors
The tokenizer directory must contain the Strata vocab/merges files and the model's chat_template.jinja.
"""
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.request as url

exe, terminal, model, tokenizer, drafter = map(lambda x: Path(x).resolve(), sys.argv[1:])
opener = url.build_opener(url.ProxyHandler({}))
root = Path.cwd()
key = secrets.token_urlsafe(32)
rows = []
for mode in ('off', 'dflash2'):
    arm = root / mode
    arm.mkdir(mode=0o700)
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
    listener.close()
    args = ['--native', str(model), '--gpu', '0', '--parallel', '1', '--max-context', '4096', '--draft', mode]
    cfg = dict(arch='inkling2_loop', exe=str(exe), args=args, cwd=str(terminal.parent),
               tokenizer=str(tokenizer), model_name='inkling-test', log=str(arm/'engine.log'), host='127.0.0.1', port=port)
    (arm/'cfg.json').write_text(json.dumps(cfg))
    env = {**os.environ, 'HF_HUB_OFFLINE':'1', 'STRATA_API_KEY':key, 'HOME':str(arm), 'STRATA_INK_DFLASH2':str(drafter)}
    def request(route, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = url.Request(f'http://127.0.0.1:{port}/v1/{route}', data=data,
                          headers={'Content-Type':'application/json', 'Authorization':'Bearer '+key})
        with opener.open(req, timeout=90) as resp:
            return json.load(resp)
    with (arm/'server.log').open('x') as log:
        process = subprocess.Popen([sys.executable, str(terminal/'serve/server.py'), '--engine', 'strata',
                    '--config', str(arm/'cfg.json'), '--host', '127.0.0.1', '--port', str(port)],
                    cwd=terminal, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError(f'{mode}: frontend exited ({process.returncode})')
                try: request('models'); break
                except OSError: time.sleep(1)
            else: raise RuntimeError('readiness timeout')
            def chat(text, limit, **extra):
                return request('chat/completions', dict(model='inkling-test', messages=[dict(role='user',content=text)],
                               reasoning_effort='none', temperature=0, seed=42, max_tokens=limit, **extra))
            warm = chat('What is 17 times 23? Give just the integer.',256)
            arithmetic = (warm['choices'][0]['message'].get('content') or '').strip().strip('.') == '391'
            tools = [{'type':'function','function':{'name':'weather','description':'Look up the temperature for a city.',
                     'parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}]
            r = chat('Use the weather tool to look up the temperature in Paris.',256,tools=tools)
            calls = r['choices'][0]['message'].get('tool_calls') or []
            tool_ok = len(calls)==1 and calls[0]['function']['name']=='weather' and json.loads(calls[0]['function']['arguments']).get('city','').lower()=='paris'
            for i in range(3):
                t = time.monotonic()
                r = chat('Explain how merge sort works, with a detailed example using twelve numbers.',128)
                elapsed = time.monotonic()-t
                c = r['choices'][0]; text = c['message'].get('content') or ''; n = r['usage']['completion_tokens']
                row = dict(mode=mode, repeat=i+1, arithmetic_ok=arithmetic, weather_tool_ok=tool_ok,
                           completion_tokens=n, wall_seconds=elapsed, end_to_end_tokens_per_second=n/elapsed,
                           finish_reason=c['finish_reason'], answer_sha256=hashlib.sha256(text.encode()).hexdigest())
                rows.append(row); print(json.dumps(row),flush=True)
        finally:
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired: pass
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait()
    time.sleep(2)
(root/'release-chat.json').write_text(json.dumps(rows,indent=2)+'\n')
