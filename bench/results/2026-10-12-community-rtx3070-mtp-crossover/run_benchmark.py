import argparse, pathlib, json, os, sys, time, threading, hashlib, subprocess, traceback

H = pathlib.Path.home()
p = argparse.ArgumentParser(description='Fresh-prompt MTP on/off latency experiment; pins no default changes.')
p.add_argument('--strata-checkout', type=pathlib.Path, required=True)
p.add_argument('--engine', type=pathlib.Path, required=True)
p.add_argument('--config', type=pathlib.Path, required=True)
p.add_argument('--mtp', type=pathlib.Path, required=True)
p.add_argument('--output-dir', type=pathlib.Path, required=True)
cli = p.parse_args()
S = cli.strata_checkout.resolve()
D = cli.output_dir.resolve()
D.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(S), str(S / 'tools')]
from serve.server import StrataEngine
from serve.frontend import ChatTemplate
from strata_tokenizer import Tokenizer

cfg = json.loads(cli.config.read_text())
args = list(cfg['args'])
first = pathlib.Path(args[args.index('--native') + 1])
pack = pathlib.Path(args[args.index('--pack') + 1])
tok = Tokenizer.from_gguf(first)
template = ChatTemplate(pack / 'tokenizer/chat_template.jinja')
exe = cli.engine.resolve()
results = {
    'hardware_label': 'rtx3070', 'model': str(first), 'kv': 'int8',
    'commit': subprocess.check_output(['git', '-C', str(S), 'rev-parse', 'HEAD'], text=True).strip(),
    'binary_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
    'output_limit': 8192, 'cases': [], 'status': 'running',
    'notes': 'Fresh prompts; no reuse; sequential arms; MTP spec=4 vs off spec=2 matching previous benchmark. First experiment is one run per condition.'
}
def save():
    (D / 'results.json').write_text(json.dumps(results, indent=2))
def option(a, key, value):
    while key in a:
        i = a.index(key)
        del a[i:i+2]
    a.extend([key, str(value)])
def encode(text):
    return tok.encode(template.render([{'role': 'user', 'content': text}], enable_thinking=False), parse_special=True)
tail = ('Write a large Python module containing a bounded LRU cache, a retry decorator, a streaming CSV validator, '
        'a thread-safe task queue, a dependency scheduler, a disk cache, a JSON schema validator, a rate limiter, '
        'an event bus, a metrics registry, a configuration loader, and a file watcher. Include extensive docstrings '
        'and at least 20 unittest tests for EACH component. Output code only, and continue until all components '
        'and their tests are fully implemented. Do not abbreviate or summarize.')
head = 'Reference log records follow. Treat them as data.\n' + '\n'.join(
    'Record %d: sensor reading stable; timestamp valid; processing completed.' % i for i in range(100))
try:
    for target in [4096, 16384, 65536]:
        pad = target - len(encode(head + '\n' + tail))
        for _ in range(12):
            text = head + ' x' * pad + '\n' + tail
            ids = encode(text)
            if len(ids) == target:
                break
            pad += target - len(ids)
        assert len(ids) == target, len(ids)
        (D / ('prompt-%d.txt' % target)).write_text(text)
        # Alternate order to reduce systematic order bias.
        for mode in ([False, True] if target != 16384 else [True, False]):
            name = '%d-%s' % (target, 'on' if mode else 'off')
            a = list(args)
            if '--mtp' in a:
                i = a.index('--mtp'); del a[i:i+2]
            for flag, value in [('--max-context', target + 9216), ('--prompt-cache', 0),
                                ('--conversation-cache-mib', 0), ('--suffix-draft', 0),
                                ('--spec', 4 if mode else 2), ('--prefill', 'auto:8192'),
                                ('--expert-profile', S / 'data/expert-profile.bin')]:
                option(a, flag, value)
            if mode:
                a += ['--mtp', str(cli.mtp.resolve()), '--spec-min-p', '0.5']
            env = dict(os.environ)
            env['STRATA_DECODE_TIMING'] = '1'
            env['LD_LIBRARY_PATH'] = ':'.join(cfg.get('lib_dirs', []) + [env.get('LD_LIBRARY_PATH', '')])
            case = {'name': name, 'mtp': mode, 'input_target': target, 'args': a, 'status': 'starting'}
            results['cases'].append(case); save()
            engine = None
            try:
                engine = StrataEngine(str(exe), a, str(S), str(D / (name + '.log')), env)
                list(engine.generate(encode('Reply with READY.'), 16, {'temperature': 0}, threading.Event()))
                case['status'] = 'running'; save()
                start = time.monotonic(); arrivals = []; emitted = []
                for token in engine.generate(ids, 8192, {'temperature': 0}, threading.Event()):
                    if token is not None:
                        emitted.append(token); arrivals.append(time.monotonic() - start)
                timing = dict(engine.last)
                case.update(status='complete', input_tokens=len(ids), output_tokens=len(emitted),
                            timings=timing, wall_seconds=time.monotonic() - start,
                            first_token_seconds=arrivals[0] if arrivals else None,
                            decode_tps=len(emitted)*1000/timing['decode_ms'] if timing.get('decode_ms') else None,
                            prefill_tps=len(ids)*1000/timing['prompt_ms'] if timing.get('prompt_ms') else None,
                            output_sha256=hashlib.sha256(tok.decode(emitted).encode()).hexdigest())
                assert timing.get('reused', 0) == 0, timing
                if not mode:
                    assert timing.get('drafts_offered', 0) == 0, timing
                (D / (name + '-arrivals.json')).write_text(json.dumps(arrivals))
                (D / (name + '-ids.json')).write_text(json.dumps(emitted))
                (D / (name + '-output.txt')).write_text(tok.decode(emitted))
            except Exception as e:
                case.update(status='failed', error=repr(e)); traceback.print_exc()
            finally:
                if engine is not None:
                    engine.close()
                save()
    results['status'] = 'complete' if all(c['status'] == 'complete' for c in results['cases']) else 'failed'
except Exception as e:
    results.update(status='failed', error=repr(e)); traceback.print_exc()
finally:
    save()
