"""Compare ordinary and active-layer prefill on one otherwise idle GPU.

Use a working native engine config and a fresh output directory. The source
binary must contain the experimental scheduler. No serving process is changed.
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from serve.server import StrataEngine, child_env, engine_args
from serve.frontend import ChatTemplate
from strata_tokenizer import Tokenizer


def check_staging_profiles(logs):
    profiles = {arm: ('auto-RAM' if 'stager takes the RAM profile' in log else 'default/explicit')
                for arm, log in logs.items()}
    if len(set(profiles.values())) != 1:
        raise ValueError(f'staging profiles differ: {profiles}; warm the model files and rerun both arms')
    return profiles


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--tokens', nargs='+', type=int, default=[4096, 8192])
    ap.add_argument('--chunk', default='auto')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--generate', type=int, default=64)
    ap.add_argument('--reference-exe', help='Optional unmodified binary for the off arm')
    ap.add_argument('--cancel-test', action='store_true', help='Cancel active prefill and check the next request')
    opts = ap.parse_args()
    if opts.repeats < 1 or opts.generate < 1 or min(opts.tokens) < 512:
        ap.error('positive repeats/output count and prompt sizes >= 512 required')
    out = Path(opts.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    base = json.loads(Path(opts.config).read_text())
    args = base['args']
    def set_arg(flag, value):
        if flag in args:
            args[args.index(flag) + 1] = str(value)
        else:
            args.extend([flag, str(value)])
    for flag, value in [('--prefill', opts.chunk), ('--prompt-cache', 0),
                        ('--conversation-cache-mib', 0), ('--adapt-every', 0)]:
        set_arg(flag, value)
    base.setdefault('env', {}).update(STRATA_PREFILL_CPU_SHARE='0', STRATA_PREFILL_ROOFLINE='1')
    for key in ('STRATA_PREFILL_DUMP_R', 'STRATA_PREFILL_DUMP_R_ALL'):
        base['env'].pop(key, None)
    tok = Tokenizer.from_gguf(Path(args[args.index('--native') + 1]))
    template = ChatTemplate(Path(base['tokenizer']) / 'chat_template.jinja')
    before, after = template.render([{'role': 'user', 'content':
        'Background notes:\nFILLER_HERE\nEnd of notes.\nTask: Explain how a Linux administrator diagnoses CPU, memory, disk and network bottlenecks.'}],
        enable_thinking=False).split('FILLER_HERE')
    aa, bb = tok.encode(before, parse_special=True), tok.encode(after, parse_special=True)
    filler = tok.encode('Archived maintenance notes contain background information about ordinary testing and documentation.\n')
    prompts = {n: aa + (filler * ((n - len(aa) - len(bb) + len(filler) - 1) // len(filler)))[:n-len(aa)-len(bb)] + bb
               for n in set([512] + opts.tokens)}
    assert all(len(p) == n for n, p in prompts.items())
    report = dict(options=vars(opts), prompts=prompts, runs=[], complete=False)
    def save():
        (out / 'results.json').write_text(json.dumps(report, indent=2))
    try:
        for enabled in [False, True]:
            cfg = json.loads(json.dumps(base))
            label = 'on' if enabled else 'off'
            cfg['env']['STRATA_PREFILL_LAYER_CACHE'] = str(int(enabled))
            if not enabled and opts.reference_exe:
                cfg['exe'] = str(Path(opts.reference_exe).resolve())
            cfg['log'] = str(out / (label + '.log'))
            (out / (label + '-config.json')).write_text(json.dumps(cfg, indent=2))
            engine = StrataEngine(cfg['exe'], engine_args(cfg), cfg['cwd'], cfg['log'], child_env(cfg))
            try:
                plan = [(512, -1)] + [(n, repeat) for n in opts.tokens for repeat in range(opts.repeats)]
                for n, repeat in plan:
                    offset = Path(cfg['log']).stat().st_size
                    started = time.monotonic()
                    timer = threading.Timer(900, engine.proc.kill)
                    timer.start()
                    try:
                        # Warm decode too, so captured MTP graphs occupy the same VRAM in both arms.
                        output = [t for t in engine.generate(prompts[n], opts.generate,
                                  {'temperature': 0}, threading.Event()) if t is not None]
                    finally:
                        timer.cancel()
                    row = dict(enabled=enabled, tokens=n, repeat=repeat, output=output,
                               timing=dict(engine.last), wall_s=time.monotonic()-started,
                               log=Path(cfg['log']).read_bytes()[offset:].decode(errors='replace'))
                    report['runs'].append(row)
                    save()
                    assert output and row['timing'].get('reused', 0) == 0
                    print(label, n, repeat, 'prompt_ms', row['timing']['prompt_ms'], flush=True)
                if enabled and opts.cancel_test:
                    cancel = threading.Event()
                    trigger = threading.Timer(2, cancel.set)
                    watchdog = threading.Timer(60, engine.proc.kill)
                    started = time.monotonic()
                    trigger.start()
                    watchdog.start()
                    try:
                        cancelled = [t for t in engine.generate(prompts[max(opts.tokens)], opts.generate,
                                     {'temperature': 0}, cancel) if t is not None]
                    finally:
                        trigger.cancel()
                        watchdog.cancel()
                    report['cancellation'] = dict(output=cancelled, timing=dict(engine.last), wall_s=time.monotonic()-started)
                    save()
                    assert not cancelled and engine.last.get('finish') == 'cancel', report['cancellation']
                    recovery = [t for t in engine.generate(prompts[512], 1, {'temperature': 0}, threading.Event()) if t is not None]
                    assert recovery == report['runs'][0]['output'][:1], 'post-cancellation state leaked'
                    report['cancellation']['recovery_output'] = recovery
                    print('cancellation and recovery passed', flush=True)
            finally:
                engine.close()
        for n in opts.tokens:
            rows = [r for r in report['runs'] if r['tokens'] == n and r['repeat'] >= 0]
            assert all(r['output'] == rows[0]['output'] for r in rows), f'token mismatch at {n}'
        assert any(r['enabled'] and 'strata layer cache: window' in r['log'] for r in report['runs']), 'scheduler never activated'
        report['staging_profiles'] = check_staging_profiles(
            {arm: (out / (arm + '.log')).read_text(errors='replace') for arm in ['off', 'on']})
        report['complete'] = True
    finally:
        save()


if __name__ == '__main__':
    main()
