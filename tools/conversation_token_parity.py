"""Exact token-ID parity probe for fresh, RAM, and disk-restored conversations.

Dry-run by default. Run with a private model config and an explicit GPU window:

  python tools/conversation_token_parity.py --config model.json --engine build/strata \
      --output /tmp/token-parity --tier ram-fallback --temperature 0.7 --seed 12345 --run
"""
import argparse
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from serve.server import StrataEngine, child_env
from serve.frontend import ChatTemplate
import strata_tokenizer as ST


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_tokenizer(path):
    vocab = json.loads((path / 'vocab.json').read_text(encoding='utf-8'))
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    return ST.Tokenizer(tokens, (path / 'merges.txt').read_text(encoding='utf-8').split('\n'),
                        json.loads((path / 'token_type.json').read_text(encoding='utf-8')))


def engine_args(cfg, tier, cache_mib, spill_dir, spec):
    # Replace the relevant config defaults so the test cannot silently inherit another cache or routing policy.
    value_flags = {'--conversation-cache-mib', '--conversation-cache-slots', '--conversation-cache-spill-dir',
                   '--conversation-cache-disk-mib', '--conversation-cache-min-free-mib', '--prompt-cache',
                   '--adapt-swaps', '--pcie-frac', '--spec', '--mtp-max-t', '--suffix-draft', '--spec-min-p'}
    flag_only = {'--conversation-cache-disk-only', '--reproducible'}
    args = []
    skip_value = False
    for item in cfg['args']:
        if skip_value:
            skip_value = False
            continue
        if item in value_flags:
            skip_value = True
            continue
        if item in flag_only:
            continue
        args.append(item)
    args += ['--reproducible', '--prompt-cache', '6', '--conversation-cache-min-free-mib', '0',
             '--spec', str(max(2, spec)), '--mtp-max-t', str(spec), '--suffix-draft', '0', '--spec-min-p', '0']
    if tier == 'off':
        args += ['--conversation-cache-mib', '0', '--conversation-cache-slots', '1']
    elif tier == 'ram':
        args += ['--conversation-cache-mib', str(cache_mib), '--conversation-cache-slots', '1']
    elif tier == 'disk':
        args += ['--conversation-cache-disk-only', '--conversation-cache-spill-dir', str(spill_dir),
                 '--conversation-cache-disk-mib', '4096']
    elif tier == 'ram-fallback':
        args += ['--conversation-cache-mib', '1', '--conversation-cache-slots', '1',
                 '--conversation-cache-spill-dir', str(spill_dir), '--conversation-cache-disk-mib', '4096']
    return args


def verify_results(results, cache_tier):
    names = ('A-first', 'B', 'A-return', 'A-repeat')
    baseline, candidate = results['off'], results[cache_tier]
    require([r['name'] for r in baseline] == list(names), 'incomplete fresh run')
    require([r['name'] for r in candidate] == list(names), 'incomplete cached run')
    for record in baseline + candidate:
        require(record['finish'] in ('length', 'stop'), f"{record['name']} did not finish normally")
        require(record['ids'], f"{record['name']} returned no token IDs")
    for label, records in (('fresh', baseline), ('cached', candidate)):
        first, returned, repeated = (next(r for r in records if r['name'] == name)['ids']
                                     for name in ('A-first', 'A-return', 'A-repeat'))
        require(first == returned == repeated, f'{label}: same prompt changed token IDs across repeats')
    for name in ('A-first', 'A-return', 'A-repeat'):
        fresh = next(r for r in baseline if r['name'] == name)
        cached = next(r for r in candidate if r['name'] == name)
        require(fresh['ids'] == cached['ids'], f'{name}: cache path changed token IDs')
    returned = next(r for r in candidate if r['name'] == 'A-return')
    require(returned.get('reused', 0) > 0, f'{cache_tier}: A did not resume from a saved prefix')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True, help='new output directory; existing paths refused')
    ap.add_argument('--tier', choices=('ram', 'disk', 'ram-fallback'), default='ram')
    ap.add_argument('--cache-mib', type=int, default=8192)
    ap.add_argument('--paragraphs', type=int, default=128)
    ap.add_argument('--max-new', type=int, default=64)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--top-k', type=int, default=20)
    ap.add_argument('--top-p', type=float, default=0.95)
    ap.add_argument('--spec', type=int, default=1, choices=range(1, 9))
    ap.add_argument('--run', action='store_true')
    a = ap.parse_args()
    if a.cache_mib <= 0 or a.paragraphs < 1 or a.max_new < 1:
        ap.error('cache-mib, paragraphs, and max-new must be positive')
    if a.temperature < 0 or a.top_k < 1 or not 0 < a.top_p <= 1:
        ap.error('temperature must be nonnegative, top-k positive, and top-p in (0,1]')
    if not a.run:
        print(f'Dry run: exact token parity; greedy or seeded sampling; tier={a.tier}.')
        print('No model loaded. Use --run only with a separately available GPU/test window.')
        return

    cfg = json.loads(a.config.read_text(encoding='utf-8'))
    tokenizer = load_tokenizer(Path(cfg['tokenizer']))
    template = ChatTemplate(Path(cfg['tokenizer']) / 'chat_template.jinja')
    def prompt(label):
        text = label + ': remember this list.\n' + '\n'.join(
            f'Record {i}: blue square, green triangle, red circle.' for i in range(a.paragraphs))
        return tokenizer.encode(template.render([{'role': 'user', 'content': text}], enable_thinking=False),
                                parse_special=True)
    A, B = prompt('Conversation A'), prompt('Unrelated conversation B')
    a.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    sampling = {'temperature': a.temperature}
    if a.temperature > 0:
        sampling.update(top_k=a.top_k, top_p=a.top_p, seed=a.seed)
    results = {}
    for tier in ('off', a.tier):
        log = a.output / f'{tier}.log'
        spill = a.output / f'{tier}-spill'
        args = engine_args(cfg, tier, a.cache_mib, spill, a.spec)
        engine = StrataEngine(str(a.engine.resolve()), args, cwd=cfg.get('cwd'), log=str(log), env=child_env(cfg))
        records = []
        def generate(ids, count, name):
            out = [t for t in engine.generate(ids, count, sampling, threading.Event()) if t is not None]
            records.append({'name': name, 'ids': out, **engine.last})
            return out
        try:
            generate(A, a.max_new, 'A-first')
            generate(B, 1, 'B')
            generate(A, a.max_new, 'A-return')
            generate(A, a.max_new, 'A-repeat')
        finally:
            engine.close()
        results[tier] = records
    (a.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    verify_results(results, a.tier)
    mode = 'greedy' if a.temperature == 0 else f'seeded sampling (seed={a.seed}, temperature={a.temperature})'
    print(f'PASS: {mode}; fresh and {a.tier} token IDs match across repeated requests.')
    print(f'Results: {a.output / "results.json"}')


if __name__ == '__main__':
    main()
