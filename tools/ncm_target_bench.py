#!/usr/bin/env python3
"""Target pilot for the NCM feasibility study (docs/NCM.md): MTP, default lookup, greedy reference and oracle runs.

Runs the existing engine binary ten times (two prompts, five modes), 64 greedy tokens each, and writes the
commands, stdout, stderr, routing traces and results.json into --out. Saved presets are not edited. The
recorded run is in docs/benchmarks/ncm-2026-10-19-rtx5070/.

    python tools/ncm_target_bench.py --pack DATA/packs/swift-iq2_xs --config strata-swift-iq2_xs.json \
        --out /tmp/strata-ncm-target-v2
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = {
    'prose': 'Explain how a bicycle stays balanced in three short paragraphs.',
    'code': 'Change only the function name add to sum_pair and return the complete file.\n```python\ndef add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n\nassert add(2, 3) == 5\n```',
}
MODES = ['plain', 'mtp', 'default', 'oracle4', 'oracle8']


def unique_experts(path, expected_t):
    """Distinct routed experts per layer per complete window, from a --dump-routing trace."""
    raw = path.read_bytes()
    records = []
    at = 0
    while at < len(raw):
        layer, count = struct.unpack_from('<ii', raw, at)
        at += 8
        ids = struct.unpack_from(f'<{count}i', raw, at)
        at += count * 8                                   # int32 IDs, then float32 weights
        records.append((layer, ids))
    groups, group, previous = [], {}, -1
    for layer, ids in records:
        if layer < previous:
            groups.append(group)
            group = {}
        group.setdefault(layer, []).append(ids)
        previous = layer
    if group:
        groups.append(group)
    windows = [g for g in groups if len(g) == 48 and all(len(rows) == expected_t for rows in g.values())]
    counts = [[len(set(x for row in g[layer] for x in row)) for layer in range(48)] for g in windows]
    return {'full_windows': len(windows), 'unique_per_layer_per_window': counts,
            'mean_unique_per_layer': statistics.mean(x for row in counts for x in row) if counts else None}


def load_tokenizer(pack):
    spec = importlib.util.spec_from_file_location('strata_tokenizer', ROOT / 'tools/strata_tokenizer.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    vocab = json.loads((pack / 'tokenizer/vocab.json').read_text())
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    return module.Tokenizer(tokens, (pack / 'tokenizer/merges.txt').read_text().splitlines(),
                            json.loads((pack / 'tokenizer/token_type.json').read_text()))


def mode_args(base, mode, name, out):
    args = list(base)
    if mode != 'default':
        args += ['--suffix-draft', '0']
    if mode == 'plain' or mode.startswith('oracle'):
        i = args.index('--mtp')
        del args[i:i + 2]
    if mode == 'plain':
        # Native IQ packs require T >= 2, even without MTP. Use dummy
        # drafts for a greedy reference; this is not a plain-decode rate.
        args[args.index('--spec') + 1] = '2'
        args += ['--spec-corrupt', '1']
    elif mode.startswith('oracle'):
        t = int(mode[6:])
        args[args.index('--spec') + 1] = str(t)
        args[args.index('--spec-min-p') + 1] = '0'
        args += ['--spec-oracle', str(out / f'{name}.oracle.ids'), '--dump-routing', str(out / f'{name}.{mode}.routing.bin')]
    return args


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--pack', type=Path, required=True, help='prepared pack directory (has tokenizer/)')
    ap.add_argument('--config', type=Path, default=ROOT / 'strata-swift-iq2_xs.json', help='saved engine arguments')
    ap.add_argument('--engine', type=Path, default=ROOT / 'engine/strata')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from serve.frontend import ChatTemplate

    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(a.pack)
    template = ChatTemplate(a.pack / 'tokenizer/chat_template.jinja')
    base = list(json.loads(a.config.read_text())['args'])
    base[base.index('--max-context') + 1] = '4096'
    base[base.index('--kv-resident') + 1] = '0'
    base += ['--mmap-experts', '--max-new', '64', '--greedy']

    results = []
    for name, prompt in PROMPTS.items():
        ids = tokenizer.encode(template.render([{'role': 'user', 'content': prompt}], enable_thinking=False), parse_special=True)
        token_path = out / f'{name}.prompt.ids'
        token_path.write_text(' '.join(map(str, ids)))
        (out / f'{name}.prompt.txt').write_text(prompt)
        baseline_ids = None
        for mode in MODES:
            command = [str(a.engine), *mode_args(base, mode, name, out), '--tokens-file', str(token_path)]
            (out / f'{name}.{mode}.command.json').write_text(json.dumps(command, indent=2))
            print(f'START {name} {mode}', flush=True)
            start = time.perf_counter()
            with (out / f'{name}.{mode}.stdout').open('w') as stdout, (out / f'{name}.{mode}.stderr').open('w') as stderr:
                run = subprocess.run(command, stdout=stdout, stderr=stderr, cwd=ROOT, timeout=600)
            output = (out / f'{name}.{mode}.stdout').read_text()
            result = {'prompt': name, 'mode': mode, 'exit_code': run.returncode,
                      'wall_s': time.perf_counter() - start, 'command': command}
            if run.returncode:
                result['error_tail'] = (out / f'{name}.{mode}.stderr').read_text()[-3000:]
                results.append(result)
                (out / 'results.json').write_text(json.dumps(results, indent=2))
                print(json.dumps(result), flush=True)
                raise SystemExit(run.returncode)
            line = re.search(r'^output\s*:(.*)$', output, re.M)
            generated = list(map(int, line.group(1).split()))
            if mode == 'plain':
                baseline_ids = generated
                (out / f'{name}.oracle.ids').write_text(' '.join(map(str, generated)))
            result['tokens'] = generated
            result['equals_plain'] = generated == baseline_ids
            decode = re.search(r'^decode\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s', output, re.M)
            result.update(generated=int(decode[1]), decode_ms=float(decode[2]), tps=float(decode[3]))
            result['speculation'] = re.findall(r'^speculation.*$', output, re.M)
            result['verify_window'] = re.findall(r'^verify window.*$', output, re.M)
            if mode.startswith('oracle'):
                result['routing'] = unique_experts(out / f'{name}.{mode}.routing.bin', int(mode[6:]))
            results.append(result)
            (out / 'results.json').write_text(json.dumps(results, indent=2))
            print(json.dumps({k: v for k, v in result.items() if k not in ('command', 'tokens', 'routing')}), flush=True)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
