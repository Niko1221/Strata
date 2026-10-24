#!/usr/bin/env python3
"""How much faster decode could be if the right experts were in VRAM ahead of time (docs/EXPERT_LOOKAHEAD.md).

Subcommands; the first three need no GPU:

    unions TRACE...                          distinct routed experts per layer for windows of 1..64 tokens
    simulate TRACE... --profile P [--slots]  copies per token a perfect lookahead prefetcher needs (Belady eviction)
    oracle-profile TRACE --tokens N --profile P --out O
                                             a profile whose first entries are exactly the experts TRACE's first N
                                             tokens route to, then P's ranking (pre-fills the VRAM cache with them)
    prefill-sweep --config C --pack DIR --out D [--chunks 4,16,...]
                                             one 1,023-token prompt read in chunks of T tokens: the cost of
                                             checking T tokens at once
    oracle-runs --config C --out D --prompt NAME=IDS:ORACLE_IDS:PROFILE:N ...
                                             greedy decode with the normal profile, with the adaptive tier off,
                                             and with an oracle profile

TRACE is a --dump-routing file: per record int32 layer, int32 k, k int32 expert ids, k float32 weights.
The recorded run is in docs/benchmarks/expert-lookahead-2026-10-19-rtx5070/.
"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
N_LAYERS = 48
BLOB_MB = 1.4            # Swift 1.5 IQ2_XS expert blob (the engine reports "largest blob 1.51 MB")
PCIE_GBS = 28.4          # pinned host -> device on the RTX 5070, tools/h2d_bench.cu


def rows_per_layer(path):
    """The routed expert ids per layer, one row per position in trace order."""
    raw = Path(path).read_bytes()
    at, per = 0, {layer: [] for layer in range(N_LAYERS)}
    while at < len(raw):
        layer, count = struct.unpack_from('<ii', raw, at)
        at += 8
        per[layer].append(struct.unpack_from(f'<{count}i', raw, at))
        at += count * 8                                   # int32 ids, then float32 weights
    return per


def read_profile(path):
    raw = Path(path).read_bytes()
    hdr = struct.unpack_from('<5I', raw, 4)
    n = hdr[4]
    ranked = [struct.unpack_from('<HH', raw, 24 + 4 * i) for i in range(n)]
    return raw, hdr, ranked


def cmd_unions(a):
    for trace in a.traces:
        per = rows_per_layer(trace)
        n = min(len(v) for v in per.values())
        print(f'{trace}: {n} positions per layer, last 64 used')
        for t in (1, 2, 4, 8, 16, 32, 64):
            counts = []
            for layer in range(N_LAYERS):
                r = per[layer][-64:]
                for s in range(0, len(r) - t + 1, t):
                    counts.append(len(set(x for row in r[s:s + t] for x in row)))
            print(f'  T={t:2d} windows/layer={len(counts) // N_LAYERS:2d} mean distinct experts/layer={statistics.mean(counts):6.1f} of 512')


def generated_sets(path):
    per = rows_per_layer(path)
    m = min(len(v) for v in per.values())
    return [{(layer, e) for layer in range(N_LAYERS) for e in per[layer][t]} for t in range(m - 64, m)]


def belady(seq, initial, slots, look):
    res, loads = set(initial), []
    for t, need in enumerate(seq):
        missing = need - res
        if missing:
            def nxt(x):
                for d in range(1, look + 1):
                    if t + d < len(seq) and x in seq[t + d]:
                        return d
                return 10 ** 9
            free = len(res) + len(missing) - slots
            if free > 0:
                for x in sorted((x for x in res if x not in need), key=nxt, reverse=True)[:free]:
                    res.discard(x)
            res |= missing
        loads.append(len(missing))
    return loads


def cmd_simulate(a):
    _, _, ranked = read_profile(a.profile)
    initial = set(ranked[:a.slots])
    for trace in a.traces:
        seq = generated_sets(trace)
        static = [len(s - initial) for s in seq]
        print(f'{trace}: routed per token {statistics.mean(len(s) for s in seq):.0f} (layer, expert) pairs; '
              f'misses against the static profile {statistics.mean(static):.1f}/token ({statistics.mean(static) / N_LAYERS:.2f}/layer)')
        for look in (1, 16, 32, 64):
            ld = belady(seq, initial, a.slots, look)
            m = statistics.mean(ld[8:])                   # the first tokens' cold loads are left out
            print(f'  lookahead {look:2d}: copies/token {statistics.mean(ld):6.1f} (after token 8: {m:6.1f}) '
                  f'-> PCIe {m * BLOB_MB / PCIE_GBS:5.2f} ms/token at {PCIE_GBS} GB/s')


def cmd_oracle_profile(a):
    raw, hdr, ranked = read_profile(a.profile)
    per = rows_per_layer(a.trace)
    cnt = Counter((layer, e) for layer in range(N_LAYERS) for row in per[layer][:a.tokens] for e in row)
    first = [k for k, _ in cnt.most_common()]
    chosen = set(first)
    new = first + [k for k in ranked if k not in chosen]
    assert len(new) == len(ranked) and len(set(new)) == len(new)
    out = b'STRP' + struct.pack('<5I', *hdr) + b''.join(struct.pack('<HH', l, e) for l, e in new) + raw[24 + 4 * len(ranked):]
    Path(a.out).write_bytes(out)
    print(f'{a.out}: {len(first)} experts first ({a.tokens} positions of {a.trace})')


def base_args(config):
    """The saved run config with the diagnostic overrides of 2026-10-19: context 4096, KV residency 0, mapped experts."""
    args = list(json.loads(Path(config).read_text())['args'])
    args[args.index('--max-context') + 1] = '4096'
    args[args.index('--kv-resident') + 1] = '0'
    return args + ['--mmap-experts']


def greedy_reference(args):
    """No MTP; native IQ packs need a window of 2, so dummy drafts are verified (not a plain-decode rate)."""
    args = list(args)
    i = args.index('--mtp')
    del args[i:i + 2]
    args[args.index('--spec') + 1] = '2'
    return args


def run(cmd, out, name):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=900)
    (out / f'{name}.command.json').write_text(json.dumps(cmd, indent=2))
    (out / f'{name}.stdout').write_text(r.stdout)
    (out / f'{name}.stderr').write_text(r.stderr)
    return r


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


def cmd_prefill_sweep(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ids_path = out / 'p1024.ids'
    if not ids_path.exists():
        tok = load_tokenizer(Path(a.pack))
        ids = tok.encode((ROOT / 'docs/HOW_IT_WORKS.md').read_text(), parse_special=False)[:1024]
        ids_path.write_text(' '.join(map(str, ids)))
    args = greedy_reference(base_args(a.config))
    if '--prefill' in args:
        j = args.index('--prefill')
        del args[j:j + 2]
    args += ['--max-new', '1', '--greedy', '--suffix-draft', '0', '--spec-corrupt', '1', '--tokens-file', str(ids_path)]
    results = []
    for i, t in enumerate(int(x) for x in a.chunks.split(',')):
        r = run([str(a.engine), *args, '--prefill', str(t)], out, f'prefill{t}.{i}')
        m = re.search(r'^prefill\s+(\d+) tokens in ([\d.]+) ms', r.stdout, re.M)
        row = {'T': t, 'rc': r.returncode}
        if m:
            n, ms = int(m[1]), float(m[2])
            row.update(tokens=n, ms=ms, ms_per_chunk=round(ms * t / n, 1), tok_s=round(n / ms * 1000, 1))
        results.append(row)
        print(json.dumps(row), flush=True)
    (out / 'prefill_results.json').write_text(json.dumps(results, indent=1))


def cmd_oracle_runs(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = {}
    for spec in a.prompt:
        name, rest = spec.split('=', 1)
        ids, oracle_ids, profile, n = rest.split(':')
        prompts[name] = (ids, oracle_ids, profile, int(n))
    base = greedy_reference(base_args(a.config))
    results = []
    order = [(p, m) for rep in range(2) for p in prompts
             for m in (['normal', 'control', 'oracle'] if rep == 0 else ['oracle', 'control', 'normal'])]
    for p, mode in order:
        ids, oracle_ids, profile, n = prompts[p]
        cmd = [str(a.engine), *base, '--max-new', str(n), '--greedy', '--suffix-draft', '0', '--spec-corrupt', '1',
               '--tokens-file', ids]
        if mode == 'oracle':
            cmd[cmd.index('--expert-profile') + 1] = profile
        if mode in ('oracle', 'control'):
            cmd += ['--adapt-min-gain', '1e9']            # the adaptive tier swaps nothing
        cmd += ['--stats']
        r = run(cmd, out, f'or.{p}.{mode}.{len(results)}')
        o = r.stdout
        row = {'prompt': p, 'mode': mode, 'rc': r.returncode}
        m = re.search(r'^decode\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s', o, re.M)
        if m:
            row.update(tokens=int(m[1]), ms_per_tok=round(float(m[2]) / int(m[1]), 2), tok_s=float(m[3]))
        m = re.search(r'CPU expert pool\s+([\d.]+) ms/token', o)
        row['pool_ms'] = float(m[1]) if m else None
        m = re.search(r'expert-cache hits\s+(\d+) of (\d+)', o)
        row['hit'] = round(int(m[1]) / int(m[2]), 4) if m else None
        m = re.search(r'^output\s*:(.*)$', o, re.M)
        row['same_ids'] = (m[1].split() if m else []) == Path(oracle_ids).read_text().split()[:n]
        results.append(row)
        print(json.dumps(row), flush=True)
    (out / 'oracle_results.json').write_text(json.dumps(results, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('unions')
    p.add_argument('traces', nargs='+')
    p.set_defaults(fn=cmd_unions)
    p = sub.add_parser('simulate')
    p.add_argument('traces', nargs='+')
    p.add_argument('--profile', default=str(ROOT / 'data/expert-profile.bin'))
    p.add_argument('--slots', type=int, default=4866, help='VRAM expert cache slots (the engine prints them)')
    p.set_defaults(fn=cmd_simulate)
    p = sub.add_parser('oracle-profile')
    p.add_argument('trace')
    p.add_argument('--tokens', type=int, required=True)
    p.add_argument('--profile', default=str(ROOT / 'data/expert-profile.bin'))
    p.add_argument('--out', required=True)
    p.set_defaults(fn=cmd_oracle_profile)
    for name, fn in (('prefill-sweep', cmd_prefill_sweep), ('oracle-runs', cmd_oracle_runs)):
        p = sub.add_parser(name)
        p.add_argument('--config', default=str(ROOT / 'strata-swift-iq2_xs.json'))
        p.add_argument('--engine', default=str(ROOT / 'engine/strata'))
        p.add_argument('--out', required=True)
        if name == 'prefill-sweep':
            p.add_argument('--pack', required=True, help='prepared pack directory (has tokenizer/)')
            p.add_argument('--chunks', default='4,16,32,64,64,32,16,8,4')
        else:
            p.add_argument('--prompt', action='append', required=True, metavar='NAME=IDS:ORACLE_IDS:PROFILE:N')
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
