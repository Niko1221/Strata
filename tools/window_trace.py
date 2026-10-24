#!/usr/bin/env python3
"""Read a STRATA_WINDOW_TRACE file (one record per decode window of the serial serve loop) and print or export it.

  python3 tools/window_trace.py TRACE            per-request summary and the slowest windows
  python3 tools/window_trace.py TRACE --csv OUT  every window as CSV

Times are wall milliseconds on the engine's thread: t (window start since the request's decode start), total (start to the
record), pre (the adaptive tier's wait before the window), ram_adapt (STRATA_RAM_ADAPT moves), tick (--adapt-async step),
setup (penalties, K/V growth), run (the verify window), commit (commit and emit), draft (the MTP draft), join (after the
draft: PCIe share update, the synchronous adaptive round's join).  wait/pool/jobs are the verifier's and the expert pool's
own sums during the window (overlapping threads, not wall time).  Counts are deltas over the window: vram_hits, cpu_entries
and cpu_misses (CPU expert work), pcie, ram_reads and file_reads (expert blobs by tier), file_bytes, ram_moved
(STRATA_RAM_ADAPT moves), ram_staged (of them, moved without a file read; -1: this engine does not count it) and
gpu_swapped (adaptive tier swaps).  flags: 1 first window, 2 merged with the prompt, 4 prompt-lookup window, 8 chained."""
import argparse, csv, struct, sys
FIELDS = ('req win T accepted flags pad pos t total pre ram_adapt tick setup run commit draft join wait pool jobs '
          'vram_hits cpu_entries cpu_misses pcie ram_reads file_reads file_bytes ram_moved ram_staged gpu_swapped').split()
FMT = '<6iq13d10q'
MAGIC = b'STRATA_WT1\n\0'

def read(path):
    with open(path, 'rb') as f: data = f.read()
    if data[:12] != MAGIC: raise SystemExit(f'{path}: not a STRATA_WINDOW_TRACE file')
    size = struct.unpack_from('<I', data, 12)[0]
    if size != struct.calcsize(FMT): raise SystemExit(f'{path}: record size {size}, this reader knows {struct.calcsize(FMT)}')
    body = data[16:]
    return [dict(zip(FIELDS, struct.unpack_from(FMT, body, i))) for i in range(0, len(body) - len(body) % size, size)]

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0]); ap.add_argument('trace'); ap.add_argument('--csv')
    a = ap.parse_args(); rows = read(a.trace)
    if a.csv:
        with open(a.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=[k for k in FIELDS if k != 'pad'], extrasaction='ignore'); w.writeheader(); w.writerows(rows)
        print(f'{len(rows)} windows -> {a.csv}'); return
    reqs = {}
    for r in rows: reqs.setdefault(r['req'], []).append(r)
    print(f'{len(rows)} windows, {len(reqs)} requests')
    keys = ('total', 'pre', 'ram_adapt', 'tick', 'setup', 'run', 'commit', 'draft', 'join')
    n = len(rows)
    print('ms/window: ' + ', '.join(f'{k} {sum(r[k] for r in rows) / n:.2f}' for k in keys))
    print('per window: ' + ', '.join(f'{k} {sum(r[k] for r in rows) / n:.2f}' for k in ('vram_hits', 'cpu_entries', 'ram_reads', 'file_reads', 'ram_moved', 'gpu_swapped')))
    tot = sorted(r['total'] for r in rows)
    print('total ms p50 %.2f p90 %.2f p99 %.2f max %.2f' % tuple(tot[min(n - 1, int(q * n))] for q in (0.5, 0.9, 0.99, 1.0)))

if __name__ == '__main__':
    main()
