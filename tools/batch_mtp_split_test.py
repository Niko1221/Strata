#!/usr/bin/env python3
"""Exact batch-MTP lifecycle checks, including a completed or cancelled slot returning to solo.

Run with a split config and --extra "--batch-mtp --batch-groups 1 --no-prefill-borrow
--pcie-frac 0 --adapt-every 1000000 --prompt-cache 4096". This checks serial split batching;
BYIELD and pipelined groups are separate modes. Use batch_test.py for larger rotating cohorts.
"""
import argparse
import json
import re
import sys
from pathlib import Path

from batch_test import Engine, tokenizer
from batch_interleave_test import LONG, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--extra', default='')
    ap.add_argument('--keys', default='')
    ap.add_argument('--max-new', type=int, default=64)
    ap.add_argument('--long', type=int, default=1200)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg['tokenizer'])

    def chat(q):
        return tok.encode(f'<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',
                          parse_special=True)

    ids = lambda p: ','.join(map(str, p))
    A = chat('Write a Python function that merges overlapping intervals, then explain it.')
    B = chat('Compare TCP and QUIC: handshake, congestion control, multiplexing.')
    C = chat(LONG * (a.long // len(tok.encode(LONG)) + 1) + '\nSummarize the text above.')
    eng = Engine(a.exe, cfg, 3, {'STRATA_IQ_MT_MIN': '1'}, a.extra.split())
    eng.pending = []
    out = eng.lines()
    checks = []

    def request(p, n=None, slot=None):
        cmd = f'GEN {n or a.max_new}' if slot is None else f'BGEN {slot} {n or a.max_new}'
        return run(eng, out, f'{cmd} {a.keys} {ids(p)}', slot)

    def take():
        l = eng.pending.pop(0) if eng.pending else next(out)
        if l.startswith('ERR'):
            raise RuntimeError(l)
        return l

    def drain(got, done):
        while eng.pending or len(done) < len(got):
            l = take()
            if l.startswith('BT '):
                _, s, y = l.split()
                got[int(s)].append(int(y))
            elif l.startswith('BDONE '):
                done.add(int(l.split()[1]))

    def equal(name, got, ref):
        ok = got == ref
        checks.append({'check': name, 'pass': ok, 'tokens': got, 'reference': ref})
        print(name + ': ' + ('IDENTICAL' if ok else 'DIFFERS') + f' ({len(got)} tokens)', flush=True)

    try:
        refs = [request(p)[0] for p in (A, B, C)]
        A2 = A + refs[0] + chat('Now do the same in Rust.')
        ref2 = request(A2)[0]
        got = {0: [], 1: [], 2: []}
        done = set()
        first, active = request(A, slot=0)
        got[0] += first
        while active and len(got[0]) < 8:
            l = take()
            if l.startswith('BT 0 '):
                got[0].append(int(l.split()[2]))
            elif l.startswith('BDONE 0 '):
                active = False
        if not active:
            raise RuntimeError('A ended before staggered admission')
        for s, p in ((1, B), (2, C)):
            first, active = request(p, slot=s)
            got[s] += first
            if not active:
                done.add(s)
        drain(got, done)
        for s in got:
            equal(f'staggered slot {s}', got[s], refs[s])
        # Completed slot retains target and draft KV for the next turn.
        first, active = request(A2, slot=0)
        got2 = {0: first}
        drain(got2, set() if active else {0})
        equal('completed slot next turn', got2[0], ref2)
        # Move a completed slot back to GEN; demand exact continuation.
        A3 = A2 + got2[0] + chat('Explain its complexity.')
        request(chat('Unrelated session reset.'))
        solo3 = request(A3)[0]
        first, active = request(A2, slot=0)
        got_reset = {0: first}
        drain(got_reset, set() if active else {0})
        back3 = request(A3)[0]
        equal('completed slot to solo next turn', back3, solo3)
        # Cancel after eight tokens, then resume the exact prefix on GEN.
        first, active = request(A, slot=1)
        stopped = list(first)
        sent = False
        while active:
            l = take()
            if l.startswith('BT 1 '):
                stopped.append(int(l.split()[2]))
                if len(stopped) >= 8 and not sent:
                    eng.send('BSTOP 1')
                    sent = True
            elif l.startswith('BDONE 1 '):
                active = False
        if not sent or len(stopped) >= a.max_new:
            raise RuntimeError('cancellation did not leave a continuation to test')
        tail = request(A + stopped, a.max_new - len(stopped))[0]
        equal('cancelled slot to solo', stopped + tail, refs[0])
        first, active = request(B, slot=1)
        reused = {1: first}
        drain(reused, set() if active else {1})
        equal('cancelled slot reuse', reused[1], refs[1])
    finally:
        if eng.p.poll() is None:
            eng.send('QUIT')
            eng.p.wait(timeout=180)
    Path(eng.log_path + '.results.json').write_text(json.dumps(checks, indent=2))
    log = Path(eng.log_path).read_text(errors='replace')
    # Exactness is vacuous if the option silently fell back to ordinary batching.
    enabled = '--batch-mtp is off' not in log
    split = log.count('batch windows of up to 3 sequences (layers [') >= 2
    grouped = any(any(x == y for x, y in zip(rows, rows[1:]))
                  for rows in (m.split(',') for m in re.findall(
                      r'captured the batch window over slots ([0-9,]+)', log)))
    print(f'feature exercised: enabled={enabled}, split={split}, grouped={grouped}', flush=True)
    if not (enabled and split and grouped):
        raise RuntimeError('expected serial split batch MTP was not exercised')
    return 0 if checks and all(c['pass'] for c in checks) else 2


if __name__ == '__main__':
    sys.exit(main())
