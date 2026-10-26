#!/usr/bin/env python3
"""A sequence on its own answers what sequence 0 does (docs/DSV41_BATCHING.md, M2's per-sequence regions).

    python tools/dsv41/batch_seq_parity.py --exe build/strata-dsv41.batch --pack <pack> [--tokens 12] [--long 140]

M2 puts each sequence's window ring at its own rows of the window buffer and its compressed KV and index keys at kv_base
rows of each source's table; with one sequence every address is row 0, and a row counted wrong does not crash - it answers
subtly wrong.  So this runs the same prompts on sequence 0 and on sequence 1 (STRATA_DS4_VSEQ=1, which brings its own
nseq = 2) in two engine loads and compares what the model writes: the two token streams must be identical.

The prompts are picked for the region's edges: a short one (the prompt's own read and a few tokens), a long one (300
tokens read, 140 written: past the 128-token window ring and across many compression groups), and a third that shares
a prefix with the first (the model's own checkpoint / restore, which now goes through the sequence's ring).

Exit 0 when both sequences answered identically, 1 with the checks that differ.
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time


class Engine:
    """The engine's stdout on a queue (the batch probe's reader)."""

    def __init__(self, argv, env):
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, bufsize=1, env=env)
        self.q = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            self.q.put(line.rstrip("\n"))
        self.q.put(None)

    def start(self, timeout):
        end = time.time() + timeout
        while True:
            left = end - time.time()
            if left <= 0:
                raise SystemExit("the engine did not say READY within %d s" % timeout)
            line = self._next(left)
            if line.startswith("READY"):
                return line

    def _next(self, timeout):
        try:
            line = self.q.get(timeout=max(0.1, timeout))
        except queue.Empty:
            raise SystemExit("the engine was silent for %.0f s" % timeout)
        if line is None:
            raise SystemExit("the engine ended (exit %s)" % self.p.poll())
        return line

    def line(self, timeout=900):
        while True:
            line = self._next(timeout)
            if line.startswith(("PP ", "STAT")):
                continue
            return line

    def gen(self, ids, max_new, timeout=900):
        self.p.stdin.write("GEN %d %s\n" % (max_new, ",".join(str(int(t)) for t in ids)))
        self.p.stdin.flush()
        toks = []
        while True:
            line = self.line(timeout)
            if line.startswith("T "):
                toks.append(int(line.split()[1]))
            elif line.startswith("DONE"):
                return toks
            elif line.startswith("ERR"):
                raise SystemExit("GEN: " + line)

    def close(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=15)
        except Exception:
            self.p.kill()


def run(exe, pack, seq, items, timeout):
    argv = [exe, "--pack", pack, "--serve", "--max-context", "8192", "--eos", "999999", "--dspark", "off",
            "--tp", "2", "--experts", "6", "--mmap-experts", "auto"]
    env = dict(os.environ)
    env["STRATA_DS4_VSEQ"] = str(seq)
    e = Engine(argv, env)
    try:
        e.start(timeout)
        return [e.gen(ids, n) for _, ids, n in items]
    finally:
        e.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tokens", type=int, default=12, help="tokens for the short prompt's answer")
    ap.add_argument("--long", type=int, default=140, help="tokens for the long prompt's answer (past the 128-ring)")
    ap.add_argument("--timeout", type=int, default=1800, help="how long the model may take to load")
    a = ap.parse_args()

    short = [1000 + i for i in range(40)]
    long_ = [20000 + (i * 37) % 4096 for i in range(300)]
    shared = short[:20] + [7000 + i for i in range(12)]
    items = [("short", short, a.tokens), ("long", long_, a.long), ("shared prefix", shared, 8)]

    print("sequence 0: loading", flush=True)
    ref = run(a.exe, a.pack, 0, items, a.timeout)
    print("sequence 1: loading", flush=True)
    got = run(a.exe, a.pack, 1, items, a.timeout)

    fails = []
    for (name, _, n), r, g in zip(items, ref, got):
        ok = r == g
        print(("ok   " if ok else "FAIL ") + "%s: sequence 1 writes what sequence 0 does (%d tokens)" % (name, n) +
              ("" if ok else "  (0: %s vs 1: %s)" % (r[:8], g[:8])), flush=True)
        if not ok:
            fails.append(name)
    if fails:
        print("\n%d check(s) failed: %s" % (len(fails), ", ".join(fails)))
        return 1
    print("\nbatch_seq_parity: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
