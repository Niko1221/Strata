#!/usr/bin/env python3
"""The per-row chunk path answers what the contiguous one does (STRATA_DS4_STEPROWS=1).

    python3 tools/dsv41/steprows_check.py --exe build/strata-dsv41.batch --pack <pack>

Two engine runs of the same prompt - one with the rows described one by one (STRATA_DS4_STEPROWS=1, M2's step shape) and
one without - with small prompt chunks (--prefill-chunk 8) so every chunk takes the per-row path.  The tokens must be
identical.
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
            try:
                line = self.q.get(timeout=max(0.1, left))
            except queue.Empty:
                raise SystemExit("the engine was silent")
            if line is None:
                raise SystemExit("the engine ended")
            if line.startswith("READY"):
                return line

    def gen(self, ids, max_new, timeout=900):
        self.p.stdin.write("GEN %d %s\n" % (max_new, ",".join(str(int(t)) for t in ids)))
        self.p.stdin.flush()
        toks = []
        while True:
            try:
                line = self.q.get(timeout=timeout)
            except queue.Empty:
                raise SystemExit("the engine was silent during GEN")
            if line is None:
                raise SystemExit("the engine ended during GEN")
            if line.startswith("PP "):
                continue
            if line.startswith("T "):
                toks.append(int(line.split()[1]))
            elif line.startswith("DONE"):
                return toks
            elif line.startswith("ERR"):
                raise SystemExit(line)

    def close(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=15)
        except Exception:
            self.p.kill()


def run(exe, pack, steprows, prompt, n, timeout, tp=4):
    argv = [exe, "--pack", pack, "--serve", "--max-context", "8192", "--eos", "999999", "--dspark", "off",
            "--tp", str(tp), "--prefill-chunk", "8"]
    env = dict(os.environ)
    env['STRATA_PREFILL_CPU_SHARE'] = '0'   # (the CPU prefill share makes small chunks non-deterministic)
    if steprows:
        env["STRATA_DS4_STEPROWS"] = "1"
    e = Engine(argv, env)
    try:
        e.start(timeout)
        return e.gen(prompt, n)
    finally:
        e.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tokens", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--tp", type=int, default=4, help="tensor-parallel group size (4: one group of the four cards)")
    a = ap.parse_args()
    prompt = [20000 + (i * 37) % 4096 for i in range(300)]
    print("contiguous rows: loading", flush=True)
    ref = run(a.exe, a.pack, False, prompt, a.tokens, a.timeout, a.tp)
    print("contiguous rows again (is the engine deterministic?): loading", flush=True)
    ref2 = run(a.exe, a.pack, False, prompt, a.tokens, a.timeout, a.tp)
    print("per-row (STEPROWS): loading", flush=True)
    got = run(a.exe, a.pack, True, prompt, a.tokens, a.timeout, a.tp)
    print(("ok   " if ref == ref2 else "note ") + "the contiguous path against itself" +
          (" (deterministic)" if ref == ref2 else " (NOT deterministic: %s vs %s)" % (ref[:8], ref2[:8])), flush=True)
    ok = ref == got and ref == ref2
    print(("ok   " if ok else "FAIL ") + "the per-row chunk path writes what the contiguous one does (%d tokens)" % a.tokens +
          ("" if ok else "  (contiguous %s vs rows %s)" % (ref[:8], got[:8])), flush=True)
    print("\nsteprows_check: " + ("all checks passed" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
