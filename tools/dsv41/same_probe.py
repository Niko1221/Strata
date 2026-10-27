#!/usr/bin/env python3
"""One prompt for every slot: the slots must all answer the same tokens (a row of a pass must not see another row's
state).  The bursts are the control; the step is the pass under test.

    same_probe.py --exe build/strata-dsv41.step15 --pack <pack> --slots 16 --tokens 32 [--step]

Exit 0 when every slot's tokens are slot 0's, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time


class Engine:
    def __init__(self, argv, env, stderr_path):
        self.err = open(stderr_path, "ab")
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True,
                                  bufsize=1, env=env)
        self.q = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.info = {}

    def _read(self):
        for line in self.p.stdout:
            self.q.put(line.rstrip("\n"))
        self.q.put(None)

    def next(self, timeout):
        try:
            line = self.q.get(timeout=max(0.1, timeout))
        except queue.Empty:
            raise SystemExit("the engine was silent for %.0f s" % timeout)
        if line is None:
            raise SystemExit("the engine ended (exit %s)" % self.p.poll())
        return line

    def start(self, timeout):
        end = time.time() + timeout
        while True:
            line = self.next(end - time.time())
            if line.startswith("INFO "):
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    self.info[k] = v
            elif line.startswith("READY"):
                return line

    def send(self, text):
        self.p.stdin.write(text + "\n")
        self.p.stdin.flush()

    def close(self):
        try:
            self.send("QUIT")
            self.p.wait(timeout=20)
        except Exception:
            self.p.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--max-context", type=int, default=1048576)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--read-timeout", type=int, default=1800)
    ap.add_argument("--extra", default="")
    ap.add_argument("--step", action="store_true")
    ap.add_argument("--alt", action="store_true", help="two prompts: the even slots one, the odd slots another")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="", help="write every slot's tokens to this JSON")
    a = ap.parse_args()

    argv = [a.exe, "--pack", a.pack, "--serve", "--batch", str(a.slots), "--max-context", str(a.max_context),
            "--eos", "999999", "--dspark", "off", "--tp", "2", "--experts", "6", "--mmap-experts", "auto"]
    argv += [x for x in a.extra.split(" ") if x]
    env = dict(os.environ)
    env["STRATA_DS4_REUSE"] = "0"
    env["STRATA_DS4_BATCH_LIVE"] = "1"
    if a.step:
        env["STRATA_DS4_STEP"] = "1"
        env.setdefault("DS4_FLOW", "0")
    e = Engine(argv, env, "/data/works/step_bench_engine.log")
    slots, n = a.slots, a.tokens
    toks = [[] for _ in range(slots)]
    pending = []
    try:
        t_load = time.time()
        ready = e.start(a.timeout)
        print("%sREADY: %s (%.0f s)" % (a.label + ": " if a.label else "", ready, time.time() - t_load), flush=True)
        # (the same prompt every slot; --alt: two prompts, the even slots A and the odd slots B - every even slot must
        # answer slot 0 and every odd slot slot 1, which a wrongly-addressed per-sequence region does not give)
        pa = [1000 + i for i in range(a.prompt_tokens)]
        pb = [2000 + 3 * i for i in range(a.prompt_tokens)]
        t0 = time.time()
        for k in range(slots):
            p = pa
            if a.alt and k % 2 == 1:
                p = pb
            e.send("BGEN %d %d %s" % (k, n, ",".join(str(x) for x in p)))
        ends = 0
        while ends < slots:
            line = e.next(a.read_timeout)
            if line.startswith("PP ") or line.startswith("STAT"):
                continue
            f = line.split()
            if line.startswith("T "):
                pending.append(int(f[1]))
            elif line.startswith("BADM "):
                if pending:
                    toks[int(f[1])].append(pending[0])
                pending = []
            elif line.startswith("BT "):
                toks[int(f[1])].append(int(f[2]))
            elif line.startswith("BDONE "):
                ends += 1
            elif line.startswith("ERR"):
                raise SystemExit("batch: " + line)
        wall = time.time() - t0
    finally:
        e.close()

    same = True
    for k in range(slots):
        ref_k = toks[0 if (not a.alt or k % 2 == 0) else 1]
        what = "slot %d" % (0 if (not a.alt or k % 2 == 0) else 1)
        if k == (0 if (not a.alt or k % 2 == 0) else 1):
            continue
        if toks[k] != ref_k:
            same = False
            d = next((i for i, (x, y) in enumerate(zip(toks[k], ref_k)) if x != y), min(len(toks[k]), len(ref_k)))
            print("  slot %d differs from %s at %d: %s vs %s" % (k, what, d, toks[k][max(0, d - 2):d + 3],
                                                                  ref_k[max(0, d - 2):d + 3]), flush=True)
    total = sum(len(t) for t in toks)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"slots": {str(k): {"tokens": toks[k]} for k in range(slots)}, "wall": wall}, f)
        print("wrote %s" % a.out, flush=True)
    print("%d slot(s), %d tokens each, %d in all in %.2f s (%.1f tok/s); every slot its prompt's tokens: %s%s" %
          (slots, n, total, wall, total / wall if wall else 0, same, " (two prompts, alternating)" if a.alt else ""),
          flush=True)
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
