#!/usr/bin/env python3
"""N slots at one context, the single-launch step against the bursts: the first token's latency and the aggregate rate.

    python3 tools/dsv41/batch_step_bench.py --exe build/strata-dsv41.step8 --pack <pack> --slots 8 --tokens 96

One engine with --batch N (STRATA_DS4_BATCH_LIVE=1: a slot its own sequence), N BGENs sent at once - the engine reads
one prompt at a time, so the admissions are sequential and everything after them may be one pass (the step) or a burst a
slot (the default).  Read to each slot's BDONE and print, per slot, the admission's first token's latency, its tokens,
its rate, and in all the wall, the aggregate rate, the spread of the first tokens and the handovers between the slots.
Exit 0 when every slot answered, 1 otherwise; with --ref FILE it also compares every slot's tokens to a run's.
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
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=96, help="tokens every slot decodes (its max_new)")
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--max-context", type=int, default=1048576)
    ap.add_argument("--timeout", type=int, default=1800, help="how long the model may take to load")
    ap.add_argument("--read-timeout", type=int, default=1800)
    ap.add_argument("--extra", default="", help="more engine arguments, space separated")
    ap.add_argument("--ref", default="", help="a JSON of an earlier run's slots: compare the tokens")
    ap.add_argument("--out", default="", help="write the tokens and the timings to this JSON")
    ap.add_argument("--label", default="", help="what the run is, for the log")
    a = ap.parse_args()

    argv = [a.exe, "--pack", a.pack, "--serve", "--batch", str(a.slots), "--max-context", str(a.max_context),
            "--eos", "999999", "--dspark", "off", "--tp", "2", "--experts", "6", "--mmap-experts", "auto"]
    argv += [x for x in a.extra.split(" ") if x]
    env = dict(os.environ)
    env["STRATA_DS4_REUSE"] = "0"
    env["STRATA_DS4_BATCH_LIVE"] = "1"
    e = Engine(argv, env, "/data/works/step_bench_engine.log")
    slots = a.slots
    out = {k: {"t_first": None, "first": None, "done": None, "badm": None, "bt": [], "bdone": None, "t_end": None}
           for k in range(slots)}
    try:
        t_load = time.time()
        ready = e.start(a.timeout)
        print("%sREADY: %s (%.0f s)" % (a.label + ": " if a.label else "", ready, time.time() - t_load), flush=True)
        prompts = [[1000 + 7 * k + i for i in range(a.prompt_tokens)] for k in range(slots)]
        n = a.tokens
        t0 = time.time()
        for k in range(slots):
            e.send("BGEN %d %d %s" % (k, n, ",".join(str(x) for x in prompts[k])))
        pending = []   # (time, token) of the T lines since the last BADM: the admission's own token
        done_line = None
        order = []
        ends = 0
        while ends < slots:
            line = e.next(a.read_timeout)
            now = time.time() - t0
            if line.startswith("PP ") or line.startswith("STAT"):
                continue
            f = line.split()
            if line.startswith("T "):
                pending.append((now, int(f[1])))
            elif line.startswith("DONE"):
                done_line = f
            elif line.startswith("BADM "):
                slot = int(f[1])
                if pending:
                    out[slot]["t_first"] = pending[0][0]
                    out[slot]["first"] = pending[0][1]
                out[slot]["done"] = done_line
                out[slot]["badm"] = f
                pending, done_line = [], None
            elif line.startswith("BT "):
                slot = int(f[1])
                out[slot]["bt"].append(int(f[2]))
                order.append(slot)
            elif line.startswith("BDONE "):
                slot = int(f[1])
                out[slot]["bdone"] = f
                out[slot]["t_end"] = now
                ends += 1
            elif line.startswith("ERR"):
                raise SystemExit("batch: " + line)
        wall = time.time() - t0
    finally:
        e.close()

    firsts = [out[k]["t_first"] for k in range(slots) if out[k]["t_first"] is not None]
    total = sum((1 if out[k]["first"] is not None else 0) + len(out[k]["bt"]) for k in range(slots))
    ok = all(out[k]["bdone"] is not None for k in range(slots))
    print("%d slot(s), max_new %d, prompt %d tokens, context %d%s" %
          (slots, n, a.prompt_tokens, a.max_context, (", " + a.extra) if a.extra else ""), flush=True)
    for k in range(slots):
        s = out[k]
        tok = (1 if s["first"] is not None else 0) + len(s["bt"])
        dt = (s["t_end"] or 0) - (s["t_first"] or 0)
        print("  slot %2d: first token %6.2f s, %3d tokens, %6.2f s of decoding, %5.1f tok/s%s" %
              (k, s["t_first"] or -1, tok, dt, (tok / dt) if dt > 0 else 0,
               "" if s["bdone"] is not None else "  <no BDONE>"), flush=True)
    spread = (max(firsts) - min(firsts)) if firsts else 0.0
    switches = sum(1 for x, y in zip(order, order[1:]) if x != y)
    print("wall %.2f s | every slot answered: %s | %d tokens in all: %.1f tok/s | first tokens within %.2f s | %d "
          "handovers" % (wall, ok, total, total / wall if wall else 0, spread, switches), flush=True)
    if a.out:
        save = {"slots": {str(k): {"tokens": ([] if out[k]["first"] is None else [out[k]["first"]]) + out[k]["bt"],
                                   "t_first": out[k]["t_first"], "t_end": out[k]["t_end"]} for k in range(slots)},
                "wall": wall, "total": total, "args": " ".join(argv), "label": a.label}
        with open(a.out, "w") as f:
            json.dump(save, f)
        print("wrote %s" % a.out, flush=True)
    rc = 0 if ok else 1
    if a.ref:
        ref = json.load(open(a.ref))
        same = True
        for k in range(slots):
            mine = ([] if out[k]["first"] is None else [out[k]["first"]]) + out[k]["bt"]
            theirs = ref["slots"][str(k)]["tokens"]
            if mine != theirs:
                same = False
                d = next((i for i, (x, y) in enumerate(zip(mine, theirs)) if x != y), min(len(mine), len(theirs)))
                print("  slot %d differs from %s at %d: %s vs %s" % (k, a.ref, d, mine[max(0, d - 2):d + 3],
                                                                     theirs[max(0, d - 2):d + 3]), flush=True)
        print("the tokens are the reference's: %s" % same, flush=True)
        rc = rc if same else 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
