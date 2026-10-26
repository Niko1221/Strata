#!/usr/bin/env python3
"""The engine's batch lines, checked against a live engine (docs/DSV41_BATCHING.md, M4).  No HTTP, no tokenizer.

    python tools/dsv41/batch_probe.py --exe build/strata-dsv41.batch --pack <pack> [--tokens 12] [--timeout 1800]

Starts the engine with --serve --batch N and STRATA_DS4_REUSE=0 (a GEN and a BGEN of the same prompt then both read it
from token 0: the comparison is about the batch lines, not about the prefix reuse) and checks:

  0. with --slots 2 and --slot-burst 4, two admissions at once: both slots' tokens equal their GEN references, the BT
     lines interleave, and each slot ends with its own BDONE;
  1. INFO batch_slots=N is said before READY;
  2. GEN and BGEN of the same prompt give the same greedy tokens;
  3. BGEN with max_new 1 answers BADM 0 (the admission's token was the whole answer) and no BT line;
  4. BSTOP 0 ends a slot: BDONE ... cancel, fewer tokens than asked for;
  5. a request after a cancelled slot still works, and a second slot after a finished one does too;
  6. a BGEN for a slot that does not exist is refused with ERR BGEN.

Exit 0 when every check passes, 1 with the failed ones named.
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
    """The engine's stdout on a queue: every check reads lines at its own pace."""

    def __init__(self, argv, env):
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, bufsize=1, env=env)
        self.q = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.info = {}

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
            if line.startswith("INFO "):
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    self.info[k] = v
            elif line.startswith("READY"):
                return line

    def _next(self, timeout):
        try:
            line = self.q.get(timeout=max(0.1, timeout))
        except queue.Empty:
            raise SystemExit("the engine was silent for %.0f s" % timeout)
        if line is None:
            raise SystemExit("the engine ended (exit %s)" % self.p.poll())
        return line

    def send(self, text):
        self.p.stdin.write(text + "\n")
        self.p.stdin.flush()

    def line(self, timeout=600):
        # PP (a prompt's progress) and STAT (the Monitor's) are not part of any check's stream
        while True:
            line = self._next(timeout)
            if line.startswith(("PP ", "STAT")):
                continue
            return line

    def gen(self, ids, max_new, timeout=600):
        self.send("GEN %d %s" % (max_new, ",".join(str(int(t)) for t in ids)))
        toks, done = [], None
        while True:
            line = self.line(timeout)
            if line.startswith("T "):
                toks.append(int(line.split()[1]))
            elif line.startswith("DONE"):
                done = line.split()
                break
            elif line.startswith("ERR"):
                raise SystemExit("GEN: " + line)
        return toks, done

    def bgen(self, slot, ids, max_new, stop_after=None, timeout=600):
        """One admission and its slot, read to BDONE.  stop_after: send BSTOP after this many BT lines."""
        self.send("BGEN %d %d %s" % (slot, max_new, ",".join(str(int(t)) for t in ids)))
        first, done, badm = [], None, None
        while True:
            line = self.line(timeout)
            if line.startswith("T "):
                first.append(int(line.split()[1]))
            elif line.startswith("DONE"):
                done = line.split()
            elif line.startswith("BADM "):
                badm = line.split()
                break
            elif line.startswith("ERR"):
                raise SystemExit("BGEN: " + line)
        bt, bdone, asked = [], None, False
        if badm[2] == "1":
            while True:
                line = self.line(timeout)
                if line.startswith("BT "):
                    bt.append(int(line.split()[2]))
                    if stop_after is not None and not asked and len(bt) >= stop_after:
                        self.send("BSTOP %d" % slot)
                        asked = True
                elif line.startswith("BDONE "):
                    bdone = line.split()
                    break
                elif line.startswith("ERR"):
                    raise SystemExit("slot: " + line)
        return {"first": first, "done": done, "badm": badm, "bt": bt, "bdone": bdone}

    def bgen_many(self, reqs, timeout=600):
        """reqs: [(slot, ids, max_new)] - every admission sent at once, then the mixed stream read to each BDONE.

        The control lines carry no slot id (the server reads them one admission at a time): a T/DONE block is given to
        the slot of the BADM that follows it.  BT and BDONE carry their slot."""
        for slot, ids, n in reqs:
            self.send("BGEN %d %d %s" % (slot, n, ",".join(str(int(t)) for t in ids)))
        out = {slot: {"first": [], "done": None, "badm": None, "bt": [], "bdone": None} for slot, _, _ in reqs}
        order = []
        pending = {"first": [], "done": None}
        ends = 0
        while ends < len(reqs):
            line = self.line(timeout)
            f = line.split()
            if line.startswith("T "):
                pending["first"].append(int(f[1]))
            elif line.startswith("DONE"):
                pending["done"] = f
            elif line.startswith("BADM "):
                slot = int(f[1])
                out[slot]["first"] = pending["first"]
                out[slot]["done"] = pending["done"]
                out[slot]["badm"] = f
                pending = {"first": [], "done": None}
            elif line.startswith("BT "):
                out[int(f[1])]["bt"].append(int(f[2]))
                order.append(int(f[1]))
            elif line.startswith("BDONE "):
                out[int(f[1])]["bdone"] = f
                ends += 1
            elif line.startswith("ERR"):
                raise SystemExit("batch: " + line)
        return out, order

    def close(self):
        try:
            self.send("QUIT")
            self.p.wait(timeout=15)      # (a QUIT the engine does not take must not hold the window)
        except Exception:
            self.p.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tokens", type=int, default=12)
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800, help="how long the model may take to load")
    ap.add_argument("--extra", default="", help="more engine arguments, space separated")
    a = ap.parse_args()

    argv = [a.exe, "--pack", a.pack, "--serve", "--batch", str(a.slots), "--max-context", "8192", "--eos", "999999",
            "--dspark", "off", "--tp", "2", "--experts", "6", "--mmap-experts", "auto"]
    if a.slots >= 2:   # (a burst small enough that the slots must take turns)
        argv += ["--slot-burst", str(max(2, a.tokens // 3))]
    argv += [x for x in a.extra.split(" ") if x]
    env = dict(os.environ)
    env["STRATA_DS4_REUSE"] = "0"
    e = Engine(argv, env)
    fails = []

    def check(name, ok, detail=""):
        print(("ok   " if ok else "FAIL ") + name + (("  (" + detail + ")") if detail else ""), flush=True)
        if not ok:
            fails.append(name)

    ready = e.start(a.timeout)
    print("READY:", ready, flush=True)
    check("INFO batch_slots=%d before READY" % a.slots, e.info.get("batch_slots") == str(a.slots),
          "info=%s" % e.info.get("batch_slots"))

    prompt = [1000 + i for i in range(40)]
    n = a.tokens
    ref, ref_done = e.gen(prompt, n)
    check("GEN gives %d tokens" % n, len(ref) == n, "got %d (%s)" % (len(ref), ref_done[:6]))

    if a.slots >= 2:
        # every admission sent at once: the engine gives each slot the model in turn (a burst each), so their BT lines
        # interleave and each slot's tokens are what a GEN of its own prompt gives
        prompts = [prompt] + [[2000 * (k + 1) + i for i in range(40)] for k in range(1, a.slots)]
        refs = [e.gen(p, n)[0] for p in prompts]
        both, order = e.bgen_many([(k, p, n) for k, p in enumerate(prompts)])
        for k in range(a.slots):
            check("slot %d says BADM %d 1" % (k, k), both[k]["badm"][:3] == ["BADM", str(k), "1"],
                  " ".join(both[k]["badm"] or ["<none>"]))
        for k in range(a.slots):
            check("slot %d's tokens equal its GEN" % k, both[k]["first"] + both[k]["bt"] == refs[k],
                  "1 + %d of %d" % (len(both[k]["bt"]), n))
            check("slot %d ends with its own BDONE" % k,
                  both[k]["bdone"] is not None and both[k]["bdone"][1:4] == [str(k), str(n), "length"],
                  " ".join(both[k]["bdone"] or ["<none>"]))
        switches = sum(1 for x, y in zip(order, order[1:]) if x != y)
        check("the slots take turns (their BT lines interleave)", switches >= a.slots,
              "".join(str(x) for x in order[:60]) + " (%d switches)" % switches)

    got = e.bgen(0, prompt, n)
    check("BGEN admission gives exactly one token", len(got["first"]) == 1, "got %d" % len(got["first"]))
    check("BGEN says BADM 0 1 (the slot decodes the rest)", got["badm"][:3] == ["BADM", "0", "1"], " ".join(got["badm"]))
    check("BGEN gives %d tokens in all" % n, len(got["first"]) + len(got["bt"]) == n,
          "1 + %d" % len(got["bt"]))
    diff = next((i for i, (x, y) in enumerate(zip(got["first"] + got["bt"], ref)) if x != y), None)
    check("BGEN's tokens equal GEN's", got["first"] + got["bt"] == ref, "first differ at %s" % diff)
    check("BDONE counts every token and says length",
          got["bdone"] is not None and got["bdone"][1:4] == ["0", str(n), "length"],
          " ".join(got["bdone"] or ["<none>"]))

    one = e.bgen(0, prompt, 1)
    check("BGEN max_new 1: BADM 0 0", one["badm"][:3] == ["BADM", "0", "0"], " ".join(one["badm"]))
    check("BGEN max_new 1: one T token, no BT, no BDONE",
          len(one["first"]) == 1 and one["bt"] == [] and one["bdone"] is None,
          "T %d BT %d" % (len(one["first"]), len(one["bt"])))

    cut = e.bgen(0, prompt, 64, stop_after=3)
    check("BSTOP ends the slot (BDONE cancel)", cut["bdone"] is not None and cut["bdone"][3] == "cancel",
          " ".join(cut["bdone"] or ["<none>"]))
    check("BSTOP ends it early", cut["bdone"] is not None and int(cut["bdone"][2]) < 64,
          "%s tokens" % (cut["bdone"][2] if cut["bdone"] else "?"))

    after, _ = e.gen(prompt, 4)
    check("a GEN after a cancelled slot still answers", len(after) == 4, "got %d" % len(after))

    other = [2000 + i for i in range(32)]
    second = e.bgen(0, other, 6)
    check("a second slot after a finished one works", len(second["first"]) + len(second["bt"]) == 6,
          "1 + %d" % len(second["bt"]))

    e.send("BGEN 3 4 1,2,3")
    line = e.line(120)
    check("BGEN for a slot that does not exist is refused", line.startswith("ERR BGEN"), line)

    e.close()
    if fails:
        print("\n%d check(s) failed: %s" % (len(fails), ", ".join(fails)))
        return 1
    print("\nbatch_probe: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())