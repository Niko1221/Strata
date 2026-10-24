"""q36_snapshot_check.py EXE [engine args] -- P.ids PNEXT.ids [P2.ids PNEXT2.ids ...] : the q36 engine's recurrent-state
snapshots (--snapshots) over the engine protocol, against a fresh read.  Per pair: prompt P (8 greedy tokens: the engine then holds P and its own
reply), then PNEXT (P and a reply other than the one generated, then more): the engine must restore the snapshot it took at
the end of P and read only the rest (DONE's reused == len(P)); then a one-token prompt (the session starts over and drops
its snapshots) and PNEXT again from the start.  The two reads' greedy tokens are compared.  --snapshots 2 is added to the
engine args unless they give --snapshots.  The two reads take the prompt in other chunks (and, in the hybrid mode, see
other slot tables), so their tokens can part at a near tie; the bit-exact check of a restore is STRATA_Q36_SNAPSHOT_TEST's
logits against the same split without the detour (docs/QWEN36.md).

  python tools/qwen35/q36_snapshot_check.py build/strata --native MODEL.gguf --max-context 131072 --prefill auto --mtp 2 \\
      --kv int8 -- p1.ids p1next.ids p2.ids p2next.ids
"""
import os
import subprocess
import sys
import threading
import time

N = 32


def ids_of(path):
    return [int(x) for x in open(path).read().replace("\n", ",").split(",") if x.strip()]


class Engine:
    def __init__(self, argv):
        args = list(argv[1:]) + ([] if "--snapshots" in argv else ["--snapshots", "2"])
        self.p = subprocess.Popen([argv[0], "--serve", *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)
        for line in self.p.stdout:
            if line.startswith("READY"):
                return
        raise SystemExit("the engine ended before READY")

    def _watch(self, seconds):
        """A read that waits longer than `seconds` ends the engine (the read then sees EOF): never hold the GPU on a hang."""
        t = threading.Timer(seconds, self.p.kill)
        t.daemon = True
        t.start()
        return t

    def gen(self, ids, n):
        t0 = time.time()
        self.p.stdin.write(f"GEN {n} temperature=0 " + ",".join(map(str, ids)) + "\n")
        self.p.stdin.flush()
        toks = []
        w = self._watch(600)
        for line in self.p.stdout:
            if line.startswith("T "):
                toks.append(int(line.split()[1]))
            elif line.startswith("DONE"):
                f = line.split()
                w.cancel()
                return toks, int(f[8]), float(f[3]), time.time() - t0
            elif line.startswith("ERR"):
                raise SystemExit(line.strip())
        raise SystemExit("the engine ended")

    def close(self):
        """Ended, not waited for at EOF (the engine's teardown can sit in its CPU pool)."""
        self.p.terminate()
        try:
            self.p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()


def main():
    if "--" not in sys.argv:
        raise SystemExit(__doc__)
    k = sys.argv.index("--")
    files = sys.argv[k + 1:]
    e = Engine(sys.argv[1:k])
    try:
        return run(e, files)
    finally:
        e.close()


def run(e, files):
    bad = 0
    for p_file, next_file in zip(files[0::2], files[1::2]):
        p, nxt = ids_of(p_file), ids_of(next_file)
        assert nxt[:len(p)] == p, "PNEXT must start with P"
        g, r0, ms0, _ = e.gen(p, 8)
        held = p + g[:-1]                       # (the last token emitted is not read)
        want = len(held) if nxt[:len(held)] == held else len(p)    # the model wrote the recorded reply's start: no snapshot
        snap, r1, ms1, _ = e.gen(nxt, N)
        e.gen([1, 2, 3], 1)
        fresh, r2, ms2, _ = e.gen(nxt, N)
        same = next((i for i in range(min(len(snap), len(fresh))) if snap[i] != fresh[i]), min(len(snap), len(fresh)))
        ok = r1 == want and r2 == 0 and same == len(fresh)
        bad += not ok
        print(f"{os.path.basename(next_file)}: P {len(p)} tokens (read in {ms0:.0f} ms); PNEXT {len(nxt)} tokens: from the "
              f"snapshot {r1} reused, read in {ms1:.0f} ms; fresh {r2} reused, read in {ms2:.0f} ms; greedy tokens equal "
              f"{same}/{len(fresh)} {'OK' if ok else 'CHECK'}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
