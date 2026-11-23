"""Live test on a Mac: the Metal Inkling engine keeps a prompt-end checkpoint (#6627) and resumes from it.

Drives the real engine through its native protocol (the same lines serve sends), with a real model, one slot:

  1. request A (3000 tokens)                           -> nothing to reuse
  2. request B = A's first 3000 tokens + a new tail    -> the whole-stream cache does not match (the slot holds A plus the tokens it
     generated), so only the prompt-end checkpoint can help: it must resume at 2999 tokens (the checkpoint is one token short of the
     end of the prompt)
  3. request C = B + another new tail                  -> resumes at len(B) - 1, the checkpoint B's own request saved
  4. the same B and C read cold by a fresh engine      -> the greedy tokens of the resumed requests must equal the cold ones

Before the fix the Metal backend returned false from save_prompt_checkpoint, so step 2 and 3 reused 0 tokens (the agent-session
failure of the 0.2.21 release grid: median reuse 0%).

  python3 test_metal_prompt_ckpt_live.py --engine ~/.../strata-inkling --model .../Inkling-2-Mini-SQ-10.strata \
      --vision .../Inkling-2-Mini-Vision.strata [--draft on|off] [--ctx 8192]

Exit 0 = pass.  The engine uses the GPU: run it under the Mac locks (run_locked.sh).
"""
import argparse
import random
import subprocess
import sys

N_NEW = 24
PROMPT_LEN = 3000


def ids(seed, n):
    r = random.Random(seed)
    return [r.randrange(1000, 60000) for _ in range(n)]


class Engine:
    def __init__(self, a):
        cmd = [a.engine, "--serve", "--native", a.model, "--vision", a.vision, "--ctx", str(a.ctx), "--parallel", "1",
               "--kv", "fp16", "--draft", a.draft]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.info = {}
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise SystemExit("engine exited before READY")
            if line.startswith("INFO "):
                self.info = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
            if line.startswith("READY "):
                break

    def gen(self, prompt):
        self.p.stdin.write("GEN %d temperature=0 %s\n" % (N_NEW, ",".join(map(str, prompt))))
        self.p.stdin.flush()
        toks, resumed = [], 0
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise SystemExit("engine exited during a request")
            if line.startswith("ERR "):
                raise SystemExit("engine error: " + line.strip())
            if line.startswith("RESUME "):
                resumed = int(line.split()[1])
            elif line.startswith("T "):
                toks.append(int(line.split()[1]))
            elif line.startswith("DONE "):
                return toks, int(line.split()[-1])

    def close(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.stdin.close()
            self.p.wait(timeout=60)
        except Exception:
            self.p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--vision", required=True)
    ap.add_argument("--draft", default="on", choices=["on", "off"])
    ap.add_argument("--ctx", type=int, default=8192)
    a = ap.parse_args()

    A = ids(1, PROMPT_LEN)
    B = A + ids(2, 40)
    C = B + ids(3, 40)
    bad = []

    e = Engine(a)
    print("engine:", e.info.get("engine"), "backend", e.info.get("backend"), "drafting", e.info.get("drafting"))
    tA, rA = e.gen(A)
    tB, rB = e.gen(B)
    tC, rC = e.gen(C)
    e.close()
    print("A reused %d, B reused %d (want %d), C reused %d (want %d)" % (rA, rB, PROMPT_LEN - 1, rC, len(B) - 1))
    if rA != 0:
        bad.append("A reused %d tokens of an empty slot" % rA)
    if rB != PROMPT_LEN - 1:
        bad.append("B reused %d, wanted %d (the checkpoint at the end of A's prompt)" % (rB, PROMPT_LEN - 1))
    if rC != len(B) - 1:
        bad.append("C reused %d, wanted %d (the checkpoint at the end of B's prompt)" % (rC, len(B) - 1))

    cold = Engine(a)
    cB, crB = cold.gen(B)
    cold.close()
    cold = Engine(a)
    cC, crC = cold.gen(C)
    cold.close()
    print("cold reads reused %d / %d" % (crB, crC))
    for name, got, ref in (("B", tB, cB), ("C", tC, cC)):
        same = sum(1 for x, y in zip(got, ref) if x == y)
        first = 0
        while first < min(len(got), len(ref)) and got[first] == ref[first]:
            first += 1
        print("%s resumed vs cold: %d / %d tokens equal, first difference at %s" % (name, same, len(ref), first if first < len(ref) else "none"))
        if first < 8:
            bad.append("%s: the resumed greedy tokens differ from a cold read within the first 8 (%s vs %s)" % (name, got[:8], ref[:8]))
    if bad:
        print("FAIL")
        for b in bad:
            print("  -", b)
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
