#!/usr/bin/env python3
"""Sampling test through the --serve protocol (the Mac engine; any engine that has STRATA_DUMP_FIRST_LOGITS).

The request's sampling keys (temperature, top_p, top_k, min_p, seed, penalty_*, logit_bias) must act as on the CUDA
engine: the chain penalties -> top_k -> top_p -> min_p -> temperature, one Philox 4x32-10 draw at (seed, position),
and greedy when temperature is 0 or absent.

Engine 1 (no --batch; logit_bias needs it):
  1. reproducible: the same sampled request twice gives the same tokens; another seed gives others; both differ from
     the greedy tokens.
  2. draw for draw: many one-token requests with seeds 1..N; each request's first-window logits are dumped
     (STRATA_DUMP_FIRST_LOGITS) and a reference sampler written here from the CUDA kernels (src/kernels/cuda/sampler.cu)
     picks from them with the same seed and position.  Every pick must be the reference's; the empirical distribution
     is compared with the reference probabilities (total variation distance, reported).
  3. penalties: greedy with penalty_repeat changes the tokens; the first pick (greedy and sampled) equals the
     reference's, which counts the prompt's last penalty_last_n tokens.
  4. logit_bias: banning the greedy first token moves the pick to the reference's; sampled picks never take a banned
     token and equal the reference's.
Engine 2:
  5. the sampled request of 1. gives the same tokens in a second engine process.

  python3 tools/metal_sampling_test.py --exe /abs/build-metal/strata --config strata-<model>.json [--draws 100]
  python3 tools/metal_sampling_test.py ... --bench      # decode speed: greedy vs sampled

Exit 0 when every check passes, 1 otherwise.  STRATA_IQ_MT_MIN=1 (as batch_test.py) for exact tokens.
"""
import argparse, json, math, os, sys, tempfile, time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_test as BT  # noqa: E402

M32 = 0xFFFFFFFF
F32 = np.float32


# ---------------------------------------------------------------- the reference sampler (src/kernels/cuda/sampler.cu)
def philox_uniform(seed, counter):
    """philox_uniform: Philox 4x32-10, counter in (c0, c1), seed in (c2, c3), key (round, 0); 24 bits of c0."""
    c0, c1, c2, c3 = counter & M32, (counter >> 32) & M32, seed & M32, (seed >> 32) & M32
    for i in range(10):
        p0, p1 = 0x9E3779B9 * c0, 0xBB67AE85 * c2
        c0, c1, c2, c3 = ((p1 >> 32) ^ c1 ^ i) & M32, p1 & M32, ((p0 >> 32) ^ c3) & M32, p0 & M32
    return (c0 >> 8) * (1.0 / 16777216.0)


class Params:
    """A request's keys as the engine parses them (strtof: float32), with the engine's defaults."""

    def __init__(self, keys):
        kv = dict(t.split("=", 1) for t in keys.split())
        self.temperature = F32(kv.get("temperature", "0"))
        self.top_p = F32(kv.get("top_p", "1"))
        self.top_k = int(kv.get("top_k", "20"))
        self.min_p = F32(min(max(float(F32(kv.get("min_p", "0"))), 0.0), 1.0))
        self.seed = int(kv.get("seed", "0"))
        self.pen_n = max(int(kv.get("penalty_last_n", "0")), 0)
        self.rep = F32(kv.get("penalty_repeat", "1"))
        self.freq = F32(kv.get("penalty_freq", "0"))
        self.pres = F32(kv.get("penalty_present", "0"))
        self.bias = {}
        if "logit_bias" in kv:
            for e in kv["logit_bias"].split(","):
                i, v = e.split(":")
                self.bias[int(i)] = -np.inf if float(v) <= -100 else F32(v)


def scores(logits, p, history):
    """logit (+ bias), then the penalties over the last min(penalty_last_n, 4096) history tokens."""
    s = logits.astype(F32).copy()
    for i, v in p.bias.items():
        s[i] = F32(s[i] + F32(v))
    if p.pen_n > 0 and history:
        for i, c in Counter(history[-min(p.pen_n, 4096):]).items():
            if not 0 <= i < len(s):
                continue
            x = s[i]
            x = F32(x * p.rep) if x <= 0 else F32(x / p.rep)
            s[i] = F32(x - F32(F32(F32(c) * p.freq) + F32(p.pres)))
    return s


def chain(logits, p, history=()):
    """(ids, probs) the pick is drawn from: the sampled chain's survivors and their final probabilities, or the
    greedy pick with probability 1."""
    s = scores(logits, p, list(history))
    if p.temperature <= 0:
        with np.errstate(invalid="ignore"):
            best = int(np.argmax(np.where(np.isnan(s), -np.inf, s)))
        return [best if s[best] > -np.inf else 0], [1.0]
    k = p.top_k if 0 < p.top_k < 64 else 64
    k = min(k, len(s))
    order = np.argsort(-s, kind="stable")   # score descending, ties to the lower id
    ids = [int(i) for i in order[:k] if s[i] > -np.inf]
    lg = [F32(s[i]) for i in ids]
    while len(ids) < k:                     # a round that finds nothing stores (0, -inf)
        ids.append(0)
        lg.append(F32(-np.inf))
    n_keep = k
    mx = max(lg)
    if p.top_p < 1:
        tot = sum(math.exp(float(v) - float(mx)) for v in lg)
        cum, cut = 0.0, k
        for i, v in enumerate(lg):
            cum += math.exp(float(v) - float(mx)) / tot
            if cum >= float(p.top_p):
                cut = i + 1
                break
        n_keep = max(cut, 1)
    if p.min_p > 0:
        thresh = F32(lg[0] + F32(np.log(p.min_p)))
        for i in range(n_keep):
            if lg[i] < thresh:
                n_keep = i
                break
    inv_t = F32(F32(1) / p.temperature)
    sc = [F32(v * inv_t) for v in lg[:max(n_keep, 1)]]
    smx = max(sc[:max(n_keep, 1)])
    e = [math.exp(float(v) - float(smx)) for v in sc[:n_keep]]
    tot = sum(e)
    return ids[:n_keep], [x / tot for x in e]


def ref_pick(logits, p, counter, history=()):
    ids, probs = chain(logits, p, history)
    if p.temperature <= 0:
        return ids[0]
    u = philox_uniform(p.seed, counter)
    cum = 0.0
    for i, q in zip(ids, probs):
        cum += q
        if u < cum:
            return i
    return ids[-1]


# ---------------------------------------------------------------- the engine
class Proto:
    def __init__(self, exe, cfg, batch, dump_dir, extra):
        env = dict(cfg.get("env") or {})
        env.update({"STRATA_IQ_MT_MIN": "1", "STRATA_DECODE_TIMING": "1"})
        if dump_dir:
            env["STRATA_DUMP_FIRST_LOGITS"] = str(Path(dump_dir) / "fl")
        self.dump_dir = dump_dir
        base = os.environ.get("SAMPLING_TEST_LOG") or os.path.join(tempfile.gettempdir(), "sampling_test_engine")
        os.environ["BATCH_TEST_LOG"] = f"{base}.batch{batch}.log"   # (BT.Engine's log; one file per engine)
        t0 = time.time()
        self.eng = BT.Engine(exe, cfg, batch, env, extra.split())
        print(f"engine up in {time.time() - t0:.0f} s (batch {batch})", flush=True)
        self.out = self.eng.lines()

    def line(self):
        line = next(self.out, None)
        if line is None:
            raise SystemExit(f"the engine ended - see {self.eng.log_path}")
        return line

    def gen(self, ids, max_new, keys=""):
        """GEN: (tokens, DONE fields as a dict)"""
        self.eng.send(" ".join(x for x in ("GEN", str(max_new), keys, ",".join(map(str, ids))) if x))
        got = []
        while True:
            line = self.line()
            if line.startswith("T "):
                got.append(int(line.split()[1]))
            elif line.startswith("ERR"):
                raise SystemExit(f"engine: {line}")
            elif line.startswith("DONE"):
                f = line.split()
                return got, {"gen": int(f[1]), "decode_ms": float(f[4]), "acc": int(f[6]), "off": int(f[7])}

    def last_logits(self):
        files = sorted(Path(self.dump_dir).glob("fl.*"), key=lambda q: int(q.suffix[1:]))
        return np.fromfile(files[-1], dtype=np.float32)

    def quit(self):
        self.eng.send("QUIT")
        self.eng.p.wait(timeout=180)


def wrap(user):
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


FAILS = []


def check(ok, what):
    print(f"  {'PASS' if ok else 'FAIL'}  {what}", flush=True)
    if not ok:
        FAILS.append(what)


def draws(P, ids, keys_of, n, history=()):
    """n one-token requests (seed s = 1..n): engine picks, reference picks, reference probabilities"""
    got, want, mass = [], [], Counter()
    pos = len(ids) - 1   # the first generated token is drawn at the last prompt position
    for s in range(1, n + 1):
        keys = keys_of(s)
        toks, _ = P.gen(ids, 1, keys)
        lg = P.last_logits()
        p = Params(keys)
        got.append(toks[0])
        want.append(ref_pick(lg, p, pos, history))
        for i, q in zip(*chain(lg, p, history)):
            mass[i] += q / n
    emp = Counter(got)
    tv = 0.5 * sum(abs(emp[i] / n - mass[i]) for i in set(emp) | set(mass))
    wrong = sum(a != b for a, b in zip(got, want))
    return got, want, wrong, tv, mass


def run_tests(a, cfg, tok):
    enc = lambda t: tok.encode(t, parse_special=True)
    A = enc(wrap("Write a short poem about the sea."))
    P_ = enc(wrap("Pick a random everyday object and describe it in one sentence."))
    R = enc(wrap("Write the word 'hello' twenty times, separated by spaces."))
    C = enc(wrap(BT.QUESTIONS[0]))
    D = enc(wrap(BT.QUESTIONS[1]))
    K = "temperature=0.8 top_p=0.95 top_k=20"
    dump = tempfile.mkdtemp(prefix="sampling_test_")

    print("engine 1: no --batch", flush=True)
    P = Proto(a.exe, cfg, 0, dump, a.extra)
    g, _ = P.gen(A, a.max_new)
    s1, _ = P.gen(A, a.max_new, K + " seed=1")
    s1b, _ = P.gen(A, a.max_new, K + " seed=1")
    s2, _ = P.gen(A, a.max_new, K + " seed=2")
    print("   greedy :", ascii(tok.decode(g)[:100]))
    print("   seed 1 :", ascii(tok.decode(s1)[:100]))
    print("   seed 2 :", ascii(tok.decode(s2)[:100]))
    check(s1 == s1b, "1. the same seed twice gives the same tokens")
    check(s1 != s2, "1. another seed gives other tokens")
    check(s1 != g and s2 != g, "1. sampled tokens differ from the greedy ones")

    for keys_of, name, n in ((lambda s: f"temperature=1.0 top_k=40 top_p=0.95 seed={s}", "t1.0 k40 p0.95", a.draws),
                             (lambda s: f"temperature=0.7 top_k=0 top_p=1.0 min_p=0.05 seed={s}", "t0.7 k0 min_p0.05",
                              a.draws // 2)):
        got, want, wrong, tv, mass = draws(P, P_, keys_of, n)
        top = sorted(mass.items(), key=lambda x: -x[1])[:5]
        print(f"   {name}: {len(set(got))} distinct first tokens in {n} draws; reference top: "
              + ", ".join(f"{ascii(tok.decode([i]))} {q:.2f}" for i, q in top) + f"; TV distance {tv:.3f}")
        check(wrong == 0, f"2. {name}: every pick equals the reference sampler's on the same logits ({wrong} of {n} differ)")
        check(len(set(got)) > 1, f"2. {name}: more than one token drawn")

    gp, _ = P.gen(R, a.max_new)
    pen = "penalty_last_n=64 penalty_repeat=1.5"
    rp, _ = P.gen(R, a.max_new, pen)
    lg = P.last_logits()
    print("   greedy            :", ascii(tok.decode(gp)[:100]))
    print("   greedy + penalties:", ascii(tok.decode(rp)[:100]))
    check(rp != gp, "3. greedy with penalty_repeat 1.5 changes the tokens")
    want = ref_pick(lg, Params(pen), len(R) - 1, R)
    check(rp[0] == want, f"3. greedy + penalties: the first pick is the reference's ({rp[0]} vs {want})")
    got, want, wrong, tv, _ = draws(P, R, lambda s: f"temperature=1.0 top_k=40 {pen} seed={s}", a.draws // 3, R)
    check(wrong == 0, f"3. sampled + penalties: every pick equals the reference's ({wrong} of {len(got)} differ; TV {tv:.3f})")

    ban = g[0]
    bg, _ = P.gen(A, 4, f"logit_bias={ban}:-100")
    lg = P.last_logits()
    check(bg[0] != ban and bg[0] == ref_pick(lg, Params(f"logit_bias={ban}:-100"), len(A) - 1),
          f"4. logit_bias bans the greedy first token {ban}: the pick is the reference's ({bg[0]})")
    P.gen(P_, 1, "temperature=1.0 top_k=40 seed=1")
    top1 = chain(P.last_logits(), Params("temperature=1.0 top_k=40 seed=1"))[0][0]   # the most likely first token
    got, want, wrong, _, _ = draws(P, P_, lambda s: f"temperature=1.0 top_k=40 seed={s} logit_bias={top1}:-100", a.draws // 4)
    check(wrong == 0 and top1 not in got,
          f"4. sampled + logit_bias: the banned token {top1} is never drawn, picks equal the reference's ({wrong} differ)")
    P.quit()

    print("engine 2", flush=True)
    P = Proto(a.exe, cfg, 0, dump, a.extra)
    s1c, _ = P.gen(A, a.max_new, K + " seed=1")
    check(s1c == s1, "5. the sampled request gives the same tokens in another engine process")
    P.quit()

def bench(a, cfg, tok):
    """decode tok/s, greedy vs sampled (the 4 short prompts, MTP drafts on)"""
    enc = lambda t: tok.encode(t, parse_special=True)
    prompts = [enc(wrap(q)) for q in BT.QUESTIONS[:4]]
    S = "temperature=0.7 top_p=0.95 top_k=20 seed=1"
    P = Proto(a.exe, cfg, 0, None, a.extra)
    res = {}
    for mode, keys in (("greedy", ""), ("sampled", S)):
        rates, acc, off = [], 0, 0
        for ids in prompts:
            toks, d = P.gen(ids, a.bench_tokens, keys)
            rates.append(d["gen"] / (d["decode_ms"] / 1000.0))
            acc, off = acc + d["acc"], off + d["off"]
        res[f"solo {mode}"] = rates
        print(f"solo {mode:7s}: " + " / ".join(f"{r:.1f}" for r in rates) + f" tok/s (mean {sum(rates) / len(rates):.1f}); "
              f"drafts accepted {acc}/{off} = {acc / max(off, 1):.2f}", flush=True)
    P.quit()
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--draws", type=int, default=100, help="one-token requests per draw-for-draw check")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench-tokens", type=int, default=256)
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    if not os.path.isabs(a.exe):
        raise SystemExit("--exe must be an absolute path (the engine runs in the config's cwd)")
    cfg = json.loads(Path(a.config).read_text())
    tok = BT.tokenizer(cfg["tokenizer"])
    if a.bench:
        bench(a, cfg, tok)
        return 0
    run_tests(a, cfg, tok)
    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
