"""q36_spec_sampled.py - --mtp-sampled (sampled requests drafting with the MTP block) on the real model, over the serve
protocol (q36_harness.Eng).

    prep  --model GGUF --out DIR
          prompt ids: DIR/short.ids (open-ended chat prompts, one per line), DIR/ctx4k.ids and DIR/ctx80k.ids (one long
          chat prompt each: Strata's own sources to review, cut to about 4K / 80K tokens).  Reads the GGUF's tokenizer
          only; no GPU.
    dist  --exe E --args "..." --prompts DIR/short.ids --n 1500 --k 6 --seed0 1 --out X.json
          n short sampled continuations per prompt (the model card's sampling, seed = seed0 + i): their tokens, the
          drafts offered / accepted.  Run it once per configuration (--mtp-sampled off / match / q in --args).
    cmp   A.json B.json [--floor C.json]
          A against B: per prompt, the two-sample chi-square of each position's token (and of the first two tokens'
          pair), as a z score; with --floor, the same statistic between two runs of ONE configuration with
          different seeds (what chance gives).  Same seeds in A and B: the share of identical continuations (match
          against off should be close to all of them - the same rng draws; q is not).
    speed --exe E --args "..." --prompts DIR/ctx80k.ids --n-gen 512 --reps 3
          decode tok/s and drafts accepted per request at the card's sampling (and --greedy for temperature 0).
"""
import argparse, json, math, shlex, sys, time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CARD = "temperature=1.0 top_p=0.95 top_k=20 penalty_present=1.5 penalty_last_n=64"   # what serve sends for qwen35moe

SHORT = [
    "Write a short poem about the sea.",
    "Give me five unusual names for a pet hedgehog.",
    "Tell me a story about a robot who learns to paint. Keep it under 200 words.",
    "What should I cook tonight? I have rice, eggs, spinach and some leftover chicken.",
    "Write a Python function that checks whether a string is a palindrome, and explain it.",
    "Describe a city on Mars a hundred years from now.",
]


def chat(text):
    return "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n<think>\n"


def prep(a):
    from strata_tokenizer import Tokenizer
    tok = Tokenizer.from_gguf(a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "short.ids", "w") as f:
        for t in SHORT:
            f.write(",".join(map(str, tok.encode(chat(t), parse_special=True))) + "\n")
    root = Path(__file__).resolve().parents[2]
    files = sorted(root.glob("src/qwen35/*.cu")) + sorted(root.glob("src/core/*.cpp")) + sorted(root.glob("serve/*.py"))
    body = tok.encode("".join("\n// ==== %s\n%s" % (p.relative_to(root), p.read_text(errors="replace")) for p in files))
    head = tok.encode("<|im_start|>user\nHere is part of a code base.\n", parse_special=True)
    tail = tok.encode("\n\nReview this code: list the three riskiest functions and explain why.<|im_end|>\n"
                      "<|im_start|>assistant\n<think>\n", parse_special=True)
    for name, budget in (("ctx4k", 4096), ("ctx80k", 80 * 1024)):
        ids = head + body[:budget - len(head) - len(tail)] + tail
        with open(out / (name + ".ids"), "w") as f:
            f.write(",".join(map(str, ids)) + "\n")
        print("%s: %d tokens (of %d)" % (name, len(ids), len(body) + len(head) + len(tail)), flush=True)


def done_fields(done):
    # DONE produced prompt prompt_ms decode_ms finish accepted offered reuse ...
    f = done.split()
    return {"produced": int(f[1]), "decode_ms": float(f[4]), "finish": f[5], "accepted": int(f[6]), "offered": int(f[7])}


def dist(a):
    from q36_harness import Eng, load_prompts
    prompts = load_prompts(a.prompts)
    eng = Eng(a.exe, shlex.split(a.args), log=a.log)
    res = {"args": a.args, "keys": a.keys, "seed0": a.seed0, "k": a.k, "runs": []}
    t0 = time.time()
    for pi, ids in enumerate(prompts):
        seqs, acc, off, toks, ms = [], 0, 0, 0, 0.0
        for i in range(a.n):
            toks_i, done, _, _ = eng.solo(ids, a.k, a.keys + " seed=%d" % (a.seed0 + i))
            d = done_fields(done)
            seqs.append(toks_i)
            acc += d["accepted"]
            off += d["offered"]
            toks += d["produced"]
            ms += d["decode_ms"]
        res["runs"].append({"prompt": pi, "seqs": seqs, "accepted": acc, "offered": off,
                            "decode_tok_s": 1000.0 * toks / max(1e-9, ms)})
        print("prompt %d: %d continuations, drafts accepted %d of %d (%.0f%%), decode %.1f tok/s, %.0f s" %
              (pi, a.n, acc, off, 100.0 * acc / max(1, off), 1000.0 * toks / max(1e-9, ms), time.time() - t0), flush=True)
    json.dump(res, open(a.out, "w"))
    eng.close()


def chi2_two(xa, xb):
    """two-sample chi-square over categories (cells with < 10 together pooled) -> (stat, dof, z)"""
    ca, cb = Counter(xa), Counter(xb)
    na, nb = len(xa), len(xb)
    stat, dof, pa, pb = 0.0, -1, 0, 0
    for c in set(ca) | set(cb):
        oa, ob = ca[c], cb[c]
        if oa + ob < 10:
            pa, pb = pa + oa, pb + ob
            continue
        stat += term(oa, ob, na, nb)
        dof += 1
    if pa + pb > 0:
        stat += term(pa, pb, na, nb)
        dof += 1
    k = max(1, dof)
    z = ((stat / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    return stat, dof, z


def term(oa, ob, na, nb):
    t = oa + ob
    ea, eb = t * na / (na + nb), t * nb / (na + nb)
    return (oa - ea) ** 2 / ea + (ob - eb) ** 2 / eb


def cmp_runs(A, B, label):
    worst = -1e9
    for ra, rb in zip(A["runs"], B["runs"]):
        k = min(A["k"], B["k"])
        zs = []
        for j in range(k):
            xa = [s[j] for s in ra["seqs"] if len(s) > j]
            xb = [s[j] for s in rb["seqs"] if len(s) > j]
            zs.append(chi2_two(xa, xb)[2])
        pair = chi2_two([tuple(s[:2]) for s in ra["seqs"] if len(s) > 1], [tuple(s[:2]) for s in rb["seqs"] if len(s) > 1])[2]
        same = sum(1 for x, y in zip(ra["seqs"], rb["seqs"]) if x == y)
        worst = max(worst, max(zs), pair)
        print("  %s prompt %d: z by position %s, pair z %.2f, identical %d of %d" %
              (label, ra["prompt"], " ".join("%.2f" % z for z in zs), pair, same, len(ra["seqs"])))
    return worst


def cmp(a):
    A, B = json.load(open(a.a)), json.load(open(a.b))
    for X in (A, B):
        acc = sum(r["accepted"] for r in X["runs"])
        off = sum(r["offered"] for r in X["runs"])
        print("%s: drafts accepted %d of %d (%.0f%%)" % (X["args"], acc, off, 100.0 * acc / max(1, off)))
    w = cmp_runs(A, B, "A/B")
    print("A/B worst z %.2f (above ~3.5 over a few dozen tests is a real difference)" % w)
    if a.floor:
        C = json.load(open(a.floor))
        print("floor worst z %.2f" % cmp_runs(A, C, "A/A'"))


def speed(a):
    from q36_harness import Eng, load_prompts
    ids = load_prompts(a.prompts)[0]
    eng = Eng(a.exe, shlex.split(a.args), log=a.log)
    keys = "" if a.greedy else (a.keys or CARD)
    tot_t, tot_ms, tot_a, tot_o = 0, 0.0, 0, 0
    for r in range(a.reps):
        toks, done, ttft, wall = eng.solo(ids, a.n_gen, (keys + " seed=%d" % (r + 1)).strip())
        d = done_fields(done)
        print("rep %d: prompt %d tokens, %d generated, decode %.1f tok/s, drafts accepted %d of %d (%s)" %
              (r, len(ids), d["produced"], 1000.0 * d["produced"] / max(1e-9, d["decode_ms"]), d["accepted"], d["offered"],
               d["finish"]), flush=True)
        tot_t, tot_ms, tot_a, tot_o = tot_t + d["produced"], tot_ms + d["decode_ms"], tot_a + d["accepted"], tot_o + d["offered"]
    steps = max(1, tot_t - tot_a)   # every step emits its kept drafts and one token more
    print("mean: %.1f tok/s, drafts accepted %.0f%%, %.2f tokens per step, %.2f draft rows per step" %
          (1000.0 * tot_t / max(1e-9, tot_ms), 100.0 * tot_a / max(1, tot_o), tot_t / steps, tot_o / steps), flush=True)
    eng.close()


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("prep"); p.add_argument("--model", required=True); p.add_argument("--out", required=True)
p = sub.add_parser("dist")
for k, v in (("--exe", None), ("--args", ""), ("--prompts", None), ("--out", None), ("--log", None)):
    p.add_argument(k, default=v, required=v is None and k != "--log")
p.add_argument("--n", type=int, default=1500); p.add_argument("--k", type=int, default=6)
p.add_argument("--seed0", type=int, default=1); p.add_argument("--keys", default=CARD)
p = sub.add_parser("cmp"); p.add_argument("a"); p.add_argument("b"); p.add_argument("--floor")
p = sub.add_parser("speed")
for k, v in (("--exe", None), ("--args", ""), ("--prompts", None), ("--log", None)):
    p.add_argument(k, default=v, required=v is None and k != "--log")
p.add_argument("--n-gen", type=int, default=512); p.add_argument("--reps", type=int, default=3)
p.add_argument("--greedy", action="store_true"); p.add_argument("--keys", default="")
a = ap.parse_args()
{"prep": prep, "dist": dist, "cmp": cmp, "speed": speed}[a.cmd](a)
