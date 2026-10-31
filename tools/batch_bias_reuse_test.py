#!/usr/bin/env python3
"""Batch slots that are used again with a different logit_bias: no bias may stay behind in a slot.

Waves of the same prompts: in wave w the prompts with (i + w) even carry a ban of every Han/Kana/Hangul token
(and ask for Chinese), the others carry none, so every slot changes from banned to free and back between waves.
Each wave runs the prompts alone (GEN) and together in the batch slots (BGEN); a slot's tokens must equal its solo tokens.

  python3 tools/batch_bias_reuse_test.py --exe engine/strata --config strata-<model>.json --batch 4 --n 4 --waves 3 \\
      --extra "--layer-split 12,24,36 --trim-stage-weights --pcie-frac 0 --adapt-every 1000000"
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_test as BT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n", type=int, default=4, help="prompts (<= --batch)")
    ap.add_argument("--waves", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--keys", default="", help='sampling keys for every request, e.g. "temperature=0.7 top_k=20 seed=7"')
    ap.add_argument("--extra", default="", help="more engine arguments in one string")
    ap.add_argument("--mt-min", default="1", help="STRATA_IQ_MT_MIN for the engine (1: exact; empty: the default)")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok = BT.tokenizer(cfg["tokenizer"])
    vocab_n = len(json.loads((Path(cfg["tokenizer"]) / "vocab.json").read_text(encoding="utf-8")))
    bias_text, n_banned = BT.han_bias(tok, vocab_n)
    print(f"logit_bias: {n_banned} tokens banned; {a.waves} waves, banned and free prompts swap every wave", flush=True)
    env = {"STRATA_DECODE_TIMING": "1"}
    if a.mt_min:
        env["STRATA_IQ_MT_MIN"] = a.mt_min
    eng = BT.Engine(a.exe, cfg, a.batch, env, a.extra.split())
    out = eng.lines()
    ok = True
    for w in range(a.waves):
        biased = lambda i: (i + w) % 2 == 0   # noqa: E731
        keys = lambda i: " ".join(x for x in (a.keys, ("logit_bias=" + bias_text) if biased(i) else "") if x)   # noqa: E731
        prompts = []
        for i, q in enumerate(BT.QUESTIONS[: a.n]):
            q = q + (" Answer in Chinese." if biased(i) else "")
            prompts.append(tok.encode(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True))
        solo = []
        for i, ids in enumerate(prompts):
            eng.send(" ".join(x for x in ("GEN", str(a.max_new), keys(i), ",".join(map(str, ids))) if x))
            got = []
            for line in out:
                if line.startswith("T "):
                    got.append(int(line.split()[1]))
                elif line.startswith("DONE") or line.startswith("ERR"):
                    if line.startswith("ERR"):
                        print("solo:", line); return 1
                    break
            solo.append(got)
        got = {i: [] for i in range(len(prompts))}
        done = set()
        for i, ids in enumerate(prompts):
            eng.send(" ".join(x for x in ("BGEN", str(i), str(a.max_new), keys(i), ",".join(map(str, ids))) if x))
            for line in out:
                if line.startswith("T "):
                    got[i].append(int(line.split()[1]))
                elif line.startswith("BT "):
                    _, s, y = line.split()
                    got[int(s)].append(int(y))
                elif line.startswith("BDONE "):
                    done.add(int(line.split()[1]))
                elif line.startswith("ERR"):
                    print("batch:", line); return 1
                elif line.startswith("BADM "):
                    if line.split()[2] == "0":
                        done.add(i)
                    break
        while len(done) < len(prompts):   # (not "for line in out": it would wait for one more line when all are done)
            line = next(out)
            if line.startswith("BT "):
                _, s, y = line.split()
                got[int(s)].append(int(y))
            elif line.startswith("BDONE "):
                done.add(int(line.split()[1]))
            elif line.startswith("ERR"):
                print("batch:", line); return 1
        for i in range(len(prompts)):
            same = got[i] == solo[i]
            ok &= same
            han = len(BT.CJK.findall(tok.decode(got[i])))
            first_diff = next((k for k in range(min(len(got[i]), len(solo[i]))) if got[i][k] != solo[i][k]), None)
            print(f"wave {w} slot {i}: {'banned' if biased(i) else 'free  '} {len(got[i])} tokens, {han} Han/Kana/Hangul characters: "
                  f"{'IDENTICAL to solo' if same else f'DIFFERS at {first_diff}'}", flush=True)
    eng.send("QUIT")
    eng.p.wait(timeout=180)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
