#!/usr/bin/env python3
"""The quant quality yardstick (MODEL-UPDATE.md, Phase 0): teacher-forced top-1 agreement and KL of a quant against
Q8_0, on one fixed text set (data/quant_quality/v1: code, docs, English, other languages, an agent transcript).

    python tools/quant_quality.py ids --tokenizer DIR [--set data/quant_quality/v1] [--out set.ids.json]
        tokenizes the set (each segment cut to --seg-tokens, default 1000) - the ids file is committed with the set
        (data/quant_quality/v1/set.ids.json), so a box without the tokenizer scores the same positions
    python tools/quant_quality.py run --config CFG.json --out RUN.tsv [--ids set.ids.json] [--env K=V ...] [--gpu-experts]
                                       [--kv FMT]   FMT is the KV cache format (default fp16); run one arm per
                                                    --kv to score KV quantization against fp16 K/V with the weights fixed
        one `strata --serve` engine (the config's exe and args), every segment one request whose tokens are all read
        through the decode windows (--short-read), where STRATA_LOGPOS / STRATA_LOGPOS_TOPK=64 write, per position,
        the log-probability of the text's next token and the 64 most likely tokens.  Fixed: the experts on the CPU
        (one VRAM slot, --expert-cache 1: the verify windows need a profile-filled tier, so the config must have
        --expert-profile; --gpu-experts: 1500 slots), no expert adaptation, no PCIe share, no suffix drafts, no prompt
        cache, K/V as --kv says (fp16 by default; the config's own --kv/--kv-resident are dropped, so the flag decides),
        STRATA_ROUTE_TAIL_SKIP=0, STRATA_PREFILL_CPU_SHARE=0.  RUN.tsv + RUN.tsv.seg.json (line ranges).
    python tools/quant_quality.py compare REF.tsv RUN.tsv [...]   (REF = the Q8_0 run)
        per segment and overall: top-1 agreement (same argmax), mean / p99 KL(REF || RUN) over REF's 64 most likely
        tokens (both renormalized to that set; a token RUN does not list gets RUN's 64th log-probability), and the
        perplexity of the text under each.

    python tools/quant_quality.py fwe --tokenizer DIR [--set data/quant_quality/fwe-2k] [--seq-tokens 2048]
                                      [--samples 8] [--alpha 2.0] [--answer-words 3] [--seed 42]
        builds the decision-test set: RULER's frequent-words-extraction task (arXiv:2404.06654 sec. 3.3) - a long
        list of six-letter coded words drawn from a Zeta(alpha) rank distribution, the top-ranked word replaced
        by '...' (the noise the question says to ignore), then the question and the `--answer-words` most frequent
        words.  RULER's FWE asks for three, so that is the default and data/quant_quality/fwe-2k is reproducible
        from it byte for byte.  RULER's CWE asks for ten, and `--answer-words 10` into data/quant_quality/cwe-2k is
        the same generator and the same first three answer words with ~3.4x the answer-region rows.  One instance
        per segment, `--seq-tokens` long, with the answer appended so this harness scores it.  The instances differ
        (RULER's own script re-seeds one RNG for every sample, so its num_samples are identical).  The drawn counts
        are exact integers, so the gold order is the realized order except where two ranks tie (fwe-2k: first tie
        at rank 12-13; cwe-2k: rank 11 in one of the eight segments, which makes that one slot ambiguous).
    python tools/quant_quality.py areport IDS.json REF.tsv RUN.tsv [...]   (REF = the fp16 run)
        the answer region only (`pos + 1 >= answer_at`): how often each arm's argmax is the text's own next
        token (the task's accuracy, and REF's value is the headroom), how often it agrees with REF, and the KL
        at those positions.  Measured on fwe-2k (8 segments, 16,355 positions, 84 answer rows): the answer-region
        accuracy has no resolution at all (fp16 = int8 = q4_0 = 80.95%; int8 reproduces fp16's argmax at 84/84 rows
        and q4_0 at 83/84), and the answer-region KL does not separate int8 from q4_0 either (paired over segments,
        t = 0.69).  The whole-segment KL on the same runs does (int8 0.00518 against q4_0 0.01100; paired over
        segments t = 3.50, 8 of 8 segments in the same direction).  The answer region is the weak readout here
        because the answer copies words that are already in the context, so emitting them is near-deterministic:
        only 1 of the 84 decisions sits near a boundary.

The RUN.tsv rows are the engine's STRATA_LOGPOS format: pos, target id, logp(target), argmax id, logp(argmax),
argmax==target, logp(<|im_end|>), logp(target without <|im_end|>), then 64 `id:logp` fields (natural log).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import string
import struct
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
SET_DIR = HERE.parent / "data" / "quant_quality" / "v1"
SEGMENTS = ["code", "docs", "english", "multilingual", "agent"]
TOPK = 64
FIXED = ["--expert-cache", "1", "--adapt-every", "100000", "--pcie-frac", "0", "--suffix-draft", "0",
         "--prompt-cache", "0"]
FIXED_ENV = {"STRATA_ROUTE_TAIL_SKIP": "0", "STRATA_PREFILL_CPU_SHARE": "0", "STRATA_PCIE_FRAC_DEFAULT": "old"}
# config args the yardstick replaces (value-taking) or drops (flags).  --kv is dropped here and set by the `--kv`
# flag instead, so the same config scores every KV format against the fp16 default.
DROP_VALUE = {"--expert-cache", "--adapt-every", "--pcie-frac", "--suffix-draft", "--prompt-cache", "--max-context",
              "--prefill", "--prompt-cache-every", "--short-read", "--conversation-cache-mib", "--eos-ids",
              "--kv", "--kv-resident"}   # (--resident-budget-gib stays: a PC whose RAM does not hold the experts)
DROP_FLAG = {"--serve", "--vision", "--resident-experts", "--resident-cpu-experts", "--prefill-head-alias"}


# ------------------------------------------------------------------------------------------------ ids
def load_tokenizer(path: str):
    """The pack's `tokenizer/` directory, as `tools/strata_tokenizer.py` reads it."""
    sys.path.insert(0, str(HERE))
    import strata_tokenizer as ST  # noqa: E402
    tp = pathlib.Path(path)
    vocab = json.loads((tp / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    return ST.Tokenizer(toks, (tp / "merges.txt").read_text(encoding="utf-8").split("\n"),
                        json.loads((tp / "token_type.json").read_text()))


def cmd_ids(a) -> int:
    tok = load_tokenizer(a.tokenizer)
    segs = []
    wanted = a.segments.split(",") if isinstance(a.segments, str) else list(a.segments)
    for name in wanted:
        name = name.strip()
        text = (pathlib.Path(a.set) / f"{name}.txt").read_text(encoding="utf-8")
        ids = tok.encode(text, parse_special=(name == "agent"))
        if len(ids) < a.seg_tokens:
            raise SystemExit(f"{name}: only {len(ids)} tokens (< {a.seg_tokens})")
        segs.append({"name": name, "ids": ids[:a.seg_tokens]})
        print(f"{name}: {len(ids)} tokens, kept {a.seg_tokens}")
    out = pathlib.Path(a.out or (pathlib.Path(a.set) / "set.ids.json"))
    out.write_text(json.dumps({"version": pathlib.Path(a.set).name, "seg_tokens": a.seg_tokens, "vocab_size": len(toks), "segments": segs}))
    print(f"wrote {out}")
    return 0


# ------------------------------------------------------------------------------------------------ fwe
FWE_VOCAB_PER = 50      # RULER: vocab_size = max_seq_length // 50
FWE_WORDLEN = 6         # RULER --coded_wordlen
FWE_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
             10: "ten", 20: "twenty", 30: "thirty"}
FWE_TEMPLATE = (
    "Read the following coded text and track the frequency of each coded word. Find the {num} most frequently "
    "appeared coded words. {context}\n"
    "Question: Do not provide any explanation. Please ignore the dots '....'. What are the {num} most frequently "
    "appeared words in the above coded text?\n"
    "Answer: According to the coded text above, the {num} most frequently appeared words are:")


def riemann_zeta(alpha: float, terms: int = 200000) -> float:
    """sum k^-alpha, which is what scipy.special.zeta(alpha) gives RULER (the infinite sum, not a truncated one)."""
    return sum(k ** -alpha for k in range(1, terms + 1))


def fwe_vocab(size: int, rng: random.Random) -> list[str]:
    vocab = ["".join(rng.choices(string.ascii_lowercase, k=FWE_WORDLEN)) for _ in range(size)]
    while len(set(vocab)) < size:
        vocab.append("".join(rng.choices(string.ascii_lowercase, k=FWE_WORDLEN)))
    vocab = sorted(set(vocab))
    rng.shuffle(vocab)
    vocab[0] = "..."        # RULER: the top-ranked word is the noise the question tells the model to ignore
    return vocab


def fwe_context(vocab: list[str], num_words: int, alpha: float, zeta: float, seed: int) -> str:
    """RULER's sample: word k appears `num_words * k^-alpha / zeta(alpha)` times, then one shuffle of all of them."""
    words = [w for k, w in enumerate(vocab, 1) for _ in range(int(num_words * k ** -alpha / zeta))]
    random.Random(seed).shuffle(words)
    return " ".join(words)


def cmd_fwe(a) -> int:
    tok = load_tokenizer(a.tokenizer)
    out = pathlib.Path(a.set)
    out.mkdir(parents=True, exist_ok=True)
    zeta = riemann_zeta(a.alpha)
    vocab_size = max(16, a.seq_tokens // FWE_VOCAB_PER)
    segs, gold = [], []
    for i in range(a.samples):
        name = "s%02d" % i
        vocab = fwe_vocab(vocab_size, random.Random(a.seed + i))
        answer = vocab[1:1 + a.answer_words]

        def prompt(n: int) -> str:
            return FWE_TEMPLATE.format(num=FWE_WORDS.get(a.answer_words, str(a.answer_words)),
                                       context=fwe_context(vocab, n, a.alpha, zeta, a.seed * 1000 + i))

        def full(n: int) -> str:
            return prompt(n) + " " + " ".join(answer)

        lo, hi = 1, a.seq_tokens     # a coded word is at least one token, so 1..seq_tokens is the whole range
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(tok.encode(full(mid))) <= a.seq_tokens:
                lo = mid
            else:
                hi = mid - 1
        text, pre = full(lo), prompt(lo)
        ids = tok.encode(text)
        answer_at = len(tok.encode(pre))
        if len(ids) > a.seq_tokens:
            raise SystemExit(f"{name}: {len(ids)} tokens, more than --seq-tokens {a.seq_tokens}")
        back = tok.decode(ids[answer_at:])
        if not all(w in back for w in answer):
            raise SystemExit(f"{name}: the answer region {back!r} does not carry {answer}")
        (out / f"{name}.txt").write_text(text)
        segs.append({"name": name, "ids": ids, "answer_at": answer_at})
        gold.append({"name": name, "answer": answer, "drawn_words": lo, "tokens": len(ids), "answer_at": answer_at})
        print(f"{name}: {len(ids)} tokens, {lo} drawn words, answer at {answer_at} ({' '.join(answer)})")
    (out / "set.ids.json").write_text(json.dumps(
        {"version": out.name, "seg_tokens": a.seq_tokens, "task": "RULER FWE", "alpha": a.alpha,
         "answer_words": a.answer_words,
         "coded_wordlen": FWE_WORDLEN, "vocab_size": vocab_size, "seed": a.seed, "segments": segs}, indent=0))
    (out / "gold.json").write_text(json.dumps(gold, indent=1))
    print(f"wrote {out}/set.ids.json, gold.json and {a.samples} s*.txt")
    return 0


# ------------------------------------------------------------------------------------------------ run
def mem_available_gb() -> float:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return float("inf")


def engine_args(cfg: dict, gpu_experts, seg_tokens: int, kv: str = "fp16",
                short_read: int | None = None, prefill: str = "512",
                prompt_cache: int = 0, max_ctx_cap: int = 131072) -> list[str]:
    base, out, i = list(cfg["args"]), [], 0
    while i < len(base):
        x = base[i]
        if x in DROP_FLAG:
            i += 1
            continue
        if x in DROP_VALUE:
            i += 2
            continue
        out.append(x)
        i += 1
    fixed = list(FIXED)
    if gpu_experts:
        fixed[1] = gpu_experts if isinstance(gpu_experts, str) else "1500"
    # The pack's context is 131,072 tokens; a 128K segment needs the whole window for the prompt plus one token.
    # Without the cap a 128K set would ask for 131,584 and the engine would refuse the model's own context.
    # --max-context raises the cap for a longer set (the model's own context is 262,144); the default keeps every
    # existing run byte-identical.
    max_ctx = min(seg_tokens + 512, max_ctx_cap)
    short = min(seg_tokens + 64, max_ctx - 1)
    if short_read is not None:   # --short-read N: 0 forces the batched prompt path (the S2 probe's route)
        short = min(short_read, max_ctx - 1)
    args = [cfg["exe"], "--serve", *out, *fixed, "--kv", kv, "--max-context", str(max_ctx),
            "--short-read", str(short), "--prefill", str(prefill)]
    if prompt_cache:   # a request's pin=N needs --prompt-cache 3 or more (generate.cpp); default stays 0
        args[args.index("--prompt-cache") + 1] = str(prompt_cache)
    return args


def check_ids_vocab(st: dict, cfg: dict, ids_path: str) -> None:
    """A wrong ids file fails here, loudly: every id must be inside the model's vocabulary (the vocabulary of the config's tokenizer
    folder when it names one, else the size the ids file recorded when `ids` made it), or the run reads nonsense - the committed
    data/quant_quality/v1/set.ids.json is tokenized for Qwen (ids up to 248,069), not for DeepSeek's 129,280."""
    vocab = None
    tdir = cfg.get("tokenizer")
    if tdir and (pathlib.Path(tdir) / "vocab.json").is_file():
        vocab = len(json.loads((pathlib.Path(tdir) / "vocab.json").read_text(encoding="utf-8")))
    elif st.get("vocab_size"):
        vocab = int(st["vocab_size"])
    if not vocab:
        return
    for s in st["segments"]:
        mx = max(s["ids"]) if s["ids"] else -1
        if mx >= vocab:
            raise SystemExit(f"{ids_path}: segment '{s['name']}' has token id {mx} >= the model's vocabulary ({vocab}): this ids file was "
                             f"made with another tokenizer; make one with `quant_quality.py ids --tokenizer <the model's tokenizer folder>`")


def cmd_run(a) -> int:
    cfg = json.load(open(a.config, encoding="utf-8-sig"))
    st = json.load(open(a.ids))
    check_ids_vocab(st, cfg, a.ids)
    out = pathlib.Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    env = dict(os.environ)
    env.update(cfg.get("env", {}))
    if cfg.get("lib_dirs"):
        # Windows loads the engine's HIP libraries from PATH (serve/server.py's child_env), Linux from LD_LIBRARY_PATH
        var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[var] = os.pathsep.join(cfg["lib_dirs"] + ([env[var]] if env.get(var) else []))
    env.update(FIXED_ENV)
    for kv in a.env or []:
        k, _, v = kv.partition("=")
        env[k] = v
    env["STRATA_LOGPOS"] = str(out)
    env["STRATA_LOGPOS_TOPK"] = str(a.topk)
    if mem_available_gb() < a.min_free_gb:
        raise SystemExit(f"only {mem_available_gb():.0f} GB available (< {a.min_free_gb}): not starting an engine")
    args = engine_args(cfg, a.gpu_experts, st["seg_tokens"], a.kv, a.short_read, a.prefill, a.prompt_cache,
                       getattr(a, "max_context", 0) or 131072)
    if getattr(a, "exe", None):
        args[0] = a.exe
    if a.wrap:
        args = a.wrap.split() + args
    errf = open(str(out) + ".stderr.txt", "w")
    print("engine:", " ".join(args), flush=True)
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf, text=True,
                         cwd=cfg.get("cwd"), env=env, bufsize=1)
    for line in p.stdout:
        if line.startswith("READY"):
            break
    else:
        raise SystemExit(f"the engine ended before READY (exit {p.wait()}); see {out}.stderr.txt")
    segs, info = [], []

    def lines() -> int:
        return sum(1 for _ in open(out)) if out.exists() else 0

    for s in st["segments"]:
        n0, t0 = lines(), time.time()
        # pin=N cuts the prompt read at N (a checkpoint edge): the part before is batched, the rest goes through the
        # verify windows when it fits --short-read (generate.cpp: cuts = {reread_to, root_at, message_at, turn_at,
        # pin_at, n-1}).  Needs --prompt-cache >= 3.  Absent: the request is exactly the one every earlier run sent.
        pin = f"pin={a.pin} " if a.pin else ""
        p.stdin.write("GEN 1 " + pin + ",".join(map(str, s["ids"])) + "\n")
        p.stdin.flush()
        done = None
        for line in p.stdout:
            if line.startswith(("DONE", "ERR")):
                done = line.strip()
                break
        if done is None or not done.startswith("DONE") or "nonfinite" in done:
            p.kill()
            raise SystemExit(f"{s['name']}: the engine answered {done!r}; see {out}.stderr.txt")
        n1 = lines()
        segs.append({"name": s["name"], "first_line": n0, "end_line": n1})
        info.append(f"{s['name']}: {n1 - n0} positions, {time.time() - t0:.1f} s")
        print(info[-1], flush=True)
        if n1 - n0 < len(s["ids"]) // 2:
            if a.allow_no_positions:   # a probe run (STRATA_KV_S2_STATS): the batched prompt path writes no positions
                print(f"{s['name']}: {n1 - n0} positions scored, allowed for a probe run", flush=True)
                continue
            p.kill()
            raise SystemExit(f"{s['name']}: only {n1 - n0} positions were scored (the prompt took the batched path?)")
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    try:
        p.wait(timeout=600)
    except subprocess.TimeoutExpired:
        p.kill()
    errf.close()
    err = pathlib.Path(str(out) + ".stderr.txt").read_text(errors="replace")
    cache = [l for l in err.splitlines() if "expert cache" in l][:3]
    pathlib.Path(str(out) + ".seg.json").write_text(json.dumps({"segments": segs, "config": a.config,
                                                                 "args": args, "env": a.env or [],
                                                                 "kv": a.kv, "topk": a.topk,
                                                                 "pin": a.pin, "prompt_cache": a.prompt_cache,
                                                                 "expert_cache_lines": cache}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ kvstats
KVSTATS_MAGIC = b"KVSTATS\x02"
KVSTATS_GROUP = struct.Struct("<5d 24d 32Q 256f")   # cnt,sum,sumsq,sum4,mx + bsum/bsumsq/bmx x8 + hist + chmax
KVSTATS_SECT = struct.Struct("<q q I")              # pos0, n_tok, ngroups, then (uint16 gidx + group) each
KVSTATS_KAPPA = [0.80 + 0.02 * i for i in range(11)]
KVSTATS_BINLO = [0.0] + [2.0 ** (b - 8) for b in range(1, 32)] + [float("inf")]   # bin b: [BINLO[b], BINLO[b+1])
KVSTATS_BINMID = [math.sqrt(a * b) if a > 0 else 0.0 for a, b in zip(KVSTATS_BINLO, KVSTATS_BINLO[1:])]


def kvstats_decode_gidx(gi: int) -> tuple[int, int, int, int]:
    tap, r = gi % 2, gi // 2
    is_v, r = r % 2, r // 2
    return r // 8, r % 8, is_v, tap


def kvstats_aggregate(raw: bytes, n_tokens: int) -> dict:
    """The STRATA_KVSTATS sections file into per-(layer,head,tensor,tap) distribution stats. Stdlib only."""
    if raw[:8] != KVSTATS_MAGIC:
        raise SystemExit("kvstats: bad magic (is STRATA_KVSTATS the raw file, not the JSON?)")
    groups: dict[tuple, dict] = {}
    windows = []
    off, n = 8, len(raw)
    while off < n:
        pos0, ntok, ng = KVSTATS_SECT.unpack_from(raw, off)
        off += KVSTATS_SECT.size
        windows.append((pos0, ntok))
        for _ in range(ng):
            gi = struct.unpack("<H", raw[off:off + 2])[0]
            off += 2
            vals = KVSTATS_GROUP.unpack_from(raw, off)
            off += KVSTATS_GROUP.size
            layer, head, is_v, tap = kvstats_decode_gidx(gi)
            g = groups.setdefault((layer, head, is_v, tap),
                                  {"cnt": 0.0, "s1": 0.0, "s2": 0.0, "s4": 0.0, "mx": 0.0,
                                   "bs": [0.0] * 8, "bsq": [0.0] * 8, "bx": [0.0] * 8,
                                   "hist": [0] * 32, "chmax": [0.0] * 256})
            g["cnt"] += vals[0]
            g["s1"] += vals[1]
            g["s2"] += vals[2]
            g["s4"] += vals[3]
            g["mx"] = max(g["mx"], vals[4])
            for b in range(8):
                g["bs"][b] += vals[5 + b]
                g["bsq"][b] += vals[13 + b]
                g["bx"][b] = max(g["bx"][b], vals[21 + b])
            for b in range(32):
                g["hist"][b] += vals[29 + b]
            for c in range(256):
                g["chmax"][c] = max(g["chmax"][c], vals[61 + c])
    out: dict = {"groups": {}, "checks": {}}
    fwht_worst, fwht_n = 0.0, 0
    pre_s2: dict[tuple, float] = {}
    for key in sorted(groups):
        layer, head, is_v, tap = key
        if not tap:
            pre_s2[(layer, head, is_v)] = groups[key]["s2"]
    for key in sorted(groups):
        layer, head, is_v, tap = key
        g = groups[key]
        nv = g["cnt"]
        mean = g["s1"] / nv
        m2 = g["s2"] / nv
        var = m2 - mean * mean
        rms = math.sqrt(max(m2, 0.0))
        kurt = (g["s4"] / nv) / (var * var) - 3.0 if var > 0 else float("nan")
        tot = sum(g["hist"]) or 1
        acc, dec = 0, {}
        for b in range(32):
            acc += g["hist"][b]
            for p in (0.50, 0.90, 0.99):
                if p not in dec and acc >= p * tot:
                    dec[p] = KVSTATS_BINLO[b]
        exc = {}
        for k in (2, 3, 4, 5):   # binned approximation by geometric bin midpoint (centers the straddle
            thr = k * rms        # error instead of maximizing it); overflow bin midpoint is +inf: always in
            exc[str(k)] = sum(h for b, h in enumerate(g["hist"]) if KVSTATS_BINMID[b] >= thr) / tot
        uncl = {}   # binned approximation by lower edge (mild overestimate; keeps unclipped[1.00] == 1.0 exact)
        if g["mx"] > 0:
            for k in KVSTATS_KAPPA:
                uncl[f"{k:.2f}"] = sum(h for b, h in enumerate(g["hist"])
                                       if KVSTATS_BINLO[b] < k * g["mx"]) / tot
        ch = sorted(g["chmax"], reverse=True)
        ctot = sum(ch) or 1.0
        name = f"L{layer:02d}.h{head}.{'V' if is_v else 'K'}.{'post' if tap else 'pre'}"
        out["groups"][name] = {"values": nv, "mean": mean, "rms": rms, "kurtosis_excess": kurt,
                               "maxabs": g["mx"], "crest": g["mx"] / rms if rms > 0 else float("nan"),
                               "bmax_p50": dec.get(0.50, float("nan")), "bmax_p90": dec.get(0.90, float("nan")),
                               "bmax_p99": dec.get(0.99, float("nan")), "exceed_rms": exc,
                               "unclipped_kappa": uncl,
                               "channels": {"top1_mass": ch[0] / ctot, "top8_mass": sum(ch[:8]) / ctot,
                                            "top8_idx": sorted(range(256),
                                                               key=lambda c: -g["chmax"][c])[:8]}}
        if tap:
            a = pre_s2.get((layer, head, is_v))
            b = g["s2"]
            if a is not None and a > 0:
                fwht_n += 1
                fwht_worst = max(fwht_worst, abs(b - a) / a)
    cov = sum(nt for _, nt in windows)
    out["checks"] = {"tokens": n_tokens, "covered_tokens": cov, "windows": [list(w) for w in windows],
                     "ngroups": len(groups), "fwht_norm_matched": fwht_n,
                     "fwht_norm_maxreldiff": fwht_worst}
    return out


def cmd_kvstats(a) -> int:
    """One calibration text through the engine with STRATA_KVSTATS set, then aggregate to JSON."""
    cfg = json.load(open(a.config, encoding="utf-8-sig"))
    tok = load_tokenizer(cfg["tokenizer"])
    text = pathlib.Path(a.text).read_text(encoding="utf-8")
    ids = tok.encode(text)[:a.tokens]
    if len(ids) < a.tokens:
        raise SystemExit(f"{a.text}: only {len(ids)} tokens (< {a.tokens})")
    out = pathlib.Path(a.out).resolve()
    raw = pathlib.Path(a.raw or (str(out) + ".raw.bin")).resolve()
    if raw.exists():
        raw.unlink()
    env = dict(os.environ)
    env.update(cfg.get("env", {}))
    if cfg.get("lib_dirs"):
        # Windows loads the engine's HIP libraries from PATH (serve/server.py's child_env), Linux from LD_LIBRARY_PATH
        var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[var] = os.pathsep.join(cfg["lib_dirs"] + ([env[var]] if env.get(var) else []))
    env.update(FIXED_ENV)
    for kv in a.env or []:
        k, _, v = kv.partition("=")
        env[k] = v
    env["STRATA_KVSTATS"] = str(raw)
    # Force the serial verify loop (same path every measurement uses): with STRATA_LOGPOS unset the engine
    # takes the pipelined prompt path, whose windows the stats drain does not cover. os.devnull discards it.
    env["STRATA_LOGPOS"] = os.devnull
    if mem_available_gb() < a.min_free_gb:
        raise SystemExit(f"only {mem_available_gb():.0f} GB available (< {a.min_free_gb}): not starting an engine")
    args = engine_args(cfg, a.gpu_experts, a.tokens, a.kv)
    if a.wrap:
        args = a.wrap.split() + args
    errf = open(str(out) + ".stderr.txt", "w")
    print("engine:", " ".join(args), flush=True)
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf, text=True,
                         cwd=cfg.get("cwd"), env=env, bufsize=1)
    for line in p.stdout:
        if line.startswith("READY"):
            break
    else:
        raise SystemExit(f"the engine ended before READY (exit {p.wait()}); see {out}.stderr.txt")
    # No warmup GEN: the tap accumulators are allocated at engine startup (kv_stats_alloc in the serve
    # init), so the first recording already contains the tap kernels. Positions below are order markers.
    p.stdin.write("GEN 1 " + ",".join(map(str, ids)) + "\n")
    p.stdin.flush()
    done = None
    for line in p.stdout:
        if line.startswith(("DONE", "ERR")):
            done = line.strip()
            break
    if done is None or not done.startswith("DONE") or "nonfinite" in done:
        p.kill()
        raise SystemExit(f"kvstats: the engine answered {done!r}; see {out}.stderr.txt")
    print(done, flush=True)
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    try:
        p.wait(timeout=600)
    except subprocess.TimeoutExpired:
        p.kill()
    errf.close()
    agg = kvstats_aggregate(raw.read_bytes(), len(ids))
    agg["meta"] = {"text": str(a.text), "tokens": len(ids), "kv": a.kv, "config": a.config}
    out.write_text(json.dumps(agg, indent=1))
    print(f"wrote {out} ({len(agg['groups'])} groups, "
          f"fwht_norm_maxreldiff {agg['checks']['fwht_norm_maxreldiff']:.2e} over "
          f"{agg['checks']['fwht_norm_matched']} matched steps)")
    return 0


# ------------------------------------------------------------------------------------------------ prompt path
PROMPT_LENS = (256, 512, 768, 1000)


def cmd_prompt(a) -> int:
    """The batched prompt path's answer: every segment's first 256 / 512 / 768 / 1000 tokens as a prompt (one request
    each, no prompt cache), and the logits after it (STRATA_DUMP_FIRST_LOGITS: float32, the whole vocabulary).
    `--lens 256,512,768,1000,2048,4056` widens the nested set (decode-4k); `--prefill N` fixes the chunk instead
    of `auto` (a ring the user sized needs a chunk that fits its recent rows)."""
    cfg = json.load(open(a.config, encoding="utf-8-sig"))
    st = json.load(open(a.ids))
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("first.*"):
        f.unlink()
    env = dict(os.environ)
    env.update(cfg.get("env", {}))
    if cfg.get("lib_dirs"):
        # Windows loads the engine's HIP libraries from PATH (serve/server.py's child_env), Linux from LD_LIBRARY_PATH
        var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[var] = os.pathsep.join(cfg["lib_dirs"] + ([env[var]] if env.get(var) else []))
    env.update(FIXED_ENV)
    for kv in a.env or []:
        k, _, v = kv.partition("=")
        env[k] = v
    env["STRATA_DUMP_FIRST_LOGITS"] = str(out / "first")
    if mem_available_gb() < a.min_free_gb:
        raise SystemExit(f"only {mem_available_gb():.0f} GB available (< {a.min_free_gb}): not starting an engine")
    lens = tuple(int(x) for x in a.lens.replace(" ", "").split(",")) if a.lens else PROMPT_LENS
    for s in st["segments"]:
        for n in lens:
            if n > len(s["ids"]):
                raise SystemExit(f"{s['name']}: len {n} exceeds the segment's {len(s['ids'])} tokens")
    args = engine_args(cfg, a.gpu_experts, max(lens), a.kv)
    k = args.index("--short-read")
    del args[k:k + 2]                      # the default (64): these prompts take the batched prompt path
    k = args.index("--prefill")
    args[k + 1] = "auto" if a.prefill is None else str(a.prefill)
    if a.wrap:
        args = a.wrap.split() + args
    errf = open(out / "stderr.txt", "w")
    print("engine:", " ".join(args), flush=True)
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf, text=True,
                         cwd=cfg.get("cwd"), env=env, bufsize=1)
    for line in p.stdout:
        if line.startswith("READY"):
            break
    else:
        raise SystemExit(f"the engine ended before READY (exit {p.wait()}); see {out}/stderr.txt")
    index = []
    for s in st["segments"]:
        for n in lens:
            p.stdin.write("GEN 1 " + ",".join(map(str, s["ids"][:n])) + "\n")
            p.stdin.flush()
            done = None
            for line in p.stdout:
                if line.startswith(("DONE", "ERR")):
                    done = line.strip()
                    break
            if done is None or not done.startswith("DONE") or "nonfinite" in done:
                p.kill()
                raise SystemExit(f"{s['name']} {n}: the engine answered {done!r}")
            index.append({"segment": s["name"], "tokens": n})
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    try:
        p.wait(timeout=600)
    except subprocess.TimeoutExpired:
        p.kill()
    errf.close()
    (out / "index.json").write_text(json.dumps({"requests": index, "lens": lens, "args": args, "env": a.env or []}, indent=1))
    print(f"{len(index)} prompts")
    return 0


def cmd_pcompare(a) -> int:
    import numpy as np
    ref = pathlib.Path(a.ref)
    idx = json.loads((ref / "index.json").read_text())["requests"]

    def logp(d, i):
        x = np.fromfile(d / f"first.{i}", dtype=np.float32).astype(np.float64)
        x -= x.max()
        return x - np.log(np.exp(x).sum())

    print(f"reference: {ref}")
    for run in a.runs:
        r = pathlib.Path(run)
        kls, agree = [], 0
        for i in range(len(idx)):
            if not (r / f"first.{i}").exists():
                continue
            lr, la = logp(ref, i), logp(r, i)
            pr = np.exp(lr)
            kls.append(float((pr * (lr - la)).sum()))
            agree += int(lr.argmax() == la.argmax())
        kls.sort()
        n = len(kls)
        print(f"{r.name:40s} prompts {n:3d}  top-1 {100 * agree / max(n, 1):6.2f}%  KL mean {sum(kls) / max(n, 1):.5f}  "
              f"median {kls[n // 2] if n else float('nan'):.5f}  max {kls[-1] if n else float('nan'):.5f}")
    return 0


# ------------------------------------------------------------------------------------------------ compare
def read_run(path: str) -> dict:
    rows = open(path).read().splitlines()
    meta = json.load(open(path + ".seg.json"))
    segs = meta["segments"]
    width = int(meta.get("topk", TOPK))   # runs written before --topk existed carry 64 fields
    out = {}
    for s in segs:
        d = {}
        for line in rows[s["first_line"]:s["end_line"]]:
            f = line.split("\t")
            if len(f) < 8 + width:
                continue
            d[int(f[0])] = (int(f[1]), float(f[2]), int(f[3]), f[5] == "1",
                            [(int(x), float(y)) for x, y in (z.split(":") for z in f[8:8 + width])])
        out[s["name"]] = d
    return out


def kl_between(kr, ka, kl_topk: int = TOPK) -> float:
    """KL(REF || RUN) over REF's kl_topk most likely tokens; a token RUN does not list gets RUN's kl_topk-th
    log-probability.  kl_topk=64 is the published definition."""
    kr, ka = kr[:kl_topk], ka[:kl_topk]
    da = dict(ka)
    floor = min(b for _, b in ka)
    pr = [math.exp(b) for _, b in kr]
    pa = [math.exp(da.get(i, floor)) for i, _ in kr]
    sr, sa = sum(pr), sum(pa)
    return sum((x / sr) * math.log((x / sr) / (y / sa)) for x, y in zip(pr, pa) if x > 0)


def compare(ref: dict, run: dict, kl_topk: int = TOPK) -> dict:
    res, allk, tot = {}, [], {"n": 0, "agree": 0, "nll_r": 0.0, "nll_a": 0.0, "kl": 0.0}
    for name, rr in ref.items():
        ra = run.get(name, {})
        pos = sorted(set(rr) & set(ra))
        n = agree = 0
        kls, nll_r, nll_a = [], 0.0, 0.0
        for q in pos:
            tr, lr, topr, _hit, kr = rr[q]
            ta, la, topa, _hita, ka = ra[q]
            if tr != ta:
                raise SystemExit(f"{name} pos {q}: target {tr} != {ta} (not the same text)")
            n += 1
            agree += topr == topa
            nll_r -= lr
            nll_a -= la
            if len(kr) < kl_topk or len(ka) < kl_topk:
                raise SystemExit(f"{name} pos {q}: only {len(kr)}/{len(ka)} neighbours, {kl_topk} asked "
                                 f"(write the runs with --topk {kl_topk})")
            kls.append(kl_between(kr, ka, kl_topk))
        allk += kls
        if n:
            res[name] = {"positions": n, "top1": agree / n, "kl": sum(kls) / n,
                         "ppl_ref": math.exp(nll_r / n), "ppl": math.exp(nll_a / n)}
        tot["n"] += n
        tot["agree"] += agree
        tot["nll_r"] += nll_r
        tot["nll_a"] += nll_a
        tot["kl"] += sum(kls)
    n = max(tot["n"], 1)
    allk.sort()
    res["all"] = {"positions": tot["n"], "top1": tot["agree"] / n, "kl": tot["kl"] / n,
                  "kl_p99": allk[int(0.99 * (len(allk) - 1))] if allk else float("nan"),
                  "ppl_ref": math.exp(tot["nll_r"] / n), "ppl": math.exp(tot["nll_a"] / n)}
    return res


def cmd_compare(a) -> int:
    ref = read_run(a.ref)
    anchor = anchor_of(a.ref)
    print(f"reference: {a.ref} (anchor: {anchor}; KL support: REF's {a.kl_topk} most likely tokens)")
    if anchor == "other":
        print("  warning: this reference is neither fp16 nor int8 - name the question these numbers answer")
    print(f"{'run':40s} {'segment':13s} {'pos':>5s} {'top1':>7s} {'KL':>8s} {'KL p99':>8s} {'PPL':>7s} {'PPL ref':>7s}")
    allres = {}
    for path in a.runs:
        r = compare(ref, read_run(path), a.kl_topk)
        allres[path] = r
        for seg, v in r.items():
            if seg != "all" and not a.segments:
                continue
            print(f"{pathlib.Path(path).name[:40]:40s} {seg:13s} {v['positions']:5d} {100 * v['top1']:6.2f}% "
                  f"{v['kl']:8.5f} {v.get('kl_p99', float('nan')):8.4f} {v['ppl']:7.3f} {v['ppl_ref']:7.3f}")
    paired_out = paired_tables(a, ref, allres) if a.paired else None
    if a.paired and len(a.runs) > 1:
        series = {pathlib.Path(p).name: kl_series(ref, read_run(p), a.kl_topk) for p in a.runs}
        names = list(series)
        print("\nper-position paired differences between arms (a pair against the anchor is not a paired "
              "difference: the anchor's own KL is 0 by construction):")
        print(f"  {'pair':54s} {'n':>6s} {'delta':>9s} {'95% interval':>23s} {'t':>6s} {'bound':>7s} {'worse':>8s}")
        pos_out = {}
        for i, na in enumerate(names):
            for nb in names[i + 1:]:
                p = paired_positions(series[na], series[nb])
                if p is None:
                    continue
                pos_out[f"{na} - {nb}"] = p
                print(f"  {na[:26]:26s} - {nb[:26]:26s} {p['n']:6d} {p['mean']:+9.5f} "
                      f"[{p['ci95_low']:+.5f}, {p['ci95_high']:+.5f}] {p['t']:+6.2f} "
                      f"{p['parity_bound']:7.5f} {p['segments_a_worse']:>4d}/{p['segments']}")
        print("  'bound' is the one-sided 95 % resolution at true parity (1.645*sd/sqrt(n)); a difference below "
              "it is not a result about the arm, it is a property of the corpus.")
        if paired_out is not None:
            paired_out["positions"] = pos_out
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(allres, indent=1))
    if a.paired_json and paired_out is not None:
        pathlib.Path(a.paired_json).write_text(json.dumps(paired_out, indent=1))
    return 0


def paired_stats(sa: dict, sb: dict) -> dict:
    """The per-segment difference A - B over the segments both cover.  Every arm sees the identical text, so the
    segment-to-segment spread of the reference - which is most of the spread of a single arm - cancels out: this is
    the right test for "is the difference between two formats real", and a plain mean over positions cannot see it
    (it has no notion of which differences came from the same text)."""
    segs = sorted(set(sa) & set(sb))
    d = [sa[s] - sb[s] for s in segs]
    if len(d) < 2:
        return None
    m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
    sem = sd / math.sqrt(len(d))
    return {"n": len(d), "mean": m, "sd": sd, "sem": sem, "t": m / sem if sem else float("nan"),
            "up": sum(1 for x in d if x > 0), "down": sum(1 for x in d if x < 0)}


def anchor_of(path: str) -> str:
    """Which reference a KL number is against.  The two anchors answer different questions and are not
    interchangeable: fp16 is the honest absolute unit, int8 is K1's anchor (and charges the arm for int8's own
    errors as well).  A KL number without its anchor is not comparable to another one."""
    stem = pathlib.Path(path).stem.lower()
    if "fp16" in stem:
        return "fp16"
    if "int8" in stem:
        return "int8"
    return "other"


def kl_series(ref: dict, run: dict, kl_topk: int = TOPK) -> dict:
    """{(segment, pos): KL(ref || run)} - the per-position series a paired test needs."""
    out = {}
    for name, rr in ref.items():
        ra = run.get(name, {})
        for q in sorted(set(rr) & set(ra)):
            out[(name, q)] = kl_between(rr[q][4], ra[q][4], kl_topk)
    return out


def paired_positions(ka: dict, kb: dict) -> dict:
    """The paired per-position difference A - B, with the one-sided bound at true parity.

    The bound is the number that makes a KL difference readable: with a paired spread `sd` over `n` positions,
    a difference of zero is still measured as up to `1.645 * sd / sqrt(n)` (one-sided 95 %).  Anything below that
    is not a result about the arm, it is a property of the corpus.  On cwe-4k with 32,746 positions and a paired
    spread of 0.22 that is 0.0020 nats, which is why a 0.001 effect is not decidable here."""
    keys = [k for k in ka if k in kb]
    n = len(keys)
    if n < 2:
        return None
    d = [ka[k] - kb[k] for k in keys]
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    sem = sd / math.sqrt(n)
    seg = {}
    for k in keys:
        seg.setdefault(k[0], []).append(ka[k] - kb[k])
    return {"n": n, "mean": m, "sd": sd, "sem": sem, "t": m / sem if sem else float("nan"),
            "ci95_low": m - 1.96 * sem, "ci95_high": m + 1.96 * sem,
            "parity_bound": 1.645 * sem,
            "segments": len(seg),
            "segments_a_worse": sum(1 for s in seg if sum(seg[s]) / len(seg[s]) > 0)}


def answer_kl_by_segment(run: dict, ref: dict, answer_at: dict, kl_topk: int = TOPK) -> dict:
    out = {}
    for name, rr in run.items():
        ra, at = ref.get(name, {}), answer_at.get(name, 0)
        kls = [kl_between(ra[q][4], rr[q][4], kl_topk) for q in sorted(rr) if q + 1 >= at and q in ra]
        if kls:
            out[name] = sum(kls) / len(kls)
    return out


def paired_tables(a, ref: dict, allres: dict) -> dict:
    """The paired whole-segment and answer-region tables.  The answer region needs the set's `answer_at`, which only
    `--ids` can supply; without it the second table is skipped and said so."""
    names = [pathlib.Path(a.ref).name[:40]] + [pathlib.Path(p).name[:40] for p in allres]
    # The reference pairs against itself: KL(ref || ref) is 0 on every segment by construction, and taking the
    # series from an actual `compare` pass keeps the segment list honest rather than assuming it.
    series = {names[0]: {s: v["kl"] for s, v in compare(ref, ref).items() if s != "all"}}
    for n, r in zip(names[1:], allres.values()):
        series[n] = {s: v["kl"] for s, v in r.items() if s != "all"}
    out = {"whole": {}, "answer": None}
    print("\npaired whole-segment KL differences (same segments, so the text cancels):")
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            p = paired_stats(series[na], series[nb])
            if p is None:
                continue
            out["whole"][f"{na} - {nb}"] = p
            print(f"  {na:40s} - {nb:40s} {p['mean']:+.5f}  sd {p['sd']:.5f}  sem {p['sem']:.5f}  "
                  f"t {p['t']:+.2f}  ({p['up']}/{p['n']} segments up)")
    if not a.ids:
        print("  (answer region needs --ids: the set's answer_at tells the reader where the answer starts)")
        return out
    st = json.load(open(a.ids))
    at = {s["name"]: s.get("answer_at", 0) for s in st["segments"]}
    aser = {pathlib.Path(p).name[:40]: answer_kl_by_segment(read_run(p), ref, at, a.kl_topk)
            for p in [a.ref, *a.runs]}
    an = list(aser)
    out["answer"] = {}
    print("\npaired answer-region KL differences:")
    for i, na in enumerate(an):
        for nb in an[i + 1:]:
            p = paired_stats(aser[na], aser[nb])
            if p is None:
                continue
            out["answer"][f"{na} - {nb}"] = p
            print(f"  {na:40s} - {nb:40s} {p['mean']:+.5f}  sd {p['sd']:.5f}  sem {p['sem']:.5f}  "
                  f"t {p['t']:+.2f}  ({p['up']}/{p['n']} segments up)")
    return out


# ------------------------------------------------------------------------------------------------ areport
def answer_region(run: dict, ref: dict, answer_at: dict, kl_topk: int = TOPK) -> dict:
    """The rows whose *target* is an answer token: row `p` predicts token `p + 1`, so the answer starts at row
    `answer_at - 1` and the region is `p + 1 >= answer_at`."""
    tot = {"positions": 0, "hit": 0, "agree": 0, "kl": 0.0}
    for name, rr in run.items():
        ra, at = ref.get(name, {}), answer_at.get(name, 0)
        for q in sorted(rr):
            if q + 1 < at or q not in ra:
                continue
            tr, _lr, topr, hit, kr = rr[q]
            ta, _la, topa, _hita, ka = ra[q]
            if tr != ta:
                raise SystemExit(f"{name} pos {q}: target {tr} != {ta} (not the same text)")
            tot["positions"] += 1
            tot["hit"] += int(hit)
            tot["agree"] += int(topr == topa)
            tot["kl"] += kl_between(kr, ka, kl_topk)
    return tot


def answer_rows(run: dict, ref: dict, at: dict) -> dict:
    """The answer region in per-row form, `{(segment, pos): hit}`, for a paired McNemar test."""
    out = {}
    for name, rr in run.items():
        ra, a = ref.get(name, {}), at.get(name, 0)
        for q in sorted(rr):
            if q + 1 >= a and q in ra:
                out[(name, q)] = bool(rr[q][3])
    return out


def binom_two_sided(b: int, c: int) -> float:
    """Exact two-sided McNemar p: b discordant pairs one way, c the other, under p = 1/2."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def cmd_areport(a) -> int:
    st = json.load(open(a.ids))
    at = {s["name"]: s.get("answer_at", 0) for s in st["segments"]}
    ref = read_run(a.ref)
    print(f"reference: {a.ref}")
    print(f"{'run':34s} {'ans pos':>7s} {'top1 tgt':>8s} {'agree ref':>9s} {'KL ref':>8s}")
    res = {}
    for path in [a.ref, *a.runs]:
        v = answer_region(read_run(path), ref, at)
        res[pathlib.Path(path).name] = v
        n = max(v["positions"], 1)
        print(f"{pathlib.Path(path).name[:34]:34s} {v['positions']:7d} {100 * v['hit'] / n:7.2f}% "
              f"{100 * v['agree'] / n:8.2f}% {v['kl'] / n:8.5f}")
    if a.paired:
        rows = {pathlib.Path(p).name[:34]: answer_rows(read_run(p), ref, at) for p in [a.ref, *a.runs]}
        names = list(rows)
        print("\npaired answer accuracy (exact McNemar over the same rows; b = only the first arm right, "
              "c = only the second):")
        for i, na in enumerate(names):
            for nb in names[i + 1:]:
                keys = sorted(set(rows[na]) & set(rows[nb]))
                b = sum(1 for q in keys if rows[nb][q] and not rows[na][q])
                c = sum(1 for q in keys if rows[na][q] and not rows[nb][q])
                print(f"  {na:34s} vs {nb:34s} b {b:3d} c {c:3d}  p {binom_two_sided(b, c):.3f}")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ids")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--set", default=str(SET_DIR))
    p.add_argument("--seg-tokens", type=int, default=1000)
    p.add_argument("--segments", default=",".join(SEGMENTS), metavar="A,B,C",
                   help=f"comma-separated segments to tokenize (default: {','.join(SEGMENTS)}); a longer-context set "
                        f"may carry only some of them (docs,code,multilingual at 8K)")
    p.add_argument("--out")
    p = sub.add_parser("fwe", help="build the RULER frequent-words-extraction decision-test set")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--set", default=str(HERE.parent / "data" / "quant_quality" / "fwe-2k"))
    p.add_argument("--seq-tokens", type=int, default=2048)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--answer-words", type=int, default=3,
                   help="how many of the most frequent words the question asks for (RULER FWE asks for 3; RULER CWE "
                        "asks for 10, which gives ~3.4x the answer-region rows and smaller gaps between the counts "
                        "of the asked-about ranks at the same context length and runtime)")
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("run")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ids", default=str(SET_DIR / "set.ids.json"))
    p.add_argument("--exe", default=None, help="override the config's engine binary")
    p.add_argument("--env", action="append")
    p.add_argument("--gpu-experts", nargs="?", const=True, default=False,
                   help="--expert-cache 1500 (or the N given, e.g. auto) instead of one slot (the experts on the GPU kernels)")
    p.add_argument("--min-free-gb", type=float, default=16)
    p.add_argument("--wrap", help="a command prefix for the engine, e.g. 'systemd-run --user --scope -p MemoryMax=200G'")
    p.add_argument("--topk", type=int, default=TOPK,
                   help=f"neighbours written per position (engine cap 256); default {TOPK}")
    p.add_argument("--kv", default="fp16", metavar="FMT",
                   help=("the KV cache format for this run: fp16 (default, the reference) | int8 | q4_0 | "
                         "q4lm32 | q4lm32s | q4lm8 | q4lm8k | q4lm8v | q4lm32t | q4lm32nr | q4lm32c | q4lm32w | k8v4"))
    p.add_argument("--short-read", type=int, default=None, metavar="N",
                   help=("override the engine's --short-read: 0 sends every prompt through the batched prompt path "
                         "instead of the verify windows (the S2 channel probe only sees the batched path); "
                         "default = the segment length + 64, i.e. the verify windows"))
    p.add_argument("--allow-no-positions", action="store_true",
                   help=("a probe run (STRATA_KV_S2_STATS): the batched prompt path writes no per-position logits, "
                         "so a segment with fewer than half its positions is not an error and the run continues"))
    p.add_argument("--prefill", default="512", metavar="N|auto",
                   help=("the engine's --prefill chunk. Default 512 keeps every earlier run byte-identical; `auto` "
                         "sizes the chunk from the prompt (a ring sized by STRATA_KV_WINDOW_BATCH then holds the "
                         "whole prompt, which is what a 128K tag arm needs: ring S+B >= --short-read)"))
    p.add_argument("--pin", type=int, default=None, metavar="N",
                   help=("send pin=N in the GEN line: the prompt read is cut at N (a checkpoint edge), so the rest "
                         "goes through the verify windows when it fits --short-read. The only way a tail of a long "
                         "prompt reaches the window path; needs --prompt-cache 3 or more"))
    p.add_argument("--prompt-cache", type=int, default=0, metavar="N",
                   help="the engine's --prompt-cache (default 0, the quality runs' fixed value); --pin needs 3 or more")
    p.add_argument("--max-context", type=int, default=0, metavar="N",
                   help=("raise the engine's --max-context cap above 131,072 for a set longer than 128K (the model's "
                         "own context is 262,144); default 0 keeps every existing run byte-identical"))
    p = sub.add_parser("kvstats", help="per-layer K/V distribution stats for one calibration text "
                                       "(STRATA_KVSTATS tap + aggregation: kurtosis, crest, block-max, channels)")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True, help="the aggregated JSON")
    p.add_argument("--text", default=str(HERE.parent / "docs" / "DETAILS.md"))
    p.add_argument("--tokens", type=int, default=2048)
    p.add_argument("--raw", help="the engine's raw tap file (default: <out>.raw.bin, never committed)")
    p.add_argument("--env", action="append")
    p.add_argument("--gpu-experts", nargs="?", const=True, default=False)
    p.add_argument("--min-free-gb", type=float, default=16)
    p.add_argument("--wrap")
    p.add_argument("--kv", default="q4_0", metavar="FMT",
                   help="the KV path exercised (taps are pre-format; q4_0 rotates so both taps differ)")
    p = sub.add_parser("prompt", help="the batched prompt path's next-token logits after 20 prompts (a directory)")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ids", default=str(SET_DIR / "set.ids.json"))
    p.add_argument("--env", action="append")
    p.add_argument("--lens", help="a comma list of nested prompt lengths (default: 256,512,768,1000); the run records them in index.json")
    p.add_argument("--prefill", type=int, default=None,
                   help="a fixed prompt chunk (default: auto); a ring sized by STRATA_KV_WINDOW_BATCH needs one that fits")
    p.add_argument("--gpu-experts", action="store_true")
    p.add_argument("--min-free-gb", type=float, default=16)
    p.add_argument("--wrap")
    p.add_argument("--topk", type=int, default=TOPK,
                   help=f"neighbours written per position (engine cap 256); default {TOPK}")
    p.add_argument("--kv", default="fp16", metavar="FMT",
                   help=("the KV cache format for this run: fp16 (default, the reference) | int8 | q4_0 | "
                         "q4lm32 | q4lm32s | q4lm8 | q4lm8k | q4lm8v | q4lm32t | q4lm32nr | q4lm32c | q4lm32w | k8v4"))
    p = sub.add_parser("compare")
    p.add_argument("ref")
    p.add_argument("runs", nargs="+")
    p.add_argument("--kl-topk", type=int, default=TOPK,
                   help=f"the KL support: REF's N most likely tokens (default {TOPK}; the runs must carry that many)")
    p.add_argument("--segments", action="store_true")
    p.add_argument("--json")
    p.add_argument("--paired", action="store_true",
                   help=("also print the *paired* per-segment KL difference of every pair of arms (the test that "
                         "can resolve a small format difference, because the text cancels); needs --ids for the "
                         "answer-region table"))
    p.add_argument("--ids", help="the set's ids file, for --paired's answer-region half")
    p.add_argument("--paired-json", help="write the --paired tables as JSON here")
    p = sub.add_parser("areport", help="the FWE answer region: accuracy vs the text, and KL/agreement on the reference")
    p.add_argument("ref")
    p.add_argument("runs", nargs="+")
    p.add_argument("--ids", required=True)
    p.add_argument("--paired", action="store_true",
                   help=("also test the answer *accuracy* pairwise with exact McNemar, which the aggregate hit "
                         "counts cannot: two arms can differ by a few rows while both beating the reference"))
    p.add_argument("--json")
    p = sub.add_parser("pcompare", help="full-vocabulary KL and top-1 of `prompt` directories against a reference one")
    p.add_argument("ref")
    p.add_argument("runs", nargs="+")
    a = ap.parse_args()
    return {"ids": cmd_ids, "fwe": cmd_fwe, "run": cmd_run, "kvstats": cmd_kvstats, "prompt": cmd_prompt,
            "compare": cmd_compare, "areport": cmd_areport, "pcompare": cmd_pcompare}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
