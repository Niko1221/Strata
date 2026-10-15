#!/usr/bin/env python3
"""Needle runs (item 5): a codeword hidden in natural text, retrieved after a long prompt.

The filler is the committed quant-quality set's own segments (data/quant_quality/v1/set.ids.json: code,
docs, english, multilingual, agent - 1,000 ids each, cycled), so no new corpus lands in the repo and the
prompt is byte-reproducible from committed files.  The needle sentence and the question are tokenized with
tools/strata_tokenizer.py from the pack's tokenizer directory (the engine takes ids, not text).

Cases: {8K, 16K, 32K} x {10%, 50%, 90%} depth under q4_0 (9 runs) + int8 at 50% depth per length (3
controls).  Length = the id budget 8192/16384/32768 less 256 for the question and margin: prompt ids +
--max-new 64 must fit the template's --max-context 32768 (the first pass built full-budget prompts and the
engine refused the 32K trio: "positive --max-new and --max-context must fit the prompt and generation").
Depth is the needle's position as a fraction of the filler span.

One `strata generate` load per case (the /tmp/m1/g.sh argument template, --tokens-file instead of --tokens
so a 32K prompt never rides argv), greedy decode, --max-new 64.  Judged by: the decoded reply, with
whitespace removed and uppercased, contains the codeword.  Writes needle-results.tsv (case, prompt ids,
needle position, found, reply) and the per-case engine log.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys

WT = pathlib.Path("/Users/berton/Github/AI-Infra/Strata-metal-kv-q4")       # the worktree
HERE = WT / "bench" / "results" / "2026-10-15-mac-kv-q4"
TOKENIZER_DIR = pathlib.Path("/Users/berton/Github/AI-Infra/Strata-data/packs/q2_0/tokenizer")
CODEWORD = "MARBLE-7742-KUMQUAT"
NEEDLE_TEXT = f"The access code for the blue warehouse is {CODEWORD}. Keep it secret and do not write it down.\n\n"
QUESTION_TEXT = "\n\nAccording to the text above, what is the access code for the blue warehouse? The access code is"

ENGINE_ARGS = ["./build-metal/strata",
               "--pack", "/Users/berton/Github/AI-Infra/Strata-data/packs/q2_0",
               "--native", "/Users/berton/Github/AI-Infra/Strata-data/models/Q2_0/"
                           "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
               "--ple-gguf", "/Users/berton/Github/AI-Infra/Strata-data/models/Q2_0/"
                             "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf",
               "--expert-profile", "data/expert-profile.bin",
               "--expert-cache", "64", "--prefill", "auto", "--spec", "2", "--suffix-draft", "0",
               "--max-context", "32768", "--mmap-experts",
               "--tokens-file", "", "--max-new", "64"]


def build_tokenizer():
    sys.path.insert(0, str(WT / "tools"))
    import strata_tokenizer as ST
    vocab = json.loads((TOKENIZER_DIR / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    return ST.Tokenizer(toks, (TOKENIZER_DIR / "merges.txt").read_text(encoding="utf-8").split("\n"),
                        json.loads((TOKENIZER_DIR / "token_type.json").read_text()))


def build_prompt_ids(tok, total: int, depth: float) -> tuple[list[int], int]:
    """Filler (the set's 5 segments, cycled) with the needle at `depth` of the filler span, question last."""
    st = json.loads((WT / "data" / "quant_quality" / "v1" / "set.ids.json").read_text())
    pool = [i for s in st["segments"] for i in s["ids"]]
    needle = tok.encode(NEEDLE_TEXT)
    question = tok.encode(QUESTION_TEXT)
    filler_len = total - len(question) - len(needle)
    filler = []
    while len(filler) < filler_len:
        take = min(filler_len - len(filler), len(pool))
        filler += pool[:take]
    at = int(depth * (len(filler) - 1))
    return filler[:at] + needle + filler[at:] + question, at


def run_case(tok, tag: str, kv: str, total: int, depth: float) -> dict:
    ids_file = HERE / f"needle-prompt-{tag}.ids"
    prompt, at = build_prompt_ids(tok, total, depth)
    ids_file.write_text(",".join(map(str, prompt)))
    args = list(ENGINE_ARGS)
    args[args.index("--tokens-file") + 1] = str(ids_file)
    i = args.index("--mmap-experts") + 1
    args[i:i] = ["--kv", kv]                                # the g.sh template's --kv slot
    log = HERE / "logs" / f"needle-{tag}-{kv}.log"
    env = {k: os.environ[k] for k in ("HOME", "TMPDIR", "PATH") if k in os.environ}
    env.update({"STRATA_METAL_STREAM": "1", "STRATA_METAL_STUBS": "log"})
    with open(log, "w") as f:
        rc = subprocess.call(args, cwd=WT, stdout=f, stderr=subprocess.STDOUT, env=env)
    text = log.read_text(errors="replace")
    out = re.search(r"^output\s+:\s+(.*)$", text, re.M)
    reply = tok.decode([int(x) for x in out.group(1).split()]) if out else ""
    found = CODEWORD in re.sub(r"\s+", "", reply).upper()
    row = {"tag": tag, "kv": kv, "prompt_ids": len(prompt), "needle_at": at, "depth": depth,
           "rc": rc, "found": found, "reply": " ".join(reply.split())[:300]}
    print(f"{tag:16s} kv={kv:5s} ids={len(prompt):6d} needle@{at:6d} ({depth:.0%})  "
          f"rc={rc}  found={found}  reply={row['reply'][:80]!r}", flush=True)
    ids_file.unlink()                                       # reproducible from this script + the committed set
    return row


def main() -> int:
    tok = build_tokenizer()
    assert tok.decode(tok.encode(NEEDLE_TEXT + QUESTION_TEXT)) == NEEDLE_TEXT + QUESTION_TEXT
    (HERE / "logs").mkdir(parents=True, exist_ok=True)
    rows = [run_case(tok, f"{n}-{d}", "q4_0", n - 256, d / 100)
            for n in (8192, 16384, 32768) for d in (10, 50, 90)]
    rows += [run_case(tok, f"{n}-50", "int8", n - 256, 0.5) for n in (8192, 16384, 32768)]
    with open(HERE / "needle-results.tsv", "w") as f:
        f.write("tag\tkv\tprompt_ids\tneedle_at\tdepth\trc\tfound\treply\n")
        for r in rows:
            f.write(f"{r['tag']}\t{r['kv']}\t{r['prompt_ids']}\t{r['needle_at']}\t{r['depth']}\t"
                    f"{r['rc']}\t{r['found']}\t{r['reply']}\n")
    got = sum(r["found"] for r in rows if r["kv"] == "q4_0")
    print(f"q4_0: {got}/9 found; int8: {sum(r['found'] for r in rows if r['kv'] == 'int8')}/3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
