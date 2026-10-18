"""Identity check of engine switches: the greedy token ids of fixed prompts, with the engine started directly.

  <strata src>/.venv/bin/python strata_ident.py ROOT CONFIG OUT.json [K=V ...] [-K ...]
      ROOT    a Strata source tree (serve/ and tools/ are imported from it; its docs/*.md are the long prompts' text)
      CONFIG  a strata-*.json; "exe", "args", "env", "lib_dirs", "tokenizer" are used, "log" is replaced by OUT.log
      K=V     set an engine environment variable for this run; -K removes one the config sets
Ten short prompts (MAX_NEW tokens each), then documents of DOC_SIZES tokens (64 tokens each), greedy, no thinking,
one request at a time, in this order. Saves the ids and the engine's own times. strata-server must be stopped (two
engines do not fit in the GTT pool). Compare two outputs with `strata_ident.py --cmp A.json B.json`.
Env: MAX_NEW (128), DOC_SIZES (4000,20000).
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

PROMPTS = (
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.",
    "Explain in two paragraphs how a refrigerator moves heat from inside to outside.",
    "List twelve European capitals with one sentence about each.",
    "週末に家族で楽しめる、雨の日の過ごし方を3つ提案してください。",
    "Solve step by step: a train leaves at 9:40 and arrives at 13:05. How long is the trip in minutes?",
    "Write a haiku about a lighthouse in winter, then explain its imagery in three sentences.",
    "Translate into English: 明日の会議は午後三時からに変更になりました。資料は前日までに共有してください。",
    "Write a SQL query that returns the five customers with the highest total order value in 2025, with comments.",
    "What are the differences between TCP and UDP? Answer as a short table and one paragraph.",
    "Write a bash script that finds the ten largest files under a directory given as its first argument.",
)


def chat_ids(tok, text):
    return tok.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                      parse_special=True)


def doc_ids(tok, root, n):
    text = "\n\n".join(p.read_text(encoding="utf-8", errors="replace") for p in sorted((root / "docs").glob("*.md")))
    body = tok.encode(text, parse_special=False)
    head = tok.encode("<|im_start|>user\nRead this and summarize its first section in three sentences:\n",
                      parse_special=True)
    tail = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True)
    return head + (body * (n // len(body) + 1))[:n] + tail


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    same = 0
    for name in a["cases"]:
        x, y = a["cases"][name]["ids"], b["cases"].get(name, {}).get("ids")
        if x == y:
            same += 1
            print(f"{name}: SAME ({len(x)} ids)")
        else:
            k = next((i for i, (p, q) in enumerate(zip(x, y or [])) if p != q), min(len(x), len(y or [])))
            print(f"{name}: DIFF at token {k} ({len(x)} vs {len(y or [])} ids)")
    print(f"IDENT {same}/{len(a['cases'])}")
    return 0 if same == len(a["cases"]) else 1


def main(argv):
    if len(argv) == 4 and argv[1] == "--cmp":
        return compare(argv[2], argv[3])
    if len(argv) < 4:
        sys.exit(__doc__)
    root, cfg_path, out = Path(argv[1]).resolve(), argv[2], argv[3]
    sys.path[:0] = [str(root / "tools"), str(root)]
    from serve.server import StrataEngine, child_env, engine_args
    import strata_tokenizer as ST
    cfg = json.load(open(cfg_path))
    env = dict(cfg.get("env") or {})
    for item in argv[4:]:
        if item.startswith("-"):
            env.pop(item[1:], None)
        else:
            k, _, v = item.partition("=")
            env[k] = v
    cfg["env"] = env
    cfg["log"] = out.removesuffix(".json") + ".log"
    tpath = Path(cfg["tokenizer"])
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    tok = ST.Tokenizer(toks, (tpath / "merges.txt").read_text(encoding="utf-8").split("\n"),
                       json.loads((tpath / "token_type.json").read_text()))
    max_new = int(os.environ.get("MAX_NEW", "128"))
    cases = [(f"p{i:02d}", chat_ids(tok, p), max_new) for i, p in enumerate(PROMPTS)]
    cases += [(f"doc{n}", doc_ids(tok, root, n), 64)
              for n in [int(x) for x in os.environ.get("DOC_SIZES", "4000,20000").split(",") if x]]
    print("env:", " ".join(f"{k}={v}" for k, v in sorted(env.items())), flush=True)
    t0 = time.time()
    eng = StrataEngine(cfg["exe"], engine_args(cfg), cwd=cfg.get("cwd"), log=cfg["log"], env=child_env(cfg))
    print(f"engine up in {time.time() - t0:.0f} s", flush=True)
    result = {"env": env, "args": engine_args(cfg), "cases": {}}
    try:
        for name, ids, n in cases:
            got = [t for t in eng.generate(ids, n, {"temperature": 0}, threading.Event()) if t is not None]
            last = dict(eng.last or {})
            result["cases"][name] = {"prompt_tokens": len(ids), "ids": got, "engine": last}
            pms, dms = last.get("prompt_ms") or 0, last.get("decode_ms") or 0
            print(f"{name}: {len(ids)} -> {len(got)} ids, prompt {pms / 1000:.1f} s"
                  f" ({len(ids) / (pms / 1000) if pms else 0:.0f} tok/s), decode {len(got) / (dms / 1000) if dms else 0:.1f} tok/s",
                  flush=True)
    finally:
        proc = getattr(eng, "proc", None)
        if proc is not None:
            try:
                proc.stdin.write("QUIT\n")
                proc.stdin.flush()
                proc.stdin.close()
                proc.wait(60)
            except Exception:
                proc.kill()
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print("IDENT_RUN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
