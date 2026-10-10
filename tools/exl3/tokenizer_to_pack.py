"""Convert a HuggingFace tokenizer directory (as shipped with the EXL3 model) into the pack's
`tokenizer/` layout that tools/strata_tokenizer.py and serve/ read.

    python3 tools/exl3/tokenizer_to_pack.py --hf <hf_model_dir> --out <pack_dir>

The HF model has `tokenizer.json` (base vocab + added tokens), `merges.txt`, `tokenizer_config.json`,
`chat_template.jinja`, and `config.json` (embedding rows).  The pack wants `vocab.json` (id -> token
list), `merges.txt`, `token_type.json` (ggml token types), `tokenizer.json` (config), and the template.
The BPE is identical to the GGUF's (247,587 merges, `qwen35` pretokenizer), so no new C++ tokenizer is
needed; this just reshapes the files.  See docs/EXL3.md and the research note in the PR.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from strata_tokenizer import QWEN35_PATTERN          # noqa: E402

# ggml token types (as the pack's token_type.json uses them)
NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED = 1, 2, 3, 4, 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="directory with tokenizer.json / merges.txt / config.json")
    ap.add_argument("--out", required=True, help="pack directory; tokenizer/ is written under it")
    ap.add_argument("--check", action="store_true", help="round-trip a corpus through the pack tokenizer")
    args = ap.parse_args()

    hf = pathlib.Path(args.hf)
    tj = json.loads((hf / "tokenizer.json").read_text(encoding="utf-8"))
    base = tj["model"]["vocab"]                        # token -> id
    added = tj.get("added_tokens", [])

    cfg = json.loads((hf / "config.json").read_text(encoding="utf-8"))
    vocab_size = (cfg.get("text_config", {}) or cfg).get("vocab_size")
    max_id = max([max(base.values())] + [a["id"] for a in added])
    size = max(vocab_size or 0, max_id + 1)

    tokens = [""] * size
    types = [UNUSED] * size
    for tok, i in base.items():
        tokens[i] = tok
        types[i] = NORMAL
    special_ids: dict[str, int] = {}
    for a in added:
        i, tok, sp = a["id"], a["content"], a.get("special", False)
        tokens[i] = tok
        types[i] = CONTROL if sp else USER_DEFINED
        if sp:
            special_ids[tok] = i

    merges = [ln for ln in (hf / "merges.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]

    # The embedding has `vocab_size` rows but the tokenizer only defines up to id 248076; the tail is
    # untrained padding.  The engine's Tokenizer requires unique tokens, so name them uniquely (never
    # emitted, never matched: type UNUSED, not in the special/always alternations).
    for i in range(size):
        if not tokens[i]:
            tokens[i] = "<|reserved_%d|>" % i

    out = pathlib.Path(args.out) / "tokenizer"
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps(tokens, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps(types), encoding="utf-8")
    pack_cfg = {
        "model": "gpt2",
        "pre": "qwen35",
        "vocab_size": size,
        "n_merges": len(merges),
        "special_ids": special_ids,
        "add_bos_token": False,
        "pre_pattern": QWEN35_PATTERN,
        "pre_pattern_source": "HF tokenizer.json (same qwen35 pattern as the GGUF)",
    }
    (out / "tokenizer.json").write_text(json.dumps(pack_cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    tpl = hf / "chat_template.jinja"
    if tpl.exists():
        (out / "chat_template.jinja").write_text(tpl.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")

    print("wrote %s: vocab %d (filled %d), merges %d, added %d, specials %d"
          % (out, size, sum(1 for t in tokens if t), len(merges), len(added), len(special_ids)))

    if args.check:
        from strata_tokenizer import Tokenizer
        tk = Tokenizer(tokens, merges, types, "qwen35", special_ids)
        corpus = ["", "Hello, world!", "  leading and trailing  ", "a\n\n\nb",
                  "def f(x):\n\treturn x  # comment\n", "\u4f60\u597d\uff0c\u4e16\u754c", "\u0645\u0631\u062d\u0628\u0627",
                  "\U0001f600\U0001f680\U0001f1fa\U0001f1f8", "e\u0301\u0301 combining", "\x00\x01\x7f control",
                  "\u00a0non-breaking\u00a0space", "1234567890", "MixedCASE_and-dashes", "\r\n\r\n", "\u2028\u2029",
                  "x" * 5000]
        bad = 0
        for s in corpus:
            if tk.decode(tk.encode(s)) != s:
                bad += 1
                print("  ROUND-TRIP FAIL: %r" % s[:60])
        print("round-trip: %d/%d ok" % (len(corpus) - bad, len(corpus)))
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
