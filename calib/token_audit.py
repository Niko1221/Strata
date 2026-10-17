#!/usr/bin/env python3
"""calib/token_audit.py — A1.3 token audit for the QFN tokenizer (donor parity).

Checks the labels the decision readout relies on: every render label (letters
A-Z for choice, Yes/No for noul, digits 0-9 for score) must encode to exactly
ONE token, all label token ids must be distinct (collision-free), and no label
may be a strict prefix of another label's token sequence (prefix-free), or the
one-decode-step readout silently reads the wrong row.

Writes reports/token_audit_qfn.json (donor schema: labels, collisions,
multi_token_labels, prefix_free, verdict).

usage: token_audit.py <serve_tree_dir>   # runs where the tokenizer lives
"""
import json
import sys
from pathlib import Path

LETTERS = [chr(ord("A") + i) for i in range(26)]
NUOUL = ["No", "Yes"]
DIGITS = [str(i) for i in range(10)]


def main():
    tree = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    tok_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    sys.path.insert(0, str(tree))
    # load the REAL pack tokenizer exactly as serve/server.py main() does
    tok = None
    if tok_path and (tok_path / "vocab.json").exists():
        import strata_tokenizer as ST
        vocab = json.loads((tok_path / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for t, i in vocab.items():
            tokens[i] = t
        merges = (tok_path / "merges.txt").read_text(encoding="utf-8").split("\n")
        types = json.loads((tok_path / "token_type.json").read_text())
        tok = ST.Tokenizer(tokens, merges, types)
        tname = f"pack:{tok_path}"
    if tok is None:
        from serve.server import ByteTokenizer   # test stand-in (bytes) - FAILs multi-token by design
        tok = ByteTokenizer()
        tname = "ByteTokenizer(test-stand-in)"

    groups = {"choice_letters": LETTERS, "noul": NUOUL, "score_digits": DIGITS}
    labels, collisions, multi = {}, [], []
    seen = {}
    for g, names in groups.items():
        for nm in names:
            ids = tok.encode(nm, parse_special=True)
            labels[nm] = {"group": g, "token_ids": ids}
            if len(ids) != 1:
                multi.append({"label": nm, "token_ids": ids})
            if len(ids) == 1:
                if ids[0] in seen:
                    collisions.append({"a": seen[ids[0]], "b": nm, "id": ids[0]})
                seen[ids[0]] = nm
    prefix_free = not multi  # single-token labels cannot be prefixes of each other
    verdict = "PASS" if not collisions and not multi else "FAIL"
    out = {"tokenizer": tname, "labels": labels,
           "collisions": collisions, "multi_token_labels": multi,
           "prefix_free": prefix_free, "verdict": verdict,
           "notes": "letters A-Z + Yes/No + digits must be single-token and "
                    "collision-free (donor A1.3; PASS/FAIL verdict)"}
    p = tree / "reports"
    p.mkdir(exist_ok=True)
    (p / "token_audit_qfn.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("collisions", "multi_token_labels",
                                          "prefix_free", "verdict")}))
    print("WROTE", p / "token_audit_qfn.json")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
