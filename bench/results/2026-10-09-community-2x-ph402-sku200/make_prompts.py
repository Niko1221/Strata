"""Rebuild the long prompt text used by every long-context request in this report.

The text is three engine source files of Strata v0.1.40.3 (commit d5ea7133), each after a header line, with
newlines normalised to LF:

    ===== FILE: src/prefill/prefill.cpp =====      (whole file)
    ===== FILE: src/core/verify.cpp =====          (whole file)
    ===== FILE: src/program/generate.cpp =====     (its first 647,925 characters)

Every file's text is followed by one "\n". The result is 1,059,516 characters, 301,005 tokens by
tools/strata_tokenizer.py with the Swift 1.5 pack's tokenizer. The requests use slices of it:

    S1K  = text[:3500]           (the 1,158-token prompts)
    S30K = text[:105000]         (the 34,835-token prompts)
    MORE = text[105000:108500]   (the 1,027 tokens appended to a 34,830-token conversation)
    B    = text                  (the 301,108-token prompt)

usage: python make_prompts.py <path to a Strata git checkout that has d5ea7133> [--out long300k.txt]
"""
import argparse
import hashlib
import subprocess
import sys

COMMIT = "d5ea7133"
FILES = [("src/prefill/prefill.cpp", None), ("src/core/verify.cpp", None), ("src/program/generate.cpp", 647925)]
SHA256 = {
    "text": "bc74d8588547a5f673f1d0589615bf73bad42b9ed611ba798e3ea093c61465d5",
    "S1K": "9cb70f9940b669414dc347ec2b66b615db936d9e4a1fc79a9d78672dbd97aff4",
    "S30K": "a7f2753b06ea8229117630e30a92715582a5696e5df1f39312b2f8c6ccfe8a00",
    "MORE": "7787fb136d7e8778ab05664cf9d708b18930c249d8afd59c3dcc2ba0c7c0de24",
}


def build_text(repo):
    out = []
    for path, limit in FILES:
        raw = subprocess.run(["git", "-C", repo, "show", f"{COMMIT}:{path}"], capture_output=True, check=True).stdout
        body = raw.decode("utf-8").replace("\r\n", "\n")
        out.append(f"===== FILE: {path} =====\n" + (body if limit is None else body[:limit]) + "\n")
    return "".join(out)


def slices(text):
    return {"text": text, "S1K": text[:3500], "S30K": text[:105000], "MORE": text[105000:108500]}


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--out", default=None, help="also write the text (UTF-8, LF) to this file")
    a = ap.parse_args()
    text = build_text(a.repo)
    ok = True
    for name, s in slices(text).items():
        h = sha(s)
        good = h == SHA256[name]
        ok &= good
        print(f"{name:5} {len(s):9,} chars  sha256 {h}  {'ok' if good else 'MISMATCH'}")
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
