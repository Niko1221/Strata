#!/usr/bin/env python3
"""Verify third_party/strata-ggml against its manifest (UPSTREAM.md).   python tools/check_vendored.py [--upstream <llama.cpp dir>]

The vendored ggml / ggml-cuda sources are frozen at the llama.cpp commit named in UPSTREAM.md.  The manifest lists every file with its
kind and sha256 (text files are hashed with LF line ends, so a Windows checkout agrees with a Linux one):

  upstream   identical to the file at the pinned commit
  edited     a Strata edit of an upstream file (the edit is described in UPSTREAM.md); the row also holds the upstream sha256
  added      a file that is Strata's own (not in llama.cpp)

Checks: every listed file exists and has the listed hash; no file of the tree is missing from the list; the pin equals
third_party/ggml/VERSION.txt; the LICENSE is there; every `edited` row names its upstream hash.  With --upstream <dir> the upstream
hashes are also checked against that checkout (the files at the pin, e.g. `git archive` of the commit): `upstream` rows must equal the file
there, `edited` rows must carry its hash.  Standard library only.  Exit 0 clean, 1 on any problem.
"""
import argparse
import hashlib
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TREE = os.path.join("third_party", "strata-ggml")
BEGIN, END = "<!-- vendored-files:begin -->", "<!-- vendored-files:end -->"
NOT_LISTED = {"UPSTREAM.md"}          # the manifest itself
TEXT_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".txt", ".cmake", ".in", ".md", ".inc"}


def sha(path):
    with open(path, "rb") as f:
        data = f.read()
    if os.path.splitext(path)[1].lower() in TEXT_SUFFIXES or os.path.basename(path) == "LICENSE":
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def parse(manifest):
    text = open(manifest, encoding="utf-8").read()
    if BEGIN not in text or END not in text:
        raise SystemExit("UPSTREAM.md: the %s block is missing" % BEGIN)
    block = text.split(BEGIN, 1)[1].split(END, 1)[0]
    rows = []
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("```") or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise SystemExit("UPSTREAM.md: bad row: %r" % line)
        kind, h, up, rel = parts[0], parts[1], parts[2], " ".join(parts[3:])
        rows.append((kind, h, up, rel))
    m = re.search(r"^pin:\s*([0-9a-f]{40})\s*$", text, re.M)
    return rows, (m.group(1) if m else None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", help="a llama.cpp tree at the pinned commit (file paths as in the manifest)")
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args(argv)
    tree = os.path.join(a.root, TREE)
    problems = []
    rows, pin = parse(os.path.join(tree, "UPSTREAM.md"))
    ver = os.path.join(a.root, "third_party", "ggml", "VERSION.txt")
    if os.path.exists(ver):
        vpin = open(ver, encoding="utf-8").read().split()[0]
        if vpin != pin:
            problems.append("pin %s in UPSTREAM.md differs from third_party/ggml/VERSION.txt (%s)" % (pin, vpin))
    if not os.path.exists(os.path.join(tree, "LICENSE")):
        problems.append("LICENSE is missing")
    listed = set()
    for kind, h, up, rel in rows:
        if kind not in ("upstream", "edited", "added"):
            problems.append("%s: unknown kind %r" % (rel, kind))
            continue
        listed.add(rel)
        p = os.path.join(tree, rel)
        if not os.path.isfile(p):
            problems.append("%s: listed but missing" % rel)
            continue
        if sha(p) != h:
            problems.append("%s: sha256 differs from the manifest (%s)" % (rel, kind))
        if kind == "upstream" and up != h:
            problems.append("%s: an `upstream` row must carry its own hash as the upstream hash" % rel)
        if kind == "edited" and (up == h or not re.fullmatch(r"[0-9a-f]{64}", up)):
            problems.append("%s: an `edited` row needs the upstream hash, different from the file's" % rel)
        if kind == "added" and up != "-":
            problems.append("%s: an `added` row has `-` as the upstream hash" % rel)
        if a.upstream and kind in ("upstream", "edited"):
            q = os.path.join(a.upstream, rel)
            if not os.path.isfile(q):
                problems.append("%s: not in the upstream tree %s" % (rel, a.upstream))
            elif sha(q) != up:
                problems.append("%s: upstream hash in the manifest differs from %s" % (rel, q))
    for d, _dirs, files in os.walk(tree):
        for f in files:
            rel = os.path.relpath(os.path.join(d, f), tree).replace(os.sep, "/")
            if rel in NOT_LISTED or rel == "LICENSE":
                continue
            if rel not in listed:
                problems.append("%s: in the tree but not in UPSTREAM.md" % rel)
    for p in problems:
        print("check_vendored: " + p)
    if problems:
        print("check_vendored: %d problem(s)" % len(problems))
        return 1
    n = {k: sum(1 for r in rows if r[0] == k) for k in ("upstream", "edited", "added")}
    print("check_vendored: ok - %d files (%d upstream, %d edited, %d added), pin %s" % (len(rows), n["upstream"], n["edited"], n["added"], (pin or "?")[:9]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
