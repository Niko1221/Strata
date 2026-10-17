#!/usr/bin/env python3
"""verify_weights.py - inventory model weight files on disk and check them against Hugging Face.

For every *.gguf / *.safetensors under the given folders:
  - size on disk, and the Hugging Face repo file with the same name (searched in --repo and the known repos below)
  - size match against that file, and with --sha the SHA-256 against Hugging Face's LFS hash (reads every byte)
  - per folder, Strata's tools/quantscope.py summary of the real storage types and bits per weight (headers only)

    python3 verify_weights.py ~/models /srv                       # sizes + types, a few seconds
    nohup python3 verify_weights.py ~/models /srv --sha > ~/weights.out 2>&1 &   # + full hashes, minutes per 100 GB

Writes ~/weights-inventory.json and prints a table. Stdlib only. Hugging Face is queried anonymously (set HF_TOKEN
for gated repos).
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

KNOWN_REPOS = [
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF",
    "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
    "unsloth/Qwen3.8-Flash-Next-GGUF",
    "nvidia/Qwen3.8-Flash-Next-NVFP4",
]
EXT = (".gguf", ".safetensors")


def hf_tree(repo, rev="main"):
    """All files of a repo: {basename: [entry,...]} with path, size, sha256 (LFS oid)."""
    out, url = {}, f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=1&expand=false"
    headers = {"User-Agent": "verify_weights/1"}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    while url:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r:
            items = json.load(r)
            link = r.headers.get("Link") or ""
        for it in items:
            if it.get("type") != "file":
                continue
            lfs = it.get("lfs") or {}
            out.setdefault(os.path.basename(it["path"]), []).append(
                {"repo": repo, "rev": rev, "path": it["path"], "size": lfs.get("size", it.get("size")),
                 "sha256": lfs.get("oid")})
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            h.update(b)
    return path, h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--repo", action="append", default=[], help="extra Hugging Face repo to match against")
    ap.add_argument("--sha", action="store_true", help="hash every file and compare with Hugging Face (slow)")
    ap.add_argument("--jobs", type=int, default=4, help="parallel hash workers")
    ap.add_argument("--strata", type=Path, default=Path.home() / "src/Strata", help="Strata checkout (for quantscope)")
    ap.add_argument("--out", type=Path, default=Path.home() / "weights-inventory.json")
    a = ap.parse_args()

    files = sorted({p.resolve() for r in a.roots if r.exists() for p in r.rglob("*")
                    if p.is_file() and p.suffix in EXT and not p.is_symlink()})
    if not files:
        sys.exit("no .gguf or .safetensors files under " + ", ".join(map(str, a.roots)))
    print(f"{len(files)} weight files, {sum(p.stat().st_size for p in files) / 1e9:,.1f} GB", flush=True)

    index, errors = {}, []
    for repo in KNOWN_REPOS + a.repo:
        try:
            for k, v in hf_tree(repo).items():
                index.setdefault(k, []).extend(v)
        except Exception as e:  # offline, gated, renamed: report and continue
            errors.append(f"{repo}: {e}")
    for e in errors:
        print("HF lookup failed:", e, flush=True)

    rows = []
    for p in files:
        size = p.stat().st_size
        cands = index.get(p.name, [])
        same = [c for c in cands if c["size"] == size]
        # a same-name file of a different size is usually another model's file (mmproj-F16.gguf exists in many
        # repos); report it as a name-only match instead of a size failure
        m = same[0] if same else None
        rows.append({"path": str(p), "size": size, "mtime": time.strftime("%Y-%m-%d", time.localtime(p.stat().st_mtime)),
                     "hf": m, "hf_candidates": len(cands), "name_only": [c["repo"] + ":" + c["path"] for c in cands] if not same else [],
                     "size_ok": None if not m else m["size"] == size, "sha256": None, "sha_ok": None})

    if a.sha:
        todo = [r["path"] for r in rows if r["hf"] and r["hf"].get("sha256")]
        print(f"hashing {len(todo)} files with {a.jobs} workers ...", flush=True)
        by = {r["path"]: r for r in rows}
        t0 = time.time()
        with ProcessPoolExecutor(a.jobs) as ex:
            for path, digest in ex.map(sha256, todo):
                r = by[path]
                r["sha256"], r["sha_ok"] = digest, digest == r["hf"]["sha256"]
                print(f"  {'OK  ' if r['sha_ok'] else 'FAIL'} {path} ({time.time() - t0:.0f} s)", flush=True)

    scope = {}
    qs = a.strata / "tools" / "quantscope.py"
    py = a.strata / ".venv" / "bin" / "python"
    for d in sorted({str(Path(r["path"]).parent) for r in rows if r["path"].endswith(".gguf")}):
        if qs.exists():
            try:
                p = subprocess.run([str(py if py.exists() else sys.executable), str(qs), d],
                                   capture_output=True, text=True, timeout=300)
                scope[d] = p.stdout if p.stdout.strip() else "quantscope: " + (p.stderr.strip()[-400:] or "no output")
            except Exception as e:
                scope[d] = f"quantscope failed: {e}"

    print()
    print(f"{'GB':>7}  {'size':5} {'sha':5}  {'Hugging Face file':58}  local path")
    for r in rows:
        hf = (f"{r['hf']['repo'].split('/')[0]}:{r['hf']['path']}" if r["hf"] else
              "(name matches a known repo, size differs)" if r["name_only"] else "(no match: local or derived file)")
        flag = lambda x: "-" if x is None else ("ok" if x else "DIFF")
        print(f"{r['size'] / 1e9:7.2f}  {flag(r['size_ok']):5} {flag(r['sha_ok']):5}  {hf[:58]:58}  {r['path']}")
    for d, s in scope.items():
        print(f"\n=== quantscope {d}\n{s.strip()}")
    a.out.write_text(json.dumps({"files": rows, "quantscope": scope, "hf_errors": errors}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
