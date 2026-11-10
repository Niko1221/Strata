#!/usr/bin/env python3
"""Pin the ISTA-DAS Coder set to a Hugging Face repository: the ONE place the repo id lives is data/dsv41/coder-set.json.

    python tools/dsv41/pin_coder.py helcig/tmp-oct26-4                # read the repo now, write data/dsv41/coder-set.json
    python tools/dsv41/pin_coder.py NEW/REPO --revision <40 hex>      # a commit other than the repository's current one
    python tools/dsv41/pin_coder.py --check                           # read the JSON, compare it with the Hub, change nothing

What it does (only range requests for the GGUF headers, never a model download):
  1. asks the Hub for the repository's current commit and for every file's size and LFS sha256 (public repositories only);
  2. reads the header of each .gguf: the shard without tensors is the metadata shard, the shard with the single per_layer_token_embd tensor is the
     Engram table, the others are the model; the mmproj is the file whose clip.projector_type is deepseek41v;
  3. takes the routed expert count from the metadata and checks that every layer's expert tensors and router agree with it
     (a set whose layers have different counts, or whose router and experts disagree, is refused: the engine reads one number a model);
  4. writes data/dsv41/coder-set.json.  setup.py, strata_control/dsv41_fit.py and the app read only that file, so a new URL is this one command.

Everything the repository serves is data: names are checked against a plain pattern, sizes are capped, nothing is executed or written outside
the JSON.  The repository's own license text is not read here; the metadata's general.license is copied into the JSON so a person can see it.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "dsv41" / "coder-set.json"
HUB = "https://huggingface.co"
NAME_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.gguf$")
REPO_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
FOLDER_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,60}$")
MAX_HEADER = 64 << 20          # a GGUF header (metadata + tensor table) larger than this is not read


class Remote:
    """A growing prefix of a remote file read with Range requests."""

    def __init__(self, url: str, chunk: int = 1 << 21):
        self.url, self.chunk, self.buf, self.pos = url, chunk, b"", 0
        self.total = None
        self._more(chunk)

    def _more(self, need: int) -> None:
        start = len(self.buf)
        end = start + max(self.chunk, need) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "strata-pin"})
        with urllib.request.urlopen(req, timeout=120) as r:
            self.buf += r.read()
            cr = r.headers.get("Content-Range")
            if cr and "/" in cr:
                self.total = int(cr.rsplit("/", 1)[1])

    def read(self, n: int) -> bytes:
        if self.pos + n > MAX_HEADER:
            raise ValueError("header too large")
        while self.pos + n > len(self.buf):
            self._more(self.pos + n - len(self.buf))
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b


_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def _u(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]


def _s(f) -> str:
    n = _u(f, "Q")
    if n > 1 << 24:
        raise ValueError("string too long")
    return f.read(n).decode("utf-8", "replace")


def _val(f, t):
    if t == 8:
        return _s(f)
    if t == 9:
        et, n = _u(f, "I"), _u(f, "Q")
        if n > 1 << 22:
            raise ValueError("array too long")
        if et == 8:
            return [_s(f) for _ in range(n)][:4]
        if et == 9:
            raise ValueError("nested arrays")
        sz = struct.calcsize(_SCALARS[et])
        raw = f.read(sz * n)
        return list(struct.unpack("<%d%s" % (min(n, 8), _SCALARS[et]), raw[:sz * min(n, 8)]))
    return _u(f, _SCALARS[t])


def read_header(url: str) -> dict:
    f = Remote(url)
    if f.read(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    version, n_tensors, n_kv = _u(f, "I"), _u(f, "Q"), _u(f, "Q")
    if version not in (2, 3) or n_tensors > 100000 or n_kv > 100000:
        raise ValueError("unexpected GGUF header")
    kv = {}
    for _ in range(n_kv):
        k = _s(f)
        kv[k] = _val(f, _u(f, "I"))
    tensors = {}
    for _ in range(n_tensors):
        name = _s(f)
        dims = [_u(f, "Q") for _ in range(_u(f, "I"))]
        tensors[name] = dims
        _u(f, "I"), _u(f, "Q")                                    # type, offset
    return {"kv": kv, "tensors": tensors, "total": f.total}


def get_json(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "strata-pin"}), timeout=60) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def build(repo: str, revision: str | None) -> dict:
    if not REPO_OK.match(repo):
        raise SystemExit(f"not a repository id: {repo!r}")
    info = get_json(f"{HUB}/api/models/{repo}" + (f"/revision/{revision}" if revision else "") + "?blobs=true")
    if info.get("private") or info.get("gated"):
        raise SystemExit("the repository is private or gated: setup downloads without a login, so it cannot serve it")
    rev = info["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", rev):
        raise SystemExit("the Hub gave no commit id")
    files, paths = {}, {}
    for s in info["siblings"]:
        path = s["rfilename"]
        folder, _, name = path.rpartition("/")
        # the repository root, or ONE folder level down (the official Coder repository keeps its model shards in IQ2_XXS/ and the Engram and the picture file at the root)
        if name.endswith(".gguf") and folder.count("/") == 0 and (not folder or FOLDER_OK.match(folder)):
            if not NAME_OK.match(name) or not s.get("size") or not (s.get("lfs") or {}).get("sha256"):
                raise SystemExit(f"{path}: no size or LFS hash from the Hub")
            if name in files:
                raise SystemExit(f"{name} is in two folders of the repository")
            files[name] = [int(s["size"]), s["lfs"]["sha256"].lower()]
            paths[name] = path
    if not files:
        raise SystemExit("no .gguf files in the repository")
    base = f"{HUB}/{repo}/resolve/{rev}/"
    heads = {n: read_header(base + urllib.parse.quote(paths[n])) for n in sorted(files)}
    mmproj = [n for n, h in heads.items() if h["kv"].get("clip.projector_type") == "deepseek41v"]
    models = {n: h for n, h in heads.items() if n not in mmproj}      # (the later shards of a split carry only split.* keys, no architecture)
    meta = [n for n, h in models.items() if not h["tensors"]]
    engram = [n for n, h in models.items() if list(h["tensors"]) == ["per_layer_token_embd.weight"]]
    body = sorted(n for n, h in models.items() if n not in engram and h["tensors"])
    if len(mmproj) != 1 or len(meta) != 1 or len(engram) != 1 or not body:
        raise SystemExit(f"expected 1 metadata shard, 1 Engram shard, 1 picture file and model shards; got meta={meta} engram={engram} "
                         f"mmproj={mmproj} model={body}")
    kv = models[meta[0]]["kv"]
    if kv.get("general.architecture") != "deepseek41":
        raise SystemExit(f"{meta[0]}: architecture {kv.get('general.architecture')!r}, not deepseek41")
    n_exp = int(kv["deepseek41.expert_count"])
    layers = {}
    for n in body:
        for t, dims in models[n]["tensors"].items():
            m = re.match(r"blk\.(\d+)\.(ffn_(?:gate|up|down)_exps\.weight|ffn_gate_inp\.weight|exp_probs_b\.bias)$", t)
            if m:
                got = dims[2] if m.group(2).endswith("exps.weight") else dims[-1]
                layers.setdefault(int(m.group(1)), set()).add(got)
    bad = {i: v for i, v in layers.items() if v != {n_exp}}
    if not layers or bad:
        raise SystemExit(f"expert counts disagree with the metadata's {n_exp}: {dict(list(bad.items())[:4])}")
    order = [meta[0]] + body
    folders = {paths[n].rpartition("/")[0] for n in order}
    if len(folders) != 1:
        raise SystemExit(f"the model shards are in different folders: {sorted(folders)}")
    return {
        "name": "ISTA-DAS-Coder",
        "label": "ISTA-DAS Coder",
        "repo": repo,
        "revision": rev,
        "pinned_from": info.get("lastModified", ""),
        "license_in_metadata": str(kv.get("general.license", "")),
        "experts_per_layer": n_exp,
        "layers": len(layers),
        "experts_used": int(kv["deepseek41.expert_used_count"]),
        "shards": order,
        "subdir": folders.pop(),                      # the folder of the model shards ("" = the repository root); setup downloads them from there
        "engram_shard": engram[0],
        "engram_path": paths[engram[0]],              # where the Engram file is in the repository (the root: setup takes the Engram as the shared file, once for every set)
        "mmproj": mmproj[0],
        "mmproj_path": paths[mmproj[0]],
        "profile": "profile-coder.bin",
        "files": {n: files[n] for n in order + engram + mmproj},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo", nargs="?", help="the Hugging Face repository id, owner/name")
    ap.add_argument("--revision", help="a commit (40 hex) instead of the current one")
    ap.add_argument("--check", action="store_true", help="compare data/dsv41/coder-set.json with the Hub and change nothing")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    cur = json.loads(Path(a.out).read_text(encoding="utf-8")) if Path(a.out).is_file() else {}
    if a.check or not a.repo:
        repo = cur.get("repo")
        if not repo:
            raise SystemExit("no repository pinned yet")
        new = build(repo, cur.get("revision"))
        keep = {k: v for k, v in cur.items() if k in new}
        same = {k: (new[k] == keep.get(k)) for k in ("revision", "experts_per_layer", "shards", "engram_shard", "mmproj", "files")}
        print(json.dumps(same))
        return 0 if all(same.values()) else 1
    new = build(a.repo, a.revision)
    Path(a.out).write_text(json.dumps(new, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"{a.repo} @ {new['revision'][:8]}: {new['experts_per_layer']} experts a layer, {len(new['shards'])} model files + Engram "
          f"{new['engram_shard']} + {new['mmproj']}; wrote {a.out}")
    for n, (size, sha) in new["files"].items():
        print(f"  {n}  {size}  {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
