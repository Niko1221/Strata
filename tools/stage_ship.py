"""Ship a stage worker's share of a model to a node (tools/stage_node.py): only the bytes its layers need.

A node that runs layers [lo, hi) reads only those layers' tensors from the model's GGUF shards (the expert arena
loads its own layers, the dense weights come from the pack).  This sends, from the PC that holds the whole model:
  - every shard's header and the tensors of layers lo..hi-1, at their own offsets in files of the shards' sizes
    (sparse on the node: the other layers are holes), plus the small non-layer tensors (token embedding, output);
    the per-layer token embeddings (the PLE, ~27 GB) only to a node that runs layer 1 - a worker never does;
  - the pack directory, the expert profile and (optional) the engine's directory, whole.
What the node has already (its /ranges) is not sent again, and every piece's sha256 is compared on arrival.

    python3 tools/stage_ship.py --agent http://192.0.2.12:7840 --token-file /opt/strata/stage-token \\
        --model-dir /data/models/<model> --pack-dir /data/packs/<pack> \\
        --profile /opt/strata/data/expert-profile.bin [--bin-dir /opt/strata/build/bin] \\
        --layers 36-47 [--data-dir <the node's data_dir>]

On the node the files land under its data_dir: models/<model dir name>/, packs/<pack name>/, data/<profile>,
bin/.  A /start that names the model ({"model": {"dir": ..., "native": ..., "pack": ...}}, as stage_tune.py sends)
runs it from there; --data-dir prints the paths for a node.json that should run it by default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_reader import GGUFFile  # noqa: E402

PIECE = 64 << 20


class Item:
    """One file to ship: the byte ranges the node needs of it (None: the whole file, sent again when it changed)."""

    def __init__(self, local: Path, rel: str, ranges: list[list[int]] | None = None):
        self.local = local
        self.rel = rel
        self.total = local.stat().st_size
        self.whole = ranges is None
        self.ranges = merge(ranges if ranges is not None else [[0, self.total]])
        self.fp: str | None = None   # a sparse file's source: what its byte ranges on the node belong to
        self.mode = "755" if local.stat().st_mode & stat.S_IXUSR else None

    @property
    def nbytes(self) -> int:
        return sum(n for _, n in self.ranges)


def merge(rs: list[list[int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for off, n in sorted(rs):
        if n <= 0:
            continue
        if out and off <= out[-1][0] + out[-1][1]:
            out[-1][1] = max(out[-1][1], off + n - out[-1][0])
        else:
            out.append([off, n])
    return out


def subtract(want: list[list[int]], have: list[list[int]]) -> list[list[int]]:
    """want minus have (both merged)."""
    out = []
    for off, n in want:
        cur, end = off, off + n
        for h0, hn in have:
            h1 = h0 + hn
            if h1 <= cur or h0 >= end:
                continue
            if h0 > cur:
                out.append([cur, h0 - cur])
            cur = max(cur, h1)
            if cur >= end:
                break
        if cur < end:
            out.append([cur, end - cur])
    return out


def gguf_items(model_dir: Path, lo: int, hi: int) -> list[Item]:
    """Every shard: its header, layers [lo, hi), the non-layer tensors but the PLE (unless layer 1 is in range)."""
    items = []
    for path in sorted(model_dir.glob("*.gguf")):
        g = GGUFFile(path)
        rs = [[0, g.data_start]]
        for t in g.tensors:
            m = re.match(r"blk\.(\d+)\.", t.name)
            if m:
                keep = lo <= int(m.group(1)) < hi
            else:
                keep = not t.name.startswith("per_layer_token_embd") or lo <= 1 < hi
            if keep:
                n = t.expected_bytes()
                if n is None:
                    raise SystemExit(f"{path.name}: {t.name}: unknown size ({t.type_name})")
                rs.append([g.data_start + t.offset, n])
        it = Item(path, f"models/{model_dir.name}/{path.name}", rs)
        with open(path, "rb") as f:   # the header names every tensor's offset: same header, same layout
            it.fp = hashlib.sha256(f.read(g.data_start)).hexdigest()[:32] + f"-{it.total}"
        items.append(it)
    return items


def tree_items(src: Path, prefix: str) -> list[Item]:
    if src.is_file():
        return [Item(src, f"{prefix}/{src.name}")]
    return [Item(p, f"{prefix}/{src.name}/{p.relative_to(src).as_posix()}")
            for p in sorted(src.rglob("*")) if p.is_file()]


class Agent:
    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.token = token

    def sha256(self, rel: str) -> dict:
        req = urllib.request.Request(f"{self.url}/sha256?path={urllib.parse.quote(rel)}",
                                     headers={"X-Stage-Token": self.token})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)

    def ranges(self, rel: str) -> dict:
        req = urllib.request.Request(f"{self.url}/ranges?path={urllib.parse.quote(rel)}",
                                     headers={"X-Stage-Token": self.token})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)

    def put(self, item: Item, off: int, data: bytes) -> dict:
        q = f"path={urllib.parse.quote(item.rel)}&offset={off}&total={item.total}"
        if item.fp:
            q += f"&fp={urllib.parse.quote(item.fp)}"
        if item.mode:
            q += f"&mode={item.mode}"
        req = urllib.request.Request(f"{self.url}/put?{q}", data=data, method="POST",
                                     headers={"X-Stage-Token": self.token,
                                              "Content-Type": "application/octet-stream"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            return {"ok": False, "error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ship(agent: Agent, items: list[Item], say=print) -> int:
    """Send what the node lacks; returns the bytes sent."""
    todo = []
    for it in items:
        have = agent.ranges(it.rel)
        same = have.get("size") == it.total and (it.fp is None or have.get("fp") in (None, it.fp))
        got = merge(have.get("ranges") or []) if same else []   # another source: everything again
        miss = subtract(it.ranges, got)
        if it.whole and not miss and it.total > 0:   # a whole file the node has: the same bytes?
            remote = agent.sha256(it.rel)
            if remote.get("sha256") != file_sha256(it.local):
                miss = list(it.ranges)
        todo.append((it, miss))
    need = sum(n for _, miss in todo for _, n in miss)
    say(f"ship: {sum(it.nbytes for it in items) / 2**30:.2f} GiB wanted, {need / 2**30:.2f} GiB to send "
        f"({len(items)} files)")
    sent, t0 = 0, time.time()
    for it, miss in todo:
        if not miss and it.total == 0:
            continue
        if it.total == 0:   # an empty file: one empty put creates it
            miss = [[0, 0]]
        with open(it.local, "rb") as f:
            for off, n in miss:
                pos, end = off, off + n
                while pos < end or n == 0:
                    f.seek(pos)
                    data = f.read(min(PIECE, end - pos))
                    if not data and n != 0:
                        raise SystemExit(f"ship: {it.local} ended at {pos} (it changed while shipping)")
                    r = agent.put(it, pos, data)
                    if not r.get("ok") or r.get("sha256") != hashlib.sha256(data).hexdigest():
                        raise SystemExit(f"ship: {it.rel} at {pos}: {r.get('error') or 'sha256 differs'}")
                    pos += len(data)
                    sent += len(data)
                    if n == 0:
                        break
                el = time.time() - t0
                say(f"  {it.rel}: {sent / 2**30:.2f} of {need / 2**30:.2f} GiB, "
                    f"{sent / max(el, 1e-6) / 1e6:.0f} MB/s")
    return sent


def plan(model_dir: Path, pack_dir: Path, profile: Path, bin_dir: Path | None, lo: int, hi: int) -> list[Item]:
    items = gguf_items(model_dir, lo, hi)
    items += tree_items(pack_dir, "packs")
    items += tree_items(profile, "data")
    if bin_dir is not None:
        items += [Item(p, f"bin/{p.name}") for p in sorted(bin_dir.iterdir()) if p.is_file()]
    return items


def node_paths(data_dir: str, model_dir: Path, pack_dir: Path, profile: Path) -> dict:
    """node.json's paths for what plan() ships."""
    shards = sorted(p.name for p in model_dir.glob("*.gguf"))
    win = bool(re.match(r"[A-Za-z]:", data_dir)) or "\\" in data_dir   # a Windows node: its own separator, .exe
    sep = "\\" if win else "/"
    d = data_dir.rstrip("/\\")
    j = lambda *parts: sep.join([d, *parts])   # noqa: E731
    return {"exe": j("bin", "strata.exe" if win else "strata"), "lib_dirs": [j("bin")],
            "pack": j("packs", pack_dir.name), "native": j("models", model_dir.name, shards[0]),
            "ple_gguf": j("models", model_dir.name, shards[1]) if len(shards) > 1 else "",
            "expert_profile": j("data", profile.name)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--token-file")
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--pack-dir", required=True, type=Path)
    ap.add_argument("--profile", required=True, type=Path)
    ap.add_argument("--bin-dir", type=Path)
    ap.add_argument("--layers", required=True, help="LO-HI, both included (the node's first and last layer)")
    ap.add_argument("--data-dir", help="the node's data_dir: print the node.json paths for what was shipped")
    a = ap.parse_args()
    token = (Path(a.token_file).read_text().strip() if a.token_file else os.environ.get("STRATA_STAGE_TOKEN", ""))
    lo, hi = (int(x) for x in a.layers.split("-"))
    items = plan(a.model_dir, a.pack_dir, a.profile, a.bin_dir, lo, hi + 1)
    t0 = time.time()
    sent = ship(Agent(a.agent, token), items)
    print(f"ship: {sent / 2**30:.2f} GiB in {time.time() - t0:.0f} s")
    if a.data_dir:
        print("node.json paths:", json.dumps(node_paths(a.data_dir, a.model_dir, a.pack_dir, a.profile), indent=1))


if __name__ == "__main__":
    main()
