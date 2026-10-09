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
        --layers 36-47 [--data-dir <the node's data_dir>] [--parallel 4]

The pieces (64 MiB) overlap: a reader thread reads and hashes them ahead, and --parallel senders (default 4) put
them, each on its own HTTP/1.1 keep-alive connection, so the disk, the hashing, the link and the node's writes run
at once and a slow link stays busy.  At most parallel + 3 pieces are in memory (448 MiB with 4).  The first piece
that fails (an HTTP error, a sha256 that differs) stops the run; what arrived is in the node's /ranges, so the same
command resumes.

On the node the files land under its data_dir: models/<model dir name>/, packs/<pack name>/, data/<profile>,
bin/.  A /start that names the model ({"model": {"dir": ..., "native": ..., "pack": ...}}, as stage_tune.py sends)
runs it from there; --data-dir prints the paths for a node.json that should run it by default.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import queue
import re
import stat
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_reader import GGUFFile  # noqa: E402

PIECE = 64 << 20
READ_AHEAD = 2   # pieces read and hashed ahead of the senders


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
    """A node agent.  Every request goes to it directly, never through an HTTP proxy from the environment (the
    /puts use http.client, which knows no proxies; the agent is a LAN or tunnel address a proxy cannot reach)."""
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.token = token
        u = urllib.parse.urlsplit(self.url)
        self.scheme, self.netloc, self.base = u.scheme, u.netloc, u.path

    def sha256(self, rel: str) -> dict:
        req = urllib.request.Request(f"{self.url}/sha256?path={urllib.parse.quote(rel)}",
                                     headers={"X-Stage-Token": self.token})
        with self.direct.open(req, timeout=600) as r:
            return json.load(r)

    def ranges(self, rel: str) -> dict:
        req = urllib.request.Request(f"{self.url}/ranges?path={urllib.parse.quote(rel)}",
                                     headers={"X-Stage-Token": self.token})
        with self.direct.open(req, timeout=60) as r:
            return json.load(r)

    def connect(self) -> http.client.HTTPConnection:
        """A connection for /put, kept alive between pieces (an agent that answers HTTP/1.0 closes it after each
        request: http.client then opens the next one itself)."""
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return cls(self.netloc, timeout=600)

    def put(self, conn: http.client.HTTPConnection, item: Item, off: int, data: bytes, sha: str | None = None) -> dict:
        """`sha`: the piece's sha256 - an agent that knows the parameter records the range only when the bytes
        that arrived hash to it (an older one ignores it; the answer's sha256 is compared here either way)."""
        q = f"path={urllib.parse.quote(item.rel)}&offset={off}&total={item.total}"
        if item.fp:
            q += f"&fp={urllib.parse.quote(item.fp)}"
        if item.mode:
            q += f"&mode={item.mode}"
        if sha:
            q += f"&sha256={sha}"
        view = memoryview(data)
        for attempt in (0, 1):
            reused = conn.sock is not None
            try:
                # the body 1 MiB at a time, with its length given (no chunked encoding): the socket's timeout is
                # then a stall's, not the whole 64 MiB's - one piece of `parallel` on a slow link takes long
                conn.request("POST", f"{self.base}/put?{q}",
                             body=(view[i:i + (1 << 20)] for i in range(0, len(view), 1 << 20)),
                             headers={"X-Stage-Token": self.token, "Content-Type": "application/octet-stream",
                                      "Content-Length": str(len(view))})
                r = conn.getresponse()
                body = r.read()
            except (OSError, http.client.HTTPException) as e:
                conn.close()
                if reused and attempt == 0:   # a kept-alive connection the agent had closed: once more, anew
                    continue
                return {"ok": False, "error": f"{type(e).__name__}: {e}"}
            if r.status != 200:
                return {"ok": False, "error": f"HTTP {r.status}: {body.decode(errors='replace')[:300]}"}
            try:
                return json.loads(body)
            except ValueError:
                return {"ok": False, "error": f"not JSON: {body[:300]!r}"}
        return {"ok": False, "error": "unreachable"}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ship(agent: Agent, items: list[Item], say=print, parallel: int = 4) -> int:
    """Send what the node lacks; returns the bytes sent.  A reader thread reads and hashes the pieces READ_AHEAD
    ahead, `parallel` senders put them (each on its own connection): at most parallel + READ_AHEAD + 1 pieces in
    memory.  The first piece that fails stops the run (SystemExit) once the pieces in flight are answered."""
    todo = []
    for it in items:
        have = agent.ranges(it.rel)
        same = have.get("size") == it.total and (it.fp is None or have.get("fp") in (None, it.fp))
        got = merge(have.get("ranges") or []) if same else []   # another source: everything again
        miss = subtract(it.ranges, got)
        if it.total == 0:   # an empty file: one empty put creates it, unless the node has it (size None: missing)
            miss = [] if have.get("size") == 0 else [[0, 0]]
        if it.whole and not miss and it.total > 0:   # a whole file the node has: the same bytes?
            remote = agent.sha256(it.rel)
            if remote.get("sha256") != file_sha256(it.local):
                miss = list(it.ranges)
        todo.append((it, miss))
    need = sum(n for _, miss in todo for _, n in miss)
    say(f"ship: {sum(it.nbytes for it in items) / 2**30:.2f} GiB wanted, {need / 2**30:.2f} GiB to send "
        f"({len(items)} files)")
    pieces = []   # (item, its range's index in left, offset, bytes)
    left = []     # each range: its pieces not yet answered (a progress line when none is left)
    for it, miss in todo:
        for off, n in miss:
            pos, end = off, off + n
            left.append(0)
            while True:
                k = min(PIECE, end - pos)
                pieces.append((it, len(left) - 1, pos, k))
                left[-1] += 1
                pos += k
                if pos >= end:
                    break
    if not pieces:
        return 0
    parallel = max(1, parallel)
    work: queue.Queue = queue.Queue(READ_AHEAD)   # (piece, its bytes, their sha256); None: no more
    done: queue.Queue = queue.Queue()             # (piece, None) answered, or (None, the error)
    stop = threading.Event()

    def offer(x) -> bool:
        while not stop.is_set():
            try:
                work.put(x, timeout=0.5)
                return True
            except queue.Full:
                pass
        return False

    def reader() -> None:
        f, cur = None, None
        try:
            for p in pieces:
                it, _, pos, n = p
                if it is not cur:
                    if f is not None:
                        f.close()
                    f, cur = open(it.local, "rb"), it
                f.seek(pos)
                data = f.read(n)
                if len(data) != n:
                    raise ValueError(f"{it.local} ended at {pos + len(data)} (it changed while shipping)")
                if not offer((p, data, hashlib.sha256(data).hexdigest())):
                    return
                del data
        except BaseException as e:   # noqa: BLE001 - every failure ends the run with its reason
            done.put((None, f"ship: {e}"))
        finally:
            if f is not None:
                f.close()
            for _ in range(parallel):
                offer(None)

    def sender() -> None:
        conn = None
        try:
            conn = agent.connect()
            while not stop.is_set():
                try:
                    x = work.get(timeout=0.5)
                except queue.Empty:
                    continue
                if x is None:
                    return
                p, data, sha = x
                del x
                it, _, pos, _ = p
                r = agent.put(conn, it, pos, data, sha)
                del data
                if not r.get("ok") or r.get("sha256") != sha:
                    done.put((None, f"ship: {it.rel} at {pos}: {r.get('error') or 'sha256 differs'}"))
                    return
                done.put((p, None))
        except BaseException as e:   # noqa: BLE001 - not a thread that ends without a word
            done.put((None, f"ship: {e}"))
        finally:
            if conn is not None:
                conn.close()

    threads = [threading.Thread(target=reader, daemon=True)]
    threads += [threading.Thread(target=sender, daemon=True) for _ in range(parallel)]
    for t in threads:
        t.start()
    sent, t0, todo_n, err = 0, time.time(), len(pieces), None
    try:
        while todo_n and err is None:
            try:
                p, err = done.get(timeout=1)   # (a wait with a timeout: Ctrl+C gets through on Windows too)
            except queue.Empty:
                continue
            if p is None:
                continue
            it, k, _, n = p
            todo_n -= 1
            sent += n
            left[k] -= 1
            if left[k] == 0:
                el = time.time() - t0
                say(f"  {it.rel}: {sent / 2**30:.2f} of {need / 2**30:.2f} GiB, "
                    f"{sent / max(el, 1e-6) / 1e6:.0f} MB/s")
    except BaseException:
        stop.set()   # Ctrl+C: not waiting for the pieces in flight (the threads end with the process)
        raise
    stop.set()       # done, or a failure: the pieces in flight are answered first
    if err:          # said now: a stalled piece in flight can take its timeout before the run ends
        say(f"{err} - waiting for the pieces in flight")
    for t in threads:   # a join with a timeout: Ctrl+C gets through on Windows while a sender waits on a node
        while t.is_alive():
            t.join(1)
    if err:
        raise SystemExit(err)
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
    ap.add_argument("--parallel", type=int, default=4,
                    help="pieces in flight at once, each on its own connection (default 4; 1: one at a time)")
    a = ap.parse_args()
    if a.parallel < 1:
        ap.error("--parallel: 1 or more")
    token = (Path(a.token_file).read_text().strip() if a.token_file else os.environ.get("STRATA_STAGE_TOKEN", ""))
    lo, hi = (int(x) for x in a.layers.split("-"))
    items = plan(a.model_dir, a.pack_dir, a.profile, a.bin_dir, lo, hi + 1)
    t0 = time.time()
    sent = ship(Agent(a.agent, token), items, parallel=a.parallel)
    print(f"ship: {sent / 2**30:.2f} GiB in {time.time() - t0:.0f} s")
    if a.data_dir:
        print("node.json paths:", json.dumps(node_paths(a.data_dir, a.model_dir, a.pack_dir, a.profile), indent=1))


if __name__ == "__main__":
    main()
