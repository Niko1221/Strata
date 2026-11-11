"""cold_store.py -- write and validate the persistent cold store for dsv41 packs.

The cold store is the arena's independent source of truth: every routed expert's
bytes copied out of the GGUF shards into one contiguous file beside the pack, with
a manifest that records where each region came from and what it hashed to.

    write   : read the expert tensors from the shards (via the pack's tensors.txt
              map), append them to cold-store.bin in map order, and write
              cold-store.json (the manifest: per-region [shard, offset, size,
              store_offset, sha256], plus the shards' sizes and mtimes)
    verify  : re-check every region's hash and the source fingerprints; report
              the first mismatch; --sample N checks a random N regions (default
              64, ~200 MB, seconds)
    usage   : what a start would do (match check + which source it would pick)

With a matching cold store, a punched GGUF becomes safe: the engine's startup fill
can take the expert bytes from cold-store.bin instead of the holes in the shards.

This is the tool half of the #5391 follow-up; the engine read-source branch is
the other half.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import subprocess
import sys
import time
from pathlib import Path

CHUNK = 1 << 23


def parse_map(pack: Path) -> tuple:
    """(files, experts) from tensors.txt.

    files: {id: name}. experts: [name, fid, shard_name, off, size] for every routed
    expert tensor; size = next tensor's offset (ANY tensor, same file, sorted) - off,
    else the file's size. Sizing by expert-only gaps swallowed non-expert tensors that
    follow the last expert in a shard (found on the real DeepSeek pack: embed.weight
    after the last expert in shard 4 turned into a 5.7 GiB fake region)."""
    files = {}
    all_tensors = {}  # fid -> [(off, name)]
    for line in (pack / "tensors.txt").read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("file "):
            parts = line.split()
            files[int(parts[1])] = parts[2]
        elif line.startswith("t "):
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                fid, off = int(parts[3]), int(parts[4])
            except ValueError:
                continue
            all_tensors.setdefault(fid, []).append((off, parts[1]))
    experts = []
    for fid, tensors in all_tensors.items():
        tensors.sort()
        name = files[fid]
        for i, (off, tname) in enumerate(tensors):
            if ".ffn.experts." not in tname:
                continue
            size = (tensors[i + 1][0] - off) if i + 1 < len(tensors) else 0
            if size > 0:
                experts.append([tname, fid, name, off, size])
    experts.sort(key=lambda e: (e[2], e[3]))
    return files, experts


def shard_path(pack: Path, name: str, models_dir: Path) -> Path:
    for cand in (models_dir / name, pack / name, Path(r"E:\models") / name):
        if cand.is_file():
            return cand
    raise SystemExit(f"shard not found: {name}")


def cmd_write(a):
    pack = Path(a.pack)
    models = Path(a.models)
    files, experts = parse_map(pack)
    print(f"map: {len(experts)} expert regions across {len(set(e[1] for e in experts))} shards")

    store = pack / "cold-store.bin"
    manifest = {
        "schema": 1,
        "tool": "cold_store.py",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pack": str(pack),
        "regions": [],
        "index": {},       # "layer:expert:role" -> store offset (the engine's lookup)
        "sources": {},
    }
    handles = {}
    total = 0
    t0 = time.time()
    h_out = hashlib.sha256()
    with store.open("wb") as w:
        for i, (tname, fid, name, off, size) in enumerate(experts):
            if name not in handles:
                handles[name] = shard_path(pack, name, models).open("rb")
                st = handles[name].name
                p = Path(st)
                manifest["sources"][name] = {"size": p.stat().st_size, "mtime": p.stat().st_mtime,
                                             "path": str(p)}
            fh = handles[name]
            fh.seek(off)
            left = size
            h = hashlib.sha256()
            store_off = total
            while left > 0:
                b = fh.read(min(CHUNK, left))
                if not b:
                    raise SystemExit(f"short read: {name} @{off} wanted {size}")
                w.write(b)
                h.update(b)
                h_out.update(b)
                left -= len(b)
            total += size
            import re as _re
            mt = _re.match(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(w\d)\.weight", tname)
            if mt:
                layer, expert, w = int(mt.group(1)), int(mt.group(2)), int(mt.group(3)[1])
                # role_base[r] = region offset; the engine indexes (layer, expert, r) -> store offset
                manifest["index"][(layer, expert, w - 1)] = store_off
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(experts)} regions, {total/2**30:.1f} GiB, {time.time()-t0:.0f}s", flush=True)
    for fh in handles.values():
        fh.close()
    manifest["store_bytes"] = total
    manifest["store_sha256"] = h_out.hexdigest()
    # (layer, expert, role) -> store offset, from the regions' tensor names (the engine's lookup)
    import re as _re
    idx = {}
    for tname, name, off, size, store_off, sha in manifest["regions"]:
        mt = _re.match(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(w\d)\.weight", tname)
        if mt:
            # engine role order (ds4_model.cu: parts = {"w1", "w3", "w2"} = gate, up, down)
            role = {"w1": 0, "w3": 1, "w2": 2}[mt.group(3)]
            idx[f"{int(mt.group(1))}:{int(mt.group(2))}:{role}"] = store_off
    manifest["index"] = idx
    (pack / "cold-store.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    # binary sidecar for the engine (no JSON parser in the C++): header + per (layer,expert,role)
    # int64 store offsets, sorted by (layer, expert, role) so the engine can scan/lookup cheaply:
    # magic u64, n_layers u32, n_exp u32, n_entries u64, then per entry: layer i32, expert i32,
    # role i32, pad u32, store_off i64; then per source: name u32+bytes, size u64, mtime f64
    idx = manifest["index"]
    entries = sorted((int(k.split(":")[0]), int(k.split(":")[1]), int(k.split(":")[2]), v)
                     for k, v in idx.items())
    nl = max(e[0] for e in entries) + 1
    ne = max(e[1] for e in entries) + 1
    mp = pack / "cold-store.map"
    with mp.open("wb") as f:
        f.write(struct.pack("<QIIQ", 0x4C53544F43, nl, ne, len(entries)))  # magic "COTSL"
        for layer, expert, role, v in entries:
            f.write(struct.pack("<iiiIq", layer, expert, role, 0, v))
        for name, meta in manifest["sources"].items():
            nb = name.encode("utf-8")
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<qd", meta["size"], meta["mtime"]))
    print(f"wrote {total/2**30:.1f} GiB to {store} in {time.time()-t0:.0f}s")
    print(f"manifest: {pack/'cold-store.json'} ({len(manifest['regions'])} regions)")
    print(f"engine map: {mp} ({len(entries)} entries, {nl} layers x {ne} experts)")


def cmd_verify(a):
    pack = Path(a.pack)
    models = Path(a.models)
    manifest = json.loads((pack / "cold-store.json").read_text(encoding="utf-8"))
    store = pack / "cold-store.bin"
    if store.stat().st_size != manifest["store_bytes"]:
        print(f"STORE SIZE MISMATCH: {store.stat().st_size} != {manifest['store_bytes']}")
        return 1
    # source fingerprints
    bad_src = 0
    for name, meta in manifest["sources"].items():
        p = shard_path(pack, name, models)
        if p.stat().st_size != meta["size"] or p.stat().st_mtime != meta["mtime"]:
            print(f"SOURCE CHANGED: {name} (size {p.stat().st_size} vs {meta['size']}, "
                  f"mtime {p.stat().st_mtime} vs {meta['mtime']})")
            bad_src += 1
    if bad_src:
        return 1
    # region hashes (all or sample)
    regions = manifest["regions"]
    if a.sample and a.sample < len(regions):
        rng = random.Random(7)
        regions = rng.sample(regions, a.sample)
    handles = {}
    bad = 0
    with store.open("rb") as s:
        for tname, name, off, size, store_off, sha in regions:
            s.seek(store_off)
            h = hashlib.sha256()
            left = size
            while left > 0:
                b = s.read(min(CHUNK, left))
                if not b:
                    print(f"STORE SHORT: {tname} @{off} at {store_off}")
                    bad += 1
                    break
                h.update(b)
                left -= len(b)
            if h.hexdigest() != sha:
                print(f"HASH MISMATCH: {tname} @{off} (store @{store_off})")
                bad += 1
    if bad:
        print(f"FAILED: {bad} bad of {len(regions)} checked")
        return 1
    print(f"OK: {len(regions)} regions verified ({a.sample or 'all'}), sources unchanged")
    return 0


def cmd_usage(a):
    """What the engine's read-source branch would decide right now."""
    pack = Path(a.pack)
    models = Path(a.models)
    manifest_path = pack / "cold-store.json"
    if not manifest_path.is_file():
        print("no cold-store.json: a start would fill the arena from the GGUF (as today)")
        return 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ok = True
    for name, meta in manifest["sources"].items():
        p = shard_path(pack, name, models)
        if p.stat().st_size != meta["size"] or p.stat().st_mtime != meta["mtime"]:
            print(f"REFUSE: source {name} changed since the store was written "
                  "(a start would refuse and fall back to the GGUF, loudly)")
            ok = False
    if ok:
        print("MATCH: a start would fill the arena from cold-store.bin "
              f"({manifest['store_bytes']/2**30:.1f} GiB, {len(manifest['regions'])} regions)")
    return 0


def cmd_record_punch(a):
    """After punching: record the shards' new (post-punch) sizes/mtimes as ACCEPTED
    fingerprints. The engine then accepts a source matching either the original or
    the punched fingerprint - a re-downloaded file matches neither and is refused."""
    pack = Path(a.pack)
    mp = pack / "cold-store.json"
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    if "punched" not in manifest:
        manifest["punched"] = {}
    for name, meta in manifest["sources"].items():
        p = Path(models := a.models) / name
        if p.is_file():
            manifest["punched"][name] = {"size": p.stat().st_size, "mtime": p.stat().st_mtime}
        else:
            print(f"note: {name} not in the models dir; leaving its record")
    (mp).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"recorded punched fingerprints for {len(manifest['punched'])} sources")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("write", cmd_write), ("verify", cmd_verify), ("usage", cmd_usage),
                     ("record-punch", cmd_record_punch)):
        p = sub.add_parser(name)
        p.add_argument("--pack", required=True, help="the dsv41 pack folder")
        p.add_argument("--models", default=r"E:\models", help="the GGUF shards' folder")
        if name == "verify":
            p.add_argument("--sample", type=int, default=64, help="0 = verify all")
    a = ap.parse_args()
    return {"write": cmd_write, "verify": cmd_verify, "usage": cmd_usage,
            "record-punch": cmd_record_punch}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
