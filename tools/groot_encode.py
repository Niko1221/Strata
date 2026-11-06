#!/usr/bin/env python3
"""Groot Gyro-M: encode the routed experts from the FP8 teacher as TQ2_T and write a
complete GGUF. Non-expert tensors are copied verbatim from the official Q4_K_M GGUF
(keeps PLE table Q5_0, dense Q4_K, embeddings...); the 48x3 stacked expert tensors are
re-encoded from the FP8 safetensors teacher at 2.25 bpw.

    .venv/bin/python tools/groot_encode.py --layers all --out FILE [--threads 22] [--pilot]

Then: iq_pack.py --gguf FILE --compat-bf16 --out PACK ; serve fully resident on one 3090.
"""
import argparse, glob, os, pathlib, struct, sys, time
import concurrent.futures as cf
import multiprocessing as mp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tqenc
from gguf_reader import GGUFFile
from safetensors import safe_open

Q4DIR = "/mnt/data4tb/models/groot-65b"
FP8DIR = Q4DIR + "/fp8"
Q4SHARDS = sorted(glob.glob(Q4DIR + "/Groot-65B-A6B-261023-Q4_K_M-0000*-of-00004.gguf"))
N_LAYERS, N_EXP = 48, 256
EXPERT_ROLES = {"ffn_gate_exps": "gate_proj", "ffn_up_exps": "up_proj", "ffn_down_exps": "down_proj"}
TQ2_T_ID = 144

def kv_skip(fh):
    """advance fh past the KV section; return nothing (pos = start of tensor info)."""
    def rd_s():
        n = struct.unpack("<Q", fh.read(8))[0]; b = fh.read(n)
        assert len(b) == n
        return b
    tmap = {0:("<B",1),1:("<b",1),2:("<H",2),3:("<h",2),4:("<I",4),5:("<i",4),6:("<f",4),7:("<?",1),10:("<Q",8),11:("<q",8),12:("<d",8)}
    n_kv = struct.unpack("<Q", fh.read(8))[0]
    for _ in range(n_kv):
        rd_s()  # key
        t = struct.unpack("<I", fh.read(4))[0]
        if t == 8: rd_s()
        elif t == 9:
            et = struct.unpack("<I", fh.read(4))[0]; n = struct.unpack("<Q", fh.read(8))[0]
            if et == 8:
                for _ in range(n): rd_s()
            else:
                fmt, sz = tmap[et]; fh.read(n * sz)
        else:
            fmt, sz = tmap[t]; fh.read(sz)

def load_st_index():
    """tensor name -> (shard path, byte offsets) from safetensors headers."""
    idx = {}
    for f in sorted(glob.glob(FP8DIR + "/*.safetensors")):
        with open(f, "rb") as h:
            n = struct.unpack("<Q", h.read(8))[0]; hdr = json_loads(h.read(n))
        for k, v in hdr.items():
            if k != "__metadata__": idx[k] = (f, tuple(v["data_offsets"]), v["dtype"], v["shape"])
    return idx

def json_loads(b):
    import json; return json.loads(b)

def read_st(idx, name, f8=None):
    f, (o0, o1), dtype, shape = idx[name]
    with open(f, "rb") as h:
        hdrn = struct.unpack("<Q", h.read(8))[0]
        h.seek(8 + hdrn + o0); raw = h.read(o1 - o0)
    if dtype == "F8_E4M3":
        import ml_dtypes
        w = np.frombuffer(raw, dtype=ml_dtypes.float8_e4m3fn).astype(np.float32).reshape(shape)
    elif dtype == "BF16":
        w = np.frombuffer(raw, dtype=np.uint16).view(ml_dtypes_bf16()).astype(np.float32).reshape(shape)
    else:
        raise ValueError("unexpected dtype " + dtype)
    return w

def ml_dtypes_bf16():
    import ml_dtypes; return ml_dtypes.bfloat16

def teacher_expert(idx, layer, role_hf, e):
    base = f"model.language_model.layers.{layer}.mlp.experts.{e}.{role_hf}"
    w = read_st(idx, base + ".weight")
    s = read_st(idx, base + ".weight_scale_inv")  # [out/128, in/128] bf16
    bs = 128
    out_, in_ = w.shape
    sc = s.repeat(bs, 0).repeat(bs, 1)[:out_, :in_]
    return (w * sc).astype(np.float32)

IDX = None
TMPD = None

def encode_task(args):
    L, gguf_role, hf_role, threads, iters = args
    assert TMPD is not None
    out_blob = TMPD / f"blk.{L}.{gguf_role}.bin"
    if out_blob.exists() and out_blob.stat().st_size > 0:
        return (L, gguf_role)  # resume: already encoded
    chunks = []
    for e in range(N_EXP):
        w = teacher_expert(IDX, L, hf_role, e)
        packed, err, tot, stats = tqenc.encode(w, tqenc.TQ2_T, iters=iters, threads=threads)
        chunks.append(packed.tobytes())
    out_blob.write_bytes(b"".join(chunks))
    return (L, gguf_role)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=22)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--iters", type=int, default=0, help="0 = production fast cfg (tq_convert), 1 = refine")
    ap.add_argument("--pilot", action="store_true", help="encode layer 0 only, report dB/speed, write nothing")
    a = ap.parse_args()
    layers = list(range(N_LAYERS)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    if a.pilot: layers = [0]

    # gather tensor infos from all Q4 shards (order preserved: shard order, then file order)
    all_infos, shard_meta = [], None
    for si, s in enumerate(Q4SHARDS):
        g = GGUFFile(pathlib.Path(s))
        if si == 0: shard_meta = s
        for t in g.tensors: all_infos.append((t.name, list(t.shape), t.type_id, si, t.offset))
    print(f"source tensors: {len(all_infos)} across {len(Q4SHARDS)} shards")
    idx = load_st_index()
    print(f"teacher tensors: {len(idx)}")

    def is_expert(name):
        return any(r in name for r in EXPERT_ROLES)

    # plan output
    plan = []
    for name, shape, tid, si, off in all_infos:
        if is_expert(name):
            n = int(np.prod(shape)); bb = n // 128 * tqenc.block_bytes(tqenc.TQ2_T)
            plan.append((name, shape, TQ2_T_ID, bb, "enc", None))
        else:
            # bytes from source: need end offset -> next tensor's offset in same shard
            plan.append((name, shape, tid, None, "copy", (si, off)))

    # copy lengths: filled after data_starts are computed below (offset deltas, exact)
    by_shard = {}
    def copy_len(si, off):
        return by_shard[si][off]
    def shard_data_start(si):
        with open(Q4SHARDS[si], "rb") as r:
            r.read(4); r.read(4); n_tensors_src = struct.unpack("<Q", r.read(8))[0]
            kv_skip(r)
            for _ in range(n_tensors_src):
                nl = struct.unpack("<Q", r.read(8))[0]; r.read(nl)
                nd = struct.unpack("<I", r.read(4))[0]; r.read(8 * nd); r.read(4); r.read(8)
            return (r.tell() + 31) // 32 * 32
    data_starts = {si: shard_data_start(si) for si in range(len(Q4SHARDS))}
    for si in range(len(Q4SHARDS)):
        g = GGUFFile(pathlib.Path(Q4SHARDS[si]))
        offs_sorted = sorted(t.offset for t in g.tensors)
        data_end = pathlib.Path(Q4SHARDS[si]).stat().st_size - data_starts[si]
        by_shard[si] = {o: ((offs_sorted[i + 1] - o) if i + 1 < len(offs_sorted) else (data_end - o))
                        for i, o in enumerate(offs_sorted)}

    # header size: kv bytes + tensor infos
    with open(shard_meta, "rb") as fh:
        fh.seek(0)
        magic = fh.read(4); ver = struct.unpack("<I", fh.read(4))[0]
        n_tensors = struct.unpack("<Q", fh.read(8))[0]
        kv_start = fh.tell()
        kv_skip(fh)
        kv_end = fh.tell()
        kv_bytes = open(shard_meta, "rb").read()[kv_start:kv_end]
    print(f"kv bytes: {len(kv_bytes)} (n_kv parsed from shard 1; n_tensors will be {len(all_infos)})")

    # encode pass (parallel across (layer, role) tasks; results to temp files, not RAM)
    global IDX, TMPD
    IDX = idx
    TMPD = tmpd = pathlib.Path(Q4DIR) / "enc-tmp"
    tmpd.mkdir(exist_ok=True)
    per_threads = max(1, a.threads // max(1, a.workers))

    tasks = [(L, gr, hr, per_threads, a.iters) for L in layers for gr, hr in EXPERT_ROLES.items()]
    t0 = time.time(); done = 0
    with cf.ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("fork")) as ex:
        for L, gr in ex.map(encode_task, tasks):
            done += 1
            dt = time.time() - t0
            if done % 4 == 0 or done == len(tasks):
                print(f"{done}/{len(tasks)} layer-roles in {dt:.0f}s ({done*1.26e9/max(dt,1)/1e6:.1f} M w/s)", flush=True)
    if a.pilot:
        print("PILOT DONE (temp files kept)")
        return

    # write GGUF
    out = pathlib.Path(a.out)
    infos_sec = b""
    offs, data_start_pad = [], None
    # first pass sizes
    sizes = []
    for name, shape, tid, bb, kind, src in plan:
        sizes.append(bb if kind == "enc" else copy_len(*src))
    hdr_fixed = 4 + 4 + 8 + len(kv_bytes)  # kv_bytes already includes the n_kv count
    tinfo = b""
    for (name, shape, tid, bb, kind, src), sz in zip(plan, sizes):
        tinfo += struct.pack("<Q", len(name.encode())) + name.encode()
        tinfo += struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in shape)
        tinfo += struct.pack("<I", tid)
        tinfo += struct.pack("<Q", 0)  # offset patched below
    hdr_total = hdr_fixed + len(tinfo)
    data_start = (hdr_total + 31) // 32 * 32
    off = 0; final_infos = b""
    for (name, shape, tid, bb, kind, src), sz in zip(plan, sizes):
        final_infos += struct.pack("<Q", len(name.encode())) + name.encode()
        final_infos += struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in shape)
        final_infos += struct.pack("<I", tid) + struct.pack("<Q", off)
        off += sz
    pad = data_start - hdr_total
    with open(out, "wb") as w:
        w.write(magic + struct.pack("<I", ver) + struct.pack("<Q", len(plan)) + kv_bytes)
        w.write(final_infos)
        w.write(b"\x00" * pad)
        assert w.tell() == data_start
        src_handles = {si: open(Q4SHARDS[si], "rb") for si in range(len(Q4SHARDS))}
        for (name, shape, tid, bb, kind, src), sz in zip(plan, sizes):
            if kind == "enc":
                role = name.split(".")[2]  # ffn_gate_exps etc
                L = name.split(".")[1]
                with open(tmpd / f"blk.{L}.{role}.bin", "rb") as r:
                    w.write(r.read())
            else:
                si, soff = src
                r = src_handles[si]
                r.seek(data_starts[si] + soff); w.write(r.read(sz))
        for r in src_handles.values(): r.close()
    print(f"WROTE {out} ({out.stat().st_size/1e9:.1f} GB)")

if __name__ == "__main__":
    main()
