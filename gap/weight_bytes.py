#!/usr/bin/env python3
"""Bytes per decode class, from a GGUF's tensor table (no GPU). Value types follow the engine's own
table (metal/src/gguf_lite.cpp:36): 0 u8 1 i8 2 u16 3 i16 4 u32 5 i32 6 f32 7 bool 8 string 9 array
10 u64 11 i64 12 f64. Tensor sizes come from consecutive offsets (alignment padding <= 32 B)."""
import struct, sys, re, collections

F1 = {0: 'B', 1: 'b', 7: 'B'}; F2 = {2: 'H', 3: 'h'}; F4 = {4: 'I', 5: 'i', 6: 'f'}; F8 = {10: 'Q', 11: 'q', 12: 'd'}

def read_infos(path):
    with open(path, 'rb') as f:
        assert f.read(4) == b'GGUF'
        ver, = struct.unpack('<I', f.read(4))
        n_t, = struct.unpack('<Q', f.read(8)); n_kv, = struct.unpack('<Q', f.read(8))
        def s():
            l, = struct.unpack('<Q', f.read(8)); return f.read(l).decode('utf-8', 'replace')
        def val(t):
            if t in F1: return struct.unpack('<' + F1[t], f.read(1))[0]
            if t in F2: return struct.unpack('<' + F2[t], f.read(2))[0]
            if t in F4: return struct.unpack('<' + F4[t], f.read(4))[0]
            if t in F8: return struct.unpack('<' + F8[t], f.read(8))[0]
            if t == 8: return s()
            if t == 9:
                at, = struct.unpack('<I', f.read(4)); n, = struct.unpack('<Q', f.read(8))
                if at == 8: return [s() for _ in range(n)]
                fmt = {**F1, **F2, **F4, **F8}[at]
                return struct.unpack('<' + str(n) + fmt, f.read(n * struct.calcsize('<' + fmt)))
            raise SystemExit(f'gguf value type {t}')
        kv = {}
        for _ in range(n_kv):
            k = s(); kv[k] = val(struct.unpack('<I', f.read(4))[0])
        ts = []
        for _ in range(n_t):
            name = s(); nd, = struct.unpack('<I', f.read(4))
            dims = struct.unpack('<' + 'Q' * nd, f.read(8 * nd))
            ty, = struct.unpack('<I', f.read(4)); off, = struct.unpack('<Q', f.read(8))
            ts.append((name, ty, dims, off))
    ts.sort(key=lambda t: t[3])
    return kv, [(n, ty, d, o, (ts[i+1][3] - o) if i+1 < len(ts) else 0) for i, (n, ty, d, o) in enumerate(ts)]

path = sys.argv[1]
kv, ts = read_infos(path)
grp = collections.defaultdict(lambda: [0, 0])
for name, ty, dims, off, size in ts:
    key = re.sub(r'\.\d+\.', '.N.', name)
    grp[key][0] += 1; grp[key][1] += size
tot = sum(g[1] for g in grp.values())
print(f"{path.split('/')[-1]}: {len(ts)} tensors, {tot/1e9:.2f} GB (last tensor size unknown)")
print("  ", {k.split('.')[-1]: v for k, v in kv.items() if any(s in k for s in
      ('block_count', 'embedding_length', 'head_count', 'expert_count', 'expert_used', 'inner_size',
       'hyper_connection', 'full_attention', 'feed_forward', 'indexer.top_k', 'key_length', 'value_length'))})
for k, (c, b) in sorted(grp.items(), key=lambda x: -x[1][1])[:22]:
    print(f"  {k:46s} {c:6d} x  {b/1e6:9.1f} MB   each {b/c/1e6:6.2f} MB")
