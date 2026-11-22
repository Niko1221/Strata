"""tools/ple_tf_screen.py - a PLE table for teacher forcing only: the rows quant_quality.py's run reads, in a native low-bit
format, at their place in a full-length table whose other rows are zero (an NTFS sparse file: about 64 KiB of disk a row,
so 4-5 GB, not the 19-21 GB of the full table). TF through it is TF through the full table of that format, so a
candidate format is measured before its full table is written (memory-goal/plan-15pct-v2.md section 1.6).

    python tools/ple_tf_screen.py --format t58 --src <IQ4_NL shard 2> --codebook <t58.codebook.json> --config <TF config.json>
                                  --out <screen-58.gguf> [--check <full t58 table>] [--exe <strata.exe>]

The rows: quant_quality.py run sends every segment of its ids file (data/quant_quality/v1/set.ids.json) as one request to
a fresh session, and the engine reads, for every token, the 16 rows ngram_rows (src/kernels/ngram.cpp) hashes from it and
its two predecessors in that segment (none before the first: session_zero sets ple_prev to -1; src/core/verify.cpp
advances it token by token). The hash constants are the model file's (`<arch>.ple.*` of the config's --native shard, as
generate.cpp takes them), else the engine's compiled ones; the hash is checked against the engine's oracle vectors
(src/kernels/ple_oracle_vectors.inc) before anything is written. The 16 rows of the one token the run generates are not
known in advance and stay zero: they come after every scored position.

The rows are encoded by ple_lowbit_emulate.py's own quantizer with the given codebook (the one the full table is or will
be written with), packed and decoded back as the engine decodes them (bit for bit, or it stops), and written at
data_start + row * row bytes of a file with the full table's size and metadata (plus strata.ple.screen). --check compares
every written row's bytes with a full table of the same format. An existing --out is overwritten in place (truncated
first), so one file serves every candidate of a row size. Also written: <out>.rows.npy (the row ids), <out>.codebook.json,
and <out>.tf.json, the config with --ple-gguf <out> and --ple-io direct (the reads, not RAM residency; TF does not depend
on which, measured), for `quant_quality.py run --config <out>.tf.json`.
"""
import argparse
import json
import os
import pathlib
import re
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ple_lowbit_emulate as E  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402

IDS = HERE.parent / "data" / "quant_quality" / "v1" / "set.ids.json"
ORACLE = HERE.parents[1] / "engine" / "src" / "kernels" / "ple_oracle_vectors.inc"
TABLE_ROWS = 320001536                      # PLE_TABLE_ROWS (ngram.hpp)
EOS, NULL = 248044, -1                      # PLE_EOS_TOKEN_ID, TOKEN_NULL
MASK = (1 << 64) - 1
# ple_artifact_consts() (ngram.cpp): the engine's own constants, for a model file that carries none
CONSTS = {"mult": [23703573157769, 20109073645365, 8052911324071],
          "vocab": [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                    20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]}
CONSTS["offset"] = [sum(CONSTS["vocab"][:h]) for h in range(16)]


def token_rows(tok, prev, c):
    """ngram_rows for one token: prev = (older, newer) predecessor, NULL where there is none -> its 16 rows."""
    ctx, cut = [tok], False
    for s in (1, 2):                        # s back: prev[2 - s] (s = 1 the newer); the cut is tested before storing
        t = NULL if cut else prev[2 - s]
        cut = cut or t < 0 or t == EOS
        ctx.append(EOS if cut else t)
    out = []
    for n in (2, 3):
        mixed = (ctx[0] * c["mult"][0]) & MASK
        for j in range(1, n):
            mixed ^= (ctx[j] * c["mult"][j]) & MASK
        out += [mixed % c["vocab"][h] + c["offset"][h] for h in range((n - 2) * 8, (n - 1) * 8)]
    return out


def segment_rows(ids, c):
    """The rows a fresh session reads for this token sequence (verify.cpp: prev = {ple_prev[0], ple_prev[1]}, then
    prev[0] = prev[1], prev[1] = token), (len(ids), 16)."""
    prev, out = [NULL, NULL], []
    for t in ids:
        out.append(token_rows(t, prev, c))
        prev = [prev[1], t]
    return np.array(out, np.int64)


def check_oracle(c):
    """The hash against the engine's oracle vectors (ref/ngram.py's, via tests/python/gen_ple_oracle.py) -> cases checked."""
    text = ORACLE.read_text()
    arrays = {m.group(1): [int(v) for v in m.group(2).replace("\n", " ").split(",") if v.strip()]
              for m in re.finditer(r"(case\d+_\w+)\[\d+\] = \{([^}]*)\}", text)}
    n = 0
    for name in sorted({k.split("_")[0] for k in arrays}):
        toks, prev, rows = arrays[name + "_tokens"], arrays[name + "_prev"], arrays[name + "_rows"]
        got = [r for i, t in enumerate(toks) for r in token_rows(t, prev[2 * i:2 * i + 2], c)]
        if got != rows:
            sys.exit("the n-gram hash disagrees with %s (%s): not writing a screen" % (ORACLE, name))
        n += 1
    if n == 0:
        sys.exit("no hash cases in %s" % ORACLE)
    return n


def model_consts(cfg):
    """The hash constants of the config's --native model file (generate.cpp), else the engine's own."""
    args = cfg.get("args", [])
    if "--native" not in args:
        return CONSTS, "the engine's compiled constants (the config has no --native)"
    path = args[args.index("--native") + 1]
    md = GGUFFile(pathlib.Path(path)).metadata
    p = md.get("general.architecture", "") + ".ple."
    keys = [p + k for k in ("layer_multipliers", "head_vocab_sizes", "head_offsets")]
    if not any(k in md for k in keys):
        return CONSTS, "the engine's compiled constants (%s has no %s*)" % (path, p)
    if not all(k in md for k in keys):
        sys.exit("%s has only some of %s: the engine refuses it" % (path, keys))
    c = {"mult": md[keys[0]], "vocab": md[keys[1]], "offset": md[keys[2]]}
    if len(c["mult"]) != 3 or len(c["vocab"]) != 16 or len(c["offset"]) != 16 or \
            any(o + v > TABLE_ROWS for o, v in zip(c["offset"], c["vocab"])):
        sys.exit("%s: PLE hash arrays the engine refuses" % path)
    return c, "%s of %s" % (p + "*", path)


# ---------------------------------------------------------------- the sparse file (Windows: NTFS)
def set_sparse(f):
    if sys.platform != "win32":
        return              # (ext4 and the like leave the holes a seek makes unallocated anyway)
    import ctypes
    import msvcrt
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p,
                                    wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    n = wintypes.DWORD()
    if not k32.DeviceIoControl(msvcrt.get_osfhandle(f.fileno()), 0x900C4, None, 0, None, 0, ctypes.byref(n), None):  # FSCTL_SET_SPARSE
        sys.exit("cannot make %s sparse (error %d): it would take the full table's space" % (f.name, ctypes.get_last_error()))


def set_size(f, size):
    """The file's length, with no data written for the gap (Python's truncate() writes zeros on Windows)."""
    f.flush()
    if sys.platform != "win32":
        os.ftruncate(f.fileno(), size)
        return
    import ctypes
    import msvcrt
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong, ctypes.c_void_p, wintypes.DWORD]
    k32.SetEndOfFile.argtypes = [wintypes.HANDLE]
    h = msvcrt.get_osfhandle(f.fileno())
    if not (k32.SetFilePointerEx(h, size, None, 0) and k32.SetEndOfFile(h)):
        sys.exit("cannot set the length of %s (error %d)" % (f.name, ctypes.get_last_error()))


def allocated(path):
    if sys.platform != "win32":
        return os.stat(path).st_blocks * 512
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCompressedFileSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
    k32.GetCompressedFileSizeW.restype = wintypes.DWORD
    hi = wintypes.DWORD()
    lo = k32.GetCompressedFileSizeW(str(path), ctypes.byref(hi))
    return (hi.value << 32) | lo


def read_rows(path, offset, rb, rows):
    """rows (sorted) of rb bytes from a table whose row 0 is at `offset` -> (n, rb) uint8."""
    out = np.empty((len(rows), rb), np.uint8)
    with open(path, "rb") as f:
        for i, r in enumerate(rows):
            f.seek(offset + int(r) * rb)
            b = f.read(rb)
            if len(b) != rb:
                sys.exit("%s: short read at row %d" % (path, r))
            out[i] = np.frombuffer(b, np.uint8)
    return out


def table_of(path):
    g = GGUFFile(pathlib.Path(path))
    t = [x for x in g.tensors if x.name == E.TENSOR]
    if len(t) != 1:
        sys.exit("%s: no %s" % (path, E.TENSOR))
    return g, t[0], g.data_start + t[0].offset


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--format", required=True, choices=sorted(E.FORMATS))
    ap.add_argument("--src", required=True, help="the IQ4_NL GGUF holding per_layer_token_embd.weight (shard 2)")
    ap.add_argument("--codebook", required=True, help="the format's codebook JSON (ple_lowbit_emulate.py writes <out>.codebook.json)")
    ap.add_argument("--config", required=True, help="the TF config (quant_quality.py run --config): its --native file's hash")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default=str(IDS), help="quant_quality.py's ids file (default: %(default)s)")
    ap.add_argument("--check", help="a full table of the same format: every written row must have its bytes")
    ap.add_argument("--exe", help="the engine for <out>.tf.json (default: the config's)")
    ap.add_argument("--workers", type=int, default=8, help="threads for the trellis search")
    a = ap.parse_args()
    t0 = time.time()

    cfg = json.load(open(a.config, encoding="utf-8-sig"))
    c, where = model_consts(cfg)
    n_cases = check_oracle(c)
    st = json.load(open(a.ids))
    per_seg = [segment_rows(s["ids"], c) for s in st["segments"]]
    reads = sum(r.size for r in per_seg)
    rows = np.unique(np.concatenate([r.ravel() for r in per_seg]))
    if rows[0] < 0 or rows[-1] >= TABLE_ROWS:
        sys.exit("row %d outside the table" % (rows[0] if rows[0] < 0 else rows[-1]))
    print("TF rows: %s: %d tokens x 16 heads = %d reads, %d distinct rows (hash: %s; %d oracle cases match)" %
          (" + ".join("%s %d" % (s["name"], len(s["ids"])) for s in st["segments"]), reads // 16, reads, len(rows), where, n_cases),
          flush=True)

    if a.format in E.TRELLIS:
        try:
            E._encoder(a.format)
        except RuntimeError as e:
            sys.exit(str(e))
    cb = E.load_codebook(a.codebook, a.format)
    grid = np.array(cb["grid"], np.float32)
    src = E.Source(a.src)
    if src.n_rows != TABLE_ROWS:
        sys.exit("%s has %d rows, not the %d the hash addresses" % (a.src, src.n_rows, TABLE_ROWS))
    rb = E.row_bytes(a.format)
    X = E.iq4nl_decode(read_rows(a.src, src.off, E.IQ_RB, rows))
    packed, num, den, sat = [], 0.0, 0.0, 0
    for i in range(0, len(rows), 8192):
        P, Y, s = E.native_rows(X[i:i + 8192], cb, grid, a.workers, "TF rows %d:%d" % (i, i + 8192))
        packed.append(P)
        x = X[i:i + 8192].astype(np.float64)
        num, den, sat = num + ((Y.astype(np.float64) - x) ** 2).sum(), den + (x ** 2).sum(), sat + s
    packed = np.concatenate(packed)
    print("encoded %d rows as %s (%d B a row): NMSE vs IQ4_NL %.5f%s, %.1f s" %
          (len(rows), a.format, rb, num / den, ", %d E4M3 scales saturated" % sat if cb["scale"] == "e4m3" else "", time.time() - t0),
          flush=True)

    kv = E.native_kv(a.format, cb, a.src, 0, TABLE_ROWS)
    kv["general.name"] += ", TF SCREEN: only the teacher-forcing rows"
    kv["strata.ple.screen"] = "%d rows of %s hold values (quant_quality.py's %s); every other row is zero" % (
        len(rows), a.format, pathlib.Path(a.ids).name)
    head = E.gguf_header(TABLE_ROWS, kv, E.GGML_I8, rb)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    size = len(head) + TABLE_ROWS * rb
    with open(out, "wb") as f:              # (an existing screen is truncated first: its space is freed, then reused)
        set_sparse(f)
        f.write(head)
        for r, p in zip(rows, packed):
            f.seek(len(head) + int(r) * rb)
            f.write(p.tobytes())
        set_size(f, size)
    if out.stat().st_size != size:
        sys.exit("%s is %d B, not %d" % (out, out.stat().st_size, size))
    g, t, off = table_of(out)               # read back through the GGUF parser: the rows are where the engine looks
    if t.type_name != "I8" or t.shape != [rb, TABLE_ROWS] or off != len(head) or \
            not np.array_equal(read_rows(out, off, rb, rows), packed) or g.metadata.get("strata.ple.format") != a.format:
        sys.exit("%s does not read back as written" % out)
    np.save(str(out) + ".rows.npy", rows.astype(np.uint32))
    pathlib.Path(str(out) + ".codebook.json").write_text(json.dumps(cb, indent=1))
    print("wrote %s: %d rows x %d B + %d B header = %d B, %.2f GB on disk (sparse); row ids: %s.rows.npy" %
          (out, TABLE_ROWS, rb, len(head), size, allocated(out) / 1e9, out), flush=True)

    if a.check:
        gf, tf, offf = table_of(a.check)
        md, ms = gf.metadata, g.metadata
        same = [k for k in ("strata.ple.format", "strata.ple.codebook", "strata.ple.scale", "strata.ple.trellis_bits", "strata.ple.code_bits")
                if md.get(k) != ms.get(k)]
        if tf.type_name != "I8" or tf.shape != [rb, TABLE_ROWS] or same:
            sys.exit("%s is not this format's full table: %s" % (a.check, same or [tf.type_name, tf.shape]))
        full = read_rows(a.check, offf, rb, rows)
        bad = int((full != packed).any(1).sum())
        print("check: %d of %d rows differ from %s's bytes; its format, codebook, scale and trellis keys are the same" %
              (bad, len(rows), a.check))
        if bad:
            sys.exit(1)

    args, out_args, i = list(cfg["args"]), [], 0
    if "--ple-gguf" not in args:
        args += ["--ple-gguf", ""]
    while i < len(args):                    # the table is --out (in the config's own place), read directly
        if args[i] == "--ple-gguf":
            out_args += ["--ple-gguf", str(out)]
        elif args[i] != "--ple-io":
            out_args.append(args[i])
            i += 1
            continue
        i += 2
    tf_cfg = dict(cfg, args=out_args + ["--ple-io", "direct"])
    if a.exe:
        tf_cfg["exe"] = a.exe
    tf_path = pathlib.Path(str(out) + ".tf.json")
    tf_path.write_text(json.dumps(tf_cfg, indent=1))
    print("TF config: %s (engine %s)\n  python terminal/tools/quant_quality.py run --config \"%s\" --out <run.tsv>\n%.1f s total" %
          (tf_path, tf_cfg["exe"], tf_path, time.time() - t0))


if __name__ == "__main__":
    main()
