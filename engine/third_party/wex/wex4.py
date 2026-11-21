# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""WEX4 reference implementation (docs/WEX4-SPEC.md). Standard library only.

Independent of c/wex4.h on purpose: the tests run both on the same bytes and
require the same verdict and the same content. Pure Python is slow, so real
files use the C codec through CLib (wex4/wex4_tools.c); the container logic
(validation, layout, hashes) is always done here.

    python -I wex4.py info FILE
    python -I wex4.py verify FILE [--dll wex4_tools.dll]   # every hash, every record decoded
"""
import ctypes
import hashlib
import io
import os
import struct
import sys

MAGIC = 0x57455834
PAGE = 4096
HEADER_HASHED = 4064
REQ_VALUE_EXACT = 1
OPT_ST_SOURCES = 1
MAX_BLOBS, MAX_RECORDS, MAX_TENSORS = 1 << 16, 1 << 24, 1 << 24
MAX_INDEX, MAX_DECODED = 1 << 28, 1 << 30
CODEC_RAW, CODEC_WNR3, CODEC_WBP1 = 0, 5, 6
REC_VALUE_EXACT, REC_EXPERT = 1, 2
BLOB_WNM2 = 1
WNM2_BYTES = 59408
BP_MAGIC = 0x31504257
U64_MAX = (1 << 64) - 1

# dtype codes 1..22 = the safetensors dtypes; other codes are opaque bytes
DTYPES = [None, 'BOOL', 'U8', 'I8', 'F8_E5M2', 'F8_E4M3', 'F8_E8M0', 'I16', 'U16', 'F16', 'BF16', 'I32', 'U32', 'F32',
          'I64', 'U64', 'F64', 'C64', 'F4', 'F6_E2M3', 'F6_E3M2', 'F8_E4M3FNUZ', 'F8_E5M2FNUZ']
BITS = [0, 8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 64, 64, 64, 64, 4, 6, 6, 8, 8]
DTYPE_CODE = {name: code for code, name in enumerate(DTYPES) if name}

HEADER = struct.Struct('<IHHQQQQQIIIIIIQQQ32s')          # 128 bytes
BLOB = struct.Struct('<IIQQ32sQ')                        # 64
RECORD = struct.Struct('<QIIHHIIII32sI')                 # 72
TENSOR = struct.Struct('<IHBB8QIIQQ')                    # 96
assert (HEADER.size, BLOB.size, RECORD.size, TENSOR.size) == (128, 64, 72, 96)


class FormatError(ValueError):
    pass


def need(ok, what):
    if not ok:
        raise FormatError(what)


def align(n):
    return (n + PAGE - 1) & ~(PAGE - 1)


def sha256(data):
    return hashlib.sha256(data).digest()


# ---- MurmurHash3_x86_32 stream hash (identity words, then the chunk's bytes) ----
M32 = 0xffffffff


def _rotl(x, r):
    return ((x << r) | (x >> (32 - r))) & M32


def _tail(h, k):
    k = (k * 0xcc9e2d51) & M32
    k = _rotl(k, 15)
    return h ^ ((k * 0x1b873593) & M32)


def _block(h, k):
    return (_rotl(_tail(h, k), 13) * 5 + 0xe6546b64) & M32


def stream_hash(ident, data):
    h = 0x33524E57
    for k in ident:
        h = _block(h, k)
    full = len(data) & ~3
    for (k,) in struct.iter_unpack('<I', bytes(data[:full])):
        h = _block(h, k)
    if len(data) & 3:
        h = _tail(h, int.from_bytes(data[full:], 'little'))
    h ^= (16 + len(data)) & M32
    h ^= h >> 16
    h = (h * 0x85ebca6b) & M32
    h ^= h >> 13
    h = (h * 0xc2b2ae35) & M32
    return h ^ (h >> 16)


# ---- rANS: 32-bit state, L = 2^23, 14-bit CDF, byte renormalisation (as WNR3) ----
RANS_L, PRECISION, TOTAL = 1 << 23, 14, 1 << 14


def quantize(counts):
    """257 cumulative values from 256 counts; the rule of wex_nr2_quantize."""
    symbols, total = len(counts), sum(counts)
    spare = TOTAL - symbols
    scaled = [c * spare if total else spare for c in counts]
    divisor = total if total else symbols
    freq = [1 + s // divisor for s in scaled]
    extra = TOTAL - sum(freq)
    for s in sorted(range(symbols), key=lambda s: (-(scaled[s] % divisor), s))[:extra]:
        freq[s] += 1
    cdf = [0]
    for f in freq:
        cdf.append(cdf[-1] + f)
    return cdf


def rans_encode(cdf, data, x0):
    state, out = x0, bytearray()
    for symbol in reversed(data):
        start = cdf[symbol]
        f = cdf[symbol + 1] - start
        limit = ((RANS_L >> PRECISION) << 8) * f
        while state >= limit:
            out.append(state & 0xff)
            state >>= 8
        state = ((state // f) << PRECISION) + state % f + start
    for _ in range(4):
        out.append(state & 0xff)
        state >>= 8
    out.reverse()
    return bytes(out)


def rans_decode(cdf, lut, buf, pos, length, count):
    """count symbols from buf[pos:pos+length]; returns (bytes, final state). The stream must end exactly."""
    need(length >= 4, 'stream shorter than its state')
    state = int.from_bytes(buf[pos:pos + 4], 'big')
    ptr, end = pos + 4, pos + length
    need(RANS_L <= state < (RANS_L << 8), 'initial state out of range')
    out = bytearray(count)
    for j in range(count):
        slot = state & (TOTAL - 1)
        symbol = lut[slot]
        start = cdf[symbol]
        state = (cdf[symbol + 1] - start) * (state >> PRECISION) + slot - start
        while state < RANS_L:
            need(ptr < end, 'stream underrun')
            state = (state << 8) | buf[ptr]
            ptr += 1
        out[j] = symbol
    need(ptr == end, 'stream not consumed')
    return out, state


# ---- WBP1 (codec 6): byte planes, order-0 rANS or raw per plane ----
def plane_bytes(decoded, planes, p):
    return (decoded - p + planes - 1) // planes if decoded > p else 0


def rotate(data, planes, left):
    """Every little-endian `planes`-byte element rotated by one bit (transform 1: exponent into the top plane)."""
    bits, out = 8 * planes, bytearray(len(data))
    for i in range(0, len(data), planes):
        v = int.from_bytes(data[i:i + planes], 'little')
        v = ((v << 1) | (v >> (bits - 1))) & ((1 << bits) - 1) if left else (v >> 1) | ((v & 1) << (bits - 1))
        out[i:i + planes] = v.to_bytes(planes, 'little')
    return bytes(out)


def bp_encode(data, planes, chunk_log2, owner, transform=0):
    """Canonical WBP1 record (padded to 4096). Byte-identical to wex4_bp_encode."""
    decoded, chunk = len(data), 1 << chunk_log2
    need(0 < decoded <= MAX_DECODED and planes in (1, 2, 4, 8) and 10 <= chunk_log2 <= 15, 'WBP1 parameters')
    need(transform in (0, 1) and (not transform or (planes > 1 and decoded % planes == 0)), 'WBP1 transform')
    if transform:
        data = rotate(data, planes, True)
    word = planes | chunk_log2 << 8 | transform << 24
    cdfs, tables, payloads, mask, g = [], [], [], 0, 0
    for p in range(planes):
        plane = bytes(data[p::planes])
        streams = (len(plane) + chunk - 1) // chunk
        if not plane:
            tables.append(b'')
            payloads.append(b'')
            continue
        counts = [0] * 256
        for b in plane:
            counts[b] += 1
        cdf = quantize(counts)
        parts = []
        for s in range(streams):
            piece = plane[s * chunk:(s + 1) * chunk]
            ident = (BP_MAGIC, owner, g + s, word)
            parts.append(rans_encode(cdf, piece, RANS_L + (stream_hash(ident, piece) & 0x7fffff)))
        coded = sum(map(len, parts))
        if 514 + 2 * streams + coded < 4 * streams + len(plane):
            mask |= 1 << p
            cdfs.append(struct.pack('<257H', *cdf))
            need(max(map(len, parts)) <= 65535, 'stream too long')
            tables.append(struct.pack(f'<{streams}H', *map(len, parts)))
            payloads.append(b''.join(parts))
        else:
            tables.append(b''.join(struct.pack('<I', stream_hash(
                (BP_MAGIC, owner, g + s, word), plane[s * chunk:(s + 1) * chunk])) for s in range(streams)))
            payloads.append(plane)
        g += streams
    body = struct.pack('<IIIBBBB', BP_MAGIC, owner, decoded, planes, chunk_log2, mask, transform) + \
        b''.join(cdfs) + b''.join(tables) + b''.join(payloads)
    return body + bytes(align(len(body)) - len(body))


def bp_decode(rec, decoded, owner):
    total = len(rec)
    need(total >= PAGE and total % PAGE == 0 and 0 < decoded <= MAX_DECODED, 'WBP1 record size')
    magic, own, dec, planes, chunk_log2, mask, transform = struct.unpack_from('<IIIBBBB', rec)
    need(magic == BP_MAGIC and own == owner and dec == decoded, 'WBP1 header')
    need(planes in (1, 2, 4, 8) and 10 <= chunk_log2 <= 15 and not mask >> planes, 'WBP1 parameters')
    need(transform in (0, 1) and (not transform or (planes > 1 and decoded % planes == 0)), 'WBP1 transform')
    word = planes | chunk_log2 << 8 | transform << 24
    chunk, at = 1 << chunk_log2, 16
    n = [plane_bytes(decoded, planes, p) for p in range(planes)]
    streams = [(x + chunk - 1) // chunk for x in n]
    cdf = [None] * planes
    for p in range(planes):
        if mask >> p & 1:
            need(n[p] and total - at >= 514, 'WBP1 CDF')
            c = struct.unpack_from('<257H', rec, at)
            at += 514
            need(c[0] == 0 and c[256] == TOTAL and all(c[i] < c[i + 1] for i in range(256)), 'WBP1 CDF values')
            cdf[p] = c
    table = [0] * planes
    for p in range(planes):
        size = streams[p] * (2 if mask >> p & 1 else 4)
        need(total - at >= size, 'WBP1 tables')
        table[p] = at
        at += size
    payload = [0] * planes
    lengths = [None] * planes
    for p in range(planes):
        size = n[p]
        if mask >> p & 1:
            lengths[p] = struct.unpack_from(f'<{streams[p]}H', rec, table[p])
            need(all(x >= 4 for x in lengths[p]), 'WBP1 stream length')
            size = sum(lengths[p])
        need(total - at >= size, 'WBP1 payload')
        payload[p] = at
        at += size
    need(align(at) == total and not any(rec[at:]), 'WBP1 padding')
    out, g = bytearray(decoded), 0
    for p in range(planes):
        pos, plane = payload[p], bytearray()
        lut = None
        if cdf[p]:
            lut = bytearray(TOTAL)
            for s in range(256):
                lut[cdf[p][s]:cdf[p][s + 1]] = bytes([s]) * (cdf[p][s + 1] - cdf[p][s])
        for s in range(streams[p]):
            count = min(chunk, n[p] - s * chunk)
            ident = (BP_MAGIC, owner, g, word)
            if cdf[p]:
                piece, state = rans_decode(cdf[p], lut, rec, pos, lengths[p][s], count)
                need(state == RANS_L + (stream_hash(ident, piece) & 0x7fffff), 'WBP1 stream hash')
                pos += lengths[p][s]
            else:
                piece = rec[pos:pos + count]
                need(stream_hash(ident, piece) == struct.unpack_from('<I', rec, table[p] + 4 * s)[0], 'WBP1 chunk hash')
                pos += count
            plane += piece
            g += 1
        out[p::planes] = plane
    return rotate(out, planes, False) if transform else bytes(out)


# ---- WNM2 model blob check (the rule of wex_nr2_model_open) ----
def wnm2_id(blob):
    need(len(blob) == WNM2_BYTES and struct.unpack_from('<8I', blob) == (0x324d4e57, 1, 14, 567, 3, 0, 0, 0), 'WNM2 header')
    at = 32
    for symbols, count in ((16, 3 * 567), (256, 3)):
        for _ in range(count):
            c = struct.unpack_from(f'<{symbols + 1}H', blob, at)
            at += 2 * (symbols + 1)
            need(c[0] == 0 and c[symbols] == TOTAL and all(c[i] < c[i + 1] for i in range(symbols)), 'WNM2 CDF')
    h = 2166136261
    for b in blob:
        h = ((h ^ b) * 16777619) & M32
    return h or 1


# ---- C backend ----
class CLib:
    """wex4/wex4_tools.c through ctypes: fast codec, and the C validator for differential tests."""

    def __init__(self, path):
        lib = ctypes.CDLL(os.path.abspath(path))   # a bare file name is not searched in the current folder on Windows
        P, U32, U64, SZ, I = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_int
        lib.wex4_tools_abi.restype = U32
        need(lib.wex4_tools_abi() == 1, 'wex4_tools ABI')
        lib.wex4_bp_encode_record.argtypes = [P, U32, U32, U32, U32, U32, P, SZ, P]
        lib.wex4_bp_decode_record.argtypes = [P, SZ, U32, U32, P]
        lib.wex4_check.argtypes = [P, U64, I, P]
        lib.wex4_fuzz_check.argtypes = [P, U64, I, P]
        lib.wex4_validate.argtypes = [P, U64, P, SZ]
        lib.wex4_decode_record_buf.argtypes = [P, U64, P, SZ, U32, P, P, I, P]
        lib.wex4_expert_table.argtypes = [P, U64, P, SZ, P]
        lib.wex4_fuzz_run.argtypes = [P, U64, U64, U32, I, P]
        lib.wex4_sha256_bytes.argtypes = [P, SZ, P]
        lib.wex4_store_probe.argtypes = [ctypes.c_char_p, P, P, U64, P]
        self.lib = lib

    def bp_encode(self, data, planes, chunk_log2, owner, transform=0):
        cap = align(len(data) * 2 + 8192)
        out, written = ctypes.create_string_buffer(cap), ctypes.c_size_t()
        need(self.lib.wex4_bp_encode_record(bytes(data), len(data), planes, chunk_log2, transform, owner, out, cap,
                                            ctypes.byref(written)) == 0, 'WBP1 encode failed')
        return out.raw[:written.value]

    def bp_decode(self, rec, decoded, owner):
        out = ctypes.create_string_buffer(decoded)
        need(self.lib.wex4_bp_decode_record(bytes(rec), len(rec), decoded, owner, out) == 0, 'WBP1 record rejected')
        return out.raw

    def check(self, data, flags=3, guard=False):
        """(status, content digest) of a whole file in memory."""
        digest = ctypes.create_string_buffer(32)
        fn = self.lib.wex4_fuzz_check if guard else self.lib.wex4_check
        return fn(bytes(data), len(data), flags, digest), digest.raw


# ---- Container ----
class Reader:
    """Validates the header page and the index, then serves records. `f` is a seekable binary file."""

    def __init__(self, f, clib=None):
        self.f, self.clib, self._blobs = f, clib, {}
        f.seek(0, os.SEEK_END)
        file_bytes = f.tell()
        need(file_bytes >= PAGE, 'file shorter than a header page')
        f.seek(0)
        self.page = page = f.read(PAGE)
        (magic, self.major, self.minor, self.required, self.optional, self.file_bytes, self.index_offset,
         self.index_bytes, nb, nr, nt, self.layers, self.experts, self.max_decoded, names_bytes, meta_bytes,
         self.total_decoded, self.index_sha) = HEADER.unpack_from(page)
        need(magic == MAGIC, 'not a WEX4 file')
        self.root = sha256(page[:HEADER_HASHED])
        need(self.root == page[HEADER_HASHED:], 'header hash mismatch')
        need(self.major == 4, 'unsupported major version')
        need(not self.required & ~REQ_VALUE_EXACT, 'unknown required feature')
        need(self.file_bytes == file_bytes and file_bytes % PAGE == 0, 'file size')
        need(nb <= MAX_BLOBS and nr <= MAX_RECORDS and nt <= MAX_TENSORS and names_bytes <= MAX_INDEX and
             meta_bytes <= MAX_INDEX, 'header counts')
        need(self.layers <= 65536 and self.experts <= 65536 and (not self.layers) == (not self.experts) and
             self.layers * self.experts <= MAX_RECORDS, 'expert grid')
        need(self.max_decoded <= MAX_DECODED and (nr or not (self.max_decoded or self.total_decoded)), 'header totals')
        size = nb * BLOB.size + nr * RECORD.size + nt * (TENSOR.size + 4) + names_bytes + meta_bytes
        need(size == self.index_bytes and size <= MAX_INDEX, 'index size')
        need(self.index_offset >= PAGE and self.index_offset % PAGE == 0 and self.index_offset <= file_bytes and
             align(size) <= file_bytes - self.index_offset and (size or self.index_offset == PAGE), 'index location')
        f.seek(self.index_offset)
        self.index = index = f.read(align(size))
        need(len(index) == align(size), 'short index read')
        need(sha256(index[:size]) == self.index_sha, 'index hash mismatch')
        need(not any(index[size:]), 'index padding')
        at = 0
        self.blobs = [BLOB.unpack_from(index, at + i * BLOB.size) for i in range(nb)]
        at += nb * BLOB.size
        self.records = [RECORD.unpack_from(index, at + i * RECORD.size) for i in range(nr)]
        at += nr * RECORD.size
        raw_tensors = [TENSOR.unpack_from(index, at + i * TENSOR.size) for i in range(nt)]
        at += nt * TENSOR.size
        order = struct.unpack_from(f'<{nt}I', index, at)
        at += nt * 4
        names = index[at:at + names_bytes]
        self.meta = index[at + names_bytes:at + names_bytes + meta_bytes]
        self._validate(raw_tensors, order, names)

    def _validate(self, raw_tensors, order, names):
        # tensors: (name, dtype, rank, shape, record, record_count, offset, nbytes)
        self.tensors, name_at = [], 0
        for t in raw_tensors:
            name_off, name_len, dtype, rank = t[:4]
            shape, (record, record_count, offset, nbytes) = t[4:12], t[12:]
            need(name_off == name_at and name_len <= len(names) - name_at, 'name pool')
            name = names[name_at:name_at + name_len]
            name_at += name_len
            try:
                name.decode('utf-8')
            except UnicodeDecodeError:
                raise FormatError('tensor name is not UTF-8') from None
            need(b'\0' not in name, 'NUL in tensor name')
            need(rank <= 8 and not any(shape[rank:]), 'tensor rank')
            elements = 1
            for d in shape[:rank]:
                need(not d or elements <= U64_MAX // d, 'tensor shape overflow')
                elements *= d
            bits = BITS[dtype] if dtype < len(BITS) else 0
            need(not bits or (elements <= U64_MAX // 64 and nbytes <= U64_MAX // 8 and elements * bits == nbytes * 8),
                 'tensor size does not match its shape')
            self.tensors.append((name, dtype, rank, shape[:rank], record, record_count, offset, nbytes))
        need(name_at == len(names), 'name pool size')
        need(all(i < len(self.tensors) for i in order), 'name order index')
        sorted_names = [self.tensors[i][0] for i in order]
        need(all(a < b for a, b in zip(sorted_names, sorted_names[1:])), 'name order')
        self.by_name = {self.tensors[i][0]: i for i in order}

        for kind, ident, offset, size, digest, reserved in self.blobs:
            need(0 < size <= MAX_DECODED and not reserved, 'blob entry')
            need(kind != BLOB_WNM2 or (size == WNM2_BYTES and ident), 'WNM2 blob entry')

        t = pending = pending_bytes = total = maximum = 0
        last_owner = -1
        for r, (offset, stored, decoded, codec, flags, blob, owner, first, count, digest, reserved) in enumerate(self.records):
            need(stored and stored % PAGE == 0 and 0 < decoded <= MAX_DECODED and stored <= align(decoded) and
                 not flags & ~(REC_VALUE_EXACT | REC_EXPERT) and not reserved and blob <= len(self.blobs), 'record entry')
            if codec == CODEC_RAW:
                need(stored == align(decoded) and not blob, 'raw record')
            else:
                need(stored < align(decoded), 'compressed record is not smaller')
            need(codec != CODEC_WBP1 or (not (blob or flags) and stored >= align(16 + 6 * ((decoded + 32767) >> 15))), 'WBP1 record')
            need(not flags & REC_VALUE_EXACT or (codec == CODEC_WNR3 and self.required & REQ_VALUE_EXACT), 'value-exact flag')
            if codec == CODEC_WNR3:
                need(flags & REC_EXPERT and blob and self.blobs[blob - 1][0] == BLOB_WNM2, 'WNR3 record')
            total += decoded
            maximum = max(maximum, decoded)
            if pending:
                need(not count and first == t - 1 and not flags & REC_EXPERT and owner == r and decoded <= pending_bytes,
                     'continuation record')
                pending_bytes -= decoded
                pending -= 1
                need(pending or not pending_bytes, 'split tensor size')
                continue
            need(count and first == t and count <= len(self.tensors) - t, 'record tensors')
            cursor = 0
            for k in range(count):
                _, _, _, _, t_record, t_count, t_offset, t_bytes = self.tensors[t + k]
                need(t_record == r and t_bytes, 'tensor record')
                if t_count == 1:
                    need(t_offset == cursor and t_bytes <= decoded - cursor, 'tensor range')
                    cursor += t_bytes
                else:
                    need(t_count >= 2 and count == 1 and not t_offset and not flags & REC_EXPERT and
                         t_count <= len(self.records) - r and t_bytes > decoded, 'split tensor')
                    pending, pending_bytes, cursor = t_count - 1, t_bytes - decoded, decoded
            need(cursor == decoded, 'record not tiled by its tensors')
            t += count
            if flags & REC_EXPERT:
                need(codec in (CODEC_RAW, CODEC_WNR3), 'expert codec')
                need(owner >> 16 < self.layers and owner & 0xffff < self.experts and owner > last_owner, 'expert owner')
                last_owner = owner
                entry = self.expert_entry(r)
                if codec == CODEC_WNR3:
                    streams = sum((a * b + 16383) // 16384 for a, b in entry['scale_shapes']) + \
                        sum((a * b * 2 + 16383) // 16384 for a, b in entry['weight_shapes'])
                    need(stored >= align(32 + streams * 6), 'WNR3 record too small')
            else:
                need(owner == r, 'record owner')
        need(not pending and total == self.total_decoded and maximum == self.max_decoded, 'record totals')
        need(not self.layers or last_owner >= 0, 'expert grid without expert records')
        for _, _, _, _, t_record, t_count, t_offset, t_bytes in self.tensors[t:]:
            need(not (t_bytes or t_count or t_record or t_offset), 'zero-size tensor entry')
        self.data_tensors = t

        # header page, blobs, records and index tile the file exactly
        pos, b, r, index_done = PAGE, 0, 0, not self.index_bytes
        while True:
            if b < len(self.blobs) and self.blobs[b][2] == pos:
                size = align(self.blobs[b][3])
                b += 1
            elif r < len(self.records) and self.records[r][0] == pos:
                size = self.records[r][1]
                r += 1
            elif not index_done and self.index_offset == pos:
                size, index_done = len(self.index), True
            else:
                break
            need(size <= self.file_bytes - pos, 'extent past the end')
            pos += size
        need(b == len(self.blobs) and r == len(self.records) and index_done and pos == self.file_bytes, 'file layout')

    def expert_entry(self, r):
        """The legacy 72-byte entry fields of an EXPERT record (validates the slab layout)."""
        offset, stored, decoded, codec, flags, blob, owner, first, count = self.records[r][:9]
        need(flags & REC_EXPERT and count == 6, 'expert record')
        scale_shapes, weight_shapes, scale_offsets, weight_offsets = [], [], [], []
        for m in range(6):
            _, dtype, rank, shape, _, t_count, t_offset, t_bytes = self.tensors[first + m]
            need(rank == 2 and t_count == 1 and all(0 < d <= 65535 for d in shape), 'expert tensor shape')
            need(dtype in ((2, 6) if m < 3 else (2, 3)), 'expert tensor dtype')
            (scale_shapes if m < 3 else weight_shapes).append(tuple(shape))
            (scale_offsets if m < 3 else weight_offsets).append(t_offset)
        cursor = 0
        for m in range(3):
            need(scale_shapes[m][0] == weight_shapes[m][0] and scale_shapes[m][1] * 16 == weight_shapes[m][1] and
                 scale_offsets[m] == cursor, 'expert slab layout')
            cursor += scale_shapes[m][0] * scale_shapes[m][1]
        scale_bytes = cursor
        for m in range(3):
            need(weight_offsets[m] == cursor, 'expert slab layout')
            cursor += weight_shapes[m][0] * weight_shapes[m][1]
        need(cursor == decoded and cursor <= 0xffffffff, 'expert slab size')
        return dict(file_offset=offset, total_bytes=stored, scale_bytes=scale_bytes, weight_bytes=cursor - scale_bytes,
                    scale_shapes=scale_shapes, weight_shapes=weight_shapes, scale_offsets=scale_offsets,
                    weight_offsets=weight_offsets, flags=codec)

    def blob(self, i, verify=True):
        if (i, verify) in self._blobs:   # a model blob is checked once, not once per record
            return self._blobs[i, verify]
        kind, ident, offset, size, digest, _ = self.blobs[i]
        self.f.seek(offset)
        data = self.f.read(align(size))
        need(len(data) == align(size), 'short blob read')
        if verify:
            need(sha256(data[:size]) == digest, 'blob hash mismatch')
            need(not any(data[size:]), 'blob padding')
        if kind == BLOB_WNM2:
            need(wnm2_id(data[:size]) == ident, 'WNM2 id')
        self._blobs[i, verify] = data[:size]
        return data[:size]

    def stored(self, r):
        offset, stored = self.records[r][:2]
        self.f.seek(offset)
        data = self.f.read(stored)
        need(len(data) == stored, 'short record read')
        return data

    def record(self, r, verify=True):
        """Decoded bytes of record r. verify also checks the record's SHA-256."""
        offset, stored, decoded, codec, flags, blob, owner, first, count, digest, _ = self.records[r]
        data = self.stored(r)
        need(not verify or sha256(data) == digest, 'record hash mismatch')
        if codec == CODEC_RAW:
            need(not any(data[decoded:]), 'raw record padding')
            return data[:decoded]
        if codec == CODEC_WBP1:
            return (self.clib or sys.modules[__name__]).bp_decode(data, decoded, owner)
        need(codec == CODEC_WNR3, f'unsupported codec {codec}')
        need(self.clib, 'codec 5 (WNR3) needs the C library: pass --dll wex4_tools.dll')
        out = ctypes.create_string_buffer(decoded)
        status = self.clib.lib.wex4_decode_record_buf(self.page, self.file_bytes, self.index, len(self.index), r, data,
                                                      self.blob(blob - 1, verify), 0, out)
        need(status == 0, f'WNR3 record rejected ({status})')
        return out.raw

    def tensor(self, name, verify=True):
        """(dtype name or None, shape, bytes) of one tensor."""
        name = name.encode() if isinstance(name, str) else name
        need(name in self.by_name, 'no such tensor')
        _, dtype, _, shape, record, count, offset, nbytes = self.tensors[self.by_name[name]]
        if count == 1:
            data = self.record(record, verify)[offset:offset + nbytes]
        else:
            data = b''.join(self.record(record + k, verify) for k in range(count))
        return (DTYPES[dtype] if dtype < len(DTYPES) else None), shape, data

    def content_digest(self, verify=True):
        """SHA-256 over every tensor's metadata and bytes in table order (the definition of wex4_check)."""
        h = hashlib.sha256()

        def meta(t):
            name, dtype, rank, shape, _, _, _, nbytes = t
            h.update(struct.pack('<H', len(name)) + name + struct.pack('<BB8QQ', dtype, rank, *shape, *([0] * (8 - rank)), nbytes))
        for i in range(len(self.blobs)):
            self.blob(i, verify)
        for r, rec in enumerate(self.records):
            data = self.record(r, verify)
            first, count = rec[7], rec[8]
            if not count:
                h.update(data)
            for t in self.tensors[first:first + count]:
                meta(t)
                h.update(data[t[6]:t[6] + t[7]] if t[5] == 1 else data)
        for t in self.tensors[self.data_tensors:]:
            meta(t)
        return h.digest()


def check(data, verify=True, clib=None):
    """Validate a whole file held in memory and decode everything; returns the content digest."""
    return Reader(io.BytesIO(data), clib).content_digest(verify)


class Tensor:
    def __init__(self, name, dtype, shape, nbytes):
        self.name = name.encode() if isinstance(name, str) else name
        self.dtype = DTYPE_CODE.get(dtype, 0) if isinstance(dtype, str) else dtype
        self.shape, self.nbytes = tuple(shape), nbytes


class Record:
    """One record to write. tensors tile its decoded bytes in order ([] = continuation part of a split tensor,
    in which case the previous Record with one tensor carries parts = number of records of that tensor).
    read() returns the decoded bytes. codec 'raw', ('bp', planes, transform) or ('pre', codec_id, stored_bytes_fn):
    'pre' stores bytes encoded elsewhere (WNR3 records). owner None = the record's index."""

    def __init__(self, tensors, decoded, read, codec='raw', flags=0, owner=None, blob=0, parts=1):
        self.tensors, self.decoded, self.read, self.codec = tensors, decoded, read, codec
        self.flags, self.owner, self.blob, self.parts = flags, owner, blob, parts


def write(path, records, blobs=(), empty=(), meta=b'', layers=0, experts=0, required=0, optional=0, chunk_log2=12,
          encoder=None, index_first=True):
    """Write a canonical WEX4 file: header page, index, blobs, records (index_first=False puts the index last,
    the layout of an upgraded WEX3 file). blobs: (kind, id, bytes). empty: zero-size Tensors. Returns the root hash."""
    encoder = encoder or sys.modules[__name__]
    tensors = [t for rec in records for t in rec.tensors] + list(empty)
    names = b''.join(t.name for t in tensors)
    index_bytes = len(blobs) * BLOB.size + len(records) * RECORD.size + len(tensors) * (TENSOR.size + 4) + len(names) + len(meta)
    need(index_bytes <= MAX_INDEX, 'index too large')
    at = PAGE + (align(index_bytes) if index_first else 0)
    tmp = os.fspath(path) + '.partial'
    blob_rows, record_rows, tensor_rows = [], [], []
    try:
        with open(tmp, 'wb') as f:
            for kind, ident, data in blobs:
                f.seek(at)
                f.write(data + bytes(align(len(data)) - len(data)))
                blob_rows.append(BLOB.pack(kind, ident, at, len(data), sha256(data), 0))
                at += align(len(data))
            t = name_at = total = maximum = 0
            last_split = None
            for r, rec in enumerate(records):
                data = rec.read()
                need(len(data) == rec.decoded and 0 < rec.decoded <= MAX_DECODED, f'record {r}: decoded size')
                raw = bytes(data) + bytes(align(len(data)) - len(data))
                codec, stored = CODEC_RAW, raw
                if rec.codec != 'raw' and rec.codec[0] == 'bp':
                    coded = encoder.bp_encode(data, rec.codec[1], chunk_log2, r,
                                              rec.codec[2] if len(data) % rec.codec[1] == 0 else 0)
                    if len(coded) < len(raw):
                        codec, stored = CODEC_WBP1, coded
                elif rec.codec != 'raw':
                    coded = rec.codec[2]()
                    if coded is not None and len(coded) < len(raw):
                        codec, stored = rec.codec[1], coded
                flags = rec.flags if codec == CODEC_WNR3 else rec.flags & ~REC_VALUE_EXACT
                blob = rec.blob if codec == CODEC_WNR3 else 0
                first = t if rec.tensors else last_split
                if rec.tensors and rec.parts > 1:
                    last_split = t
                cursor = 0
                for ten in rec.tensors:
                    tensor_rows.append(TENSOR.pack(name_at, len(ten.name), ten.dtype, len(ten.shape),
                                                   *ten.shape, *([0] * (8 - len(ten.shape))), r, rec.parts, cursor, ten.nbytes))
                    name_at += len(ten.name)
                    cursor += ten.nbytes
                f.seek(at)
                f.write(stored)
                record_rows.append(RECORD.pack(at, len(stored), rec.decoded, codec, flags, blob,
                                               r if rec.owner is None else rec.owner, first, len(rec.tensors), sha256(stored), 0))
                at += len(stored)
                t += len(rec.tensors)
                total += rec.decoded
                maximum = max(maximum, rec.decoded)
            for ten in empty:
                tensor_rows.append(TENSOR.pack(name_at, len(ten.name), ten.dtype, len(ten.shape),
                                               *ten.shape, *([0] * (8 - len(ten.shape))), 0, 0, 0, 0))
                name_at += len(ten.name)
            order = sorted(range(len(tensors)), key=lambda i: tensors[i].name)
            index = b''.join(blob_rows) + b''.join(record_rows) + b''.join(tensor_rows) + \
                struct.pack(f'<{len(order)}I', *order) + names + meta
            need(len(index) == index_bytes, 'index size')
            index_offset = PAGE if index_first or not index_bytes else at
            f.seek(index_offset)
            f.write(index + bytes(align(index_bytes) - index_bytes))
            file_bytes = max(at, index_offset + align(index_bytes))
            page = HEADER.pack(MAGIC, 4, 0, required, optional, file_bytes, index_offset, index_bytes, len(blobs),
                               len(records), len(tensors), layers, experts, maximum, len(names), len(meta), total, sha256(index))
            page += bytes(HEADER_HASHED - len(page))
            root = sha256(page)
            f.seek(0)
            f.write(page + root)
            f.flush()
            os.fsync(f.fileno())
        with open(tmp, 'rb') as f:   # never publish a file the reader refuses
            Reader(f)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, path)
    return root


def main(argv):
    if len(argv) < 3 or argv[1] not in ('info', 'verify'):
        print(__doc__)
        return 2
    clib = CLib(argv[argv.index('--dll') + 1]) if '--dll' in argv else None
    try:
        return report(argv, clib)
    except FormatError as error:
        print('REFUSED:', error)
        return 1


def report(argv, clib):
    with open(argv[2], 'rb') as f:
        rd = Reader(f, clib)
        stored = sum(rec[1] for rec in rd.records)
        codecs = {}
        for rec in rd.records:
            codecs[rec[3]] = codecs.get(rec[3], 0) + 1
        print(f'WEX4 {rd.major}.{rd.minor}  root {rd.root.hex()}')
        print(f'  file {rd.file_bytes:,} B  index {rd.index_bytes:,} B  blobs {len(rd.blobs)}  records {len(rd.records)}'
              f'  tensors {len(rd.tensors)}')
        print(f'  decoded {rd.total_decoded:,} B  stored {stored:,} B'
              f'  ratio {stored / rd.total_decoded if rd.total_decoded else 0:.4f}  codecs {codecs}')
        if argv[1] == 'verify':
            print(f'  content digest {rd.content_digest(True).hex()}  (every hash checked, every record decoded)')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
