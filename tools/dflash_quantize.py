#!/usr/bin/env python3
"""Quantize a standalone BF16 DFlash GGUF with llama.cpp's GGML quantizers.

Matrices use Q8_0, Q5_0 or Q4_0; one-dimensional norms retain their original
BF16 bits. Streaming conversion is offline and does not change the target model.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import struct
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'third_party/llama.cpp/gguf-py'))
from gguf_reader import GGUFFile, BLOCK_GEOMETRY
from gguf_writer import GGUFWriter
from gguf import GGMLQuantizationType as Q
from gguf.quants import quantize


def export(source, output, kind, chunk_rows=256):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('source and output must differ')
    if kind not in ('Q8_0', 'Q5_0', 'Q4_0') or chunk_rows < 1:
        raise ValueError('expected Q8_0, Q5_0 or Q4_0 and a positive chunk size')
    g = GGUFFile(source)
    if g.metadata.get('general.architecture') != 'dflash' or not g.tensors:
        raise ValueError('source must be a standalone DFlash GGUF')
    directory = []
    offset = 0
    for t in g.tensors:
        if t.type_id != 30 or len(t.shape) not in (1, 2):
            raise ValueError(f'{t.name}: expected original BF16 vector or matrix')
        if g.data_start + t.offset + t.expected_bytes() > source.stat().st_size:
            raise ValueError(f'{t.name}: truncated payload')
        typ = Q[kind].value if len(t.shape) == 2 else 30
        block, size = BLOCK_GEOMETRY[kind if typ != 30 else 'BF16']
        if t.shape[0] % block:
            raise ValueError(f'{t.name}: columns must be divisible by {block}')
        length = int(np.prod(t.shape)) // block * size
        directory.append((t, typ, offset, length))
        offset += (length + 31) // 32 * 32
    w = GGUFWriter()
    for key, value in g.metadata.items():
        w.add(key, value)
    w.add('general.alignment', 32)
    w.add('strata.dflash.quantization', kind)
    digest = hashlib.sha256()
    with source.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            digest.update(chunk)
    w.add('strata.dflash.source_sha256', digest.hexdigest())
    header = b'GGUF' + struct.pack('<IQQ', 3, len(directory), len(w.metadata))
    header += b''.join(w._kv_bytes(k, typ, value) for k, (typ, value) in w.metadata.items())
    for t, typ, at, length in directory:
        name = t.name.encode()
        header += struct.pack('<Q', len(name)) + name + struct.pack('<I', len(t.shape))
        header += struct.pack('<' + 'Q' * len(t.shape), *t.shape) + struct.pack('<IQ', typ, at)
    header += b'\0' * (-len(header) % 32)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + '.part')
    try:
        with source.open('rb') as src, tmp.open('wb') as dst:
            dst.write(header)
            for t, typ, at, length in directory:
                src.seek(g.data_start + t.offset)
                if typ == 30:
                    dst.write(src.read(t.expected_bytes()))
                else:
                    cols, rows = t.shape
                    for start in range(0, rows, chunk_rows):
                        n = min(chunk_rows, rows - start)
                        raw = src.read(n * cols * 2)
                        if len(raw) != n * cols * 2:
                            raise ValueError(f'{t.name}: truncated payload')
                        bits = np.frombuffer(raw, '<u2').astype(np.uint32) << 16
                        values = bits.view(np.float32).reshape(n, cols)
                        if not np.isfinite(values).all():
                            raise ValueError(f'{t.name}: non-finite weights')
                        dst.write(quantize(values, Q(typ)).tobytes())
                if dst.tell() != len(header) + at + length:
                    raise ValueError(f'{t.name}: incorrect quantized payload size')
                dst.write(b'\0' * (-length % 32))
        check = GGUFFile(tmp)
        if [(t.name, t.shape, t.type_id) for t in check.tensors] != [
                (t.name, t.shape, typ) for t, typ, _, _ in directory]:
            raise ValueError('quantized directory verification failed')
        tmp.replace(output)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    print(f'{kind}: {output} ({output.stat().st_size} bytes)', flush=True)
    return output


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('-o', '--output', required=True, type=Path)
    ap.add_argument('--type', required=True, choices=['Q8_0', 'Q5_0', 'Q4_0'])
    a = ap.parse_args()
    export(a.source, a.output, a.type)
