#!/usr/bin/env python3
"""Export output.weight alone for --dflash-head, preserving target verification.
Offline only. Uses the repository's gguf-py quantizers; no inference dependency.
Q8/Q4 made from a quantized source are re-quantizations, not recovered precision.
"""
import argparse
import hashlib
import json
import pathlib
import struct
import sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'third_party/llama.cpp/gguf-py'))
from gguf_reader import GGUFFile, BLOCK_GEOMETRY
from gguf_writer import GGUFWriter
from gguf import GGMLQuantizationType as Q
from gguf.quants import dequantize, quantize


def export(source, output, kind, chunk_rows=1024):
    g = GGUFFile(source)
    ts = [t for t in g.tensors if t.name == 'output.weight']
    if len(ts) != 1 or len(ts[0].shape) != 2:
        raise ValueError('source must contain exactly one matrix named output.weight')
    t = ts[0]; width, vocab = t.shape
    out_type = t.type_id if kind == 'copy' else Q[kind].value
    out_name = t.type_name if kind == 'copy' else kind
    block, size = BLOCK_GEOMETRY[t.type_name]
    row_bytes = width // block * size
    w = GGUFWriter()
    for key in ['general.architecture', 'qwen4exp.block_count', 'qwen4exp.embedding_length',
                'qwen4exp.expert_count', 'qwen4exp.expert_used_count',
                'qwen4exp.attention.head_count', 'qwen4exp.attention.head_count_kv']:
        w.add(key, g.metadata[key])
    w.add('general.name', 'DFlash draft-only output head')
    w.add('strata.dflash_head.source', str(pathlib.Path(source).resolve()))
    w.add('strata.dflash_head.source_type', t.type_name)
    name = b'output.weight'
    header = b'GGUF' + struct.pack('<IQQ', 3, 1, len(w.metadata))
    header += b''.join(w._kv_bytes(k, typ, val) for k, (typ, val) in w.metadata.items())
    header += struct.pack('<Q', len(name)) + name + struct.pack('<IQQIQ', 2, width, vocab, out_type, 0)
    header += b'\0' * (-len(header) % 32)
    output = pathlib.Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + '.part')
    src_hash = hashlib.sha256(); dst_hash = hashlib.sha256(); d2 = a2 = b2 = dot = 0.; maxerr = 0.
    with open(source, 'rb') as f, tmp.open('wb') as out:
        f.seek(g.data_start + t.offset); out.write(header)
        for start in range(0, vocab, chunk_rows):
            n = min(chunk_rows, vocab - start)
            raw = f.read(n * row_bytes)
            if len(raw) != n * row_bytes: raise ValueError('truncated source tensor')
            src_hash.update(raw)
            if kind == 'copy': payload = raw
            else:
                values = dequantize(np.frombuffer(raw, np.uint8).reshape(n, row_bytes), Q(t.type_id))
                if not np.isfinite(values).all(): raise ValueError('non-finite source head')
                packed = quantize(values, Q(out_type)); back = dequantize(packed, Q(out_type))
                a = values.astype(np.float64); b = back.astype(np.float64); diff = a - b
                maxerr = max(maxerr, float(np.abs(diff).max())); d2 += float(np.sum(diff * diff))
                a2 += float(np.sum(a*a)); b2 += float(np.sum(b*b)); dot += float(np.sum(a*b))
                payload = packed.tobytes()
            out.write(payload); dst_hash.update(payload)
    check = GGUFFile(tmp)
    assert check.tensors[0].shape == t.shape and check.tensors[0].type_id == out_type
    assert tmp.stat().st_size == check.data_start + check.tensors[0].expected_bytes()
    tmp.replace(output)
    rec = dict(source=str(pathlib.Path(source).resolve()), source_type=t.type_name,
               output=str(output.resolve()), output_type=out_name, shape=t.shape,
               source_tensor_sha256=src_hash.hexdigest(), output_tensor_sha256=dst_hash.hexdigest(),
               payload_bytes=output.stat().st_size - len(header), max_abs=maxerr,
               relative_l2=float(np.sqrt(d2/a2)) if a2 else 0.,
               cosine=dot/np.sqrt(a2*b2) if a2 and b2 else 1.)
    output.with_suffix(output.suffix + '.json').write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec), flush=True)
    return rec


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source'); ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--type', choices=['Q8_0', 'Q4_0', 'copy'], required=True)
    a = ap.parse_args(); export(a.source, a.output, a.type)
