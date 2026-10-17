#!/usr/bin/env python3
"""Fetch only the original Qwen output head and copy its BF16 bytes to GGUF.

Uses the checkpoint revision and strict HTTP range checks shared with mtp_fetch.
No dequantization or quantization; the DFlash verifier keeps its target head.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import mtp_fetch as source
from gguf_reader import GGUFFile
from gguf_writer import GGUFWriter

SHAPE = [2560, 248320]  # GGUF order
CHUNK = 16 << 20


def valid_cache(output):
    try:
        g = GGUFFile(output)
        t, = g.tensors
        stamp = json.loads(output.with_suffix(output.suffix + '.json').read_text())
        if (t.name != 'output.weight' or t.type_id != 30 or t.shape != SHAPE or
                g.metadata.get('strata.dflash_head.source') != source.REPO or
                output.stat().st_size != g.data_start + t.expected_bytes()):
            return False
        digest = hashlib.sha256()
        with output.open('rb') as f:
            f.seek(g.data_start)
            while chunk := f.read(CHUNK):
                digest.update(chunk)
        return digest.hexdigest() == stamp['tensor_sha256']
    except (OSError, ValueError, KeyError):
        return False


def fetch(output):
    output = Path(output)
    if valid_cache(output):
        print(f'Original BF16 DFlash head: {output} (cached)', flush=True)
        return output
    index = json.loads(source.get(source.REPO + 'model.safetensors.index.json'))['weight_map']
    names = [n for n in index if n == 'lm_head.weight' or n == 'model.lm_head.weight']
    if len(names) != 1:
        raise ValueError('checkpoint must contain one text lm_head.weight')
    name, = names
    shard = index[name]
    base, header = source.shard_header(shard)
    t = header[name]
    start, end = t['data_offsets']
    if t['dtype'] != 'BF16' or list(reversed(t['shape'])) != SHAPE or end - start != 2 * SHAPE[0] * SHAPE[1]:
        raise ValueError('original output head has unexpected dtype, shape or byte count')
    w = GGUFWriter()
    w.add('general.architecture', 'qwen4exp')
    for key, value in [('block_count', 48), ('embedding_length', 2560),
                       ('expert_count', 512), ('expert_used_count', 10),
                       ('attention.head_count', 24), ('attention.head_count_kv', 2)]:
        w.add('qwen4exp.' + key, value, 'u32')
    w.add('general.name', 'DFlash original BF16 output head')
    w.add('strata.dflash_head.source', source.REPO)
    w.add('strata.dflash_head.source_type', 'BF16')
    raw_name = b'output.weight'
    gguf = b'GGUF' + struct.pack('<IQQ', 3, 1, len(w.metadata))
    gguf += b''.join(w._kv_bytes(k, typ, v) for k, (typ, v) in w.metadata.items())
    gguf += struct.pack('<Q', len(raw_name)) + raw_name + struct.pack('<IQQIQ', 2, *SHAPE, 30, 0)
    gguf += b'\0' * (-len(gguf) % 32)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + '.part')
    digest = hashlib.sha256()
    try:
        with tmp.open('wb') as f:
            f.write(gguf)
            for offset in range(start, end, CHUNK):
                chunk = source.get(source.REPO + shard, base + offset, base + min(offset + CHUNK, end) - 1)
                f.write(chunk)
                digest.update(chunk)
                print(f'Original BF16 head: {min(offset + CHUNK, end) - start}/{end - start} bytes', flush=True)
        tmp.replace(output)
        output.with_suffix(output.suffix + '.json').write_text(json.dumps({
            'source': source.REPO, 'tensor': name, 'shard': shard,
            'tensor_sha256': digest.hexdigest(), 'dtype': 'BF16', 'shape': t['shape']}, indent=2))
    finally:
        tmp.unlink(missing_ok=True)
    print(f'Original BF16 DFlash head: {output}', flush=True)
    return output


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True)
    fetch(ap.parse_args().output)
