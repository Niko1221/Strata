#!/usr/bin/env python3
"""Quantize BF16 draft matrices to Q8_0 with the reference quantizer. Preserve other tensors and shard boundaries."""
from __future__ import annotations
import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "terminal" / "tools"))
from gguf_reader import GGUFFile
from iq_pack import Model


class Directory(GGUFFile):
    def __init__(self, path):
        self.directory_positions = []
        self.value_depth = 0
        self.file_type_position = None
        super().__init__(path)

    def _str(self, fh):
        value = super()._str(fh)
        if self.value_depth == 0 and value == 'general.file_type':
            self.file_type_position = self._pos
        return value

    def _value(self, fh):
        self.value_depth += 1
        try:
            return super()._value(fh)
        finally:
            self.value_depth -= 1

    def _unpack(self, fh, fmt):
        if fmt == '<IQ' and self.value_depth == 0:
            self.directory_positions.append(self._pos)
        return super()._unpack(fh, fmt)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gguf', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--ggml-library', type=Path, required=True)
    ap.add_argument('--dfly', action='store_true', help='quantize the standalone hy_v4_next_dfly drafter')
    ap.add_argument('--threads', type=int, default=16)
    args = ap.parse_args()
    if args.threads < 1:
        ap.error('--threads must be positive')
    model = Model(args.gguf.absolute())
    metadata = model.files[0].metadata
    if args.dfly:
        if metadata.get('general.architecture') != 'hy_v4_next_dfly':
            ap.error('--dfly needs a standalone hy_v4_next_dfly GGUF')
        scope = 'Dfly'
        prefixes = ('',)
    else:
        scope = 'MTP'
        blocks = metadata['hy_v4_flash.block_count']
        count = metadata['hy_v4_flash.nextn_predict_layers']
        if count < 1:
            ap.error('model has no MTP layer')
        prefixes = tuple(f'blk.{i}.' for i in range(blocks-count, blocks))
    targets = {name for name, (_, t, _, _) in model.where.items()
               if name.startswith(prefixes) and t.type_id == 30 and len(t.shape) >= 2
               and t.shape[0] % 32 == 0}
    if not targets:
        ap.error(f'no BF16 {scope} matrices to quantize')
    destinations = [args.out_dir / (p.stem.removesuffix('-bf16') + '-q8_0.gguf' if args.dfly else p.name)
                    for p in model.paths]
    if any(p.exists() or p.with_suffix('.gguf.tmp').exists() for p in destinations):
        ap.error('output or temporary shard already exists; choose a fresh directory')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    library = ctypes.CDLL(str(args.ggml_library.resolve()))
    quantize = library.quantize_row_q8_0_ref
    quantize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    quantize.restype = None
    manifest = {'schema': 1, 'source': str(args.gguf.absolute()), 'quantizer': str(args.ggml_library.resolve()),
                'scope': f'BF16 {scope} matrices only', 'tensors': []}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as pool:
        for source, destination in zip(model.paths, destinations):
            directory = Directory(source)
            if not any(t.name in targets for t in directory.tensors):
                # Link immutable source bytes; never edit a linked shard in place.
                try:
                    os.link(source, destination)
                    print('unchanged shard hardlink:', destination, flush=True)
                except OSError:
                    import shutil
                    shutil.copyfile(source, destination)
                for t in directory.tensors:
                    manifest['tensors'].append({'name': t.name, 'type': t.type_name, 'unchanged_shard': True,
                                               'bytes': t.expected_bytes()})
                continue
            with source.open('rb') as inp:
                header = bytearray(inp.read(directory.data_start))
                if args.dfly and directory.file_type_position is not None:
                    pos = directory.file_type_position
                    if struct.unpack_from('<I', header, pos)[0] != 4:
                        raise ValueError('general.file_type must be a uint32')
                    struct.pack_into('<I', header, pos + 4, 7)  # GGML_FTYPE_MOSTLY_Q8_0
                if len(directory.directory_positions) != len(directory.tensors):
                    raise ValueError('tensor directory position count mismatch')
                offset = 0
                for t, position in zip(directory.tensors, directory.directory_positions):
                    size = t.elements//32*34 if t.name in targets else t.expected_bytes()
                    struct.pack_into('<IQ', header, position, 8 if t.name in targets else t.type_id, offset)
                    offset += size
                    offset += (-offset) % directory.alignment
                temporary = destination.with_suffix('.gguf.tmp')
                with temporary.open('xb') as out:
                    out.write(header)
                    for t in directory.tensors:
                        inp.seek(directory.data_start+t.offset)
                        remaining = t.expected_bytes()
                        source_hash, result_hash = hashlib.sha256(), hashlib.sha256()
                        while remaining:
                            raw = inp.read(min(32 << 20, remaining))
                            if not raw:
                                raise EOFError(t.name)
                            source_hash.update(raw)
                            if t.name in targets:
                                values = (np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16).view(np.float32)
                                if not np.isfinite(values).all():
                                    raise ValueError(f'nonfinite {scope} weight: {t.name}')
                                result = np.empty(values.size//32*34, dtype=np.uint8)
                                blocks_per_job = max(1, (values.size//32 + args.threads-1)//args.threads)
                                jobs = []
                                for b in range(0, values.size//32, blocks_per_job):
                                    n = min(blocks_per_job, values.size//32-b)
                                    jobs.append(pool.submit(quantize, values.ctypes.data+b*32*4,
                                                            result.ctypes.data+b*34, n*32))
                                for job in jobs:
                                    job.result()
                                payload = result.tobytes()
                            else:
                                payload = raw
                            out.write(payload)
                            result_hash.update(payload)
                            remaining -= len(raw)
                        out.write(b'\0' * ((-out.tell()) % directory.alignment))
                        record = {'name': t.name, 'source_type': t.type_name,
                                  'type': 'Q8_0' if t.name in targets else t.type_name,
                                  'source_sha256': source_hash.hexdigest(), 'sha256': result_hash.hexdigest(),
                                  'bytes': t.elements//32*34 if t.name in targets else t.expected_bytes()}
                        if t.name not in targets:
                            assert record['source_sha256'] == record['sha256']
                        manifest['tensors'].append(record)
                        if t.name in targets:
                            print(t.name, 'BF16 -> Q8_0', record['bytes'], flush=True)
                    out.flush()
                    os.fsync(out.fileno())
                temporary.rename(destination)
            print('wrote:', destination, destination.stat().st_size, flush=True)
    Model(destinations[0])  # validate every declared payload and the split contract
    (args.out_dir/f'{scope.lower()}-quantization.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('complete:', len(targets), f'{scope} matrices quantized; other tensors preserved', flush=True)


if __name__ == '__main__':
    main()
