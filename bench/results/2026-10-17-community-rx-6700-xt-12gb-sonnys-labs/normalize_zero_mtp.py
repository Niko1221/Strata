"""Add an explicit zero-MTP metadata key without changing any tensor bytes."""
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys

KEY = 'qwen35moe.nextn_predict_layers'


def normalize(source, destination, receipt, tools):
    sys.path.insert(0, str(tools))
    from gguf_reader import GGUFFile
    source, destination, receipt = map(Path, (source, destination, receipt))
    original = GGUFFile(source)
    assert original.metadata.get('general.architecture') == 'qwen35moe'
    assert KEY not in original.metadata
    assert original.metadata['qwen35moe.block_count'] == 40
    assert not any(t.name.startswith('blk.40.') or 'nextn.' in t.name for t in original.tensors)
    if destination.exists():
        assert receipt.exists(), 'Existing compatibility file lacks verification receipt'
        saved = json.loads(receipt.read_text())
        assert saved['source_sha256'] == file_hash(source)
        assert saved['compatibility_sha256'] == file_hash(destination)
        return saved
    assert shutil.disk_usage(destination.parent).free > source.stat().st_size + 10 * 1024**3, 'Insufficient space for compatibility copy plus reserve'
    with source.open('rb') as stream:
        header = stream.read(original.data_start)
        stream.seek(16)
        count = struct.unpack('<Q', stream.read(8))[0]
        for _ in range(count):
            original._str(stream)
            original._value(stream)
        metadata_end = stream.tell()
    entry = struct.pack('<Q', len(KEY)) + KEY.encode() + struct.pack('<II', 4, 0)
    changed = header[:16] + struct.pack('<Q', count + 1) + header[24:metadata_end] + entry + header[metadata_end:]
    changed += bytes((-len(changed)) % original.alignment)
    partial = destination.with_suffix('.gguf.partial')
    with source.open('rb') as inp, partial.open('wb') as out:
        out.write(changed)
        inp.seek(original.data_start)
        shutil.copyfileobj(inp, out, 8 * 1024 * 1024)
    adapted = GGUFFile(partial)
    assert adapted.metadata == dict(original.metadata, **{KEY: 0})
    assert [(t.name,t.shape,t.type_id,t.offset) for t in original.tensors] == [(t.name,t.shape,t.type_id,t.offset) for t in adapted.tensors]
    # Compare the entire tensor-data region, including alignment gaps and trailing bytes.
    h = hashlib.sha256()
    with source.open('rb') as inp, partial.open('rb') as out:
        inp.seek(original.data_start); out.seek(adapted.data_start)
        while True:
            a,b = inp.read(8*1024*1024),out.read(8*1024*1024)
            assert a == b, 'Tensor-data region changed'
            if not a:
                break
            h.update(a)
    saved = dict(change={KEY:0}, source=str(source), compatibility_file=str(destination),
                 source_sha256=file_hash(source), compatibility_sha256=file_hash(partial),
                 tensor_data_sha256=h.hexdigest(), tensor_data_byte_identical=True,
                 original_data_offset=original.data_start, compatibility_data_offset=adapted.data_start,
                 all_other_metadata_equal=True, tensor_descriptors_equal=True)
    partial.replace(destination)
    receipt.write_text(json.dumps(saved,indent=2))
    return saved


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


if __name__ == '__main__':
    assert len(sys.argv) == 5, 'Usage: normalize_zero_mtp.py ORIGINAL.gguf COMPATIBILITY.gguf receipt.json /path/to/Strata/tools'
    normalize(*sys.argv[1:])
