"""Read/hash the existing model without loading a runtime or modifying its bytes."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from gguf_reader import GGUFFile

MODEL = Path('H:/Strata/models/新しいフォルダー/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf')
OUT = Path('H:/Strata/evidence/upstream-sync-20261116/model-9b-identity-v1.json')
assert not OUT.exists()
assert MODEL.is_file() and not MODEL.is_symlink() and not MODEL.is_junction()
before = MODEL.stat()
header = GGUFFile(MODEL)
with MODEL.open('rb') as stream:
    checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
after = MODEL.stat()
assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
assert after.st_size == 5629111584
record = {'modelPath': str(MODEL), 'bytes': after.st_size, 'sha256': checksum,
          'architecture': header.metadata.get('general.architecture'),
          'name': header.metadata.get('general.name'),
          'quantizationCounts': dict(Counter(row.type_name for row in header.tensors)),
          'tensorCount': len(header.tensors), 'unchangedDuringRead': True,
          'actualInferenceRun': False, 'separateFromQ36Baseline': True}
with OUT.open('x', encoding='utf-8') as stream:
    json.dump(record, stream, indent=2, ensure_ascii=False)
print(json.dumps({key: record[key] for key in ('bytes', 'sha256', 'architecture', 'actualInferenceRun')}))
