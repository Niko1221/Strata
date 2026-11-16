"""Freeze only the new compiler-resource correction and its explicit profile proposal."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('H:/Strata/evidence/upstream-sync-20261116/native-spill-v1/root-review-v1')
BASE = '4282fce334e0d6c1843c74bca5d4063bd784bc05'
PATHS = ['src/prefill/moe_fused_iq.cu', 'tests/cuda/prefill_fused_q36_test.cpp',
         'cmake/backends/cuda.cmake', 'tools/baselines/hakua060-sm120-cuda133-msvc1444.json']
OUT.mkdir(exist_ok=False)
subprocess.run(['git', '-C', str(ROOT), 'add', '--', *PATHS], check=True, timeout=60)
tree = subprocess.check_output(['git', '-C', str(ROOT), 'write-tree'], text=True, timeout=60).strip()
diff = subprocess.check_output(['git', '-C', str(ROOT), 'diff', '--no-color', '--full-index',
                               '--unified=10', BASE, tree, '--', *PATHS], timeout=120)
(OUT / 'review.diff').write_bytes(diff)
report = {'base': BASE, 'indexTree': tree, 'diffSHA256': hashlib.sha256(diff).hexdigest(),
          'files': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in PATHS},
          'profileDisposition': 'PROPOSAL, NOT YET ACCEPTED', 'defaultStrictDefectsRemain': 2}
(OUT / 'review.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
