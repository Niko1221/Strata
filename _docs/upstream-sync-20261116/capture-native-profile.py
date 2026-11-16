"""Capture a proposed bound compiler-resource profile. This is not release acceptance."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('H:/Strata/evidence/upstream-sync-20261116')
BUILD = BASE / 'cuda-build'
PROFILE = ROOT / 'tools/baselines/hakua060-sm120-cuda133-msvc1444.json'
REPORT = BASE / 'native-profile-proposal-v1.json'
sys.path.insert(0, str(ROOT / 'tools'))
import check_cuda_spills as spills
import cuda_spill_baseline as measured

assert not PROFILE.exists() and not REPORT.exists()
catalog = json.loads(subprocess.check_output(['C:/Program Files/CMake/bin/ctest.exe',
    '--test-dir', str(BUILD), '-C', 'Release', '-R', '^cuda_spill_guard$', '--show-only=json-v1'], text=True, timeout=30))
assert len(catalog['tests']) == 1
command = catalog['tests'][0]['command']
assert Path(command[1]).resolve() == ROOT / 'tools/check_cuda_spills.py'
index = command.index('--baseline')
assert Path(command[index + 1]).resolve() == PROFILE
PROFILE.parent.mkdir(exist_ok=True)
capture = list(command)
capture[index] = '--capture-baseline'
subprocess.run(capture, check=True, timeout=300)
record = json.loads(PROFILE.read_text())
expected = measured.bindings(ROOT, BUILD / 'cuda-spill-build-key.txt', ROOT / 'third_party/strata-ggml')
budgets = measured.load(PROFILE, expected)
assert record['bindings'] == expected
cuobjdump, demangler = spills.find_tools(command[command.index('--cuobjdump') + 1])
libraries = command[command.index('--compiler-key-file') + 2:]
assert len(libraries) == 9
exceptions = []
for library in libraries:
    rows = spills.resource_usage(cuobjdump, library)
    names = spills.demangle(demangler, sorted({function for _, _, function, _ in rows}))
    for name, instance, sm, stack, local, reason, obj in spills.find_violations(rows, names)[2]:
        function = next(function for _, _, function, _ in rows if names[function] == name)
        key = measured.entry_key(library, obj, sm, function)
        assert budgets[key] == (stack, local)
        assert sm == 'sm_120' and stack == 24 and local == 0
        assert 'fused_q36' in name and 'native_kernel<' in name
        assert '(int)17,' in name or '(int)22,' in name
        exceptions.append({'entry': key, 'name': name, 'stackBytes': stack, 'localBytes': local,
                           'defaultLimitStackBytes': 16, 'defaultReason': reason})
assert len(exceptions) == 2
result = {'status': 'PROPOSED_MEASUREMENTS_ONLY', 'releaseAccepted': False,
          'profile': str(PROFILE), 'profileSHA256': hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
          'bindings': expected, 'measuredEntries': len(budgets), 'defaultStrictExceptions': exceptions,
          'measuredLibraries': record['measuredLibraries'],
          'limits': 'Two Q36 IQ2 specializations remain above default16. No blanket allowance, default change, speed or model qualification.'}
with REPORT.open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps({'status': result['status'], 'entries': len(budgets), 'exceptions': len(exceptions), 'releaseAccepted': False}))
