"""Bounded executable probes or serial GPU sanitizer/Spin8 checks, never model inference."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

BASE = Path('H:/Strata/evidence/upstream-sync-20261116')
BUILD = BASE / 'cuda-build'
SMI = 'C:/Windows/System32/nvidia-smi.exe'
SANITIZER = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/compute-sanitizer/compute-sanitizer.exe'

def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def free_mib() -> int:
    return int(subprocess.check_output([SMI, '--id=0', '--query-gpu=memory.free',
                                       '--format=csv,noheader,nounits'], text=True, timeout=10).strip())

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('mode', choices=['probes', 'gpu-extra'])
args = parser.parse_args()
output = BASE / ('native-final-' + args.mode + '-v1')
output.mkdir(exist_ok=False)
commands = []
if args.mode == 'probes':
    for exe in ('strata', 'strata-q36'):
        for flag in ('version', 'help'):
            commands.append((exe + '-' + flag, [str(BUILD / 'Release' / (exe + '.exe')), '--' + flag], 30))
else:
    for name in ('q36_kv_lifecycle_test', 'q36_kv_cache_test'):
        commands.append((name, [SANITIZER, '--tool', 'memcheck', '--error-exitcode', '1',
                                str(BUILD / 'Release' / (name + '.exe'))], 240))
    spin8 = str(BUILD / 'spin8/Release/spin8_attention_v2_cuda_test.exe')
    commands.extend([('spin8-first-use', [spin8, '--first-use-only'], 120), ('spin8-full', [spin8], 180)])

rows = []
for name, command, seconds in commands:
    samples = [free_mib()] if args.mode == 'gpu-extra' else []
    if samples:
        assert samples[0] >= 8192, 'Fresh exclusive-task GPU admission refused'
    log = output / (name + '.txt')
    start = time.monotonic()
    with log.open('xb') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if samples:
                    samples.append(free_mib())
                    assert samples[-1] >= 768, 'GPU reserve floor'
                assert time.monotonic() - start < seconds, 'Case deadline'
                assert log.stat().st_size <= 8 * 1024**2, 'Case output limit'
                time.sleep(0.5)
            assert process.returncode == 0, 'Case process failed: ' + name
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
    text = log.read_text(encoding='utf-8', errors='replace')
    if name.endswith('-version'):
        assert '0.6.0' in text
    elif name.endswith('-help'):
        assert all(word in text for word in ('Q36_MEMORY_CAPABILITIES', 'bounded-experts-v1', 'Q36_KV_CAPABILITIES'))
    elif name.startswith('q36_'):
        assert 'ERROR SUMMARY: 0 errors' in text
    rows.append({'case': name, 'passed': True, 'command': command, 'logSHA256': sha(log),
                 'minimumFreeMiB': min(samples) if samples else None, 'reserveFloorMiB': 768 if samples else None})
record = {'passed': True, 'mode': args.mode, 'cases': rows, 'releaseQualified': False,
          'nativeBuildSHA256': sha(BUILD / 'Release/native-build.json'),
          'binaries': {name: sha(BUILD / 'Release' / name) for name in ('strata.exe', 'strata-q36.exe')},
          'scope': 'Executable capabilities and synthetic checks only; no actual-model or performance qualification.'}
with (output / 'results.json').open('x', encoding='utf-8') as stream:
    json.dump(record, stream, indent=2)
print(json.dumps({'passed': True, 'cases': len(rows), 'mode': args.mode, 'releaseQualified': False}))
