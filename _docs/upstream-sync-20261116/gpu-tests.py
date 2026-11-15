"""One serial numerical CUDA check group with fresh admission and reserve monitoring."""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import re
import subprocess
import time

BASE = Path('H:/Strata/evidence/upstream-sync-20261116')
BUILD = BASE / 'cuda-build'
CTEST = 'C:/Program Files/CMake/bin/ctest.exe'
SMI = 'C:/Windows/System32/nvidia-smi.exe'
NAMES = ('q36_kv_lifecycle_test', 'q36_kv_cache_test', 'native_cache_bridge',
         'native_cache_owner', 'expert_cache_segmented_test', 'prefill_fused_moe_test',
         'prefill_fused_iq_test', 'prefill_fused_group_test', 'prefill_fused_q36_test')


def free_mib():
    text = subprocess.check_output([SMI, '--id=0', '--query-gpu=memory.free',
                                    '--format=csv,noheader,nounits'], text=True, timeout=10)
    return int(text.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='gpu-tests-v1')
    parser.add_argument('--test', action='append')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9-]+', args.name):
        parser.error('contained unique evidence name required')
    names = tuple(args.test or NAMES)
    if not all(re.fullmatch(r'[A-Za-z0-9_.-]+', name) for name in names):
        parser.error('explicit CTest names required')
    output = BASE / (args.name + '.json')
    if output.exists():
        raise FileExistsError(output)
    catalog = json.loads(subprocess.check_output([CTEST, '--test-dir', str(BUILD),
        '-C', 'Release', '--show-only=json-v1'], text=True, timeout=30))
    registered = {row['name'] for row in catalog['tests']}
    assert set(names) <= registered, set(names) - registered
    admission = free_mib()
    if admission < 8192:
        raise RuntimeError('GPU0 needs at least 8192 MiB free for this bounded test group')
    pattern = '^(' + '|'.join(re.escape(name) for name in names) + ')$'
    command = [CTEST, '--test-dir', str(BUILD), '-C', 'Release', '-R', pattern,
               '--parallel', '1', '--no-tests=error', '--timeout', '90', '--output-on-failure',
               '--output-junit', str(BASE / (args.name + '.xml'))]
    proc = subprocess.Popen(command)
    samples = [admission]
    reason = None
    try:
        while proc.poll() is None:
            available = free_mib()
            samples.append(available)
            if available < 768:
                reason = 'GPU_RESERVE_BELOW_768_MIB'
                proc.terminate()
                break
            time.sleep(0.75)
        code = proc.wait(timeout=15)
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=15)
    report = {'owner': 'current Strata upstream integration; only serial CUDA experiment in this task',
              'command': command, 'registeredRequestedTests': list(names), 'exitCode': code,
              'reason': reason, 'admissionFreeMiB': admission, 'minimumFreeMiB': min(samples),
              'reserveFloorMiB': 768, 'samples': len(samples),
              'scope': 'Synthetic numerical/lifecycle checks on shared Windows desktop GPU; no performance or model-quality claim.'}
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if reason or code:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
