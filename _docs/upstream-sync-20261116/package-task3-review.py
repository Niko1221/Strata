"""Freeze the controller-owned integration boundary, excluding prior task scopes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '.superpowers/sdd/2026-11-16_upstream-sync-plan_Codex/task-3-review'
FORK = '6f13c7555a7d50289f0e3587ab780627c65a8000'
OFFICIAL = '09b603886925994d0d4461857b01321ef3aa431c'
PATHS = [
    'CMakeLists.txt', 'cmake', 'src', 'include', 'metal', 'sycl', 'tests/core',
    'tests/cuda', 'tests/dsv41', 'tests/qwen35', 'tests/perf131k/test_diagnostics.py',
    '.github/workflows', 'AGENTS.md', 'README*', 'VERSION', 'CONTROL_VERSION',
    'docs/UPSTREAM_0214.md', 'docs/MODELS.md', 'docs/INSTALL.md',
    'docs/data/release.json', 'packaging/README.md', 'tools/build_hakua_engine.mjs',
    'tools/check_cuda_spills.py', 'tools/cuda_spill_baseline.py',
    'tools/test_check_cuda_spills.py', 'tools/test_cuda_spill_baseline.py',
    'tools/release_manifest.py', 'tools/test_release_manifest.py',
    'apps/desktop/scripts/fetch-node.ps1', 'apps/desktop/scripts/fetch-node.sh',
    'apps/desktop/scripts/verify-release.mjs', 'apps/desktop/src-tauri/Cargo.toml',
    'apps/desktop/src-tauri/Cargo.lock', 'apps/desktop/src-tauri/src/main.rs',
    'apps/desktop/package.json', 'apps/desktop/package-lock.json',
    'apps/host/scripts/release-integrity.mjs', 'apps/host/tests/native-build-identity.test.mjs',
    'apps/host/package.json', 'apps/host/package-lock.json',
]

def git(*args: str) -> bytes:
    return subprocess.check_output(['git', '-C', str(ROOT), *args], timeout=120)

def main() -> None:
    tree = git('write-tree').decode().strip()
    record = {'fork': FORK, 'official': OFFICIAL, 'indexTree': tree, 'files': {}}
    for label, base in [('fork', FORK), ('official', OFFICIAL)]:
        target = OUT.with_name(OUT.name + '-' + label + '.diff')
        if target.exists():
            raise FileExistsError(target)
        body = git('diff', '--no-ext-diff', '--no-color', '--full-index', '--no-renames',
                   '--unified=6', base, tree, '--', *PATHS)
        header = f'Root Task3 integration. Base {base}; index tree {tree}.\nNot a release commit.\n'.encode()
        target.write_bytes(header + body)
        record['files'][label] = {'path': str(target), 'bytes': target.stat().st_size,
            'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'lines': body.count(b'\n')}
    with OUT.with_suffix('.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record, indent=2))

if __name__ == '__main__':
    main()
