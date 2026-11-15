"""Freeze Task1's staged core and actual snapshot changes for independent review."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / '.superpowers/sdd/2026-11-16_upstream-sync-plan_Codex'
BASE = '6f13c7555a7d50289f0e3587ab780627c65a8000'


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def main():
    output = WORK / 'task-1-review.diff'
    if output.exists():
        raise FileExistsError(output)
    tree = git('write-tree').decode().strip()
    paths = json.loads(Path('H:/Strata/evidence/upstream-sync-20261116/task1-semantic/staged-owned-paths.json').read_text())
    paths += ['tests/snapshot']
    excluded = {'tools/check_cuda_spills.py', 'tools/test_check_cuda_spills.py',
                'tools/cuda_spill_baseline.py', 'tools/test_cuda_spill_baseline.py',
                'tools/release_manifest.py', 'tools/test_release_manifest.py'}
    paths = [path for path in paths if path not in excluded]
    diff = git('diff', '--no-ext-diff', '--no-color', '--full-index', '--no-renames',
               '--unified=18', BASE, tree, '--', *paths)
    output.write_bytes(f'Task1 frozen staged tree {tree} (not a commit), fork base {BASE}.\n\n'.encode()
                       + git('diff', '--stat', BASE, tree, '--', *paths) + b'\n' + diff)
    report = {'base': BASE, 'indexTree': tree, 'path': str(output), 'bytes': output.stat().st_size,
              'diffLines': diff.count(b'\n'), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
