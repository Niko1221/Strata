"""Freeze a read-only Task2 review from its staged paths, independent of ongoing Python work."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '.superpowers/sdd/2026-11-16_upstream-sync-plan_Codex/task-2-review.diff'
BASE = '6f13c7555a7d50289f0e3587ab780627c65a8000'


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    tree = git('write-tree').decode().strip()
    paths = ['apps/host', 'apps/desktop/src-tauri/src',
             ':(exclude)apps/host/package.json', ':(exclude)apps/host/package-lock.json']
    stat = git('diff', '--stat', BASE, tree, '--', *paths)
    diff = git('diff', '--no-ext-diff', '--no-color', '--full-index', '--no-renames',
               '--unified=18', BASE, tree, '--', *paths)
    header = f'Task2 only. Fork base {BASE}; reviewed index tree {tree} (not a commit).\nMerge is uncommitted; root owns publication.\n\n'.encode()
    OUT.write_bytes(header + stat + b'\n' + diff)
    report = {'file': str(OUT), 'base': BASE, 'indexTree': tree, 'bytes': OUT.stat().st_size,
              'sha256': hashlib.sha256(OUT.read_bytes()).hexdigest(), 'diffLines': diff.count(b'\n')}
    OUT.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
