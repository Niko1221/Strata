"""Remove five explicitly requested integrated worktrees, preserving native evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[1]
PARENT = Path('H:/Strata/worktrees')
BACKUP = Path('H:/Strata/private-backups/retired-worktrees-20261108')
CANDIDATES = {
    'gui-consistency-20261106': '7a4eda5def3e13ff6c5178ec863b40cc48ef5345',
    'native-32gb16gb-20261106': '044722d2b314f2bd50158b9b7da6989f82a9f213',
    'hybrid-integration-20261106': '7523fdb902b1fc5f8055c283012ea64fc9cc7224',
    'spin8-cuda135-20261106': '27456e2a58fa1b1e6067a987cd0392434e53c422',
    'spin8-baseline-c50ca06a': 'c50ca06a863a98ff121cbff30365a035a90205e2',
}


def git(where: Path, *args: str) -> str:
    return subprocess.check_output(['git', '-C', str(where), *args], text=True, timeout=60).strip()


def inventory(root: Path) -> dict:
    result = {}
    pending = [root]
    while pending:
        folder = pending.pop()
        assert not folder.is_junction() and not folder.is_symlink(), 'Linked evidence tree'
        for path in folder.iterdir():
            assert not path.is_junction() and not path.is_symlink(), 'Linked evidence entry'
            if path.is_dir():
                pending.append(path)
            elif path.is_file():
                with path.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                result[path.relative_to(root).as_posix()] = {'bytes': path.stat().st_size, 'sha256': digest}
            else:
                raise ValueError('Unknown evidence entry')
    return result


def inactive(path: Path) -> None:
    needle = str(path).lower().replace('\\', '/')
    for proc in psutil.process_iter(['exe', 'cmdline']):
        try:
            exe = (proc.info['exe'] or '').lower().replace('\\', '/')
            cmd = ' '.join(proc.info['cmdline'] or []).lower().replace('\\', '/')
            assert needle not in exe and needle not in cmd, 'Worktree is in use'
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    assert git(ROOT, 'rev-parse', '--abbrev-ref', 'HEAD') == 'main'
    assert git(ROOT, 'rev-parse', 'HEAD') == git(ROOT, 'rev-parse', 'origin/main'), 'Push first'
    remote = git(ROOT, 'ls-remote', 'origin', 'refs/heads/main').split()[0]
    assert remote == git(ROOT, 'rev-parse', 'HEAD'), 'Remote main differs'
    common = git(ROOT, 'rev-parse', '--path-format=absolute', '--git-common-dir')
    rows = []
    for name, expected in CANDIDATES.items():
        path = PARENT / name
        assert path.is_dir() and not path.is_junction() and not path.is_symlink()
        assert path.resolve().parent == PARENT.resolve()
        assert git(path, 'rev-parse', '--path-format=absolute', '--git-common-dir') == common
        assert git(path, 'rev-parse', 'HEAD') == expected, 'Worktree advanced'
        assert not git(path, 'status', '--porcelain', '--untracked-files=all'), 'Unsaved work remains'
        subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', expected, remote], check=True)
        inactive(path)
        rows.append({'name': name, 'head': expected, 'ignored': git(path, 'status', '--short', '--ignored'), 'removed': False})
    if args.execute:
        assert BACKUP.parent.is_dir() and not BACKUP.exists(), 'Fresh preservation directory required'
        BACKUP.mkdir()
        sys.path.insert(0, str(ROOT))
        from terminal.strata_control import private_files
        private_files.restrict_to_current_user(BACKUP, directory=True)
        source = PARENT / 'native-32gb16gb-20261106/_docs/native-32gb16gb'
        expected = inventory(source)
        target = BACKUP / 'native-32gb16gb-evidence'
        shutil.copytree(source, target)
        assert inventory(target) == expected and inventory(source) == expected, 'Evidence preservation mismatch'
        with (BACKUP / 'evidence-inventory.json').open('x', encoding='utf-8') as stream:
            json.dump(expected, stream, indent=2)
        print(json.dumps({'preservedFiles': len(expected), 'preservedBytes': sum(v['bytes'] for v in expected.values()), 'backup': str(BACKUP)}), flush=True)
        for row in rows:
            path = PARENT / row['name']
            inactive(path)
            assert git(path, 'rev-parse', 'HEAD') == row['head']
            assert not git(path, 'status', '--porcelain', '--untracked-files=all')
            subprocess.run(['git', '-C', str(ROOT), 'worktree', 'remove', str(path)], check=True, timeout=300)
            row['removed'] = not path.exists()
            with (BACKUP / 'cleanup.json').open('w', encoding='utf-8') as stream:
                json.dump({'remoteMain': remote, 'worktrees': rows}, stream, indent=2)
            print(json.dumps({'name': row['name'], 'removed': row['removed']}), flush=True)
    else:
        print(json.dumps({'execute': False, 'remoteMain': remote, 'worktrees': rows}, indent=2))


if __name__ == '__main__':
    main()
