"""Stage only the verified Node security runtime in this isolated worktree."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('H:/Strata/evidence/upstream-sync-20261116')
SOURCE = BASE / 'node-24.22.3-v1'
TARGET = ROOT / 'apps/desktop/src-tauri/resources/node'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    receipt = BASE / 'node-stage-v1.json'
    if receipt.exists():
        raise FileExistsError(receipt)
    assert TARGET.resolve() == Path('H:/Strata/worktrees/upstream-sync-20261116/apps/desktop/src-tauri/resources/node')
    for path in (TARGET, *TARGET.parents):
        assert not path.is_symlink() and not path.is_junction(), 'Linked stage destination refused'
    verified = json.loads((SOURCE / 'receipt.json').read_text())
    files = {'node.exe': verified['nodeSHA256'], 'LICENSE': verified['licenseSHA256']}
    for name, digest in files.items():
        assert sha(SOURCE / name) == digest
        if (TARGET / name).exists():
            raise FileExistsError('Retain earlier stage: ' + str(TARGET / name))
    TARGET.mkdir(parents=True, exist_ok=True)
    for name, digest in files.items():
        shutil.copy2(SOURCE / name, TARGET / name)
        assert sha(TARGET / name) == digest
    version = subprocess.check_output([str(TARGET / 'node.exe'), '--version'], text=True).strip()
    assert version == verified['version']
    receipt.write_text(json.dumps({'version': version, 'files': files, 'target': str(TARGET),
        'isolatedWorktreeOnly': True, 'installedRuntimeChanged': False, 'source': verified}, indent=2))
    print(json.dumps({'version': version, 'target': str(TARGET), 'files': files}, indent=2))


if __name__ == '__main__':
    main()
