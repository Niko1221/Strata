"""Recover disk space by reversible NTFS compression of this task's stopped build only."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

BASE = Path('H:/Strata/evidence/upstream-sync-20261116')
TARGET = BASE / 'cuda-build'


def inventory():
    assert TARGET.resolve() == Path('H:/Strata/evidence/upstream-sync-20261116/cuda-build')
    for path in (TARGET, *TARGET.parents):
        if path.is_symlink() or path.is_junction():
            raise ValueError('Linked compression target refused')
    records = {}
    for folder, dirs, files in os.walk(TARGET, followlinks=False):
        for name in dirs + files:
            path = Path(folder) / name
            if path.is_symlink() or path.is_junction():
                raise ValueError('Linked member refused')
        for name in files:
            path = Path(folder) / name
            with path.open('rb') as stream:
                records[path.relative_to(TARGET).as_posix()] = {
                    'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}
    return records


def main():
    out = BASE / 'owned-build-compression-v1'
    out.mkdir()
    build = json.loads((BASE / 'receipts/cuda-build-v1.json').read_text())
    assert build['run']['status'] == 'RESOURCE_PRESSURE' and build['run']['job_active_after'] == 0
    before = inventory()
    free_before = shutil.disk_usage(TARGET).free
    (out / 'before.json').write_text(json.dumps(before, indent=2))
    # This storage recovery action is not a build/test admission. It preserves
    # the disk floor that rejected the build and never retries that build here.
    command = [str(Path(os.environ['SYSTEMROOT']) / 'System32/compact.exe'),
               '/c', '/a', '/q', '/s:' + str(TARGET)]
    with (out / 'compact.log').open('xb') as log:
        done = subprocess.run(command, cwd=TARGET, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    after = inventory()
    report = {'target': str(TARGET), 'command': command, 'exitCode': done.returncode,
              'fileCount': len(before), 'logicalBytes': sum(row['bytes'] for row in before.values()),
              'allFileBytesUnchanged': before == after, 'deletedFiles': [],
              'freeBefore': free_before, 'freeAfter': shutil.disk_usage(TARGET).free,
              'scope': 'Only new task-owned stopped CUDA build; no source, model, old release or user profile touched.',
              'resourceFloorsChanged': False, 'reversal': 'compact.exe /u /a /q /s:<exact target> after ensuring enough space'}
    (out / 'receipt.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    assert done.returncode == 0 and before == after


if __name__ == '__main__':
    main()
