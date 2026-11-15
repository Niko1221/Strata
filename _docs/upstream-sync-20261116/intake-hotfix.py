"""Preview immutable text diff3 for the seven final upstream drafter fixes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('H:/Strata/evidence/upstream-sync-20261116/hotfix-intake-v1')
REFS = {'base': '09b603886925994d0d4461857b01321ef3aa431c',
        'fork': '394234d911fc2056af7a9bb2b38df75f94bf0a6a',
        'official': 'baf16363b4403ef0ed4f538f5d5ed11c43fb9535'}

def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(['git', '-C', str(ROOT), *args], check=check,
                          capture_output=True, timeout=120)

def preview() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    names = git('diff', '--name-only', REFS['base'], REFS['official']).stdout.decode().splitlines()
    assert len(names) == 12
    rows = {}
    for name in names:
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts and ':' not in name
        directory = OUT / relative
        directory.mkdir(parents=True)
        inputs = {}
        for label, ref in REFS.items():
            blob = git('show', ref + ':' + name).stdout
            normalized = blob.decode('utf-8').replace('\r\n', '\n').encode()
            target = directory / label
            target.write_bytes(normalized)
            inputs[label] = {'ref': ref, 'sha256': hashlib.sha256(blob).hexdigest()}
        result = git('merge-file', '-p', '--diff3', '-L', 'fork', '-L', 'base', '-L', 'official',
                     str(directory / 'fork'), str(directory / 'base'), str(directory / 'official'), check=False)
        assert 0 <= result.returncode <= 127
        (directory / 'candidate').write_bytes(result.stdout)
        rows[name] = {'inputs': inputs, 'conflicts': result.returncode,
                      'candidateSHA256': hashlib.sha256(result.stdout).hexdigest()}
    (OUT / 'preview.json').write_text(json.dumps({'refs': REFS, 'rows': rows, 'writesSource': False}, indent=2))
    print(json.dumps({name: row['conflicts'] for name, row in rows.items()}, indent=2))

def begin_merge() -> None:
    receipt = json.loads((OUT.parent / 'receipts/snapshot-final-v3.json').read_text())
    assert receipt['run']['status'] == 'FINISHED' and receipt['run']['job_active_after'] == 0
    assert git('rev-parse', 'HEAD').stdout.decode().strip() == REFS['fork']
    assert not git('status', '--porcelain', '--untracked-files=no').stdout.strip()
    result = git('merge', '--no-commit', '--no-ff', REFS['official'], check=False)
    assert result.returncode in (0, 1)
    (OUT / 'git-merge.stdout.txt').write_bytes(result.stdout)
    (OUT / 'git-merge.stderr.txt').write_bytes(result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(git('diff', '--name-only', '--diff-filter=U').stdout.decode())

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('action', choices=['preview', 'begin-merge'])
args = parser.parse_args()
if args.action == 'preview':
    preview()
else:
    begin_merge()
