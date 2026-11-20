"""Archive obsolete build caches, verify every archived byte, then remove the exact cache."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from terminal.strata_control.private_files import private_mkdir, restrict_to_current_user

OUT = Path('H:/Strata/releases/stable-058-20261111-codex/cleanup')
BACKUP = Path('H:/Strata/private-backups/obsolete-builds-058-20261111')
SEVEN = Path('C:/Program Files/7-Zip/7z.exe')
CANDIDATES = [Path('H:/Strata/' + name) for name in (
    'releases/personal-052-local2-20261107/native-build-configure-failed-r1',
    'releases/personal-052-local2-20261107/native-build',
    'releases/local-053-20261109-opencode/native-build',
    'releases/local-053-20261109-opencode-configure-failed-r1/native-build',
    'releases/strata-0.5.0/native-build-r1',
    'builds/cuda135-comparison-20261106/cuda-13.5-vs-r2',
    'builds/cuda135-comparison-20261106/cuda-13.5',
    'builds/cuda135-comparison-20261106/cuda-13.3',
)]

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def validate(path):
    assert path in CANDIDATES and path.is_absolute() and path.resolve() == path
    assert path.is_relative_to(Path('H:/Strata/releases')) or path.is_relative_to(Path('H:/Strata/builds'))
    for part in (path, *path.parents):
        assert not part.is_symlink() and not part.is_junction()
    for proc in psutil.process_iter(['cwd', 'exe']):
        for value in proc.info.values():
            if isinstance(value, str) and value:
                assert not Path(value).resolve().is_relative_to(path), 'Build is still in use'

def inventory(path, hashes=False):
    rows = {}
    for base, dirs, names in os.walk(path, followlinks=False):
        for name in dirs + names:
            item = Path(base) / name
            assert not item.is_symlink() and not item.is_junction()
        for name in names:
            item = Path(base) / name
            rows[item.relative_to(path).as_posix()] = {'bytes': item.stat().st_size, **({'sha256': sha(item)} if hashes else {})}
    return rows

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('action', choices=('plan', 'archive-remove'))
parser.add_argument('--index', type=int)
parser.add_argument('--execute', action='store_true')
args = parser.parse_args()
if args.action == 'plan':
    rows = []
    for index, path in enumerate(CANDIDATES):
        if not path.is_dir():
            continue
        validate(path)
        files = inventory(path)
        rows.append({'index': index, 'path': str(path), 'files': len(files), 'bytes': sum(row['bytes'] for row in files.values())})
    with (OUT / 'obsolete-builds-plan.json').open('x', encoding='utf-8') as stream:
        json.dump(rows, stream, indent=2)
    print(json.dumps(rows))
else:
    assert args.execute and args.index is not None and 0 <= args.index < len(CANDIDATES)
    path = CANDIDATES[args.index]
    validate(path)
    assert path.is_dir()
    rows = inventory(path, hashes=True)
    assert shutil.disk_usage(path).free > sum(row['bytes'] for row in rows.values()) + 10 * 1024**3
    private_mkdir(BACKUP)
    restrict_to_current_user(BACKUP, directory=True)
    archive = BACKUP / (str(args.index) + '-' + path.parent.name + '-' + path.name + '.7z')
    assert not archive.exists()
    subprocess.run([str(SEVEN), 'a', '-t7z', '-mx1', '-mmt1', '-md16m', '-bd', '-bso0', str(archive), path.name],
                   cwd=path.parent, check=True, timeout=1800)
    restrict_to_current_user(archive)
    listing = subprocess.check_output([str(SEVEN), 'l', '-slt', '-ba', '-sccUTF-8', str(archive)], timeout=120).decode('utf-8')
    members = []
    for block in listing.replace('\r\n', '\n').split('\n\n'):
        values = dict(line.split(' = ', 1) for line in block.splitlines() if ' = ' in line)
        if 'Path' in values and values.get('Folder') != '+' and not values.get('Attributes', '').startswith('D'):
            name = values['Path'].replace('\\', '/')
            prefix = path.name + '/'
            assert name.startswith(prefix)
            relative = name[len(prefix):]
            assert relative in rows and int(values['Size']) == rows[relative]['bytes']
            members.append(relative)
    assert len(members) == len(rows) and set(members) == set(rows)
    child = subprocess.Popen([str(SEVEN), 'x', '-so', '-bsp0', '-bse2', str(archive)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for name in members:
        left = rows[name]['bytes']
        digest = hashlib.sha256()
        while left:
            data = child.stdout.read(min(left, 1024**2))
            assert data, 'Truncated archive stream'
            left -= len(data)
            digest.update(data)
        assert digest.hexdigest() == rows[name]['sha256'], 'Archived file differs: ' + name
    assert not child.stdout.read(1) and child.wait(timeout=120) == 0
    validate(path)
    assert inventory(path, hashes=True) == rows, 'Cache changed during archive'
    receipt = {'path': str(path), 'archive': str(archive), 'archiveSHA256': sha(archive), 'archiveBytes': archive.stat().st_size,
               'verifiedFiles': len(rows), 'originalBytes': sum(row['bytes'] for row in rows.values()), 'files': rows}
    with (OUT / f'obsolete-build-{args.index}-verified.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
    # The resolved absolute allowlisted path and every member were checked again.
    shutil.rmtree(path)
    assert not path.exists()
    with (OUT / 'obsolete-builds-removed.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({key: value for key, value in receipt.items() if key != 'files'}) + '\n')
    print(json.dumps({'removed': str(path), 'archive': str(archive), 'verifiedFiles': len(rows)}))
