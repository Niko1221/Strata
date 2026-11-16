"""Retain fork versions and refresh independent official release metadata."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('H:/Strata/evidence/upstream-sync-20261116/hotfix-intake-v1')
FORK = '394234d911fc2056af7a9bb2b38df75f94bf0a6a'
OFFICIAL = 'baf16363b4403ef0ed4f538f5d5ed11c43fb9535'

def git(*args: str) -> bytes:
    return subprocess.check_output(['git', '-C', str(ROOT), *args], timeout=90)

assert git('rev-parse', 'HEAD').decode().strip() == FORK
assert git('rev-parse', 'MERGE_HEAD').decode().strip() == OFFICIAL
assert not (OUT / 'metadata-resolution.json').exists()
# The entire upstream CMake delta is the independent Control version bump.
# Preserve the already reviewed fork source bytes, including native version 0.6.0.
cmake = git('show', FORK + ':CMakeLists.txt')
assert b'project(strata VERSION 0.6.0 ' in cmake
(ROOT / 'CMakeLists.txt').write_bytes(cmake)
native_build = json.loads((OUT.parent / 'cuda-build/Release/native-build.json').read_text())
assert hashlib.sha256(cmake).hexdigest() == native_build['sources']['CMakeLists.txt']
metadata = json.loads(git('show', FORK + ':docs/data/release.json'))
release = json.loads(subprocess.check_output(['C:/Program Files/GitHub CLI/gh.exe', 'api',
                    'repos/Niko1221/Strata/releases/latest'], timeout=90))
assert release['tag_name'] == 'v0.2.14'
snapshot = {'tag': release['tag_name'], 'published_at': release['published_at'],
            'url': release['html_url'], 'assets': [row['name'] for row in release['assets']]}
with (OUT / 'upstream-release-final.json').open('x', encoding='utf-8') as stream:
    json.dump(snapshot, stream, indent=2)
metadata['terminal_version'] = '0.6.0'
metadata['upstream'].update(snapshot)
metadata['upstream'].update(product_version='0.2.14', terminal_version='0.2.14.1',
                            control_version='0.1.60.1', engine_version='0.1.60', source_commit=OFFICIAL)
(ROOT / 'docs/data/release.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
rows = {}
for name in ('CMakeLists.txt', 'docs/data/release.json'):
    rows[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
git('add', '--', *rows)
with (OUT / 'metadata-resolution.json').open('x', encoding='utf-8') as stream:
    json.dump({'fork': FORK, 'official': OFFICIAL, 'files': rows,
               'nativeSourceBytesUnchanged': True, 'officialAssetCount': len(snapshot['assets'])}, stream, indent=2)
print(json.dumps({'ok': True, 'officialAssetCount': len(snapshot['assets']), 'nativeSourceBytesUnchanged': True}))
