"""Fetch only the pinned interpreter archive and hash-locked wheels; install nothing."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('H:/Strata/evidence/upstream-sync-20261116/control-inputs-v2')

def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main() -> None:
    assert not OUT.exists(), 'Retain previous input acquisition attempts'
    OUT.mkdir(parents=True)
    cache, wheels = OUT / 'python-cache', OUT / 'wheels'
    cache.mkdir()
    wheels.mkdir()
    spec = importlib.util.spec_from_file_location('control_input_pins', ROOT / 'packaging/package_terminal.py')
    pt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pt)
    assert pt.PYTHON_VERSION == '3.12.15'
    pin = pt.PYTHON_PINS['windows-x86_64']
    previous = OUT.with_name('control-inputs-v1') / 'python-cache' / pin['file'].replace('+', '_')
    assert digest(previous) == pin['sha256']
    shutil.copy2(previous, cache / previous.name)
    archive = pt.fetch_python('windows-x86_64', cache)
    assert digest(archive) == pin['sha256']
    lock = pt.lock_for('windows-x86_64')
    assert len(pt.lock_names(lock)) == 25
    # The same reviewed local-wheel URI selection used by install_locked().
    # --find-links alone may select a different public build of this version.
    vendor = ROOT / 'packaging/vendor/hakua-memory/hakua_memory-0.3.5-py3-none-any.whl'
    assert digest(vendor) == 'ff583d7f867cac1d9bca8e11b43cbd82705f7f6b06852722eae3f4e2f7de0cfa'
    resolved = re.sub(r'^hakua-memory==0[.]3[.]5(?=\s)',
                      lambda _match: 'hakua-memory @ ' + vendor.resolve().as_uri(),
                      lock.read_text(), flags=re.M)
    build_lock = OUT / 'runtime-resolved.txt'
    build_lock.write_text(resolved, encoding='utf-8')
    command = [sys.executable, '-m', 'pip', '--isolated', 'download', '--no-deps',
               '--require-hashes', '--only-binary=:all:', '--python-version', '3.12',
               '--implementation', 'cp', '--abi', 'cp312', '--platform', 'win_amd64',
               '--index-url', 'https://pypi.org/simple', '--dest', str(wheels), '-r', str(build_lock)]
    env = dict(os.environ)
    env.pop('PIP_NO_INDEX', None)
    subprocess.run(command, check=True, env=env, timeout=900)
    hashes = {path.name: digest(path) for path in wheels.iterdir()}
    approved = set(re.findall(r'--hash=sha256:([a-f0-9]{64})', lock.read_text()))
    assert len(hashes) == 25 and all(name.endswith('.whl') and checksum in approved for name, checksum in hashes.items())
    report = {'interpreter': {'path': str(archive), 'sha256': digest(archive)},
              'lockSHA256': digest(lock), 'wheelFiles': hashes, 'installed': False,
              'scope': 'Build inputs only. No downloaded interpreter or wheel executed.'}
    with (OUT / 'inputs.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({'ok': True, 'wheels': len(hashes), 'installed': False, 'out': str(OUT)}))

if __name__ == '__main__':
    main()
