#!/usr/bin/env python3
"""Verify the preserved header patch and exercise its original host-only tests."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
HEADER = 'include/strata/core/exchange_storage.hpp'

def main():
    manifest = json.loads((ROOT / 'provenance.json').read_text())
    for tree, key in [('before', 'before_sha256'), ('after', 'after_sha256')]:
        actual = hashlib.sha256((ROOT / (tree + '-exchange_storage.hpp')).read_bytes()).hexdigest()
        if actual != manifest['files'][HEADER][key]:
            raise RuntimeError(f'{tree} header hash mismatch')
    patch = (ROOT / 'historical-bucket-rotation.patch').read_text()
    # The first file is the self-contained ownership header.
    end = patch.index('--- a/include/strata/core/expert_source.hpp\n')
    with tempfile.TemporaryDirectory(prefix='strata-exchange-reference-') as temporary:
        work = Path(temporary)
        header = work / 'patched' / HEADER
        header.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / 'before-exchange_storage.hpp', header)
        subprocess.run(['patch', '-p1', '--fuzz=0'], input=patch[:end], text=True,
                       cwd=work / 'patched', check=True)
        if (work / 'patched' / HEADER).read_bytes() != (ROOT / 'after-exchange_storage.hpp').read_bytes():
            raise RuntimeError('patched header differs from the tested candidate')
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        flags = shlex.split(os.environ.get('CXXFLAGS', '-O2'))
        for source in sorted(ROOT.glob('exchange_*test.cpp')):
            binary = work / source.stem
            subprocess.run(compiler + ['-std=c++17', '-pthread'] + flags +
                           ['-I' + str(work / 'patched' / 'include'), str(source), '-o', str(binary)],
                           check=True, timeout=120)
            subprocess.run([str(binary)], check=True, timeout=180)
            print('PASS', source.name, flush=True)
    print('PASS: header provenance, patch round trip, four host tests')

if __name__ == '__main__':
    main()
