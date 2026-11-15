"""Apply reviewed workflow diff3 rulings; retain strict checks and both Metal targets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('H:/Strata/evidence/upstream-sync-20261116')


def main():
    receipt = BASE / 'workflow-resolution.json'
    if receipt.exists():
        raise FileExistsError(receipt)
    check = (BASE / 'merge/.github/workflows/check.yml.normalized').read_text(encoding='utf-8')
    pattern = r'^<<<<<<< fork\n(.*?)^\|\|\|\|\|\|\| base\n.*?^=======\n.*?^>>>>>>> upstream\n'
    check, count = re.subn(pattern, lambda match: match[1], check, flags=re.M | re.S)
    assert count == 1 and 'PowerShell 7 parent (diagnostic)' in check
    assert '--strict --only' in check and 'check-macos-' in check
    backend = (ROOT / '.github/workflows/backends.yml').read_text(encoding='utf-8')
    backend, count = re.subn(r'^<<<<<<< HEAD\n(.*?)^=======\n.*?^>>>>>>> [a-f0-9]+\n',
                             lambda match: match[1], backend, flags=re.M | re.S)
    assert count == 2
    obsolete = ' -DSTRATA_GGML_DIR="$PWD/llama.cpp"'
    assert backend.count(obsolete) == 1
    backend = backend.replace(obsolete, '')
    assert 'macos-26-intel' in backend and '--target strata strata_ds4' in backend
    fork_agents = subprocess.check_output(['git', '-C', str(ROOT), 'show',
        '6f13c7555a7d50289f0e3587ab780627c65a8000:AGENTS.md'])
    assert (ROOT / 'AGENTS.md').read_bytes() == fork_agents
    records = []
    for relative, content in [('.github/workflows/check.yml', check),
                              ('.github/workflows/backends.yml', backend)]:
        assert not re.search(r'^(<<<<<<<|=======|>>>>>>>)', content, re.M)
        path = ROOT / relative
        before = path.read_bytes()
        path.write_bytes(content.encode('utf-8'))
        records.append({'path': relative, 'beforeSHA256': hashlib.sha256(before).hexdigest(),
                        'afterSHA256': hashlib.sha256(path.read_bytes()).hexdigest()})
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump({'records': records, 'rulings': [
            'Retain fork --strict checks and always-upload logs; include official PowerShell diagnostic.',
            'Retain arm64 cross-build of both Metal targets on Intel macOS 26.',
            'Use the new official vendored ggml, removing the nonexistent external checkout argument.',
            'Keep fork AGENTS unchanged; historical release records are not a new policy source.',
        ]}, stream, indent=2)
    subprocess.run(['git', '-C', str(ROOT), 'add', '.github/workflows/check.yml',
                    '.github/workflows/backends.yml', 'AGENTS.md'], check=True)


if __name__ == '__main__':
    main()
