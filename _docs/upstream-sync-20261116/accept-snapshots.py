"""Copy complete, actually driven snapshots; preserve all immutable three-way inputs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('H:/Strata/evidence/upstream-sync-20261116')
sys.path.insert(0, str(ROOT))
from tests.snapshot.matrix import ENTRY_POINTS, MACHINES, MODEL_SETS


def main():
    out = BASE / 'snapshot-resolution'
    if out.exists():
        raise FileExistsError(out)
    generated = BASE / 'snapshot-all-v1'
    receipt = json.loads((BASE / 'receipts/snapshot-generate-all-v1.json').read_text())
    assert receipt['passed'] and receipt['run']['exit_code'] == 0
    expected = {f'{m.key}/{s.key}.json' for m in MACHINES for s in MODEL_SETS}
    actual = {p.relative_to(generated).as_posix() for p in generated.glob('*/*.json')}
    assert expected == actual
    rows, not_driven = [], []
    for relative in sorted(expected):
        raw = (generated / relative).read_bytes()
        cell = json.loads(raw)
        assert f'{cell["machine"]}/{cell["set"]}.json' == relative
        assert set(cell['entries']) == set(ENTRY_POINTS)
        for entry, record in cell['entries'].items():
            assert 'failed' not in record, (relative, entry, record)
            if 'not_driven' in record:
                assert entry == 'setup' and record['not_driven']
                not_driven.append({'cell': relative, 'entry': entry, 'reason': record['not_driven']})
        rows.append({'cell': relative, 'sha256': hashlib.sha256(raw).hexdigest()})
    out.mkdir()
    refs = {'base': 'f33c8c30588ff00e24b6b63bad0bf6a14da19081',
            'fork': '6f13c7555a7d50289f0e3587ab780627c65a8000',
            'upstream': '09b603886925994d0d4461857b01321ef3aa431c'}
    for row in rows:
        relative = row['cell']
        row['inputs'] = {}
        for label, ref in refs.items():
            result = subprocess.run(['git', '-C', str(ROOT), 'show',
                                     f'{ref}:tests/snapshot/golden/{relative}'], capture_output=True)
            if result.returncode:
                row['inputs'][label] = {'ref': ref, 'status': 'ABSENT'}
                continue
            backup = out / label / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(result.stdout)
            row['inputs'][label] = {'ref': ref, 'sha256': hashlib.sha256(result.stdout).hexdigest()}
        target = ROOT / 'tests/snapshot/golden' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((generated / relative).read_bytes())
    (out / 'receipt.json').write_text(json.dumps({'cells': rows, 'notDriven': not_driven,
        'source': str(generated), 'generatorReceipt': receipt['command_log'],
        'verdict': 'Copied actual driver output only; run agreement and repeat snapshot gates next.'}, indent=2))
    subprocess.run(['git', '-C', str(ROOT), 'add', 'tests/snapshot/golden'], check=True)
    print(json.dumps({'cells': len(rows), 'entries': len(rows)*len(ENTRY_POINTS),
                      'setupNotDriven': len(not_driven)}))


if __name__ == '__main__':
    main()
