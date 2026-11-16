"""Restore retained engine-setting records without changing any executable default."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('H:/Strata/evidence/upstream-sync-20261116/settings-restoration-v1.json')
sys.path.insert(0, str(ROOT / 'tools'))
import settings_register as sr

assert not OUT.exists()
current = json.loads(sr.REGISTER.read_text(encoding='utf-8'))
saved = json.loads(subprocess.check_output(['git', '-C', str(ROOT), 'show',
                   '6f13c7555a7d50289f0e3587ab780627c65a8000:data/settings-register.json'], timeout=60))
old = {entry['name']: entry for entry in saved['settings']}
have = {entry['name'] for entry in current['settings']}
missing = sorted(sr.engine_names() - have)
restored, unclassified = [], []
for name in missing:
    if name in old:
        current['settings'].append(old[name])
        restored.append(name)
    else:
        if name in ('STRATA_Q36_PHASE_METRICS', 'STRATA_TEST_NAN_STOP_AT'):
            entry = {'name': name, 'status': 'research', 'owner': 'fork diagnostics',
                     'note': 'Explicit diagnostic/test-only control. Off by default; not a model-quality or total-process-bound claim.'}
        else:
            entry = {'name': name, 'status': 'unsure', 'owner': 'retained fork extension',
                     'note': 'Retained source setting. Classification needs review; no Windows DeepSeek runtime qualification.'}
        current['settings'].append(entry)
        unclassified.append(name)
sr.write(current)
updated = sr.build(with_git=False)
sr.write(updated)
(ROOT / sr.SETTINGS_PAGE).write_text(sr.render_page(updated['settings']), encoding='utf-8', newline='\n')
with OUT.open('x', encoding='utf-8') as stream:
    json.dump({'restoredFromFrozenFork': restored, 'explicitlyClassifiedOrUnclassified': unclassified,
               'missingBefore': missing, 'settingsAfter': len(updated['settings']), 'runtimeDefaultsChanged': False}, stream, indent=2)
print(json.dumps({'restored': len(restored), 'newRecords': unclassified, 'settingsAfter': len(updated['settings']),
                  'runtimeDefaultsChanged': False}))
