"""Start missing non-publication checks on one already-pushed immutable candidate."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HEAD = '4282fce334e0d6c1843c74bca5d4063bd784bc05'
REF = 'codex/upstream-sync-20261116'
REPO = 'zapabob/Strata'
GH = 'C:/Program Files/GitHub CLI/gh.exe'
OUT = Path('H:/Strata/evidence/upstream-sync-20261116/candidate-ci-4282-v1.json')
WORKFLOWS = {'Checks': 'check.yml', 'Docs': 'docs.yml', 'Backend compile checks': 'backends.yml',
             'Native preview CPU checks': 'native-preview.yml', 'Metal build': 'metal-build.yml',
             'Portable monitor client': 'terminal-portable-monitor.yml',
             'Minimal serving runtime': 'terminal-minimal-server.yml', 'Config snapshot': 'snapshot.yml'}

def gh(*args: str) -> str:
    return subprocess.check_output([GH, *args, '--repo', REPO], cwd=ROOT, text=True, timeout=120).strip()

assert not OUT.exists()
remote = subprocess.check_output(['git', '-C', str(ROOT), 'ls-remote', 'origin', 'refs/heads/' + REF], text=True, timeout=60)
assert remote.split()[0] == HEAD
runs = json.loads(gh('run', 'list', '--commit', HEAD, '--limit', '60', '--json',
                     'databaseId,headSha,workflowName,status,conclusion,url'))
record = {'sourceCommit': HEAD, 'ref': REF, 'observedInitialRuns': runs, 'dispatches': [],
          'publicationWorkflowStarted': False, 'qualification': 'IN_PROGRESS'}
OUT.write_text(json.dumps(record, indent=2), encoding='utf-8')
seen = {row['workflowName'] for row in runs}
for name, file in WORKFLOWS.items():
    if name not in seen:
        gh('workflow', 'run', file, '--ref', REF)
        record['dispatches'].append({'name': name, 'workflow': file, 'requested': True})
        OUT.write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps({'sourceCommit': HEAD, 'alreadyPresent': sorted(seen),
                  'requested': [row['name'] for row in record['dispatches']], 'qualification': 'IN_PROGRESS'}))
