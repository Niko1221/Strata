"""Freeze exact unstaged intake source and retained guarded attempts for review."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('H:/Strata/private-backups/cleanup-20261116/upstream-c775-intake-v1')
EVIDENCE = Path('H:/Strata/evidence/upstream-sync-20261116')
PLAN = ROOT/'.superpowers/sdd/2026-11-16_upstream-sync-plan_Codex'


def git(*args, check=True):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, check=check, timeout=120)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def keep(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_bytes(raw)


preview = json.loads((OUT/'preview.json').read_text())
assert git('rev-parse','HEAD').stdout.decode().strip() == preview['refs']['fork']
assert not git('diff','--cached','--name-only').stdout.strip()
paths = set(preview['rows'])
paths.update(git('diff','--name-only').stdout.decode().splitlines())
untracked = git('ls-files','--others','--exclude-standard').stdout.decode().splitlines()
owned_new = [n for n in untracked if n in paths or n == '_docs/2026-11-16_upstream-c775-intake_Codex.md'
             or n.startswith('_docs/upstream-sync-20261116/') or n == 'apps/host/tests/helpers/verified-local-runtime.mjs']
assert set(untracked) <= set(owned_new), 'Unexpected untracked files require scope review'
paths.update(owned_new)
record = {'refs': preview['refs'], 'indexUnchanged': True, 'files': {}, 'receipts': {}}
for name in sorted(paths):
    raw = (ROOT/name).read_bytes()
    before = git('show', preview['refs']['fork']+':'+name, check=False)
    keep(OUT/'source-after'/name, raw)
    if before.returncode == 0:
        keep(OUT/'source-before-git'/name, before.stdout)
    record['files'][name] = {'afterBytes': len(raw), 'afterSHA256': sha(raw),
                            'beforeGitBytes': len(before.stdout) if before.returncode == 0 else None,
                            'beforeGitSHA256': sha(before.stdout) if before.returncode == 0 else None,
                            'physicalOriginal': preview['rows'].get(name, {}).get('originalSHA256')}
patch = git('diff','--binary','--full-index','--no-ext-diff','HEAD','--',*sorted(paths)).stdout
for name in sorted(owned_new):
    delta = git('diff','--no-index','--binary','--full-index','--','/dev/null',name,check=False)
    assert delta.returncode == 1, name
    patch += delta.stdout
keep(OUT/'intake.patch', patch)
for file in sorted((EVIDENCE/'receipts').glob('c775-*.json')):
    row = json.loads(file.read_text())
    if 'run' in row:
        assert row['run'].get('job_active_after') == 0, file.name
    else:
        assert row.get('passed') is False and row.get('error'), file.name
    record['receipts'][file.name] = {'passed': row.get('passed'), 'status': row.get('run',{}).get('status', 'NOT_LAUNCHED'),
                                   'exit': row.get('run',{}).get('exit_code'), 'ramPolicy': row.get('ramPolicy')}
for source in sorted((EVIDENCE/'receipts').glob('c775-*')):
    keep(OUT/'receipts'/source.name, source.read_bytes())
for source in sorted(EVIDENCE.glob('c775-*.xml')):
    keep(OUT/'junit'/source.name, source.read_bytes())
for source in sorted((EVIDENCE/'c775-native-cpu-v1').glob('*.exe')):
    keep(OUT/'cpu-test-binaries'/source.name, source.read_bytes())
candidate_build = OUT.parent/'native-c775-candidate-build-v1'
for name in ('before.json', 'after.json', 'run.py'):
    keep(OUT/'native-candidate-build'/name, (candidate_build/name).read_bytes())
for source in sorted((EVIDENCE/'receipts').glob('native-c775-candidate-build-v1*')):
    keep(OUT/'native-candidate-build'/source.name, source.read_bytes())
for source in sorted((OUT.parent/'native-c775-profile-v1').iterdir()):
    if source.is_file():
        keep(OUT/'proposed-native-measurements'/source.name, source.read_bytes())
for source in sorted((EVIDENCE/'receipts').glob('native-c775-profile-capture-v1*')):
    keep(OUT/'proposed-native-measurements'/source.name, source.read_bytes())
primary = OUT.parent/'official-0215-inventory-v1'
for name in ('evidence.json', 'strata-feed.json', 'strata-feed.json.sig'):
    keep(OUT/'official-release'/name, (primary/name).read_bytes())
for name in ('upstream-c775-intake-brief.md','upstream-c775-intake-report.md'):
    keep(OUT/name, (PLAN/name).read_bytes())
keep(OUT/'report.md', (ROOT/'_docs/2026-11-16_upstream-c775-intake_Codex.md').read_bytes())
record['patchSHA256'] = sha(patch)
keep(OUT/'source.json', json.dumps(record,indent=2).encode())
manifest = {p.relative_to(OUT).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())}
            for p in sorted(OUT.rglob('*')) if p.is_file()}
keep(OUT/'SHA256-MANIFEST.json', json.dumps(manifest,indent=2).encode())
print(json.dumps({'files':len(paths), 'patchSHA256':sha(patch), 'sourceSHA256':sha((OUT/'source.json').read_bytes()),
                  'manifestSHA256':sha((OUT/'SHA256-MANIFEST.json').read_bytes())}))
