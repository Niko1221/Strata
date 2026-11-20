"""Approved Stop/unload and same-settings restore through the installed Host only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = Path('H:/Strata/private-backups/native-diagnostics-20261109')
sys.path[:0] = [str(ROOT), str(ROOT / '_docs/local-install-20261106')]
from installed_state import endpoint, get, shells
from terminal.strata_control import private_files


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name: str, value: dict) -> None:
    with (PRIVATE / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)


def post(base: str, route: str, value: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(base + route, data=json.dumps(value).encode(),
        method='POST', headers={'Content-Type': 'application/json', 'Origin': base, 'x-strata-client': 'ui'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def until(check, seconds: int):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        result = check()
        if result:
            return result
        time.sleep(1)
    raise RuntimeError('Normal Host operation did not settle before the deadline')


def pause(base: str, shell) -> None:
    assert PRIVATE.parent.is_dir() and not PRIVATE.exists(), 'Fresh private evidence required'
    state = get(base, '/api/native')
    instances = state.get('instances', [])
    assert len(instances) == 1 and instances[0]['model'] == 'swift-iq2_xs'
    instance = instances[0]
    assert instance['status'] == 'ready'
    assert not [j for j in state['jobs'] if j['state'] in ['running', 'cancelling', 'queued']]
    turns = get(base, '/api/chat/active')['turns']
    assert len(turns) <= 1, 'Do not cancel a new unrelated conversation'
    source = Path(get(base, f"/api/native/models/{instance['model']}/settings-file")['path'])
    assert re.fullmatch(r'[0-9a-f-]{36}', instance['id']), 'Unexpected native instance ID'
    # The public DTO deliberately omits the private run-config path. Resolve the
    # exact instance-owned file using the installed Host's native-instances layout.
    run = Path('C:/Users/downl/.strata/data/native-instances') / (instance['id'] + '.json')
    assert run.is_file() and not run.is_symlink() and not run.is_junction()
    before = source.read_bytes()
    PRIVATE.mkdir()
    private_files.restrict_to_current_user(PRIVATE, directory=True)
    snapshot = {'shellPid': shell.pid, 'shellCreated': shell.create_time(), 'base': base,
        'instance': instance, 'sourceConfig': str(source), 'sourceSHA256': sha(source),
        'sourceBytes': before.decode('utf-8-sig'), 'runtimeConfig': json.loads(run.read_text(encoding='utf-8-sig')),
        'restoreArgs': {'model': instance['model'], 'multiple': False}, 'turns': turns}
    write('snapshot.json', snapshot)
    report = {'ok': False, 'normalApiOnly': True, 'stopRequests': 0, 'unloaded': False}
    try:
        for turn in turns:
            # The UI's Stop API addresses the chat ID, not the underlying Harness session.
            chat = turn.get('chatId')
            assert isinstance(chat, str) and chat
            post(base, '/api/chat/stop', {'sessionId': chat})
            report['stopRequests'] += 1
        until(lambda: not get(base, '/api/chat/active')['turns']
              and get(base, '/api/serving')['stats']['running'] == 0
              and get(base, '/api/serving')['stats']['waiting'] == 0, 180)
        assert sha(source) == snapshot['sourceSHA256'], 'Concurrent source settings change'
        current = get(base, '/api/native')['instances']
        assert len(current) == 1 and current[0]['id'] == instance['id']
        post(base, f"/api/native/instances/{instance['id']}/unload", {})
        until(lambda: not get(base, '/api/native')['instances'], 60)
        assert sha(source) == snapshot['sourceSHA256']
        report.update(ok=True, unloaded=True, sourceSettingsUnchanged=True)
    finally:
        write('pause.json', report)
        print(json.dumps(report), flush=True)


def restore(base: str) -> None:
    snapshot = json.loads((PRIVATE / 'snapshot.json').read_text(encoding='utf-8'))
    paused = json.loads((PRIVATE / 'pause.json').read_text())
    reconciled = PRIVATE / 'pause-reconciled.json'
    assert paused['ok'] or (reconciled.is_file() and json.loads(reconciled.read_text())['ok']), 'Pause did not settle'
    assert sha(Path(snapshot['sourceConfig'])) == snapshot['sourceSHA256'], 'Do not overwrite concurrent user edits'
    existing = get(base, '/api/native')['instances']
    if existing:
        # A user/normal workflow can reload the same model while update checks
        # finish. Verify that restoration; never start a duplicate or stop it.
        assert len(existing) == 1
        observed_id = existing[0]['id']
        assert re.fullmatch(r'[0-9a-f-]{36}', observed_id)
        def observed_restore():
            rows = get(base, '/api/native')['instances']
            assert len(rows) == 1 and rows[0]['id'] == observed_id, 'Model changed during restore observation'
            item = rows[0]
            for key in ['model', 'context', 'parallel', 'gpus', 'vision']:
                assert item.get(key) == snapshot['instance'].get(key), 'A different model/configuration is active'
            config = Path('C:/Users/downl/.strata/data/native-instances') / (observed_id + '.json')
            current = json.loads(config.read_text(encoding='utf-8-sig'))
            assert current.get('args') == snapshot['runtimeConfig'].get('args'), 'Runtime arguments differ'
            assert item['status'] != 'error', 'Externally started restoration failed'
            return item if item['status'] == 'ready' and item.get('phase') in ['idle', 'reading', 'thinking', 'generating'] else None
        item = until(observed_restore, 660)
        assert sha(Path(snapshot['sourceConfig'])) == snapshot['sourceSHA256']
        write('restored.json', {'ok': True, 'observedExternalRestore': True, 'loadRequestsSent': 0,
            'model': item['model'], 'context': item.get('context'), 'parallel': item.get('parallel'),
            'sourceSettingsUnchanged': True, 'runtimeArgsUnchanged': True, 'interruptedTurnResubmitted': False})
        print(json.dumps({'restored': True, 'observedExternalRestore': True, 'sameRuntimeArgs': True,
                          'sameModelContextParallel': True, 'loadRequestsSent': 0}), flush=True)
        return
    assert not get(base, '/api/chat/active')['turns'], 'A new user conversation is active'
    assert not get(base, '/api/native')['instances'], 'A model was loaded during diagnostics'
    job = post(base, '/api/native/operations', {'kind': 'load', 'args': snapshot['restoreArgs']})
    def finished():
        state = get(base, '/api/native')
        row = next((j for j in state['jobs'] if j['id'] == job['id']), None)
        return (row, state) if row and row['state'] in ['done', 'error', 'cancelled'] else None
    row, state = until(finished, 660)
    assert row['state'] == 'done', 'Same-settings model restore failed; inspect private Host diagnostics'
    instance = next(i for i in state['instances'] if i['model'] == snapshot['instance']['model'])
    assert instance['status'] == 'ready'
    assert instance.get('context') == snapshot['instance'].get('context')
    assert instance.get('parallel') == snapshot['instance'].get('parallel')
    assert sha(Path(snapshot['sourceConfig'])) == snapshot['sourceSHA256']
    write('restored.json', {'ok': True, 'model': instance['model'], 'context': instance.get('context'),
        'parallel': instance.get('parallel'), 'sourceSettingsUnchanged': True, 'interruptedTurnResubmitted': False})
    print(json.dumps({'restored': True, 'sameModelContextParallel': True, 'settingsUnchanged': True}), flush=True)


def main() -> None:
    global PRIVATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['pause', 'restore'])
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--label', default='initial')
    args = parser.parse_args()
    assert re.fullmatch(r'[a-z][a-z0-9-]{0,40}', args.label), 'Invalid evidence label'
    if args.label != 'initial':
        PRIVATE = PRIVATE.with_name(PRIVATE.name + '-' + args.label)
    if not args.execute:
        print(json.dumps({'wouldExecute': False, 'step': args.step})); return
    processes = shells()
    assert len(processes) == 1
    base = endpoint(processes[0])
    if args.step == 'pause': pause(base, processes[0])
    else: restore(base)


if __name__ == '__main__':
    main()
