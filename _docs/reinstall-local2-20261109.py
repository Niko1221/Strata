"""Explicit same-version overwrite using the byte-identical, already qualified local.2 installer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / '_docs/personal-install-20261107'),
               str(ROOT / '_docs/local-install-20261106'), str(ROOT / '_docs/native-maintenance-20261107')]
import common
import upgrade
from foundation import IdentityHandle
from installed_state import endpoint, get, shells
from terminal.strata_control import engine_component, private_files

ORIGINAL = common.OUT
OUT = Path('H:/Strata/releases/personal-052-local2-20261107/reinstall-20261109')
BACKUP = Path('H:/Strata/private-backups/reinstall-local2-20261109')
EXPECTED = 'be423e6955943f5f5689160a20e9e9a897e1b22b5eff38e13fa7d01179dad9cd'
COMPILED = '52048324c235e15ad34b95ec84a80dbad7e2c492'


def write(name: str, record: dict) -> None:
    with (OUT / name).open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)


def quiet(base: str) -> None:
    assert not get(base, '/api/chat/active')['turns'], 'A new user turn is active'
    stats = get(base, '/api/serving')['stats']
    assert stats['running'] == stats['waiting'] == 0, 'Requests are active'
    native = get(base, '/api/native')
    assert not native['instances'], 'Unload models through the normal API first'
    assert not any(j['state'] in ['running', 'cancelling', 'queued'] for j in native['jobs'])


def close_idle_app() -> None:
    apps = shells()
    if not apps:
        return
    assert len(apps) == 1
    app = apps[0]
    base = endpoint(app)
    for _ in range(3):
        quiet(base)
        time.sleep(1)
    hosts = [p for p in app.children() if Path(p.exe()).resolve() == (common.INSTALLED / 'node/node.exe').resolve()]
    assert len(hosts) == 1
    host = hosts[0]
    token = host.environ().get('STRATA_SHUTDOWN_TOKEN', '')
    assert len(token) >= 32, 'Real shell-held shutdown credential unavailable'
    with IdentityHandle(app.pid, terminate=True) as held:
        assert Path(held.exe).resolve() == (common.INSTALLED / 'strata-desktop.exe').resolve()
        quiet(base)
        request = urllib.request.Request(base + '/api/control/shutdown', data=b'{}', method='POST',
            headers={'Content-Type': 'application/json', 'Origin': base, 'x-strata-shutdown': token})
        with urllib.request.urlopen(request, timeout=30) as response:
            assert response.status == 202
        # Let the Host flush its stores and close its own children before ending the idle shell.
        host.wait(timeout=30)
        time.sleep(2)
        if held.alive():
            held.stop_exact()
    assert not shells()
    write('app-close.json', {'ok': True, 'hostShutdownApi': True, 'idleConfirmed': True,
                            'endedExactIdleShellOnly': True})


def execute() -> None:
    assert OUT.parent.is_dir() and BACKUP.parent.is_dir()
    assert not OUT.exists() and not BACKUP.exists(), 'Preserve prior update evidence'
    common.require_resources()
    build = common.load(ORIGINAL / 'app-build.json')
    qa = common.load(ORIGINAL / 'app-qualification.json')
    final = common.load(ORIGINAL / 'final-sanity.json')
    assert build['sourceCommit'] == qa['sourceCommit'] == COMPILED
    assert build['sha256'] == qa['installerSHA256'] == final['installerSHA256'] == EXPECTED
    assert Path(build['installer']).stat().st_size == 112090046
    assert common.sha(Path(build['installer'])) == EXPECTED
    assert qa['ok'] and qa['selfTests'] == 13 and qa['payloadFiles'] == 29476
    for name in ['functional-e2e.json', 'prism-e2e.json']:
        record = common.load(ORIGINAL / name)
        assert record['ok'] and record['sourceCommit'] == COMPILED and record['mocksUsed'] is False
    manifest = engine_component.read_manifest(upgrade.NEW_COMPONENT)
    assert engine_component.verify_signature(manifest, environ={}) == 'verified'
    engine_component.verify(manifest, full=True)
    native_before = common.inventory(upgrade.NEW_COMPONENT)
    OUT.mkdir()
    write('preflight.json', {'ok': True, 'checkpointCommit': common.git('rev-parse', 'HEAD'),
        'compiledSourceCommit': COMPILED, 'sameVersionOverwrite': True, 'installerSHA256': EXPECTED,
        'qualificationReusedForExactSameBytes': True, 'newKernelCandidateInstalled': False,
        'nativeManifestSHA256': common.sha(upgrade.NEW_COMPONENT / 'engine-component.json')})
    for name in ['app-build.json', 'app-qualification.json', 'qualified-payload.json']:
        shutil.copy2(ORIGINAL / name, OUT / name)
    close_idle_app()
    common.require_resources()
    upgrade.OUT = OUT
    upgrade.BACKUP = BACKUP
    upgrade.backup()
    upgrade.install()
    upgrade.verify_installed()
    assert common.inventory(upgrade.NEW_COMPONENT) == native_before, 'Existing signed native changed'
    manifest = engine_component.read_manifest(upgrade.NEW_COMPONENT)
    assert engine_component.verify_signature(manifest, environ={}) == 'verified'
    write('native-activated.json', {'ok': True, 'action': 'verified-existing-component',
        'activationRepeated': False, 'engineVersion': '0.5.2', 'signatureVerified': True,
        'componentFilesMatched': len(native_before), 'releaseQualified': False})
    write('install-complete.json', {'ok': True, 'sameVersionOverwrite': True,
        'productVersion': common.PRODUCT, 'controlVersion': common.CONTROL, 'nativeVersion': common.ENGINE,
        'compiledSourceCommit': COMPILED, 'checkpointCommit': common.git('rev-parse', 'HEAD'),
        'installerSHA256': EXPECTED, 'installerBytes': build['bytes'],
        'nsisExitCode': common.load(OUT / 'nsis-result.json')['exitCode'],
        'installedPayload': common.load(OUT / 'installed-payload.json'),
        'backup': common.load(OUT / 'backup.json'), 'nativeFilesUnchanged': len(native_before),
        'normalStartup': 'PENDING_SEPARATE_USER_HANDOFF', 'releaseQualified': False,
        'newKernelCandidateInstalled': False})
    print(json.dumps({'ok': True, 'nsisExitCode': 0,
        'payloadFilesMatched': common.load(OUT / 'installed-payload.json')['payloadFilesMatched'],
        'userFilesUnchanged': common.load(OUT / 'installed-payload.json')['userFilesUnchanged'],
        'nativeFilesUnchanged': len(native_before), 'productVersion': common.PRODUCT,
        'compiledSourceCommit': COMPILED, 'normalStartup': False}), flush=True)


def start_normal() -> None:
    complete = common.load(OUT / 'install-complete.json')
    assert complete['ok'] and complete['installerSHA256'] == EXPECTED
    upgrade.OUT = OUT
    upgrade.BACKUP = BACKUP
    upgrade.normal_start()
    manifest = engine_component.read_manifest(upgrade.NEW_COMPONENT)
    assert engine_component.verify_signature(manifest, environ={}) == 'verified'
    engine_component.verify(manifest, full=True)
    result = {**complete, 'normalStartup': common.load(OUT / 'normal-startup.json')}
    write('result.json', result)
    print(json.dumps({'ok': True, 'normalStartup': True, 'productVersion': common.PRODUCT,
                     'controlVersion': common.CONTROL, 'nativeVersion': common.ENGINE}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--start-only', action='store_true')
    args = parser.parse_args()
    if args.execute:
        if args.start_only: start_normal()
        else: execute()
    else:
        print(json.dumps({'wouldExecute': False, 'sameVersionOverwrite': True, 'expectedInstallerSHA256': EXPECTED}))


if __name__ == '__main__':
    main()
