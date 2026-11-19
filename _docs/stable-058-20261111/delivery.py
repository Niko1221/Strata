"""Explicit candidate extraction, private backup, standard NSIS upgrade and bundle checks."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import psutil

ROOT = Path('C:/Users/downl/Documents/ChatGPT/Strata')
OUT = Path('H:/Strata/releases/stable-058-20261111-codex')
OLD = Path('H:/Strata/releases/public-055-20261111-opencode/assets')
APP = Path('C:/Users/downl/AppData/Local/Programs/Strata')
EXTRACTED = OUT / 'extracted'
BACKUP = Path('H:/Strata/private-backups/stable-058-20261111-codex')
ASSETS = OUT / 'assets'
SEVEN = Path('C:/Program Files/7-Zip/7z.exe')
PRODUCT = '0.5.8'
ENGINE = 'strata-engine-0.5.8-windows-x86_64-cuda-cuda13.tar.gz'
CONTROL = 'strata-control-0.5.8-windows-x86_64.zip'
INSTALLER = f'Strata-{PRODUCT}-windows-x64-setup.exe'
ARCHIVE = f'Strata-{PRODUCT}-windows-x64.tar.7z'
MANAGED = ('control', 'host', 'node', 'harness', 'engine-bundle', 'local-backends', 'notices')
WEIGHTS = {'.gguf', '.bin', '.safetensors', '.pack', '.weights', '.q36epk'}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def inventory(root: Path) -> dict:
    require(root.is_dir() and not root.is_symlink() and not root.is_junction(), 'Linked or absent root refused')
    paths = []
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            path = Path(directory) / name
            require(not path.is_symlink() and not path.is_junction(), 'Linked payload refused')
        for name in names:
            path = Path(directory) / name
            paths.append(path)
    with ThreadPoolExecutor(max_workers=5) as pool:
        hashes = list(pool.map(sha, paths))
    return {path.relative_to(root).as_posix(): {'sha256': value, 'bytes': path.stat().st_size} for path, value in zip(paths, hashes)}


def copy_tree(source: Path, target: Path) -> None:
    """Bound independent file copies to five workers; never overwrite a previous attempt."""
    require(not target.exists() and not source.is_junction(), 'Fresh non-linked copy required')
    target.mkdir()
    paths = []
    for directory, dirs, names in os.walk(source, followlinks=False):
        for name in dirs + names:
            path = Path(directory) / name
            require(not path.is_symlink() and not path.is_junction(), 'Linked copy source refused')
        for name in dirs:
            (target / (Path(directory) / name).relative_to(source)).mkdir()
        paths.extend(Path(directory) / name for name in names)
    def copy(path: Path) -> None:
        shutil.copy2(path, target / path.relative_to(source))
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(copy, paths))


def payload(root: Path) -> dict:
    rows = {'strata-desktop.exe': {'sha256': sha(root / 'strata-desktop.exe'), 'bytes': (root / 'strata-desktop.exe').stat().st_size}}
    for name in MANAGED:
        if (root / name).exists():
            rows.update({name + '/' + path: value for path, value in inventory(root / name).items()})
    return rows


def quiescent() -> None:
    for process in psutil.process_iter(['exe']):
        try:
            path = Path(process.info['exe'] or '')
            require(not path.is_relative_to(APP), 'Close the installed Strata and its owned processes first')
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def user_files() -> dict:
    roots = [Path('C:/Users/downl/.strata'), Path('C:/Users/downl/AppData/Roaming/Strata'),
             Path('C:/Users/downl/AppData/Local/Strata'), Path('C:/Users/downl/Documents/Strata/Workspace')]
    rows = {}
    for index, root in enumerate(roots):
        if not root.exists():
            continue
        require(not root.is_symlink() and not root.is_junction(), 'Linked user root refused')
        for directory, dirs, names in os.walk(root, followlinks=False):
            for name in list(dirs):
                path = Path(directory) / name
                require(not path.is_symlink() and not path.is_junction(), 'Linked user folder refused')
                if path == Path('C:/Users/downl/AppData/Local/Strata/components'):
                    dirs.remove(name)
            for name in names:
                path = Path(directory) / name
                require(not path.is_symlink(), 'Linked user file refused')
                if path.suffix.lower() in WEIGHTS or '.q36epk' in name.lower():
                    continue
                rows[str(path)] = {'backup': f'user-{index}/' + path.relative_to(root).as_posix(), 'bytes': path.stat().st_size}
    with ThreadPoolExecutor(max_workers=5) as pool:
        hashes = list(pool.map(lambda name: sha(Path(name)), rows))
    for row, value in zip(rows.values(), hashes):
        row['sha256'] = value
    return rows


def extract() -> None:
    build = load(OUT / 'app-build.json')
    installer = Path(build['installer'])
    require(sha(installer) == build['installerSHA256'], 'Exact installer required')
    if not EXTRACTED.exists():
        subprocess.run([str(SEVEN), 'x', str(installer), '-o' + str(EXTRACTED), '-y'], check=True, timeout=600)
    else:
        # The prior 7z extraction completed; its subsequent serial hashing timed out.
        # Verify the existing bytes without rewriting or deleting that attempt.
        require(load(OUT / 'receipts/extract-self-test.json')['run']['status'] == 'TIMEOUT', 'Unexpected prior extraction')
    print('Hashing every extracted managed file with five bounded workers.', flush=True)
    expected = payload(EXTRACTED)
    unbundled = Path(build['shell']).read_bytes()
    require(hashlib.sha256(unbundled).hexdigest() == build['shellSHA256'], 'Unbundled shell changed')
    marker = b'__TAURI_BUNDLE_TYPE_VAR_UNK'
    require(unbundled.count(marker) == 1, 'Expected one official Tauri bundle marker')
    nsis_bytes = unbundled.replace(marker, b'__TAURI_BUNDLE_TYPE_VAR_NSS', 1)
    require((EXTRACTED / 'strata-desktop.exe').read_bytes() == nsis_bytes, 'Extracted shell differs beyond the official Tauri NSIS marker')
    binding = {'ok': True, 'unbundledSHA256': build['shellSHA256'], 'nsisSHA256': hashlib.sha256(nsis_bytes).hexdigest(),
               'onlyDifference': '__TAURI_BUNDLE_TYPE_VAR_UNK -> __TAURI_BUNDLE_TYPE_VAR_NSS', 'allOtherBytesIdentical': True,
               'source': 'https://docs.rs/crate/tauri-utils/2.10.2/source/src/platform.rs',
               'installerSHA256': build['installerSHA256']}
    save(OUT / 'nsis-shell-binding.json', binding)
    require(sha(EXTRACTED / 'harness/rustxng/rustxng.exe') == 'a509798f040722db4a3df9eebb3551437c69189ed0003bb2a0b62bf749cf6775', 'RustXNG differs in installer')
    require(load(EXTRACTED / 'harness/rustxng/SOURCE.json')['profile'] == 'full', 'Wrong search profile')
    require(not (EXTRACTED / 'engine-bundle').exists() and not (EXTRACTED / 'local-backends').exists(), 'GPU runtimes must remain separate')
    save(OUT / 'expected-payload.json', expected)
    home = OUT / 'self-test-home'
    home.mkdir()
    report = OUT / 'self-test.json'
    env = dict(os.environ)
    for name, folder in {'STRATA_HOME': '', 'STRATA_COMPONENTS_DIR': 'components', 'STRATA_SETTINGS_FILE': 'settings.json',
                         'STRATA_INSTALL_DIR': 'install', 'STRATA_DESKTOP_DATA_DIR': 'desktop', 'STRATA_PROFILE_DIR': 'profiles',
                         'STRATA_RUNTIME_DIR': 'runtime', 'STRATA_WORKSPACE': 'workspace', 'STRATA_MODELS_DIR': 'models',
                         'APPDATA': 'Roaming', 'LOCALAPPDATA': 'Local', 'DSH_HOME': 'harness', 'TEMP': 'temp', 'TMP': 'temp'}.items():
        env[name] = str(home / folder)
    for name in ('workspace', 'Roaming', 'Local', 'temp'):
        (home / name).mkdir()
    env.update(STRATA_NO_REPAIR_ACCESS='1', STRATA_NO_DIALOG='1', STRATA_NO_BUNDLED_ENGINE='1')
    subprocess.run([str(EXTRACTED / 'strata-desktop.exe'), '--self-test=' + str(report)], cwd=EXTRACTED, env=env, check=True, timeout=180)
    result = load(report)
    require(result.get('ok') is True or result.get('passed') is True, 'Packaged self-test did not pass')
    save(OUT / 'extraction.json', {'ok': True, 'files': len(expected), 'installerSHA256': build['installerSHA256'], 'selfTest': result})


def backup() -> None:
    quiescent()
    require(not BACKUP.exists(), 'Preserve the existing backup')
    BACKUP.mkdir()
    who = subprocess.check_output(['whoami.exe', '/user', '/fo', 'csv', '/nh'], text=True).strip()
    sid = next(csv.reader(io.StringIO(who)))[1]
    subprocess.run(['icacls.exe', str(BACKUP), '/inheritance:r', '/grant:r', f'*{sid}:(OI)(CI)F', '*S-1-5-18:(OI)(CI)F'], check=True, capture_output=True)
    print('Hashing user data with five bounded workers.', flush=True)
    user = user_files()
    print('Hashing the old installed payload.', flush=True)
    old = payload(APP)
    require(shutil.disk_usage(OUT).free > sum(row['bytes'] for row in user.values()) + sum(row['bytes'] for row in old.values()) + 10 * 1024**3, 'Keep the H: resource floor')
    save(BACKUP / 'source-inventory.json', {'user': user, 'installedManaged': old})
    def copy_user(item: tuple) -> None:
        source, row = item
        target = BACKUP / row['backup']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        require(sha(target) == row['sha256'], 'User backup differs')
    print('Copying verified user data and the old app.', flush=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(copy_user, user.items()))
    copy_tree(APP, BACKUP / 'installed-app')
    print('Verifying backup hashes and unchanged source membership.', flush=True)
    require(payload(BACKUP / 'installed-app') == old and user_files() == user, 'Backup membership or source changed')
    subprocess.run(['reg.exe', 'export', r'HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\Strata', str(BACKUP / 'uninstall.reg')], check=True, capture_output=True)
    save(BACKUP / 'inventory.json', {'user': user, 'installedManaged': old})
    save(OUT / 'backup.json', {'ok': True, 'path': str(BACKUP), 'files': len(user), 'bytes': sum(row['bytes'] for row in user.values()), 'modelsCopied': False})


def install() -> None:
    quiescent()
    prior = load(BACKUP / 'inventory.json')
    require(user_files() == prior['user'], 'User state changed since backup')
    expected = load(OUT / 'expected-payload.json')
    build = load(OUT / 'app-build.json')
    require(sha(Path(build['installer'])) == build['installerSHA256'], 'Installer changed')
    subprocess.run([build['installer'], '/S', '/D=' + str(APP)], check=True, timeout=600)
    require(payload(APP) == expected, 'Installed managed files differ from the exact extracted installer')
    require(user_files() == prior['user'], 'User files changed during installation')
    save(OUT / 'installed.json', {'ok': True, 'product': PRODUCT, 'files': len(expected), 'installerSHA256': build['installerSHA256'], 'userFilesUnchanged': len(prior['user']), 'exactPayloadMatch': True})


def bundle() -> None:
    require(load(OUT / 'installed.json')['ok'] and not ASSETS.exists(), 'Verified install and fresh asset folder required')
    ASSETS.mkdir()
    build = load(OUT / 'app-build.json')
    component_assets = OUT / 'component-assets'
    verified = load(OUT / 'components-verified.json')
    require(verified['ok'] and verified['sourceCommit'] == build['sourceCommit'], 'Fresh coordinated signed components required')
    ci = load(OUT / 'ci-verified.json')
    equivalence = ci.get('runtimeSourceEquivalence', {})
    require(ci['ok'] and ci['appSourceCommit'] == build['sourceCommit']
            and equivalence.get('ok') is True and equivalence['distributedRuntimeSourcesUnchanged']
            and equivalence['installerSHA256'] == build['installerSHA256'], 'CI must qualify unchanged exact distributed runtime sources')
    for name in [ENGINE, CONTROL, 'strata-feed.json', 'strata-feed.json.sig', 'install-manifest.txt', 'install.ps1', 'install.sh']:
        require(sha(component_assets / name) == verified['assets'][name]['sha256'] and (component_assets / name).stat().st_size == verified['assets'][name]['bytes'], 'Fresh component differs: ' + name)
        shutil.copy2(component_assets / name, ASSETS / name)
    shutil.copy2(Path(build['installer']), ASSETS / INSTALLER)
    for source, target in [('corresponding-source.tar.gz', 'RustXNG-0.31.1-source.tar.gz'), ('LICENSE', 'RustXNG-LICENSE.txt'), ('NOTICE', 'RustXNG-NOTICE.txt')]:
        shutil.copy2(EXTRACTED / 'harness/rustxng' / source, ASSETS / target)
    save(ASSETS / 'SOURCE.json', {**build, 'validatedSourceCommit': ci['sourceCommit'], 'runtimeSourceEquivalence': equivalence, 'unbundledShellSHA256': build['shellSHA256'], 'nsisShellBinding': load(OUT / 'nsis-shell-binding.json'), 'releaseTag': 'v' + PRODUCT, 'upstreamCommit': 'f33c8c30588ff00e24b6b63bad0bf6a14da19081', 'rustxng': load(EXTRACTED / 'harness/rustxng/SOURCE.json')})
    ui = load(OUT / 'installed-ui-verified.json')
    require(ui['ok'] and ui['actualInstalledExe'] and ui['mockModelOnly'], 'Installed WebView2 checks required')
    require(load(OUT / 'receipts' / ('installed-ui-' + ui['attempt'] + '.json'))['passed'], 'Contained UI run required')
    require(load(OUT / 'receipts/nsis-cleanup-final-r2.json')['passed'], 'NSIS cleanup scenarios required')
    final_native = load(OUT / 'native-final-verified.json')
    require(final_native['ok'] and final_native['engineSHA256'] == verified['native']['engineSHA256']
            and final_native['signedManifestSHA256'] == verified['native']['signedManifestSHA256'],
            'Complete native audit must qualify the exact locally installed component before publication')
    require(load(OUT / 'receipts/harness-group-r1.json')['passed'], 'The actual Harness regression group must pass')
    quality = {'releaseQualified': False, 'coordinatedSemVer': '0.5.8', 'components': verified,
               'cloudCIObservation': ci, 'completeNativeAuditAfterLocalInstallPreparation': final_native,
               'nativePriorEvidence': 'https://github.com/zapabob/Strata/releases/tag/v0.5.5-local.1',
               'expandedProfileContextGuard': {'passed': 43, 'failed': 0, 'skipped': 0,
                   'note': 'Actual pinned Harness, synthetic model API and isolated fixture profiles. Native model quality is not qualified by these tests.'},
               'packagedSelfTest': {'passed': 13, 'failed': 0},
               'nsisCleanupScenarios': {'passed': 6, 'failed': 0},
               'installedWebView2': {'passed': len(ui['checks']), 'checks': ui['checks'], 'consoleErrors': ui['consoleErrors'],
                                    'modelAPI': 'explicit local test fixture; does not qualify real-model or GPU performance', 'realRustXNG': True},
               'rustxng': {'version': '0.31.1', 'profile': 'full', 'realQueriesWithResults': 1},
               'installed': load(OUT / 'installed.json')}
    save(ASSETS / 'QUALITY.json', quality)
    verify_code = "import json,sys;sys.path.insert(0,sys.argv[1]);from terminal.strata_control import feed,signing;v=feed.fetch(sys.argv[2],signing.load_trusted_keys(environ={}));assert len(v.engine)==len(v.control)==1;print(json.dumps({'ok':True,'engine':len(v.engine),'control':len(v.control),'trust':'bundled public keys only'}))"
    verified = subprocess.run([str(EXTRACTED / 'control/python/python.exe'), '-I', '-B', '-c', verify_code,
                               str(EXTRACTED / 'control'), str(ASSETS)], check=True, capture_output=True, text=True, timeout=60)
    save(OUT / 'signed-feed-check.json', json.loads(verified.stdout))
    install_text = f'''# Hakua / Strata {PRODUCT} Windows stable fork release

Verify SHA256SUMS.txt before use. Extract 7z first, then the inner tar; verify
BUNDLE-SHA256SUMS.txt. Run {INSTALLER} after finishing requests, unloading models,
backing up data and quitting Strata including the tray icon. Updates use the
existing install folder. Models, chats and memories are outside managed files.

The app is a current-user Tauri NSIS package with private Node/Python/Control.
The GPU engine is separate. On the preparation screen choose I already have the
file and import {ENGINE} still compressed. Select a compatible model and Load.
Models and personal data are not included. Windows executable signing is absent;
the engine and Control feed have independent component signatures.

/btw delegates a separate tool-free question to a real subagent. In Skills, enable
workflow learning and automatic application independently. Review status and undo
are available. The same task model reviews redacted metadata in one extra request.
Manual edits, workspace scope and approvals are preserved. /deepresearch compares searches and pages;
incomplete evidence is disclosed. In Settings / Tools choose RustXNG Full (no key).
Full sends queries to enabled external providers; access must comply with their
conditions. Paid providers are not automatically enabled. General model and
research quality remain unqualified. This app is the 0.5.8 stable fork release. Historical releases and immutable tags remain available.

日本語: 7zを展開し、その中のtarを展開してください。内部チェックサムを照合し、
依頼の完了・モデルのアンロード・データのバックアップ・トレイを含むアプリ終了後に
上のEXEを実行します。準備画面で既存ファイルから署名付きエンジンを読み込んでください。
/btwは本題と別の質問、/deepresearchは出典付き調査です。設定のツールから
RustXNG Fullを選択できます。検索語は外部サービスへ送信され、利用条件が適用されます。
旧版は削除せず、問題があればモデルをアンロードして旧EXEとバックアップから戻します。

Workflow skills: https://github.com/zapabob/Strata/blob/main/docs/WORKFLOW_SKILLS.md
Hermes Agent: https://github.com/NousResearch/hermes-agent
Strata: https://github.com/Niko1221/Strata
llama.cpp: https://github.com/ggml-org/llama.cpp
RustXNG: https://github.com/sahenjp/RustXNG
RustXNG LICENSE/NOTICE and exact corresponding source are included separately,
and inside the installed harness/rustxng folder. These projects are appreciated;
this fork is maintained independently and implies no upstream endorsement.
'''
    (ASSETS / 'INSTALL.md').write_text(install_text, encoding='utf-8', newline='\n')
    members = {path.name: {'bytes': path.stat().st_size, 'sha256': sha(path)} for path in sorted(ASSETS.iterdir())}
    (ASSETS / 'BUNDLE-SHA256SUMS.txt').write_text(''.join(f'{row["sha256"]}  {name}\n' for name, row in members.items()), encoding='utf-8', newline='\n')
    members['BUNDLE-SHA256SUMS.txt'] = {'bytes': (ASSETS / 'BUNDLE-SHA256SUMS.txt').stat().st_size, 'sha256': sha(ASSETS / 'BUNDLE-SHA256SUMS.txt')}
    tar = OUT / f'Strata-{PRODUCT}-windows-x64.tar'
    with tarfile.open(tar, 'w', format=tarfile.PAX_FORMAT) as archive:
        for name in sorted(members):
            path = ASSETS / name
            info = tarfile.TarInfo(name)
            info.size = path.stat().st_size
            info.mode = 0o644
            info.mtime = 0
            with path.open('rb') as stream:
                archive.addfile(info, stream)
    subprocess.run([str(SEVEN), 'a', '-t7z', '-mx=0', '-mmt=1', str(ASSETS / ARCHIVE), str(tar)], check=True, timeout=300)
    # Read every member back through the actual 7z decoder, with bounded memory.
    child = subprocess.Popen([str(SEVEN), 'x', '-so', str(ASSETS / ARCHIVE)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    observed = {}
    with tarfile.open(fileobj=child.stdout, mode='r|') as archive:
        for info in archive:
            require(info.isfile() and info.name in members and info.name not in observed, 'Unexpected archive member')
            source = archive.extractfile(info)
            digest = hashlib.sha256()
            size = 0
            while chunk := source.read(1024*1024):
                digest.update(chunk)
                size += len(chunk)
            observed[info.name] = {'bytes': size, 'sha256': digest.hexdigest()}
    child.stdout.read()
    stderr = child.stderr.read()
    require(child.wait(timeout=30) == 0, '7z readback failed: ' + stderr.decode(errors='replace')[:200])
    require(observed == members, 'Streamed archive member hashes differ')
    sums = ''.join(f'{sha(path)}  {path.name}\n' for path in sorted(ASSETS.iterdir()))
    (ASSETS / 'SHA256SUMS.txt').write_text(sums, encoding='utf-8', newline='\n')
    records = {path.name: {'bytes': path.stat().st_size, 'sha256': sha(path)} for path in sorted(ASSETS.iterdir())}
    require(records[ARCHIVE]['bytes'] < 2*1024**3, 'Release asset exceeds 2 GiB')
    save(OUT / 'bundle-verified.json', {'ok': True, 'method': 'tar + 7z Copy mode', 'allMemberHashesMatched': True, 'members': members, 'assets': records})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['extract', 'backup', 'install', 'bundle'])
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'execute': False, 'action': args.action, 'output': str(OUT)}))
        return
    globals()[args.action]()
    print(json.dumps({'ok': True, 'action': args.action}))


if __name__ == '__main__':
    main()
