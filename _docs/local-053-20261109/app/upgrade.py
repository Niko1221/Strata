"""One-shot overwrite, prelaunch identity gates, signed test-start, normal handoff."""
from __future__ import annotations

import time
from common import *

BACKUP = Path('H:/Strata/private-backups/local-053-20261109-opencode')
LOCAL = Path('C:/Users/downl/AppData/Local/Strata')
OLD_COMPONENT = LOCAL / 'components/engine/0.5.2-windows-x86_64-cuda-cuda13'
NEW_COMPONENT = OLD_COMPONENT.parent / SIGNED.name
MANAGED = ['control', 'host', 'node', 'harness', 'engine-bundle', 'local-backends', 'notices']
WEIGHTS = {'.gguf', '.bin', '.safetensors', '.pack', '.weights', '.q36epk'}


def user_hashes() -> dict:
    """Keep names private in memory; compare exact user membership/bytes around isolated QA."""
    result = {}
    roots = [Path('C:/Users/downl/.strata'), Path('C:/Users/downl/AppData/Roaming/Strata'), LOCAL]
    for root in roots:
        if not root.exists():
            continue
        require(not root.is_symlink() and not root.is_junction(), 'Linked user root')
        for directory, dirs, names in os.walk(root, followlinks=False):
            for name in list(dirs):
                path = Path(directory) / name
                require(not path.is_symlink() and not path.is_junction(), 'Linked user directory')
                if path == LOCAL / 'components/engine':
                    dirs.remove(name)
            for name in names:
                path = Path(directory) / name
                require(not path.is_symlink() and not path.is_junction(), 'Linked user file')
                if path.suffix.lower() not in WEIGHTS and '.q36epk' not in name.lower() and path.stat().st_size < 100 * 1024**2:
                    result[str(path)] = sha(path)
    return dict(sorted(result.items()))


def quiescent() -> None:
    require(process_counts() == dict(shell=0, native=0, llama=0), 'Parent must arrange a quiescent maintenance window')


def payload(app: Path) -> dict:
    require(app.is_dir() and not app.is_symlink() and not app.is_junction(), 'Linked install root refused')
    result = {'strata-desktop.exe': sha(app / 'strata-desktop.exe')}
    for name in MANAGED:
        if (app / name).exists():
            result.update({name + '/' + p: value for p, value in inventory(app / name).items()})
    return dict(sorted(result.items()))


def acceptance(commit: str, target: str = 'candidate') -> None:
    build = load(OUT / 'app-build.json')
    qa = load(OUT / 'app-qualification.json')
    require(build['sourceCommit'] == qa['sourceCommit'] == commit and qa['ok'], 'Fresh app qualification missing')
    require(sha(Path(build['installer'])) == qa['installerSHA256'] == build['sha256'], 'Installer drift')
    for kind, actual in [('q36', 'actualNative'), ('prism', 'actualPrism041')]:
        result = load(OUT / f'{target}-{kind}-e2e.json')
        require(result['ok'] and result['sourceCommit'] == commit and result['target'] == target, 'Target-specific GUI evidence missing')
        require(result[actual] is True and result['actualWebView2'] is True and result['actualShellAndHost'] is True and result['userHashesUnchanged'] is True and result['installedExeTest'] == (target == 'installed'), 'Actual target runtime required')
        if kind == 'q36':
            files = result.get('workspaceFilesAcceptance', {})
            require(files.get('ok') is True and files.get('noMocks') is True and files.get('isolatedWorkspace') is True, 'Actual sidebar file CRUD acceptance required')
            projection = result.get('projectionAcceptance', {})
            require(projection.get('ok') is True and projection.get('supported') is False and projection.get('positiveProjectionRuntimeTested') is False and projection.get('q38RuntimeQualified') is False, 'Actual Q36 negative projection acceptance required')
        expected = (INSTALLED if target == 'installed' else OUT / 'extracted') / 'strata-desktop.exe'
        require(Path(result['shellPath']).resolve() == expected.resolve() and result['shellSHA256'] == sha(expected), 'GUI shell identity drift')


def registration() -> dict:
    import winreg
    key = r'Software\Microsoft\Windows\CurrentVersion\Uninstall\Strata'
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as handle:
        values = {name: winreg.QueryValueEx(handle, name)[0] for name in ['InstallLocation', 'DisplayVersion', 'UninstallString']}
    script = "$ErrorActionPreference='Stop';$w=New-Object -ComObject WScript.Shell;$a=@();foreach($d in @([Environment]::GetFolderPath('Desktop'),[Environment]::GetFolderPath('Programs'))){$p=Join-Path $d 'Strata.lnk';if(!(Test-Path -LiteralPath $p)){throw 'missing shortcut'};$s=$w.CreateShortcut($p);$a+=@{path=$p;target=$s.TargetPath;arguments=$s.Arguments;workingDirectory=$s.WorkingDirectory}};ConvertTo-Json -InputObject $a -Compress"
    links = json.loads(subprocess.check_output(['powershell.exe', '-NoProfile', '-Command', script], text=True, encoding='utf-8', timeout=30))
    return dict(uninstall=values, shortcuts=links)


def backup() -> None:
    quiescent()
    require(not BACKUP.exists(), 'Fresh private backup required')
    old_payload = payload(INSTALLED)
    old_component = inventory(OLD_COMPONENT)
    before_registration = registration()
    before_user_inventory = user_hashes()
    BACKUP.mkdir()
    sys.path.insert(0, str(ROOT))
    from terminal.strata_control import private_files
    private_files.restrict_to_current_user(BACKUP, directory=True)
    sources = set()
    # Retain all small user files. Skip weights and managed engine bytes, which get an exact inventory.
    for root in [Path('C:/Users/downl/.strata'), Path('C:/Users/downl/AppData/Roaming/Strata'), LOCAL]:
        if not root.exists():
            continue
        require(not root.is_symlink() and not root.is_junction(), 'Linked user root')
        for directory, dirs, names in os.walk(root, followlinks=False):
            for name in list(dirs):
                path = Path(directory) / name
                require(not path.is_symlink() and not path.is_junction(), 'Linked user directory')
                if path == LOCAL / 'components/engine':
                    dirs.remove(name)
            for name in names:
                path = Path(directory) / name
                require(not path.is_symlink() and not path.is_junction(), 'Linked user file')
                if path.suffix.lower() not in WEIGHTS and '.q36epk' not in name.lower():
                    require(path.stat().st_size < 100 * 1024**2, 'Large non-weight user file requires parent review')
                    sources.add(path)
    config_roots = {LOCAL / 'install', Path('H:/Strata/models')}
    settings = Path('C:/Users/downl/.strata/data/native-host.json')
    if settings.exists() and load(settings).get('installDir'):
        config_roots.add(Path(load(settings)['installDir']))
    for root in config_roots:
        if root.is_dir():
            require(not root.is_symlink() and not root.is_junction(), 'Linked model config root')
            for directory, dirs, names in os.walk(root, followlinks=False):
                for name in dirs:
                    path = Path(directory) / name
                    require(not path.is_symlink() and not path.is_junction(), 'Linked model config directory')
                for name in names:
                    path = Path(directory) / name
                    if path.suffix.lower() in {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.verified'}:
                        require(not path.is_symlink() and not path.is_junction() and path.stat().st_size < 100 * 1024**2, 'Invalid model config')
                        sources.add(path)
    records = []
    for index, source in enumerate(sorted(sources)):
        target = BACKUP / 'user-files' / f'{index}-{source.name}'
        target.parent.mkdir(exist_ok=True)
        value = sha(source)
        shutil.copy2(source, target)
        require(sha(target) == value == sha(source), 'User backup mismatch')
        records.append(dict(source=str(source), backup=str(target), sha256=value, bytes=source.stat().st_size))
    for index, link in enumerate(before_registration['shortcuts']):
        shutil.copy2(link['path'], BACKUP / f'shortcut-{index}.lnk')
    subprocess.run(['reg.exe', 'export', r'HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\Strata', str(BACKUP / 'uninstall.reg')], check=True, capture_output=True, timeout=30)
    quiescent()
    require(all(sha(Path(r['source'])) == r['sha256'] for r in records), 'User data changed during backup')
    require(user_hashes() == before_user_inventory, 'User membership changed during backup')
    write_new(BACKUP / 'preflight.json', dict(target=str(INSTALLED), files=records, oldComponent=old_component,
              installedManaged=old_payload, registration=before_registration, userInventory=before_user_inventory, modelsCopied=False))
    write_new(OUT / 'backup.json', dict(ok=True, backup=str(BACKUP), preflightSHA256=sha(BACKUP / 'preflight.json'), files=len(records), modelsCopied=False))


def preflight() -> dict:
    record = load(OUT / 'backup.json')
    require(sha(BACKUP / 'preflight.json') == record['preflightSHA256'], 'Private preflight drift')
    pre = load(BACKUP / 'preflight.json')
    require(all(sha(Path(r['source'])) == r['sha256'] == sha(Path(r['backup'])) for r in pre['files']), 'User files changed since backup')
    require(user_hashes() == pre['userInventory'], 'User membership/bytes changed since backup')
    require(inventory(OLD_COMPONENT) == pre['oldComponent'], 'Existing native0.5.2 changed')
    return pre


def install() -> None:
    quiescent()
    pre = preflight()
    require(payload(INSTALLED) == pre['installedManaged'] and registration() == pre['registration'], 'Installed state changed since backup')
    build = load(OUT / 'app-build.json')
    require(sha(Path(build['installer'])) == build['sha256'], 'Installer drift')
    # Exclusive claim is written before spawning. Never retry this phase automatically.
    write_new(OUT / 'nsis-install.json', dict(ok=False, invocationClaimed=True, target=str(INSTALLED)))
    run = subprocess.run([build['installer'], '/S', '/D=' + str(INSTALLED)], cwd=OUT, env=build_env(), timeout=600, check=False)
    write_new(OUT / 'nsis-result.json', dict(ok=run.returncode == 0, exitCode=run.returncode, target=str(INSTALLED), installerSHA256=build['sha256']))
    require(run.returncode == 0, 'NSIS failed; retain evidence and review')


def verify_installed() -> None:
    quiescent()
    require(load(OUT / 'nsis-result.json')['ok'], 'Successful NSIS result required')
    pre = preflight()
    expected = load(OUT / 'qualified-payload.json')
    require(sha(OUT / 'qualified-payload.json') == load(OUT / 'app-qualification.json')['payloadInventorySHA256'], 'Qualified inventory drift')
    require(payload(INSTALLED) == expected, 'Installed membership/bytes differ')
    reg = registration()
    require(reg['uninstall']['InstallLocation'].strip('"').casefold() == str(INSTALLED).casefold() and reg['uninstall']['DisplayVersion'] == PRODUCT, 'Uninstall registration differs')
    for link in reg['shortcuts']:
        require(Path(link['target']).resolve() == (INSTALLED / 'strata-desktop.exe').resolve(), 'Shortcut target differs')
    version = subprocess.check_output(['powershell.exe', '-NoProfile', '-Command', "(Get-Item -LiteralPath '" + str(INSTALLED / 'strata-desktop.exe') + "').VersionInfo.ProductVersion"], text=True, timeout=30).strip()
    require(version == PRODUCT, 'Installed ProductVersion differs')
    write_new(OUT / 'installed-payload.json', dict(ok=True, beforeFirstLaunch=True, payloadFilesMatched=len(expected),
              userFilesUnchanged=len(pre['files']), oldComponentPreserved=True, registrationTargetCorrect=True, shortcutsChecked=len(reg['shortcuts']),
              shellSHA256=expected['strata-desktop.exe'], payloadInventorySHA256=sha(OUT / 'qualified-payload.json')))


def activate() -> None:
    quiescent()
    require(load(OUT / 'installed-payload.json')['ok'], 'Prelaunch installed verification required')
    preflight()
    require(not NEW_COMPONENT.exists(), 'Existing native0.5.3 must not be overwritten')
    native = staged()
    require(shutil.disk_usage('C:/').free >= 5 * 1024**3 + sum(p.stat().st_size for p in SIGNED.rglob('*') if p.is_file()) + 1024**2, 'Component import would consume C reserve')
    q36 = load(OUT / 'candidate-q36-e2e.json')
    valid = []
    for path in (Path(q36['home']) / 'data/native-instances').glob('*.json'):
        cfg = load(path)
        if cfg.get('q36_run', {}).get('backend', {}).get('binarySha256') == native['engineSHA256'] and cfg.get('q36_memory') == 'bounded-experts-v1':
            valid.append(path)
    require(bool(valid), 'Actually qualified bounded runtime config required')
    source = max(valid, key=lambda p: p.stat().st_mtime_ns)
    cfg = load(source)
    # Desktop memory principal credentials are not portable to the update sandbox; engine API auth stays enabled.
    for key in ['hakua_harness', 'log', 'mcp', 'mcp_servers', 'mcpServers', 'hakua_memory', 'memory_harness']:
        cfg.pop(key, None)
    command = cfg['args']
    require('--vram-reserve-mib' not in command and all(flag in command and command[command.index(flag) + 1] == value for flag, value in [('--max-context', '4096'), ('--prefill', '512'), ('--expert-slots', '5'), ('--mtp', '2'), ('--cache-type-k', 'f32'), ('--cache-type-v', 'f32')]), 'Qualified test settings differ')
    private = BACKUP / 'activation'
    private.mkdir()
    test = private / 'qualified-test-config.json'
    write_new(test, cfg)
    env = normal_app_env()
    env.update(STRATA_NO_ACQUIRE='1', STRATA_REQUIRE_SIGNED_COMPONENTS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', TEMP=str(OUT / 'temp'), TMP=str(OUT / 'temp'))
    py, cli = INSTALLED / 'control/python/python.exe', INSTALLED / 'control/strata.py'
    write_new(OUT / 'native-activation-claim.json', dict(claimed=True, explicitTestConfigSHA256=sha(test), testStartDisabled=False))
    run = subprocess.run([str(py), '-B', str(cli), 'components', 'install-file', native['archive'], '--model-config', str(test), '--output', 'jsonl'], cwd=OUT, env=env, capture_output=True, text=True, encoding='utf-8', timeout=600)
    with (private / 'install-output.txt').open('x', encoding='utf-8') as stream:
        stream.write(run.stdout + '\n' + run.stderr)
    write_new(OUT / 'native-activation-result.json', dict(ok=run.returncode == 0, exitCode=run.returncode, testStartDisabled=False))
    require(run.returncode == 0, 'Signed full test-start failed; inspect private diagnostics')
    require(inventory(NEW_COMPONENT) == inventory(SIGNED), 'Imported component differs')
    selected = subprocess.run([str(py), '-B', str(cli), 'components', 'engine', '--output', 'json'], cwd=OUT, env=env, capture_output=True, text=True, timeout=60)
    require(selected.returncode == 0, 'Selected engine inspection failed')
    value = json.loads(selected.stdout)
    require(value['usable'] and value['engine']['version'] == ENGINE and Path(value['engine']['root']).resolve() == NEW_COMPONENT.resolve(), 'Selection remains pinned; parent must review explicit components use with the same model config')
    require(inventory(OLD_COMPONENT) == load(BACKUP / 'preflight.json')['oldComponent'], 'Previous component changed')
    write_new(OUT / 'native-activated.json', dict(ok=True, signatureRequired=True, fullTestStartEnabled=True, explicitTestConfigSHA256=sha(test), engineVersion=ENGINE, engineSHA256=native['engineSHA256'], releaseQualified=False))


def normal_start() -> None:
    """Run outside all helper jobs, preserve normal settings, leave the app open unloaded."""
    require(load(OUT / 'native-activated.json')['ok'], 'Activation required')
    require(not (OUT / 'normal-startup.json').exists(), 'Normal handoff already recorded')
    acceptance(git('rev-parse', 'HEAD'), 'installed')
    require(payload(INSTALLED) == load(OUT / 'qualified-payload.json'), 'Installed payload drift')
    # Reuse the successful normal-launch implementation, never its old constants or acceptance.
    text = (HISTORY / 'upgrade.py').read_text(encoding='utf-8')
    start = text.index('def normal_start()')
    end = text.index('\ndef main()', start)
    text = text[start:end]
    text = text.replace("require(process.poll() is None,'Normal app exited')", "require_resources();require(process.poll() is None,'Normal app exited')")
    text = text.replace("for _ in range(10):require(idle(base)['idle']", "for _ in range(10):require_resources();require(idle(base)['idle']")
    scope = dict(globals(), no_shell=quiescent)
    exec(compile(text, str(HISTORY / 'upgrade.py'), 'exec'), scope)
    scope['normal_start']()
