"""Prepared data-preserving overwrite and offline activation; parent review is mandatory."""
from __future__ import annotations
import argparse
import importlib.util
import json
import subprocess
import time
from common import *

BACKUP=Path('H:/Strata/private-backups/personal-052-local2-20261107')
OLD_COMPONENT=Path('C:/Users/downl/AppData/Local/Strata/components/engine/0.5.1-windows-x86_64-cuda-cuda13')
NEW_COMPONENT=OLD_COMPONENT.parent/SIGNED.name
MANAGED=['control','host','node','harness','engine-bundle','local-backends','notices']

def no_shell() -> None:
    require(process_counts()['shell']==0,'Close Strata only after active answers finish; parent owns the pause')

def acceptance(commit: str) -> None:
    for name in ['app-qualification.json','functional-e2e.json','prism-e2e.json']:
        record=load(OUT/name)
        require(record['ok'] and record['sourceCommit']==commit,'Reviewed package/actual functional E2E has not passed')
    require(load(OUT/'functional-e2e.json')['actualNative'] is True,'Actual native GUI flow required')
    require(load(OUT/'prism-e2e.json')['actualPrism041'] is True,'Actual moved-GGUF Prism GUI flow required')
    require(sha(Path(load(OUT/'app-build.json')['installer']))==load(OUT/'app-qualification.json')['installerSHA256'],'Installer changed')

def backup() -> None:
    no_shell();require(not BACKUP.exists(),'Fresh private backup required')
    require(INSTALLED.resolve()==INSTALLED and not INSTALLED.is_junction() and not INSTALLED.is_symlink(),'Linked install destination')
    BACKUP.mkdir()
    sys.path.insert(0,str(ROOT));from terminal.strata_control import private_files
    private_files.restrict_to_current_user(BACKUP,directory=True)
    local=Path('C:/Users/downl/AppData/Local/Strata');data=Path('C:/Users/downl/.strata/data')
    sources={'host-data':data,'roaming':Path('C:/Users/downl/AppData/Roaming/Strata')}
    for name in ['app-settings.json','shell-prefs.json','components/.updates/state.json','tuning','launch']:
        sources['local/'+name]=local/name
    config_dirs={local/'install'}
    native_settings=data/'native-host.json'
    if native_settings.exists():
        folder=load(native_settings).get('installDir')
        if folder:config_dirs.add(Path(folder))
    for index,folder in enumerate(sorted(config_dirs)):
        if folder.is_dir():
            require(not folder.is_symlink() and not folder.is_junction(),'Linked config folder')
            for number,path in enumerate(sorted(folder.rglob('strata-*.json'))):
                sources[f'model-configs/{index}/{number}.json']=path
    records=[]
    for label,source in sources.items():
        if not source.exists():continue
        paths=[source] if source.is_file() else sorted(source.rglob('*'))
        for path in paths:
            require(not path.is_symlink() and not path.is_junction(),'Linked user data')
            if not path.is_file():continue
            require(path.stat().st_size<100*1024**2 and path.suffix.lower() not in {'.gguf','.bin','.safetensors','.pack','.weights'},'Unexpected large/model backup input')
            name=Path(label) if source.is_file() else Path(label)/path.relative_to(source)
            target=BACKUP/name;target.parent.mkdir(parents=True,exist_ok=True)
            value=sha(path);shutil.copy2(path,target);require(sha(target)==value,'Private backup mismatch')
            records.append({'source':str(path),'backup':str(target),'sha256':value,'bytes':path.stat().st_size})
    no_shell();require(all(sha(Path(r['source']))==r['sha256'] for r in records),'User data changed during backup')
    write_new(BACKUP/'preflight.json',{'target':str(INSTALLED),'files':records,'oldComponent':inventory(OLD_COMPONENT),'modelsCopied':False})
    write_new(OUT/'backup.json',{'ok':True,'files':len(records),'bytes':sum(r['bytes'] for r in records),'backup':str(BACKUP),'preflightSHA256':sha(BACKUP/'preflight.json'),'modelsCopied':False})

def install() -> None:
    no_shell();record=load(OUT/'backup.json');require(sha(BACKUP/'preflight.json')==record['preflightSHA256'],'Backup receipt changed')
    pre=load(BACKUP/'preflight.json');require(all(sha(Path(r['source']))==r['sha256'] for r in pre['files']),'Data changed since backup')
    # Check every managed tree before the production NSIS cleanup sees it.
    for name in MANAGED:
        if (INSTALLED/name).exists():inventory(INSTALLED/name)
    receipt=OUT/'nsis-install.json';write_new(receipt,{'ok':False,'invocationClaimed':True,'target':str(INSTALLED)})
    build=load(OUT/'app-build.json');process=subprocess.Popen([build['installer'],'/S','/D='+str(INSTALLED)],cwd=OUT,env=build_env(),creationflags=subprocess.CREATE_NO_WINDOW)
    try:code=process.wait(timeout=600)
    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=15);raise
    write_new(OUT/'nsis-result.json',{'ok':code==0,'exitCode':code,'target':str(INSTALLED),'installerSHA256':build['sha256']})
    require(code==0,'NSIS failed; never repeat automatically')

def verify_installed() -> None:
    no_shell();require(load(OUT/'nsis-result.json')['ok'],'NSIS must finish successfully')
    expected=load(OUT/'qualified-payload.json');require(sha(OUT/'qualified-payload.json')==load(OUT/'app-qualification.json')['payloadInventorySHA256'],'Payload inventory changed')
    actual={'strata-desktop.exe':sha(INSTALLED/'strata-desktop.exe')}
    for name in MANAGED:
        if (INSTALLED/name).is_dir():actual.update({name+'/'+p:d for p,d in inventory(INSTALLED/name).items()})
    require(actual==expected,'Installed payload membership/bytes differ')
    pre=load(BACKUP/'preflight.json');require(all(sha(Path(r['source']))==r['sha256'] for r in pre['files']),'User data changed before first launch')
    require(inventory(OLD_COMPONENT)==pre['oldComponent'],'Old component changed')
    script=("$ErrorActionPreference='Stop';$t='C:\\Users\\downl\\AppData\\Local\\Programs\\Strata';"
        "$v=(Get-Item -LiteralPath ($t+'\\strata-desktop.exe')).VersionInfo.ProductVersion;"
        "if($v -ne '0.5.2-local.2'){throw 'version'};"
        "$r=Get-ItemProperty -LiteralPath 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Strata';"
        "if($r.InstallLocation.Trim('\"') -ine $t -or $r.DisplayVersion -ne $v){throw 'registration'};"
        "$w=New-Object -ComObject WScript.Shell;foreach($d in @([Environment]::GetFolderPath('Desktop'),[Environment]::GetFolderPath('Programs'))){"
        "if($w.CreateShortcut((Join-Path $d 'Strata.lnk')).TargetPath -ine ($t+'\\strata-desktop.exe')){throw 'shortcut'}}")
    subprocess.run(['powershell.exe','-NoProfile','-Command',script],check=True,timeout=30,capture_output=True)
    write_new(OUT/'installed-payload.json',{'ok':True,'payloadFilesMatched':len(actual),'userFilesUnchanged':len(pre['files']),'beforeFirstLaunch':True,'registrationTargetCorrect':True,'shortcutsChecked':2})

def activate() -> None:
    no_shell();require(load(OUT/'installed-payload.json')['ok'],'Prelaunch data/payload verification required')
    native=load(OUT/'staged-native.json');archive=Path(native['archive']);require(sha(archive)==native['archiveSHA256'],'Signed archive changed')
    require(not NEW_COMPONENT.exists(),'Do not replace an existing native 0.5.2 installation blindly')
    component_bytes=sum(path.stat().st_size for path in SIGNED.rglob('*') if path.is_file())
    # The manifest and small filesystem/update receipts also consume target space.
    require(shutil.disk_usage('C:/').free>=5*1024**3+component_bytes+1024**2,'New component would consume the C reserve')
    py=INSTALLED/'control/python/python.exe';cli=INSTALLED/'control/strata.py'
    env={k:v for k,v in os.environ.items() if not k.upper().startswith(('STRATA_','DSH_','WEBVIEW2_'))}
    env.update(STRATA_NO_ACQUIRE='1',STRATA_REQUIRE_SIGNED_COMPONENTS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TEMP=str(OUT/'temp'),TMP=str(OUT/'temp'))
    write_new(OUT/'native-activation-claim.json',{'claimed':True,'releaseQualified':False})
    run=subprocess.run([str(py),'-B',str(cli),'components','install-file',str(archive),'--output','jsonl'],cwd=OUT,env=env,capture_output=True,text=True,timeout=600)
    require(run.returncode==0,'Offline signed component install failed')
    require(inventory(NEW_COMPONENT)==inventory(SIGNED),'Installed component differs from signed copy')
    selected=subprocess.run([str(py),'-B',str(cli),'components','engine','--output','json'],cwd=OUT,env=env,capture_output=True,text=True,timeout=60)
    require(selected.returncode==0,'Selected engine inspection failed')
    value=json.loads(selected.stdout);require(value['usable'] and value['engine']['version']==ENGINE,'New component is not selected/usable')
    require(Path(value['engine']['root']).resolve()==NEW_COMPONENT.resolve(),'Selected component root differs')
    write_new(OUT/'native-activated.json',{'ok':True,'signatureRequired':True,'engineVersion':ENGINE,'engineSHA256':sha(NEW_COMPONENT/'strata.exe'),'releaseQualified':False})

def normal_start() -> None:
    require(load(OUT/'native-activated.json')['ok'],'Verify the new component activation first');no_shell()
    old_helpers=ROOT/'_docs/local-install-20261106';sys.path.insert(0,str(old_helpers))
    from installed_state import endpoint, idle, get
    state=Path('C:/Users/downl/.strata/data/state.json');registrations=Path('C:/Users/downl/.strata/data/local-models.json')
    before=load(state);registry_hash=sha(registrations)
    env=normal_app_env()
    process=subprocess.Popen([str(INSTALLED/'strata-desktop.exe')],cwd=INSTALLED,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,close_fds=True,creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
    end=time.monotonic()+90;base=None
    while time.monotonic()<end:
        require(process.poll() is None,'Normal app exited')
        try:base=endpoint(psutil.Process(process.pid));break
        except (RuntimeError,OSError,psutil.NoSuchProcess):time.sleep(.5)
    require(base is not None,'Normal installed Host unavailable')
    for _ in range(10):require(idle(base)['idle'],'Normal startup must remain unloaded');time.sleep(1)
    build=get(base,'/api/app/build');require(build['host']==build['shell']==PRODUCT,'Installed version mismatch')
    control=get(base,'/api/control/status');require(control['control']['bundle']['version']==CONTROL and control['engine']['usable'] and control['engine']['engine']['version']==ENGINE,'Installed component versions differ')
    after=load(state);require(all(after.get(k)==before.get(k) for k in ['current','beginner','expert']) and sha(registrations)==registry_hash,'Saved settings/registrations changed')
    write_new(OUT/'normal-startup.json',{'ok':True,'leftOpen':True,'debuggingEnabled':False,'manualUnloaded':True,'settingsRetained':True,'productVersion':PRODUCT,'controlVersion':CONTROL,'nativeVersion':ENGINE,'releaseQualified':False})

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step',choices=['backup','install','verify','activate','normal-start']);parser.add_argument('--execute',action='store_true');parser.add_argument('--reviewed-commit')
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps({'wouldExecute':False,'step':args.step,'target':str(INSTALLED),'requiresParentReview':True,'requiresActualGuiE2E':True,'releaseQualified':False}));return
    commit=approved_source(args.reviewed_commit);require_resources();acceptance(commit)
    {'backup':backup,'install':install,'verify':verify_installed,'activate':activate,'normal-start':normal_start}[args.step]()

if __name__=='__main__':main()
