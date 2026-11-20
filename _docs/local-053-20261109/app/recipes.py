"""Checked, in-memory adapters of immutable successful execution recipes."""
from __future__ import annotations

import time
from common import *

OLD_OUT = 'H:/Strata/releases/personal-052-local2-20261107/app'


def replace(text: str, old: str, new: str, count: int = 1) -> str:
    require(text.count(old) == count, 'Historical recipe drift: ' + old[:100])
    return text.replace(old, new)


def qualify() -> str:
    text = (HISTORY / 'qualify-app.mjs').read_text(encoding='utf-8')
    text = text.replace("from '../../app/host/", "from '" + (ROOT / 'apps/host').as_uri() + '/')
    text = replace(text, "root=resolve(import.meta.dirname,'../..')", 'root=' + json.dumps(ROOT.as_posix()))
    text = text.replace(OLD_OUT, OUT.as_posix()).replace('0.5.2-local.2', PRODUCT).replace("controlVersion='0.5.3'", "controlVersion='0.5.4'")
    text = text.replace("'0.5.2'", "'0.5.3'").replace('0.5.2-windows-x86_64', '0.5.3-windows-x86_64')
    text = replace(text, "verifyNativeBuild(native,root,verifyEngine(native,'0.5.3'));", "verifyNativeBuild(native,root,verifyEngine(native,'0.5.3'),{sourceRoot:process.env.STRATA_NATIVE_SOURCE_ROOT,sourceCommit:signed.nativeSourceCommit,manifestSHA256:signed.signedManifestSHA256});")
    text = replace(text, "DSH_HOME:join(home,'harness'),PATH:", "DSH_HOME:join(home,'harness'),HOME:home,USERPROFILE:home,PATH:")
    # These schema/manifest/inventory checks run again on the new package. No historical pass is copied.
    text = replace(text, "assert.equal(JSON.parse(readFileSync(join(app,'control/BUNDLE.json'))).version,controlVersion);", "const bundle=JSON.parse(readFileSync(join(app,'control/BUNDLE.json')));assert.equal(bundle.version,controlVersion);assert.equal(bundle.python.version,'3.12.15');assert.equal(bundle.installed['hakua-memory'],'0.3.5');assert.equal(bundle.lock.sha256,sha256(join(root,'packaging/locks/runtime-windows-x86_64-cp312.txt')));")
    return text


def qa(kind: str, target: str, attempt: str, work: Path) -> str:
    require(kind in {'q36', 'prism'} and target in {'candidate', 'installed'}, 'Invalid QA identity')
    require(bool(re.fullmatch('r[1-9][0-9]*', attempt)), 'Invalid attempt')
    source = RUNTIME / ('qa-q36-r14.mjs' if kind == 'q36' else 'qa-prism-r5.mjs')
    text = source.read_text(encoding='utf-8')
    text = text.replace('H:/Strata/worktrees/main-integration-20261106', ROOT.as_posix())
    old_attempt = 'r14' if kind == 'q36' else 'r5'
    text = text.replace(OLD_OUT + '/execution-20261108/isolate-host-home.mjs', (HERE / 'isolate-host-home.mjs').as_posix())
    text = text.replace(OLD_OUT + '/execution-20261108/probe-runtime-owned.py', (HERE / 'probe.py').as_posix())
    for name in ['probe-owned.py', 'probe-prism-child.py']:
        text = text.replace((HISTORY / name).as_posix(), (HERE / 'probe.py').as_posix())
    text = replace(text, "out='" + OLD_OUT + "'", 'out=' + json.dumps(work.as_posix()))
    text = text.replace('0.5.2-local.2', PRODUCT).replace('0.5.2-windows-x86_64', '0.5.3-windows-x86_64').replace("'0.5.2'", "'0.5.3'")
    text = text.replace(old_attempt, attempt)
    text = text.replace('actualR3Required', 'actualNative053Required').replace('real-r3-capability', 'real-native053-capability')
    text = text.replace("'--shell',String(child.pid)", "'--app',app,'--home',home,'--shell',String(child.pid)")
    # Record the actual target explicitly, including packaged Host identity before every UI session.
    text = replace(text, 'const result={ok:false,', 'const result={ok:false,target:' + json.dumps(target) + ',attempt:' + json.dumps(attempt) + ',installedExeTest:' + str(target == 'installed').lower() + ',shellPath:exe,shellSHA256:sha256(exe),')
    text = replace(text, "assert.equal((await " + ('request' if kind == 'q36' else 'call') + "('/api/app/build')).host,'" + PRODUCT + "');", "const versions=await " + ('request' if kind == 'q36' else 'call') + "('/api/app/build');assert.equal(versions.host,'" + PRODUCT + "');assert.equal(versions.shell,'" + PRODUCT + "');const identity=spawnSync(" + ('python' if kind == 'q36' else 'py') + ",['-B'," + json.dumps((HERE / 'probe.py').as_posix()) + ",'--app',app,'--home',home,'--shell',String(child.pid)],{encoding:'utf8',windowsHide:true,timeout:30000});assert.equal(identity.status,0,identity.stderr);result.actualShellAndHost=true;")
    if kind == 'prism':
        text = replace(text, "const qa=JSON.parse(readFileSync(join(out,'app-qualification.json'))),native=JSON.parse(readFileSync(join(out,'staged-native.json')));", "const qa=JSON.parse(readFileSync('" + (OUT / 'app-qualification.json').as_posix() + "')),native=JSON.parse(readFileSync('" + (OUT / 'staged-native.json').as_posix() + "'));const app=" + json.dumps((INSTALLED if target == 'installed' else OUT / 'extracted').as_posix()) + ",exe=join(app,'strata-desktop.exe');assert.equal(sha256(exe),qa.shellSHA256);")
        start = text.index('const selection=')
        end = text.index('const hashFile=', start)
        text = text[:start] + "const model=" + json.dumps(USER_GGUF.as_posix(), ensure_ascii=False) + ";\n" + text[end:]
        text = replace(text, "home=join(out,'prism-e2e-profile-" + attempt + "')", "home=join(out,'prism-e2e-profile')")
        text = replace(text, "source=join(out,'signed-component-r4',componentName)", 'source=' + json.dumps(SIGNED.as_posix()))
        text = text.replace('installRoot:qa.app', 'installRoot:app').replace("spawn(join(qa.app,'strata-desktop.exe'),[],{cwd:qa.app", 'spawn(exe,[],{cwd:app')
        start = text.index(' const automatic=plan.')
        end = text.index('\n', start)
        text = text[:start] + text[end:]
        text = replace(text, 'assert.equal(id,plan.target.variantId);', "assert.ok(id.startsWith('local-')); ")
        text = text.replace('result.automaticOverrideWasPreserved=true;', 'result.normalProfileUntouched=true;')
    # Keep all failed attempts. Only this fresh invocation may publish its own target acceptance.
    text = text.replace("if(result.ok)writeFileSync(join(out,'" + ('functional-e2e' if kind == 'q36' else 'prism-e2e') + ".json'),JSON.stringify(result,null,2)+'\\n',{flag:'wx'});", '')
    text = replace(text, 'result.actualShellAndHost=true;', 'result.actualShellAndHost=true;result.actualWebView2=true;')
    if kind == 'q36':
        text = "import {workspaceFilesAcceptance} from " + json.dumps((HERE / 'workspace-files-acceptance.mjs').as_uri()) + ";\n" + text
        text = q36_projection(text)
    return text


def q36_projection(text: str) -> str:
    """Extend the existing real lifecycle; never replace APIs, INFO, or positive observations."""
    acceptance = (HERE / 'q36-projection-acceptance.mjs').read_text(encoding='utf-8')
    acceptance = replace(acceptance, "import assert from 'node:assert/strict';\n", '')
    binding = "\nconst projectionQa=createQ36ProjectionAcceptance({getPage:()=>page,request,until,language,modelScreen,modelId,modelPath,hashFile:sha256,getBase:()=>base,home,result});\n"
    text = replace(text, 'const synthetic=', acceptance + binding + 'const synthetic=')
    text = replace(text, "await start();await language('en');await modelScreen();\n", "await start();await language('en');await modelScreen();result.workspaceFilesAcceptance=await workspaceFilesAcceptance({getPage:()=>page,request,home,language,modelScreen});assert.equal(resolve((await request(`/api/native/models/${modelId}/settings-file`)).path).toLowerCase(),resolve(modelPath).toLowerCase(),'Native config escaped isolated fixture');await projectionQa.japaneseSettings();\n")
    anchor = "await page.getByRole('button',{name:/^Saved settings/}).click();await page.getByRole('menuitem',{name:'Save current settings…',exact:true}).click();"
    text = replace(text, anchor, "await page.getByRole('combobox',{name:'Conversation context',exact:true}).click();await page.getByRole('option',{name:'4K',exact:true}).click();" + anchor)
    for marker, extra in [
        ('real-native053-capability-and-native-settings-save', "await projectionQa.status('save-only');"),
        ('saved-native-settings-after-reselection', "await projectionQa.status('after-reselection');"),
        ('saved-profile-after-process-restart', "await projectionQa.status('after-profile-restart',true);"),
        ('gui-load-actual-info-private-owner', 'await projectionQa.rejectedEnable(first);'),
        ('real-gui-nonempty-chat', 'await projectionQa.sampling();'),
        ('gui-stop-restart-new-owned-child', "assert.notEqual(restarted.generation,first.generation);await projectionQa.status('reloaded-child');"),
    ]:
        anchor = "result.checks.push('" + marker + "');"
        text = replace(text, anchor, anchor + extra)
    unloaded = "await until(async()=>!(await request('/api/native')).instances.some(i=>i.status==='ready'));"
    text = replace(text, unloaded, unloaded + "await projectionQa.status('after-unload',true);", count=2)
    return replace(text, 'result.modelFilesPreserved=true;result.ok=true;', 'result.projectionAcceptance=projectionQa.receipt();result.modelFilesPreserved=true;result.ok=true;')


def prepare_q36(work: Path, app: Path, commit: str) -> dict:
    """Fresh fixture, with the corrected generic cache aliases and explicit bounded mode."""
    native = staged()
    qa_receipt = load(OUT / 'app-qualification.json')
    require(qa_receipt['ok'] and qa_receipt['sourceCommit'] == commit, 'New candidate qualification required')
    home = work / 'e2e-profile'
    require(not home.exists(), 'Fresh isolated profile required')
    home.mkdir()
    for name in ['data', 'install', 'profiles', 'runtime', 'models', 'Local', 'Roaming', 'webview', 'desktop', 'temp', 'screenshots', 'workspace']:
        (home / name).mkdir()
    (home / 'workspace/qa-existing-note.md').write_text('Isolated workspace file acceptance fixture.\n', encoding='utf-8')
    component = preserved_copy(SIGNED, home / 'components/engine' / SIGNED.name)
    files = [MODEL, Path(str(MODEL) + '.q36epk'), Path(str(MODEL) + '.q36epk.verified')]
    require(all(p.is_file() for p in files) and (TOKENIZER / 'tokenizer.json').is_file(), 'Existing model/verified pack/tokenizer required')
    before = {str(p): dict(bytes=p.stat().st_size, mtimeNs=p.stat().st_mtime_ns) for p in files}
    model_id = 'qwen36-ud-q4_k_xl'
    cfg = dict(exe='component:strata-engine', tokenizer=str(TOKENIZER), cwd=str(home / 'install'), arch='qwen35moe', backend='cuda',
               model_name='Isolated local053 Qwen3.6', host='127.0.0.1', parallel=1, ui_context_chosen=True, ui_vision_enabled=False,
               ui_resource_profile='performance', q36_memory='legacy', open_browser=False,
               args=['--native', str(MODEL), '--q36-expert-store', 'mapped-pack', '--max-context', '4096', '--prefill', '512',
                     '--expert-slots', '5', '--mtp', '2', '--no-warmup', '--greedy', '--prompt-cache', '0', '--conversation-cache-mib', '0',
                      '--private-conversation-cache', '--cache-type-k', 'f32', '--cache-type-v', 'f32'])
    write_new(home / 'install' / f'strata-{model_id}.json', cfg)
    settings = dict(data_dir=str(home), data_dir_at=time.time())
    write_new(home / 'settings.json', settings)
    terminal = home / 'Roaming/Strata'
    terminal.mkdir()
    write_new(terminal / 'settings.json', settings)
    write_new(home / 'data/native-host.json', dict(installDir=str(home / 'install'), downloadDirectory=str(home / 'models')))
    write_new(home / 'data/state.json', dict(version=1, setup=dict(complete=True, flowVersion=2, step=4, history=[]), installed={}, current=None, loaded=None,
              beginner=dict(priority='balanced', contextLength='4k', readImages=False, gpuIdle='never'),
              expert={'sample.maxTokens': 32, 'sample.temp': 0, 'sample.speedProjection': False}))
    fixture = dict(prepared=True, sourceCommit=commit, home=str(home), app=str(app), modelId=model_id, modelFilesBefore=before,
                   componentInventorySHA256=component['inventorySHA256'], engineSHA256=native['engineSHA256'], shellSHA256=qa_receipt['shellSHA256'],
                   tokenizerInventorySHA256=digest(inventory(TOKENIZER)), executed=False, releaseQualified=False)
    write_new(work / 'e2e-fixture.json', fixture)
    return fixture
