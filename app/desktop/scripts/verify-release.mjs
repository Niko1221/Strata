/** Refuse to package a current shell around stale GUI, Control or engine payloads. */
import {existsSync,readFileSync,readdirSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {resolve,join,relative} from 'node:path';
import {files,sha256,verifyHost,verifyTree,verifyEngine,verifyNativeBuild,verifyControlData} from '../../host/scripts/release-integrity.mjs';
import {HARNESS_BUDGET_PATCH} from './harness-budget.mjs';
import {missingHarnessImports} from './harness-prune.mjs';
import {LOCAL_BACKENDS,findLocalBackend,verifyLocalBackend} from '../../host/server/local-backends.mjs';
import {RUSTXNG_VERSION,RUSTXNG_SHA256} from '../../host/server/rustxng-runtime.mjs';
const root=resolve(import.meta.dirname,'../../..');
const resources=join(root,'app/desktop/src-tauri/resources');
const control=join(resources,'control');
const read=path=>readFileSync(path,'utf8').replace(/^\uFEFF/,'');
const json=path=>JSON.parse(read(path));
const product=read(join(root,'VERSION')).trim();
// The shipped engine component is 0.6.3; engine/CMakeLists.txt now carries the
// independent upstream engine version (0.1.62), so read the component's own
// manifest version instead of the CMake project version.
const engine=json(engineManifestPath()).version;
function engineManifestPath(){
 const dir=join(root,'app/desktop/src-tauri/resources/engine-bundle');
 const folder=readdirSync(dir).filter(n=>existsSync(join(dir,n,'engine-component.json')));
 if(folder.length!==1)throw new Error('Stage exactly one Windows engine package');
 return join(dir,folder[0],'engine-component.json');
}
const controlVersion=read(join(root,'CONTROL_VERSION')).trim();
if(!/^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(controlVersion))throw new Error('Invalid Control release version');
verifyHost(join(root,'app/host'));
const uiBuild=json(join(root,'app/host/dist/release-build.json'));
if(uiBuild.recipe?.nodeVersion!=='24.22.3'||uiBuild.recipe?.publicPackage!==true||uiBuild.recipe?.publicAssetCommit!=='03f11e2d6adc319ea21e9a1d5820a99cd3767257')throw new Error('Rebuild with the public-package assets and system-font policy');
if(Object.keys(uiBuild.assets).some(name=>/gtwalsheim/i.test(name)))throw new Error('Unqualified supplied font in the release UI');
for(const name of ['icon.png','icon.ico'])if(sha256(join(root,'app/desktop/src-tauri/icons',name))!==uiBuild.recipe.publicAssets?.[name])throw new Error('Stage the approved public shell icon before bundling');
const nsis=json(join(root,'app/desktop/src-tauri/tauri.conf.json')).bundle?.windows?.nsis;
if(nsis?.headerImage||nsis?.sidebarImage)throw new Error('Unqualified supplied installer artwork');
const shellConfig=json(join(root,'app/desktop/src-tauri/tauri.conf.json'));
if(process.platform==='win32')for(const name of ['engine-bundle','local-backends'])if(shellConfig.bundle?.resources?.['resources/'+name])throw new Error('The Windows app must distribute GPU runtimes as a separate engine component');
if(JSON.stringify(shellConfig.plugins?.updater?.endpoints)!==JSON.stringify(['https://github.com/zapabob/Strata/releases/latest/download/latest.json']))throw new Error('The Hakua shell must use the fork update endpoint');
for(const name of ['app/host/package.json','app/desktop/package.json']) if(json(join(root,name)).version!==product) throw new Error(`Product version mismatch: ${name}`);
const terminal=join(root,'terminal');  // new layout: the Terminal lives under terminal/, staged Control mirrors its top level
const appPath=(...parts)=>[terminal,join(root,'engine'),root].map(base=>join(base,...parts)).find(p=>existsSync(p))??join(root,...parts);
for(const name of ['strata_cli','strata_control','serve']) verifyTree(join(terminal,name),join(control,name));
for(const name of ['strata.py','requirements.txt','requirements-monitor.txt','setup.py']) if(sha256(join(terminal,name))!==sha256(join(control,name))) throw new Error(`Stale Control: ${name}`);
for(const path of files(join(control,'tools')).filter(path=>path.endsWith('.py'))) {
 const name=relative(control,path).replaceAll('\\','/');
 const source=appPath(name);
 if(!existsSync(source) || sha256(source)!==sha256(path)) throw new Error(`Stale Control tool: ${name}`);
}
for(const name of ['hakua_hermes_memory_stack.py','strata_memory_mcp.py']) if(!existsSync(join(control,'tools',name))) throw new Error(`Missing Hakua tool: ${name}`);
const vectorProvenance=verifyControlData(terminal,control),qwenLicenseHash=vectorProvenance.license.sha256;
const notices=join(resources,'notices'),noticeRecord=json(join(notices,'NOTICES.json'));
if(noticeRecord.schema!==1||!noticeRecord.files||!Object.keys(noticeRecord.files).length)throw new Error('Stage the third-party notices');
for(const name of ['Strata-MIT.txt','ggml-MIT.txt','NVIDIA-EULA.txt','NVIDIA-LICENSE.txt','Node-LICENSE.txt','Microsoft-VisualStudio-Community-2022.docx','Microsoft-Build-Tools-2022.docx','Microsoft-Visual-C-Runtime-2015-2022.docx','Qwen-Community-License-1.0.txt'])if(!noticeRecord.files[name])throw new Error('Missing third-party notice: '+name);
const noticeNames=files(notices).map(path=>relative(notices,path).replaceAll('\\','/')).filter(name=>name!=='NOTICES.json' && name!=='.gitignore').sort();
if(JSON.stringify(noticeNames)!==JSON.stringify(Object.keys(noticeRecord.files).sort()))throw new Error('Third-party notice membership changed');
for(const name of noticeNames)if(sha256(join(notices,name))!==noticeRecord.files[name])throw new Error('Third-party notice changed: '+name);
const libstrixNotice=join(root,'engine/third_party/libstrix/NOTICE.md');
if(existsSync(libstrixNotice)&&noticeRecord.files['libstrix-NOTICE.md']!==sha256(libstrixNotice))throw new Error('Missing or stale upstream libstrix attribution');
if(JSON.stringify(noticeRecord.microsoftSources)!==JSON.stringify(json(join(root,'tools/packaging/microsoft-2022-notice-sources.json'))))throw new Error('Pinned Microsoft notice provenance changed');
if(JSON.stringify(noticeRecord.qwenSource)!==JSON.stringify(vectorProvenance.license)||sha256(join(notices,'Qwen-Community-License-1.0.txt'))!==qwenLicenseHash)throw new Error('Missing or changed Qwen notice');
if(sha256(join(notices,'README.md'))!==sha256(join(root,'tools/packaging/hakua-third-party-notices.md')))throw new Error('Stale third-party notice index');
const bundle=json(join(control,'BUNDLE.json'));
if(bundle.version!==controlVersion || read(join(control,'VERSION')).trim()!==controlVersion) throw new Error('Stale Control package version');
if(bundle.installed?.['hakua-memory']!=='0.3.5') throw new Error('Missing locked Hakua-memory runtime');
const py=process.platform==='win32'?join(control,'python/python.exe'):join(control,'python/bin/python3');
const probe=spawnSync(py,['-I','-B','-c',"import importlib.metadata as m; from hakua_memory.ebbinghaus.store import EbbinghausMemoryStore; from hakua_memory.composite import CompositeMemory; assert m.version('hakua-memory') == '0.3.5'"],{encoding:'utf8',timeout:30000,windowsHide:true});
if(probe.error || probe.status!==0) throw new Error(`Hakua runtime probe failed: ${probe.error??probe.stderr}`);
if(!existsSync(join(root,'app/host/hakua-chatbot.enabled'))) throw new Error('Missing Hakua enable marker');
const harness=join(resources,'harness'),harnessBundle=json(join(harness,'BUNDLE.json'));
// Reviewed integration revision, documented in docs/work/IMPLEMENTATION_STATUS.md.
if(harnessBundle.commit!=='571902028a75c64c1d767f4ae5fd099485f0ddf0')throw new Error('Stage the reviewed Harness source revision');
const patch=harnessBundle.patches?.find(item=>item.id===HARNESS_BUDGET_PATCH.id);
if(!patch || patch.beforeSha256!==HARNESS_BUDGET_PATCH.beforeSha256 || patch.afterSha256!==sha256(join(harness,HARNESS_BUDGET_PATCH.file)))throw new Error('Stale Harness output-budget patch');
if(missingHarnessImports(harness).length)throw new Error('Missing Harness runtime dependencies');
if(process.platform==='win32') {
 const search=join(harness,'rustxng'),searchSource=json(join(search,'SOURCE.json'));
 if(searchSource.version!==RUSTXNG_VERSION || searchSource.profile!=='full' || searchSource.sourceCommit!=='a8ba33d8c9ceed91236f3abf029254d0f84995bb' || sha256(join(search,'rustxng.exe'))!==RUSTXNG_SHA256)throw new Error('Stage the pinned RustXNG Full distribution');
 for(const name of ['LICENSE','NOTICE','corresponding-source.tar.gz'])if(!searchSource.files?.[name] || sha256(join(search,name))!==searchSource.files[name].sha256)throw new Error('RustXNG source or attribution changed: '+name);
 const node=join(resources,'node/node.exe');
 if(sha256(join(resources,'node/LICENSE'))!=='ed34dd8e3f0a78dbaf00d0444ce8e285b015b765379c2e17880455f70370f8e9')throw new Error('Stage the official Node 24.22.3 licence');
 // nodejs.org/dist/v24.22.3/SHASUMS256.txt: win-x64/node.exe.
 if(sha256(node)!=='86fc4785327db5f5c7558dc6156c543b433c313e813a222e088086222ab4afa6')throw new Error('Stage the verified Node 24.22.3 Windows runtime');
 const nodeProbe=spawnSync(node,['--version'],{encoding:'utf8',timeout:15000,windowsHide:true});
 if(nodeProbe.status!==0||nodeProbe.stdout.trim()!=='v24.22.3')throw new Error('Bundled Node runtime version mismatch');
 for(const choice of Object.keys(LOCAL_BACKENDS)) {
  const binary=findLocalBackend(choice,{installRoot:resources});
  const manifest=await verifyLocalBackend(binary,choice);
  if(manifest.version!==LOCAL_BACKENDS[choice].packageVersion)throw new Error(`Stale ${choice} runtime version`);
  if(choice==='TurboQuant' && (JSON.stringify(manifest.cacheTypesK)!==JSON.stringify(['f16','bf16','q8_0','q4_0']) || JSON.stringify(manifest.cacheTypesV)!==JSON.stringify(['f16','bf16','q8_0','q4_0','turbo2','turbo3','turbo4'])))throw new Error('Stage the TurboQuant runtime with independently qualified K/V capabilities');
  const probe=spawnSync(binary.path,['--version'],{encoding:'utf8',timeout:30000,windowsHide:true,env:{...process.env,PATH:join(process.env.SystemRoot??'C:/Windows','System32')}});
  if(probe.error||probe.status!==0||!((probe.stdout+probe.stderr).match(/\b[0-9a-f]{7,40}\b/gi)??[]).some(value=>manifest.source.commit.startsWith(value.toLowerCase())))throw new Error(`Bundled ${choice} executable does not match its source revision`);
 }
  // The Windows installer does not bundle the GPU engine. For the reviewed
  // portable Q38 preview, verify the exact companion payload.
  // SIGNATURE VERIFICATION REMOVED by user decision 2026-11-22: this fork ships
  // rebuilt/unsigned engines (0.6.4, 9B, Inkling-2) and there is no signing key.
  // The bundled public-key signature check is replaced by the manifest-pinned
  // SHA-256 digest below plus e.verify(full=True) file-by-file digests.
  const preview=process.env.STRATA_Q38_PREVIEW_COMPONENT_DIR;
  if(preview) {
   const folder=resolve(preview), manifestPath=join(folder,'engine-component.json');
   const verify="import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); from strata_control import engine_component as e; c=e.read_manifest_any(Path(sys.argv[2])); e.verify(c,full=True); print(len(c.files))";
   const checked=spawnSync(py,['-I','-B','-c',verify,control,folder],{encoding:'utf8',timeout:180000,windowsHide:true});
   if(checked.error||checked.status!==0||checked.stdout.trim()!=='152')throw new Error(`Q38 preview payload verification failed: ${checked.error??checked.stderr}`);
   const run=spawnSync(join(folder,'strata-q38.exe'),['--version'],{encoding:'utf8',timeout:15000,windowsHide:true});
   if(run.error||run.status!==0||!run.stdout.includes('SM120 cudart=13050 cublas=131000'))throw new Error('Q38 preview version probe failed');
   console.log('Verified the Q38 preview companion payload (152 files, digest-based)');
  } else {
  const manifests=files(join(resources,'engine-bundle')).filter(path=>path.endsWith('engine-component.json'));
 if(manifests.length!==1) throw new Error('Stage exactly one current Windows engine package');
 const path=manifests[0],folder=resolve(path,'..');
 const manifest=verifyEngine(folder,engine);
 const nativeSource=verifyNativeBuild(folder,root,manifest,{
  sourceRoot:process.env.STRATA_NATIVE_SOURCE_ROOT,
  sourceCommit:process.env.STRATA_NATIVE_SOURCE_COMMIT,
  manifestSHA256:process.env.STRATA_NATIVE_MANIFEST_SHA256,
 });
 if(nativeSource.sourceCommit) {
  // SIGNATURE VERIFICATION REMOVED by user decision 2026-11-22 (no signing key;
  // unsigned rebuilt engines). verifyNativeBuild above still checks the pinned
  // source commit and file digests, which remain the integrity check.
  console.log('Verified separate native source (digest-based): '+JSON.stringify(nativeSource));
 }
 const run=spawnSync(join(folder,manifest.entrypoint),['--version'],{encoding:'utf8',timeout:15000,windowsHide:true});
  if(run.error || run.status!==0 || !run.stdout.includes(engine)) throw new Error('Bundled executable version does not match engine manifest');
  }
}
console.log(`Release payload verified: Hakua ${product}, engine ${engine}, Control ${controlVersion}`);
