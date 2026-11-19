/** Qualify the exact NSIS archive, then run its production shell self-tests. */
import assert from 'node:assert/strict';
import {existsSync,mkdirSync,readFileSync,writeFileSync,readdirSync} from 'node:fs';
import {spawn,spawnSync} from 'node:child_process';
import {resolve,join,relative} from 'node:path';
import {verifyTree,verifyHost,sha256,verifyControlData,verifyEngine,verifyNativeBuild} from '../../app/host/scripts/release-integrity.mjs';
const root=resolve(import.meta.dirname,'../..');
const out='H:/Strata/releases/local-052-20261106-opencode';
const build=JSON.parse(readFileSync(join(out,'app-build.json'),'utf8'));
const resources=join(root,'apps/desktop/src-tauri/resources');
const attempt=process.argv[2]??'r2';assert.match(attempt,/^r[2-9][0-9]*$/);
const app=join(out,'extracted-'+attempt),home=join(out,'selftest-profile-'+attempt);
const report={ok:false,installerSHA256:sha256(build.installer),product:'0.5.2-local.1',control:'0.5.2',native:'0.5.1'};
const checkpoint=phase=>{global.gc?.();report.phase=phase;writeFileSync(join(out,'app-qualification-'+attempt+'.json'),JSON.stringify(report,null,2)+'\n');};
const allFiles=dir=>readdirSync(dir,{withFileTypes:true}).flatMap(e=>{assert.equal(e.isSymbolicLink(),false);return e.isDirectory()?allFiles(join(dir,e.name)):[join(dir,e.name)];});
try {
 assert.equal(build.ok,true);assert.equal(report.installerSHA256,build.sha256);
 assert.equal(existsSync(app),false);assert.equal(existsSync(home),false);
 const extraction=spawnSync('C:/Program Files/7-Zip/7z.exe',['x',build.installer,'-o'+app,'-y'],{encoding:'utf8',windowsHide:true,timeout:180000});
 writeFileSync(join(out,'extraction.txt'),extraction.stdout+'\n'+extraction.stderr);
 assert.equal(extraction.status,0);
 checkpoint('extracted');
 verifyHost(join(root,'apps/host'));
 for(const name of ['server','dist'])verifyTree(join(root,'apps/host',name),join(app,'host',name));
 for(const name of ['package.json','hakua-chatbot.enabled'])assert.equal(sha256(join(root,'apps/host',name)),sha256(join(app,'host',name)));
 checkpoint('host');
 for(const name of ['control','node','harness','notices']){verifyTree(join(resources,name),join(app,name));checkpoint(name);}
 for(const name of ['strata_cli','strata_control','serve'])verifyTree(join(root,name),join(app,'control',name));
 verifyControlData(root,join(app,'control'));
 assert.equal(JSON.parse(readFileSync(join(app,'control/BUNDLE.json'))).version,'0.5.2');
 assert.equal(JSON.parse(readFileSync(join(app,'host/package.json'))).version,'0.5.2-local.1');
 assert.equal(readFileSync(join(app,'control/COMMIT'),'utf8').trim(),build.sourceCommit);
 assert.equal(sha256(join(app,'node/node.exe')),'1c331340fa6cac5e4294725959070a143fa0507195a78ad77eed7943f719aac9');
 const unbundled=readFileSync(join(build.target,'release/strata-desktop.exe'));
 const marker=Buffer.from('__TAURI_BUNDLE_TYPE_VAR_UNK'),at=unbundled.indexOf(marker);
 assert.ok(at>=0);assert.equal(unbundled.indexOf(marker,at+1),-1);
 Buffer.from('__TAURI_BUNDLE_TYPE_VAR_NSS').copy(unbundled,at);
 assert.ok(unbundled.equals(readFileSync(join(app,'strata-desktop.exe'))));
 report.shellSHA256=sha256(join(app,'strata-desktop.exe'));
 const version=spawnSync('powershell.exe',['-NoProfile','-Command',`(Get-Item -LiteralPath '${join(app,'strata-desktop.exe')}').VersionInfo.ProductVersion`],{encoding:'utf8',timeout:15000,windowsHide:true});
 assert.equal(version.status,0);assert.equal(version.stdout.trim(),'0.5.2-local.1');
 report.shellVersion=version.stdout.trim();
 for(const name of ['engine-bundle','local-backends'])assert.equal(existsSync(join(app,name)),false);
 const names=allFiles(app).map(p=>relative(app,p).replaceAll('\\','/')).sort();
 const managed=names.filter(n=>/^(control|node|host|harness|notices)\//.test(n)||n==='strata-desktop.exe');
 assert.deepEqual(managed.filter(n=>n!=='control/data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf').filter(n=>/\.(?:gguf|db|sqlite|sqlite3|log)$/i.test(n)||/(?:^|\/)\.env(?:\.|$)/i.test(n)),[]);
 // Pin all shipped files before any runtime can create or regenerate bytecode.
 const inventory=Object.fromEntries(managed.map(n=>[n,sha256(join(app,n))]));
 writeFileSync(join(out,'qualified-payload.json'),JSON.stringify(inventory,null,2)+'\n');
 report.payloadInventorySHA256=sha256(join(out,'qualified-payload.json'));report.payloadFiles=managed.length;
 checkpoint('inventory');
 const native=join(resources,'engine-bundle/0.5.1-windows-x86_64-cuda-cuda13');
 const manifest=verifyEngine(native,'0.5.1');
 assert.throws(()=>verifyNativeBuild(native,root,manifest),/Native engine sources changed/);
 report.defaultNativeDriftRejected=true;
 report.nativeBinding=verifyNativeBuild(native,root,manifest,{sourceRoot:'H:/Strata/worktrees/upstream-0211-latest-20261105',sourceCommit:build.nativeSourceCommit,manifestSHA256:build.nativeManifestSHA256});
 checkpoint('native');
 for(const dir of [home,join(home,'Local'),join(home,'Roaming'),join(home,'Temp')])mkdirSync(dir,{recursive:true});
 const env=Object.fromEntries(Object.entries(process.env).filter(([k])=>!/^(STRATA_|DSH_|WEBVIEW2_)/i.test(k)&&!/(?:API_KEY|TOKEN|SECRET|PASSWORD|ATTEST_KEY)$/i.test(k)));
 Object.assign(env,{USERPROFILE:home,APPDATA:join(home,'Roaming'),LOCALAPPDATA:join(home,'Local'),TEMP:join(home,'Temp'),TMP:join(home,'Temp'),STRATA_HOME:join(home,'data'),STRATA_COMPONENTS_DIR:join(home,'components'),STRATA_SETTINGS_FILE:join(home,'settings.json'),STRATA_DESKTOP_DATA_DIR:join(home,'desktop'),STRATA_PROFILE_DIR:join(home,'profiles'),STRATA_RUNTIME_DIR:join(home,'runtime'),DSH_HOME:join(home,'harness'),PATH:'C:/Windows/System32',PYTHONDONTWRITEBYTECODE:'1'});
 const selfReport=join(home,'app-selftest.json');
 const child=spawn(join(app,'strata-desktop.exe'),['--self-test='+selfReport],{env,cwd:home,windowsHide:true,stdio:'ignore'});
 const code=await new Promise((res,rej)=>{const timer=setTimeout(()=>{child.kill();rej(new Error('Self-test deadline'));},180000);child.on('error',rej);child.on('exit',code=>{clearTimeout(timer);res(code);});});
 assert.equal(code,0);
 const self=JSON.parse(readFileSync(selfReport,'utf8'));
 assert.equal(self.ok,true);assert.equal(self.steps.length,13);assert.ok(self.steps.every(s=>s.ok));
 Object.assign(report,{ok:true,selfTests:13,gpuRuntimesEmbedded:false,sourceCommit:build.sourceCommit,app});
} catch(error) {report.error=error.stack;process.exitCode=1;}
finally {checkpoint('finished');if(report.ok)writeFileSync(join(out,'app-qualification.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({ok:report.ok,selfTests:report.selfTests,payloadFiles:report.payloadFiles,shellSHA256:report.shellSHA256}));}
