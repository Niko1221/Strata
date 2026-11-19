/** Exact archive/source/resource verification and 13 production self-tests; parent gate only. */
import assert from 'node:assert/strict';
import {existsSync,mkdirSync,readFileSync,writeFileSync,readdirSync} from 'node:fs';
import {spawn,spawnSync,execFileSync} from 'node:child_process';
import {resolve,join,relative} from 'node:path';
import {verifyTree,verifyHost,sha256,verifyControlData,verifyEngine,verifyNativeBuild} from '../../app/host/scripts/release-integrity.mjs';
const root=resolve(import.meta.dirname,'../..'),out='H:/Strata/releases/personal-052-local2-20261107/app';
const product='0.5.2-local.2',controlVersion='0.5.3';
const argv=process.argv.slice(2);
if(!argv.includes('--execute')){console.log(JSON.stringify({wouldQualify:false,productionSelfTests:13,productVersion:product}));process.exit(0);}
const commit=argv[argv.indexOf('--reviewed-commit')+1];assert.match(commit??'',/^[0-9a-f]{40}$/);
assert.equal(execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim(),commit);
assert.equal(execFileSync('git',['diff','HEAD','--name-only'],{cwd:root,encoding:'utf8'}).trim(),'');
assert.equal(execFileSync('git',['ls-files','--others','--exclude-standard'],{cwd:root,encoding:'utf8'}).trim(),'');
const build=JSON.parse(readFileSync(join(out,'app-build.json')));assert.equal(build.sourceCommit,commit);
const app=join(out,'extracted'),home=join(out,'selftest-profile'),resources=join(root,'apps/desktop/src-tauri/resources');
assert.equal(existsSync(app),false);assert.equal(existsSync(home),false);assert.equal(existsSync(join(out,'app-qualification.json')),false);
const result={ok:false,sourceCommit:commit,productVersion:product,controlVersion,nativeVersion:'0.5.2',releaseQualified:false};
const allFiles=dir=>readdirSync(dir,{withFileTypes:true}).flatMap(e=>{assert.equal(e.isSymbolicLink(),false);return e.isDirectory()?allFiles(join(dir,e.name)):[join(dir,e.name)];});
try{
 assert.equal(sha256(build.installer),build.sha256);
 const unpack=spawnSync('C:/Program Files/7-Zip/7z.exe',['x',build.installer,'-o'+app,'-y'],{encoding:'utf8',windowsHide:true,timeout:240000});assert.equal(unpack.status,0);
 verifyHost(join(root,'apps/host'));
 for(const name of ['server','dist'])verifyTree(join(root,'apps/host',name),join(app,'host',name));
 assert.equal(sha256(join(root,'apps/host/package.json')),sha256(join(app,'host/package.json')));
 for(const name of ['control','node','harness','notices']){verifyTree(join(resources,name),join(app,name));global.gc?.();}
 for(const name of ['strata_cli','strata_control','serve'])verifyTree(join(root,name),join(app,'control',name));
 verifyControlData(root,join(app,'control'));
 assert.equal(JSON.parse(readFileSync(join(app,'control/BUNDLE.json'))).version,controlVersion);
 assert.equal(readFileSync(join(app,'control/COMMIT'),'utf8').trim(),commit);
 assert.equal(JSON.parse(readFileSync(join(app,'host/package.json'))).version,product);
 assert.equal(sha256(build.unbundledShell),build.unbundledShellSHA256);
 const expected=readFileSync(build.unbundledShell),mark=Buffer.from('__TAURI_BUNDLE_TYPE_VAR_UNK'),offset=expected.indexOf(mark);
 assert.ok(offset>=0);assert.equal(expected.indexOf(mark,offset+1),-1);Buffer.from('__TAURI_BUNDLE_TYPE_VAR_NSS').copy(expected,offset);
 assert.ok(expected.equals(readFileSync(join(app,'strata-desktop.exe'))));
 const version=spawnSync('powershell.exe',['-NoProfile','-Command',`(Get-Item -LiteralPath '${join(app,'strata-desktop.exe')}').VersionInfo.ProductVersion`],{encoding:'utf8',timeout:15000,windowsHide:true});assert.equal(version.status,0);assert.equal(version.stdout.trim(),product);
 for(const name of ['engine-bundle','local-backends'])assert.equal(existsSync(join(app,name)),false);
 const names=allFiles(app).map(p=>relative(app,p).replaceAll('\\','/')).filter(n=>/^(control|node|host|harness|notices)\//.test(n)||n==='strata-desktop.exe').sort();
 assert.deepEqual(names.filter(n=>n!=='control/data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf').filter(n=>/\.(?:gguf|db|sqlite|sqlite3|log)$/i.test(n)||/(?:^|\/)\.env(?:\.|$)/i.test(n)),[]);
 const inventory=Object.fromEntries(names.map(n=>[n,sha256(join(app,n))]));writeFileSync(join(out,'qualified-payload.json'),JSON.stringify(inventory,null,2)+'\n',{flag:'wx'});
 const native=join(resources,'engine-bundle/0.5.2-windows-x86_64-cuda-cuda13'),signed=JSON.parse(readFileSync(join(out,'staged-native.json')));
 assert.equal(signed.signatureVerified,true);assert.equal(sha256(join(native,'engine-component.json')),signed.signedManifestSHA256);
 assert.equal(sha256(join(native,'local-backends/prism/runtime.json')),signed.prismManifestSHA256);
 verifyNativeBuild(native,root,verifyEngine(native,'0.5.2'));
 const signature=spawnSync(join(app,'control/python/python.exe'),['-I','-B','-c',"import sys,json; sys.path.insert(0,sys.argv[1]); from terminal.strata_control import signing,engine_component; m=json.load(open(sys.argv[2],encoding='utf-8')); signing.verify_bytes(signing.load_trusted_keys(environ={}),engine_component.signing_payload(m),m.get('signature'))",join(app,'control'),join(native,'engine-component.json')],{encoding:'utf8',windowsHide:true,timeout:30000});
 assert.equal(signature.status,0);
 for(const dir of [home,join(home,'Local'),join(home,'Roaming'),join(home,'Temp')])mkdirSync(dir,{recursive:true});
 const env=Object.fromEntries(Object.entries(process.env).filter(([k])=>!/^(STRATA_|DSH_|WEBVIEW2_)/i.test(k)&&!/(?:API_KEY|TOKEN|SECRET|PASSWORD)$/i.test(k)));
 Object.assign(env,{APPDATA:join(home,'Roaming'),LOCALAPPDATA:join(home,'Local'),TEMP:join(home,'Temp'),TMP:join(home,'Temp'),STRATA_HOME:home,STRATA_COMPONENTS_DIR:join(home,'components'),STRATA_SETTINGS_FILE:join(home,'settings.json'),STRATA_DESKTOP_DATA_DIR:join(home,'desktop'),STRATA_PROFILE_DIR:join(home,'profiles'),STRATA_RUNTIME_DIR:join(home,'runtime'),DSH_HOME:join(home,'harness'),PATH:'C:/Windows/System32',PYTHONDONTWRITEBYTECODE:'1'});
 const report=join(home,'selftest.json'),child=spawn(join(app,'strata-desktop.exe'),['--self-test='+report],{env,cwd:home,stdio:'ignore',windowsHide:true});
 const code=await new Promise((res,rej)=>{const timer=setTimeout(()=>{child.kill();rej(Error('Self-test deadline'));},180000);child.on('error',e=>{clearTimeout(timer);rej(e);});child.on('exit',c=>{clearTimeout(timer);res(c);});});assert.equal(code,0);
 const tests=JSON.parse(readFileSync(report));assert.equal(tests.ok,true);assert.equal(tests.steps.length,13);assert.ok(tests.steps.every(s=>s.ok));
 Object.assign(result,{ok:true,app,installerSHA256:build.sha256,shellSHA256:sha256(join(app,'strata-desktop.exe')),payloadFiles:names.length,payloadInventorySHA256:sha256(join(out,'qualified-payload.json')),selfTests:13,gpuRuntimesEmbedded:false});
}catch(error){result.errorType=error.name;result.errorCode=error.code??null;process.exitCode=1;}
finally{writeFileSync(join(out,'app-qualification.json'),JSON.stringify(result,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify(result));}
