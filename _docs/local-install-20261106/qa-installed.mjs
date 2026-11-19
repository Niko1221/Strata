/** Normal installed profile: observe startup only, with no private screenshots or content logging. */
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {readFileSync,writeFileSync} from 'node:fs';
import {createServer} from 'node:net';
import {createRequire} from 'node:module';
import {join,resolve} from 'node:path';
import {setTimeout as delay} from 'node:timers/promises';
import {sha256} from '../../app/host/scripts/release-integrity.mjs';
import {createHash} from 'node:crypto';
const root=resolve(import.meta.dirname,'../..'),out='H:/Strata/releases/local-052-20261106-opencode';
const attempt=process.argv[2]??'r3';assert.match(attempt,/^r[3-9][0-9]*$/);
const installed='C:/Users/downl/AppData/Local/Programs/Strata';
assert.equal(JSON.parse(readFileSync(join(out,'installed-payload.json'))).ok,true);
const statePath='C:/Users/downl/.strata/data/state.json',registryPath='C:/Users/downl/.strata/data/local-models.json';
const before=JSON.parse(readFileSync(statePath,'utf8')),registryHash=sha256(registryPath);
const socket=createServer();await new Promise(r=>socket.listen(0,'127.0.0.1',r));const port=socket.address().port;await new Promise(r=>socket.close(r));
const env=Object.fromEntries(Object.entries(process.env).filter(([k])=>!/^(STRATA_|DSH_|WEBVIEW2_|NODE_OPTIONS$)/i.test(k)));
env.WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS='--remote-debugging-address=127.0.0.1 --remote-debugging-port='+port;
const child=spawn(join(installed,'strata-desktop.exe'),[],{env,windowsHide:true,stdio:'ignore'});
const result={passed:false,normalInstalledProfile:true,modelLoadRequested:false,checks:0,pageErrors:0,consoleErrors:0,consoleDiagnostics:[],failedResponses:[]};
const hash=value=>createHash('sha256').update(value).digest('hex');
const classify=url=>{try{const path=new URL(url).pathname;return {pathHash:hash(path),api:path.startsWith('/api/'),staticAsset:path.startsWith('/assets/'),missingImage:/\.(?:png|svg|jpg|webp|ico)$/i.test(path),modelRoute:path.startsWith('/api/models'),nativeRoute:path.startsWith('/api/native'),chatRoute:path.startsWith('/api/chat'),memoryRoute:path.startsWith('/api/memory'),controlRoute:path.startsWith('/api/control')};}catch{return {unknown:true};}};
const {chromium}=createRequire(join(root,'apps/host/package.json'))('playwright-core');let browser,page,base;
const responseChecks=[];
const get=async route=>{const r=await fetch(base+route,{signal:AbortSignal.timeout(60000)});assert.equal(r.status,200);return r.json();};
const isIdle=async()=>{const [models,turns,serving]=await Promise.all([get('/api/models'),get('/api/chat/active'),get('/api/serving')]);return models.loaded===null&&models.engine.status==='stopped'&&turns.turns.length===0&&serving.stats.running===0&&serving.stats.waiting===0;};
try {
 for(let i=0;i<120;i++){
  assert.equal(child.exitCode,null);
  try{if((await fetch('http://127.0.0.1:'+port+'/json/version',{signal:AbortSignal.timeout(1000)})).ok){browser=await chromium.connectOverCDP('http://127.0.0.1:'+port);break;}}catch{}
  await delay(500);
 }
 assert.ok(browser);
 for(let i=0;i<120;i++){page=browser.contexts().flatMap(c=>c.pages()).find(p=>/^http:\/\/127\.0\.0\.1:\d+/.test(p.url()));if(page)break;await delay(500);}
 assert.ok(page);base=new URL(page.url()).origin;
 page.on('pageerror',()=>result.pageErrors++);page.on('console',m=>{if(m.type()==='error'){result.consoleErrors++;result.consoleDiagnostics.push({messageHash:hash(m.text()),resourceError:m.text().startsWith('Failed to load resource:'),status:Number(/status of (\d+)/.exec(m.text())?.[1])||null,...classify(m.location().url)});}});
 page.on('response',r=>{if(r.status()>=400){const row={status:r.status(),...classify(r.url()),expectedNoTurn:false};result.failedResponses.push(row);responseChecks.push((async()=>{if(r.status()===404&&new URL(r.url()).pathname==='/api/chat/attach')row.expectedNoTurn=(await r.json()).code==='no-turn';})().catch(()=>{}));}});
 await page.waitForLoadState('domcontentloaded');await page.locator('main').first().waitFor();
 const build=await get('/api/app/build');assert.equal(build.host,'0.5.2-local.1');assert.equal(build.shell,'0.5.2-local.1');
 result.hostVersion=build.host;result.shellVersion=build.shell;
 for(let i=0;i<10;i++){assert.equal(await isIdle(),true,'Installed startup must remain idle and unloaded');result.checks++;await delay(1000);}
 assert.equal(sha256(registryPath),registryHash);
 const after=JSON.parse(readFileSync(statePath,'utf8'));assert.equal(after.current,before.current);assert.deepEqual(after.beginner,before.beginner);assert.deepEqual(after.expert,before.expert);
 result.modelSelectionRetained=true;result.modelContextSettingsRetained=true;result.localRegistrationsRetained=true;
 const control=await get('/api/control/status');assert.equal(control.packaged,true);assert.equal(control.control.bundle.version,'0.5.2');assert.equal(control.engine.usable,true);assert.equal(control.engine.engine.version,'0.5.1');
 result.controlVersion=control.control.bundle.version;result.nativeVersion=control.engine.engine.version;result.nativeUsable=true;
 assert.equal(sha256('C:/Users/downl/AppData/Local/Strata/components/engine/0.5.1-windows-x86_64-cuda-cuda13/strata.exe'),'e75716a9d5deb6289e87741d42e7003ba82b7696f6121d5203e615d9fe596a96');
 result.nativeBytesUnchanged=true;assert.equal(result.pageErrors,0);
 await Promise.all(responseChecks);
 // The existing restart contract deliberately returns 404/no-turn for a lost
 // in-memory attachment. Require that exact response; all other errors still fail.
 assert.ok(result.failedResponses.every(r=>r.expectedNoTurn));
 assert.ok(result.consoleDiagnostics.every(r=>r.resourceError&&r.status===404&&r.pathHash===hash('/api/chat/attach')));
 assert.equal(result.consoleErrors,result.failedResponses.length);
 result.consoleClean=result.consoleErrors===0;result.handledNoTurnResponses=result.failedResponses.length;result.passed=true;
} catch(error){result.errorType=error.name;result.errorCode=error.code??null;process.exitCode=1;}
finally {
 // This process is ours. Do not close it if a person has begun a request meanwhile.
 if(base&&await isIdle().catch(()=>false)){
  await browser?.close().catch(()=>{});if(child.exitCode===null)child.kill();
  for(let i=0;i<40&&child.exitCode===null&&child.signalCode===null;i++)await delay(250);
  result.ownedIdleShellStopped=child.exitCode!==null||child.signalCode!==null;
 } else {await browser?.close().catch(()=>{});child.unref();result.ownedIdleShellStopped=false;result.passed=false;process.exitCode=1;}
 writeFileSync(join(out,'installed-startup-'+attempt+'.json'),JSON.stringify(result,null,2)+'\n');
 if(result.passed)writeFileSync(join(out,'installed-startup-qualified.json'),JSON.stringify(result,null,2)+'\n');
 console.log(JSON.stringify({passed:result.passed,checks:result.checks,hostVersion:result.hostVersion,shellVersion:result.shellVersion,controlVersion:result.controlVersion,nativeVersion:result.nativeVersion,modelContextSettingsRetained:result.modelContextSettingsRetained,ownedIdleShellStopped:result.ownedIdleShellStopped}));
}
