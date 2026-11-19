/** Real extracted NSIS/WebView2 QA, isolated profile, existing GGUF metadata only. */
import assert from 'node:assert/strict';
import {spawn,spawnSync} from 'node:child_process';
import {createServer} from 'node:net';
import {createRequire} from 'node:module';
import {mkdirSync,readFileSync,writeFileSync,existsSync,readdirSync,realpathSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {setTimeout as delay} from 'node:timers/promises';
import {sha256} from '../../app/host/scripts/release-integrity.mjs';
const root=resolve(import.meta.dirname,'../..'),out='H:/Strata/releases/local-052-20261106-opencode';
const attempt=process.argv[2]??'r1';assert.match(attempt,/^r[1-9][0-9]*$/);
const qa=JSON.parse(readFileSync(join(out,'app-qualification.json')));assert.equal(qa.ok,true);
const exe=join(qa.app,'strata-desktop.exe');assert.equal(sha256(exe),qa.shellSHA256);
const home=join(out,'webview-'+attempt);assert.equal(existsSync(home),false);
mkdirSync(home);for(const dir of ['data','models','local','roaming','desktop','webview','temp'])mkdirSync(join(home,dir));
writeFileSync(join(home,'data/state.json'),JSON.stringify({version:1,setup:{complete:true,flowVersion:2,step:4,history:[]},installed:{},current:null,loaded:null,beginner:{priority:'balanced',contextLength:'128k',readImages:false,gpuIdle:'never'},expert:{}}));
const socket=createServer();await new Promise(r=>socket.listen(0,'127.0.0.1',r));const port=socket.address().port;await new Promise(r=>socket.close(r));
const env=Object.fromEntries(Object.entries(process.env).filter(([k])=>!/^(STRATA_|DSH_|WEBVIEW2_)/i.test(k)&&!/(?:API_KEY|TOKEN|SECRET|PASSWORD|ATTEST_KEY)$/i.test(k)));
Object.assign(env,{USERPROFILE:home,HOME:home,STRATA_HOME:home,STRATA_WORKSPACE:join(home,'workspace'),STRATA_MODELS_DIR:join(home,'models'),STRATA_COMPONENTS_DIR:join(home,'components'),STRATA_SETTINGS_FILE:join(home,'settings.json'),STRATA_RUNTIME_DIR:join(home,'runtime'),STRATA_DESKTOP_DATA_DIR:join(home,'desktop'),STRATA_REAL_ENGINE:'1',STRATA_NO_ACQUIRE:'1',STRATA_NO_BUNDLED_ENGINE:'1',STRATA_NO_REPAIR_ACCESS:'1',STRATA_AGENT:'off',STRATA_NO_DIALOG:'1',DSH_HOME:join(home,'harness'),LOCALAPPDATA:join(home,'local'),APPDATA:join(home,'roaming'),TEMP:join(home,'temp'),TMP:join(home,'temp'),WEBVIEW2_USER_DATA_FOLDER:join(home,'webview'),WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS:'--remote-debugging-port='+port,PYTHONDONTWRITEBYTECODE:'1'});
// Keep Windows identity discovery intact, as in the previously qualified recipe.
// Every application state path and the WebView profile remain explicitly isolated.
env.USERPROFILE=process.env.USERPROFILE;if(process.env.HOME)env.HOME=process.env.HOME;else delete env.HOME;
env.WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS+=' --remote-debugging-address=127.0.0.1 --enable-logging --log-file='+join(home,'webview-debug.log');
const child=spawn(exe,[],{env,cwd:qa.app,windowsHide:true,stdio:'ignore'});
const {chromium}=createRequire(join(root,'apps/host/package.json'))('playwright-core');
const result={passed:false,shellSHA256:qa.shellSHA256,actualWebView2:false,isolatedProfile:true,modelLoadRequested:false,checks:[],pageErrors:0,consoleErrors:0};
let browser,page;
const deadline=setTimeout(()=>{child.kill();},240000);
try {
 for(let i=0;i<120;i++){
  assert.equal(child.exitCode,null);
  try{if((await fetch('http://127.0.0.1:'+port+'/json/version',{signal:AbortSignal.timeout(1000)})).ok){browser=await chromium.connectOverCDP('http://127.0.0.1:'+port);break;}}catch(error){result.lastCDPError=error.message;result.lastCDPCause=error.cause?.code;}
  await delay(500);
 }
 assert.ok(browser);result.actualWebView2=true;
 for(let i=0;i<120;i++){page=browser.contexts().flatMap(c=>c.pages()).find(p=>/^http:\/\/127\.0\.0\.1:\d+/.test(p.url()));if(page)break;await delay(500);}
 assert.ok(page);page.setDefaultTimeout(20000);const base=new URL(page.url()).origin;
 page.on('pageerror',()=>result.pageErrors++);page.on('console',m=>{if(m.type()==='error')result.consoleErrors++;});
 await page.addInitScript(()=>localStorage.setItem('strata.prefs.v2',JSON.stringify({language:'en'})));
 await page.reload({waitUntil:'domcontentloaded'});
 assert.equal((await fetch(base+'/api/health')).status,200);result.checks.push('actual-shell-host');
 const build=await (await fetch(base+'/api/app/build')).json();assert.equal(build.host,'0.5.2-local.1');assert.equal(build.shell,'0.5.2-local.1');
 const paths=['H:/Strata/models/models/qwen36-UD-Q4_K_XL/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf','C:/Users/downl/Desktop/SO8T/gguf_models/zapabobouj/Hikari-Bonsai-27B-Nimble-LoRA-GGUF/Hikari-LoRA-fixed-scale-PQ2_0.gguf','H:/Strata/models/SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ2_XS-00001-of-00002.gguf'];
 assert.ok(paths.every(p=>existsSync(p)));const ids=[];
 for(const path of paths){const r=await fetch(base+'/api/local-models',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({path}),signal:AbortSignal.timeout(30000)});assert.equal(r.status,200);ids.push((await r.json()).variantId);}
 const repeated=await fetch(base+'/api/local-models',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({path:paths[0]})});assert.equal((await repeated.json()).variantId,ids[0]);
 const library=await (await fetch(base+'/api/models')).json();assert.equal(library.loaded,null);assert.equal(library.engine.status,'stopped');
 for(let i=0;i<ids.length;i++){const v=library.variants.find(v=>v.id===ids[i]);assert.ok(v);assert.equal(v.local.path,realpathSync(paths[i]));assert.equal(v.downloadGb,0);if(i===2)assert.equal(v.local.files.length,2);}
 assert.equal(readdirSync(join(home,'models'),{recursive:true}).some(p=>/\.gguf$/i.test(p)),false);result.checks.push('three-existing-gguf-metadata','split-shards','idempotent-import','no-weight-copy');
 await page.goto(base+'/#/models/local?variant='+encodeURIComponent(ids[1]));await page.getByRole('button',{name:'Manage models',exact:true}).waitFor();
 assert.match(await page.getByRole('combobox',{name:'Model version',exact:true}).innerText(),/Hikari.*PQ2_0/);result.checks.push('hikari-name');
 await page.goto(base+'/#/models/local?variant='+encodeURIComponent(ids[0]));await page.getByRole('button',{name:'Manage models',exact:true}).waitFor();
 await page.getByRole('group',{name:'Context presets',exact:true}).getByRole('button',{name:'96K',exact:true}).click();assert.equal(await page.getByRole('spinbutton',{name:'Custom context in K tokens'}).inputValue(),'96');
 await page.getByRole('group',{name:'Context presets',exact:true}).getByRole('button',{name:'128K',exact:true}).click();
 await page.getByRole('button',{name:'Expert mode',exact:true}).click();await page.getByRole('textbox',{name:'MoE layers in CPU memory',exact:true}).fill('28');
 await page.getByRole('tab',{name:'JSON',exact:true}).click();const config=JSON.parse(await page.getByRole('textbox',{name:'Run configuration JSON'}).inputValue());
 assert.equal(config.expert['gpu.cpuMoeLayers'],28);assert.equal(config.beginner.contextLength,'128k');result.checks.push('context-moe-editor');
 await page.getByRole('button',{name:'Saved settings',exact:true}).click();await page.getByRole('menuitem',{name:'Save current settings…',exact:true}).click();await page.getByRole('textbox',{name:'Settings name'}).fill('Local candidate isolated QA');await page.getByRole('dialog').getByRole('button',{name:'Save settings',exact:true}).click();await page.getByRole('dialog').waitFor({state:'hidden'});
 const saved=await page.evaluate(()=>JSON.parse(localStorage.getItem('strata.run-profiles.v1')));assert.equal(saved.profiles.at(-1).config.expert['gpu.cpuMoeLayers'],28);
 await page.reload();await page.getByRole('button',{name:'Manage models',exact:true}).waitFor();assert.deepEqual(await page.evaluate(()=>JSON.parse(localStorage.getItem('strata.run-profiles.v1'))),saved);result.checks.push('webview-settings-reload');
 for(const width of [1440,768,375]){await page.setViewportSize({width,height:1000});await page.goto(base+'/#/models/local?variant='+encodeURIComponent(ids[0]));await page.getByRole('button',{name:'Manage models',exact:true}).waitFor();assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));result.checks.push('layout-'+width);}
 await page.setViewportSize({width:1440,height:1000});
 for(const route of ['chat','monitor','skills','tasks']){await page.goto(base+'/#/'+route);await page.locator('main').first().waitFor();result.checks.push('route-'+route);}
 await page.goto(base+'/#/skills');await page.getByRole('button',{name:'Import SKILL.md',exact:true}).click();
 await page.getByRole('textbox',{name:'Or paste SKILL.md',exact:true}).fill('---\nname: Local candidate QA\ndescription: Isolated instruction-only test\n---\nSummarize the provided text.');
 await page.getByRole('button',{name:'Preview skill',exact:true}).click();await page.getByRole('textbox',{name:'Instructions',exact:true}).waitFor();
 await page.getByRole('textbox',{name:/^Trigger phrases/}).fill('isolated-qa');await page.getByRole('switch',{name:'Skill enabled',exact:true}).click();
 await page.getByRole('dialog').getByRole('button',{name:'Import',exact:true}).click();
 await page.getByRole('dialog',{name:'Import SKILL.md',exact:true}).waitFor({state:'hidden'});result.checks.push('integrated-skill-preview-import');
 await page.goto(base+'/#/chat');await page.getByRole('button',{name:'Search web',exact:true}).click();await page.getByRole('textbox',{name:'Search query',exact:true}).waitFor();
 await page.getByRole('dialog').getByRole('button',{name:'Cancel',exact:true}).click();result.checks.push('integrated-explicit-search-dialog');
 const final=await (await fetch(base+'/api/models')).json();assert.equal(final.loaded,null);assert.equal(final.engine.status,'stopped');
 assert.equal(result.pageErrors,0);result.passed=true;
} catch(error){result.error=error.stack;process.exitCode=1;
 if(child.exitCode===null){const probe=spawnSync('C:/Users/downl/AppData/Local/Programs/Python/Python312/python.exe',['-B',join(import.meta.dirname,'diagnose_webview.py'),String(child.pid),String(port)],{encoding:'utf8',timeout:15000,windowsHide:true});if(probe.status===0)result.diagnostic=JSON.parse(probe.stdout);}
}
finally {
 clearTimeout(deadline);await browser?.close().catch(()=>{});if(child.exitCode===null)child.kill();
 for(let i=0;i<40&&child.exitCode===null&&child.signalCode===null;i++)await delay(250);
 result.ownedShellStopped=child.exitCode!==null||child.signalCode!==null;
 writeFileSync(join(out,'ui-result-'+attempt+'.json'),JSON.stringify(result,null,2)+'\n');if(result.passed)writeFileSync(join(out,'packaged-ui.json'),JSON.stringify(result,null,2)+'\n');
 console.log(JSON.stringify({passed:result.passed,checks:result.checks.length,pageErrors:result.pageErrors,consoleErrors:result.consoleErrors,ownedShellStopped:result.ownedShellStopped}));
}
