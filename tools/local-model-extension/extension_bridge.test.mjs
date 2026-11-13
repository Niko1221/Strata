import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve, join, dirname, basename } from 'node:path';
import { createHash } from 'node:crypto';
import { createExtensionBridge, loadExtension, PREFIX } from './extension_bridge.mjs';

const sha = data => createHash('sha256').update(data).digest('hex');
async function fixture(t, { appVersion='0.2.0-beta', bridgeApi=1, features }={}) {
  const root = await mkdtemp(join(tmpdir(),'strata-extension-test-'));
  t.after(async () => {
    assert.equal(dirname(root),resolve(tmpdir())); assert.ok(basename(root).startsWith('strata-extension-test-'));
    await rm(root,{recursive:true,force:true});
  });
  const extensionRoot=join(root,'strata-local-models'), officialWebRoot=join(root,'official');
  await mkdir(extensionRoot,{recursive:true}); await mkdir(officialWebRoot);
  const html='<html><head><title>Official App</title><script src="/assets/official-new.js"></script></head><body><main id="root">Official Model Catalog</main></body></html>';
  await writeFile(join(officialWebRoot,'index.html'),html);
  const payload={'ui/entry.mjs':'export const entry=true;', 'ui/index.html':'<html>Extension Panel</html>', 'ui/panel.mjs':'export const panel=true;'};
  const manifest={schema:1,id:'strata-local-models',version:'1.0.0',bridgeApi,app:{major:0,minor:2},entry:'ui/entry.mjs',panel:'ui/index.html',
    files:Object.entries(payload).map(([path,data])=>({path,bytes:Buffer.byteLength(data),sha256:sha(data)}))};
  if(features!==undefined)manifest.features=features;
  const raw=JSON.stringify(manifest), hash=sha(raw), release='1.0.0-'+hash.slice(0,12);
  const releaseRoot=join(extensionRoot,'releases',release);
  await mkdir(join(releaseRoot,'ui'),{recursive:true});
  await writeFile(join(releaseRoot,'manifest.json'),raw);
  for(const [name,data] of Object.entries(payload))await writeFile(join(releaseRoot,name),data);
  const state={schema:1,enabled:true,active:{release,sha256:hash},previous:null};
  const saveState=()=>writeFile(join(extensionRoot,'active.json'),JSON.stringify(state)); await saveState();
  const delegated=[];
  const bridge=createExtensionBridge({extensionRoot,officialWebRoot,ownerDirectory:root,appVersion});
  const original=async(req,res)=>{delegated.push(req.url);res.writeHead(200,{'content-type':req.url==='/'?'text/html':'application/json'});res.end(req.url==='/'?await readFile(join(officialWebRoot,'index.html')):JSON.stringify({official:true,path:req.url}));};
  const server=http.createServer(async(req,res)=>{if(!await bridge.handle(req,res))await original(req,res);});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  t.after(async()=>{server.closeAllConnections();await new Promise(resolve=>server.close(resolve));});
  return {root,extensionRoot,officialWebRoot,releaseRoot,manifest,state,saveState,bridge,html,delegated,base:'http://127.0.0.1:'+server.address().port};
}

test('adds only its entry to current official HTML, keeping official assets and catalog',async t=>{
  const f=await fixture(t), response=await fetch(f.base+'/'), html=await response.text();
  assert.match(html,/Official Model Catalog/); assert.match(html,/assets\/official-new.js/);
  assert.match(html,/data-strata-local-models-extension/); assert.doesNotMatch(html,/__desktop_ui|__STRATA_LOCAL_MODEL__/);
  assert.equal(html.replace(/<script type="module" data-strata-local-models-extension[^>]*><\/script>\n/,''),f.html);
  assert.equal(Number(response.headers.get('content-length')),Buffer.byteLength(html));
  assert.equal((await (await fetch(f.base+'/assets/official-new.js')).json()).official,true);
  assert.equal((await (await fetch(f.base+'/api/native')).json()).official,true);
  assert.equal((await (await fetch(f.base+'/api/updates/check')).json()).official,true);
  const newer='<html><head><script src="/assets/official-after-update.js"></script></head><body>New Upstream UI</body></html>';
  await writeFile(join(f.officialWebRoot,'index.html'),newer);
  const updated=await (await fetch(f.base+'/')).text();
  assert.match(updated,/official-after-update/); assert.match(updated,/New Upstream UI/); assert.doesNotMatch(updated,/official-new.js/);
});

test('extension assets are separate, allowlisted and cannot expose adjacent files',async t=>{
  const f=await fixture(t); await writeFile(join(f.root,'secret.txt'),'PRIVATE FILE');
  const entry=await fetch(f.base+PREFIX+'ui/entry.mjs'); assert.match(entry.headers.get('content-type'),/javascript/);
  assert.equal(await entry.text(),'export const entry=true;');
  for(const path of ['ui/not-listed.mjs','ui/..%2F..%2Fsecret.txt','manifest.json','active.json']){
    const response=await fetch(f.base+PREFIX+path); assert.equal(response.status,404); assert.doesNotMatch(await response.text(),/PRIVATE FILE/);
  }
  const status=await (await fetch(f.base+PREFIX+'api/status')).json();
  assert.equal(status.compatible,true); assert.equal(status.bridgeApi,1); assert.equal(status.extension.version,'1.0.0');
  assert.ok(!JSON.stringify(status).includes(f.root));
});

test('disabled extension immediately leaves the official UI and updater available',async t=>{
  const f=await fixture(t); f.state.enabled=false;await f.saveState();
  assert.equal(await (await fetch(f.base+'/')).text(),f.html);
  const result=await (await fetch(f.base+PREFIX+'api/status')).json();assert.equal(result.reason,'disabled');
  assert.equal((await fetch(f.base+PREFIX+'api/model')).status,503);
  assert.equal((await (await fetch(f.base+'/api/updates/check')).json()).official,true);
});

test('unsupported App and Bridge versions fail open to the official frontend',async t=>{
  for(const options of [{appVersion:'0.3.0'},{appVersion:'dev'},{bridgeApi:2}]){
    const f=await fixture(t,options);
    assert.equal(await (await fetch(f.base+'/')).text(),f.html);
    const state=await (await fetch(f.base+PREFIX+'api/status')).json();assert.equal(state.compatible,false);
    assert.match(state.reason,/unsupported/); assert.equal((await fetch(f.base+PREFIX+'api/model')).status,503);
  }
});

test('missing, corrupt or modified packages do not block the official App',async t=>{
  const f=await fixture(t);
  await writeFile(join(f.releaseRoot,'ui/entry.mjs'),'tampered');
  assert.equal(await (await fetch(f.base+'/')).text(),f.html);
  assert.equal((await (await fetch(f.base+PREFIX+'api/status')).json()).reason,'invalid_package');
  await writeFile(join(f.extensionRoot,'active.json'),'{broken');
  assert.equal(await (await fetch(f.base+'/')).text(),f.html);
  assert.equal((await (await fetch(f.base+'/api/native')).json()).official,true);
});

test('upstream HTML layout/CSP changes fall back rather than serving an old frontend',async t=>{
  const f=await fixture(t);
  for(const html of ['<html><body>Changed Template</body></html>', '<html><head><meta http-equiv="Content-Security-Policy" content="default-src none"></head><body>New Policy</body></html>']){
    await writeFile(join(f.officialWebRoot,'index.html'),html);
    assert.equal(await (await fetch(f.base+'/')).text(),html);
    assert.equal((await (await fetch(f.base+PREFIX+'api/status')).json()).reason,'unsupported_host_layout');
  }
});

test('cross-origin and non-loopback extension requests cannot reach model controls',async t=>{
  const f=await fixture(t);
  assert.equal((await fetch(f.base+PREFIX+'api/model',{headers:{origin:'https://example.com'}})).status,403);
  const foreignHost = await new Promise((resolve,reject)=>{
    http.get(f.base+PREFIX+'ui/entry.mjs',{headers:{host:'example.com'}},response=>{response.resume();resolve(response.statusCode);}).on('error',reject);
  });
  assert.equal(foreignHost,403);
  assert.equal((await fetch(f.base+PREFIX+'api/context',{method:'POST',headers:{'content-type':'application/json'},body:'{"tokens":262144}'})).status,403);
});

test('unsupported Node runtime is incompatible without loading Extension code',async t=>{
  const f=await fixture(t);
  const state=loadExtension({extensionRoot:f.extensionRoot,appVersion:'0.2.0-beta',nodeVersion:'20.0.0'});
  assert.equal(state.compatible,false);assert.equal(state.reason,'unsupported_app_version');
});

test('unsupported package feature declarations fail open',async t=>{
  for(const features of [{chatEffort:true},{unknown:true},[]]){
    const invalid=await fixture(t,{features});
    assert.equal(invalid.bridge.status().compatible,false);
    assert.equal(await (await fetch(invalid.base+'/')).text(),invalid.html);
  }
});
