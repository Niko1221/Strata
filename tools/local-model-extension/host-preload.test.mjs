import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {createServer} from 'node:net';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
const sha=data=>createHash('sha256').update(data).digest('hex');
async function fixture(t,{enabled=true,foreign=false,version='0.2.0-beta',badConfig=false}={}){
 const root=await mkdtemp(join(tmpdir(),'strata-preload-fixture '));
 await mkdir(join(root,'host/server'),{recursive:true});await mkdir(join(root,'host/dist'));
 const original='<html><head><title>Official Fixture</title></head><body>Official Catalog</body></html>';
 await writeFile(join(root,'host/dist/index.html'),original);
 const payload={'ui/entry.mjs':'export const active=true;','ui/index.html':'<html>Example Extension</html>'};
 const manifest={schema:1,id:'strata-local-models',version:'1.0.0',bridgeApi:1,app:{major:0,minor:2},entry:'ui/entry.mjs',panel:'ui/index.html',
  files:Object.entries(payload).map(([path,data])=>({path,bytes:Buffer.byteLength(data),sha256:sha(data)}))};
 const raw=JSON.stringify(manifest),hash=sha(raw),release='1.0.0-'+hash.slice(0,12),extensionRoot=join(root,'strata-local-models');
 await mkdir(join(extensionRoot,'releases',release,'ui'),{recursive:true});
 await writeFile(join(extensionRoot,'releases',release,'manifest.json'),raw);
 for(const [name,data] of Object.entries(payload))await writeFile(join(extensionRoot,'releases',release,name),data);
 await writeFile(join(extensionRoot,'active.json'),JSON.stringify({schema:1,enabled:true,active:{release,sha256:hash},previous:null}));
 const config=join(root,'extension.json');await writeFile(config,badConfig?'not json':JSON.stringify({app_executable:join(root,'app.exe'),extension_root:extensionRoot,owner_directory:root}));
 const probe=createServer();await new Promise(r=>probe.listen(0,'127.0.0.1',r));const port=probe.address().port;await new Promise(r=>probe.close(r));
 const entry=join(root,foreign?'other.mjs':'host/server/index.mjs');
 await writeFile(entry,`import http from 'node:http';const s=http.createServer((req,res)=>{res.end(${JSON.stringify(original)});});s.listen(Number(process.env.STRATA_PORT),'127.0.0.1',()=>console.log('ready'));`);
 const env={...process.env,STRATA_PORT:String(port),STRATA_APP_VERSION:version};
 delete env.STRATA_LOCAL_MODELS_CONFIG;
 if(enabled)env.STRATA_LOCAL_MODELS_CONFIG=config;
 const child=spawn(process.execPath,['--import',new URL('./host-preload.mjs',import.meta.url).href,entry],{env,windowsHide:true,stdio:['ignore','pipe','pipe']});
 let errors='';child.stderr.on('data',part=>errors+=part);
 const exited=once(child,'exit');
 t.after(async()=>{if(child.exitCode===null){child.kill();await exited;}await rm(root,{recursive:true,force:true,maxRetries:3});});
 await Promise.race([once(child.stdout,'data'),exited.then(()=>{throw new Error('Fixture host exited: '+errors);}),new Promise((_,reject)=>{const timer=setTimeout(()=>reject(new Error('Fixture host timeout')),5000);timer.unref();})]);
 return {base:'http://127.0.0.1:'+port,original};
}
test('the exact opted-in host serves its installed package without modifying official files',async t=>{
 const f=await fixture(t),html=await (await fetch(f.base+'/')).text();assert.match(html,/data-strata-local-models-extension/);
 assert.equal((await (await fetch(f.base+'/__strata_extension/api/status')).json()).compatible,true);
});
test('default launch, inherited child tools, malformed config and incompatible App versions keep the official page',async t=>{
 for(const options of [{enabled:false},{foreign:true},{badConfig:true},{version:'0.3.0'}]){
  const f=await fixture(t,options);assert.equal(await (await fetch(f.base+'/')).text(),f.original);
 }
});
