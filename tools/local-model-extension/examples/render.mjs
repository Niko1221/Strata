/** Actual app components + extension assets, with anonymous simulated state. No inference. */
import {createRequire} from 'node:module';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {resolve,join,extname,sep} from 'node:path';
import http from 'node:http';
const repo=resolve(import.meta.dirname,'../../..'),root=join(repo,'apps/host'),out=import.meta.dirname;
const require=createRequire(join(root,'package.json'));
const {build}=require('esbuild'),{chromium}=require('playwright-core');
const temporary=join(out,'.fixture-build');mkdirSync(temporary,{recursive:true});
function stub(file,overrides={}){
 const source=readFileSync(join(root,'src/lib',file),'utf8');
 const names=new Set([...source.matchAll(/export\s+(?:async\s+)?(?:function\*?|const|let|var|class)\s+([A-Za-z_$][\w$]*)/g)].map(m=>m[1]));
 for(const m of source.matchAll(/export\s*\{([^}]*)\}/g))for(const part of m[1].split(','))names.add(part.split(/\s+as\s+/).pop().trim());
 names.delete('');
 return [...[...names].filter(n=>!(n in overrides)).map(n=>`export const ${n}=(...a)=>Promise.resolve({});`),
  ...Object.entries(overrides).map(([n,v])=>`export const ${n}=${v};`)].join('\n');
}
const native=JSON.parse(readFileSync(join(root,'tests/fixtures/ui-preview-native.json'),'utf8'));
native.repo='C:/Example/Strata';native.downloadDirectory='C:/Models/Strata';
native.installed={models:[],data_folder:'C:/Models/Strata',engine:{version:'0.1.45.3'}};
native.instances=[];native.jobs=[];native.defaultId=null;
native.hardware={cpu:{cores:16},ram_gb:64,gpus:[{index:0,name:'Example GPU',vram_gb:16,usable:true}]};
native.catalog.this_pc=native.hardware;
native.catalog.contexts=[8192,32768,65536,131072,204800,262144,393216,524288];
const store={setup:{complete:true,choices:{},app:{modelsDir:'C:/Models/Strata'}},hardware:null,imageModels:{variants:[]},
 library:{disk:{modelsDir:'C:/Models/Strata'},variants:[],downloads:{active:null},loaded:null},runtime:{instances:[]}};
const stubs={native:stub('native.ts',{useNative:'()=>globalThis.__native',nativeState:'()=>globalThis.__native'}),
 store:stub('store.ts',{useStore:'sel=>sel(globalThis.__store)',getState:'()=>globalThis.__store',lifecycleStatus:'()=>({allowed:true})'}),
 client:stub('client.ts'),router:stub('router.ts'),
 'kept-state':`import {useState} from 'react';\n`+stub('kept-state.ts',{useKept:'(key,initial)=>useState(globalThis.__kept?.[key]??initial)'})};
const plugin={name:'anonymous-state-only',setup(b){
 b.onResolve({filter:/\/lib\/(native|store|client|router|kept-state)$/},args=>({path:args.path.split('/').pop(),namespace:'fixture'}));
 b.onLoad({filter:/.*/,namespace:'fixture'},args=>({contents:stubs[args.path],loader:'js',resolveDir:root}));
}};
await build({stdin:{contents:`import React from 'react';import {createRoot} from 'react-dom/client';
 import {NativeModels} from './src/components/NativeModels';import {SettingsStorage} from './src/components/SettingsStorage';
 import {Screen} from './src/components/layout';
 globalThis.__native=${JSON.stringify(native)};globalThis.__store=${JSON.stringify(store)};
 globalThis.__kept={'models.family':'qwen','models.size':'q2_0','models.context':65536};
 const storage=location.pathname==='/storage.html';
 createRoot(document.getElementById('app')).render(storage?<Screen title="Storage"><SettingsStorage/></Screen>:<NativeModels onOtherModels={()=>{}}/>);`,
 resolveDir:root,loader:'tsx'},bundle:true,format:'esm',platform:'browser',outfile:join(temporary,'fixture.mjs'),
 jsx:'automatic',logLevel:'silent',plugins:[plugin],loader:{'.css':'empty','.svg':'dataurl','.png':'dataurl'}});
const css=[...readFileSync(join(root,'dist/index.html'),'utf8').matchAll(/href="(\/assets\/[^"?]+\.css)"/g)].map(m=>m[1]);
if(!css.length)throw new Error('Build The App Styles Before Rendering Examples.');
const caption='Sanitized Example · Simulated Hardware And State · No Inference';
const html=`<!doctype html><html lang="en" data-theme="dark"><head><meta charset="utf-8">${css.map(p=>`<link rel="stylesheet" href="${p}">`).join('')}
 <style>.main{width:100%;height:calc(100vh - 44px);margin:0}.example-caption{position:fixed;bottom:0;inset-inline:0;padding:12px 24px;background:var(--bg);border-top:1px solid var(--border);font:13px system-ui;color:var(--muted);z-index:2147483647}</style></head>
 <body><main class="main" id="app"></main><div class="example-caption">${caption}</div><script type="module" src="/fixture.mjs"></script><script type="module" src="/__strata_extension/ui/entry.mjs"></script></body></html>`;
const dto={available:true,model:{id:'example-qwen-local',name:'Qwen 3.8 Flash-Next',variant:'Example Quant · Local Server',loaded:false,loading:false,apiBaseUrl:'http://127.0.0.1:8080/v1'},
 context:{choices:native.catalog.contexts,selected:131072,effective:131072,phase:'ready',activeRequests:0,inFlight:0},autoUnload:{choices:[0,300,600,1800],selected:1800,effective:1800}};
const posts=[];
const server=http.createServer((req,res)=>{
 const path=new URL(req.url,'http://fixture').pathname;
 if(path.startsWith('/__strata_extension/api/')){
  const action=path.split('/').at(-1);if(req.method!=='GET'){posts.push(action);res.writeHead(409);res.end('{"error":"Read-Only Screenshot Fixture"}');return;}
  res.writeHead(200,{'content-type':'application/json'});res.end(JSON.stringify(action==='status'?{enabled:true,compatible:true,bridgeApi:1,extension:{id:'strata-local-models',version:'1.0.0'}}:dto));return;
 }
 if(['/models.html','/storage.html'].includes(path)){res.writeHead(200,{'content-type':'text/html'});res.end(html);return;}
 let file;
 if(path==='/fixture.mjs')file=join(temporary,'fixture.mjs');
 else if(path.startsWith('/__strata_extension/ui/'))file=resolve(out,'../package/ui',path.split('/').at(-1));
 else if(/^\/(assets|fonts|icons)\//.test(path))file=resolve(root,'dist',path.slice(1));
 if(!file||![resolve(root,'dist')+sep,resolve(out,'../package/ui')+sep,temporary+sep].some(p=>file.startsWith(p))){res.writeHead(404);res.end();return;}
 try{res.writeHead(200,{'content-type':({'.mjs':'application/javascript','.html':'text/html','.css':'text/css','.woff2':'font/woff2','.svg':'image/svg+xml'})[extname(file)]??'application/octet-stream'});res.end(readFileSync(file));}catch{res.writeHead(404);res.end();}
});
await new Promise(r=>server.listen(0,'127.0.0.1',r));const origin='http://127.0.0.1:'+server.address().port;
let browser;
try{
 browser=await chromium.launch({executablePath:process.env.CHROME_PATH??'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',headless:true,args:['--disable-gpu','--force-device-scale-factor=1']});
 const page=await browser.newPage({viewport:{width:1440,height:1100},deviceScaleFactor:1,colorScheme:'dark'}),errors=[];
 page.on('pageerror',error=>errors.push(error.message));
 await page.route('**/*',route=>new URL(route.request().url()).origin===origin?route.continue():route.abort());
 async function screenshot(name,fullPage=false){
  await page.evaluate(()=>document.fonts.ready);
  const texts=[];for(const frame of page.frames())texts.push(await frame.locator('body').innerText());
  if(/[a-z]:[\\/]Users[\\/]/i.test(texts.join('\n')))throw new Error('Private User Path In Example');
  if(errors.length)throw new Error('Fixture Render Failed: '+errors.join('; '));
  await page.screenshot({path:join(out,name+'.png'),fullPage});
 }
 await page.goto(origin+'/models.html',{waitUntil:'networkidle'});
 await page.getByRole('button',{name:'Manage models',exact:true}).click();
 const tile=page.locator('strata-connected-local-models');
 await tile.getByRole('button',{name:'Use In Chat',exact:true}).waitFor();
 await page.waitForFunction(()=>!document.querySelector('strata-connected-local-models')?.shadowRoot.querySelector('button.primary').disabled);
 await tile.getByRole('button',{name:'Use In Chat',exact:true}).scrollIntoViewIfNeeded();
 await screenshot('manage-models');
 await page.getByRole('button',{name:'Done',exact:true}).click();
 await page.getByRole('button',{name:'Current Model',exact:true}).click();
 const panel=page.frameLocator('iframe[title="Current Model Controls"]');
 await panel.getByRole('button',{name:'Use In Chat',exact:false}).waitFor();
 await page.waitForFunction(()=>{const f=document.querySelector('strata-local-models-extension')?.shadowRoot.querySelector('iframe');return f?.contentDocument?.getElementById('load')?.disabled===false;});
 const choices=await panel.locator('#context-choice option').allTextContents();
 if(JSON.stringify(choices)!==JSON.stringify(['8K','32K','64K','128K','200K','256K','384K','512K']))throw new Error('The Example Must Show The Official Catalog Choices');
 await screenshot('current-model-in-app');
 await page.keyboard.press('Escape');
 await page.goto(origin+'/storage.html',{waitUntil:'networkidle'});
 await page.waitForFunction(()=>!document.querySelector('strata-connected-local-models')?.shadowRoot.querySelector('button.primary').disabled);
 await screenshot('storage');
 await page.setViewportSize({width:1100,height:1180});
 await page.goto(origin+'/__strata_extension/ui/index.html',{waitUntil:'networkidle'});
 await page.waitForFunction(()=>document.getElementById('load')?.disabled===false);
 await page.evaluate(caption=>{
  const footer=document.createElement('p');footer.textContent=caption;
  footer.style.cssText='padding:12px 28px;margin:0 auto;max-width:1000px;font:12px system-ui;color:#b0bfc1';
  document.body.append(footer);
 },caption);
 await screenshot('current-model',true);
 if(posts.length)throw new Error('Screenshots Must Not Mutate Model State');
 writeFileSync(join(temporary,'evidence.json'),JSON.stringify({caption,actualComponents:['NativeModels','SettingsStorage'],actualExtension:true,inference:false,contextChoices:choices,mutations:posts.length,errors},null,2)+'\n');
 console.log(JSON.stringify({images:['manage-models.png','current-model.png','storage.png'],contextChoices:choices,mutations:0,errors}));
}finally{await browser?.close();server.closeAllConnections();await new Promise(r=>server.close(r));}
