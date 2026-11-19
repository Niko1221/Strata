/** Mounted picker interactions. Host load/trust verification remains a separate server contract. */
import './helpers/isolate-profile.mjs';
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { chromium } from 'playwright-core';
import { resolve } from 'node:path';
import { testBrowserExecutable, TEST_BROWSER_LAUNCH_TIMEOUT_MS } from './helpers/browser-executable.mjs';

const root = resolve(import.meta.dirname, '..');
const boundaries = {
  store: `import {useSyncExternalStore} from 'react';
const listeners=new Set();const put=patch=>{window.__state={...window.__state,...patch};listeners.forEach(f=>f());};
export const useStore=select=>select(useSyncExternalStore(f=>{listeners.add(f);return()=>listeners.delete(f)},()=>window.__state));
 export const useLifecycle=()=>({allowed:true});
 export const getState=()=>window.__state;
export const selectRemoteModel=value=>{window.__selections.push(value);put({remoteModel:value});};
export const loadVariant=async id=>{window.__loads.push(id);put({library:{...window.__state.library,loaded:id,variants:window.__state.library.variants.map(v=>({...v,state:v.id===id?'loaded':'installed'}))}});return true;};
export const go=()=>{},setSettingsSection=()=>{};`,
  native: `import {useSyncExternalStore} from 'react';
 const listeners=new Set();const put=patch=>{window.__native={...window.__native,...patch};listeners.forEach(f=>f());};
 export const useNative=()=>useSyncExternalStore(f=>{listeners.add(f);return()=>listeners.delete(f)},()=>window.__native);
 export const isNativeJobBusy=j=>j.state==='running'||j.state==='cancelling';
 export const loadNativeChatModel=model=>{window.__nativeLoads.push(model);return new Promise((resolve,reject)=>{window.__finishNative=ok=>{if(!ok){reject(new Error('Model load failed'));return;}const id='instance-'+model;put({instances:[{id,model,label:'Q38 test',status:'ready',gpus:[]}]});resolve(id);};});};
 export const selectNativeChatInstance=async id=>{window.__nativeSelections.push(id);put({defaultId:id});};`,
  i18n: 'export const localeTag=()=>\'en-US\';export const tr=(value,args)=>typeof value===\'string\'?value.replace(/\\{(\\d+)\\}/g,(_,i)=>args?.[i]??_):value;',
  ui: `import React from 'react';export const Icon=()=>null,Bar=()=>null,ErrorNotice=({message})=><p role="alert">{message}</p>;
export const Btn=({children,onClick,disabled})=><button onClick={onClick} disabled={disabled}>{children}</button>;
export const Dialog=({open,title,children,actions})=>open?<section role="dialog" aria-label={title}>{children}{actions}</section>:null;`,
  FloatingMenu: 'export const FloatingMenu=({children})=>children;',
};
const bundle = await build({
  stdin:{contents:`import React from 'react';import {createRoot} from 'react-dom/client';import {ChatModelPicker} from './src/components/ChatModelPicker';createRoot(document.getElementById('root')).render(<ChatModelPicker/>);`,resolveDir:root,loader:'tsx'},
  bundle:true,write:false,format:'iife',platform:'browser',jsx:'automatic',define:{'process.env.NODE_ENV':'"test"'},
  plugins:[{name:'controlled-picker-boundaries',setup(b){
    b.onResolve({filter:/\/lib\/(store|native|i18n)$|^\.\/(ui|FloatingMenu|i18n)$/},a=>({path:a.path.split('/').at(-1),namespace:'boundary'}));
    b.onLoad({filter:/.*/,namespace:'boundary'},a=>({contents:boundaries[a.path],loader:'jsx',resolveDir:root}));
  }}],
});
let browserPromise;
after(async()=>{if(browserPromise)await browserPromise.then(b=>b.close(),()=>{});});
async function fixture(t,{backend='Automatic',loaded=null,native=null}={}) {
  const env={...process.env};
  if(process.env.STRATA_TEST_WINDOWS_PROFILE){env.USERPROFILE=process.env.STRATA_TEST_WINDOWS_PROFILE;delete env.HOME;}
  browserPromise??=chromium.launch({executablePath:testBrowserExecutable(),headless:true,env,timeout:TEST_BROWSER_LAUNCH_TIMEOUT_MS,args:['--disable-gpu','--disable-background-networking','--disable-component-update','--no-first-run']});
  const context=await (await browserPromise).newContext();t.after(()=>context.close());
  const page=await context.newPage();page.setDefaultTimeout(5000);
  const errors=[];page.on('pageerror',e=>errors.push(e.message));t.after(()=>assert.deepEqual(errors,[]));
  await page.route(/^https?:/,r=>r.abort());
  await page.route('http://127.0.0.1/',r=>r.fulfill({contentType:'text/html',body:'<!doctype html><div id="root"></div>'}));
  await page.goto('http://127.0.0.1/');
  await page.evaluate(({backend,loaded,native})=>{
    window.__selections=[];window.__loads=[];
    window.__native=native;window.__nativeLoads=[];window.__nativeSelections=[];
    window.__state={activeSession:'chat-one',remoteModel:null,streaming:false,hardware:null,
      engine:{beginner:{contextLength:'32K'},expert:[{id:'runtime.backend',value:backend}]},
      providers:{connections:[{id:'remote',name:'Provider',models:[{id:'api',name:'Remote API'}]}]},
      library:{mode:'real',loaded,current:'local',variants:[{id:'local',name:'Imported model',family:'Imported family',local:{path:'fixture.gguf'},quant:'Q4_K_M',state:loaded?'loaded':'installed',diskGb:1,contextMax:32768,ramClass:{minGb:1,maxGb:2}}]}};
  },{backend,loaded,native});
  await page.addScriptTag({content:bundle.outputFiles[0].text});
  return page;
}
const open=p=>p.getByRole('button',{name:/^Choose model:/}).click();
for(const backend of ['PrismML','TurboQuant']) {
  test(`${backend}: loaded local -> remote API -> same local without reloading`,async t=>{
    const p=await fixture(t,{backend,loaded:'local'});
    await open(p);await p.getByRole('menuitemradio',{name:/Remote API/}).click();
    await p.getByRole('button',{name:'Choose model: Remote API'}).waitFor();
    await open(p);await p.getByRole('menuitemradio',{name:/Imported family/}).click();
    await p.getByRole('button',{name:'Choose model: Imported model'}).waitFor();
    assert.deepEqual(await p.evaluate(()=>window.__selections),[{connectionId:'remote',modelId:'api'},null]);
    assert.deepEqual(await p.evaluate(()=>window.__loads),[]);
  });
  test(`${backend}: installed import still uses the confirmed load action`,async t=>{
    const p=await fixture(t,{backend});await open(p);
    await p.getByRole('menuitemradio',{name:/Imported family/}).click();
    await p.getByRole('dialog').waitFor();assert.deepEqual(await p.evaluate(()=>window.__loads),[]);
    await p.getByRole('button',{name:'Switch version',exact:true}).click();
    await p.getByRole('button',{name:'Choose model: Imported model'}).waitFor();
    assert.deepEqual(await p.evaluate(()=>window.__loads),['local']);
  });
}
test('Automatic does not offer an unloaded generic import',async t=>{
  const p=await fixture(t);await open(p);
  assert.equal(await p.getByRole('menuitemradio',{name:/Imported family/}).count(),0);
  await p.getByRole('menuitemradio',{name:/Remote API/}).click();await open(p);
  assert.equal(await p.getByRole('menuitemradio',{name:/Imported family/}).count(),0);
  assert.deepEqual(await p.evaluate(()=>window.__loads),[]);
});
test('a loaded import remains reachable when the next-load backend setting is Automatic',async t=>{
  const p=await fixture(t,{loaded:'local'});await open(p);
  await p.getByRole('menuitemradio',{name:/Remote API/}).click();await open(p);
  await p.getByRole('menuitemradio',{name:/Imported family/}).click();
  await p.getByRole('button',{name:'Choose model: Imported model'}).waitFor();
  assert.deepEqual(await p.evaluate(()=>window.__loads),[]);
});

const nativeFixture = {
  instances: [{ id: 'another-instance', model: 'another', label: 'Other native', status: 'ready', gpus: [] }],
  defaultId: 'another-instance', jobs: [], installed: { models: [{ model: 'q38', model_name: 'Q38 test', ready: true }] },
  servingPolicies: { q38: { id: 'q38dense' } },
};
test('installed Q38 remains in the picker with another native model loaded and switches only after readiness', async t=>{
  const p=await fixture(t,{native:nativeFixture});
  await open(p);await p.getByRole('menuitemradio',{name:/Remote API/}).click();
  await open(p);await p.getByRole('menuitem',{name:/Q38 test/}).click();
  await p.getByRole('status').waitFor();
  await p.getByRole('button',{name:'Choose model: Remote API'}).waitFor();
  assert.deepEqual(await p.evaluate(()=>window.__nativeSelections),[]);
  await p.evaluate(()=>window.__finishNative(true));
  await p.getByRole('button',{name:'Choose model: Q38 test'}).waitFor();
  assert.deepEqual(await p.evaluate(()=>window.__nativeLoads),['q38']);
  assert.deepEqual(await p.evaluate(()=>window.__nativeSelections),['instance-q38']);
  assert.equal(await p.evaluate(()=>window.__state.remoteModel),null);
});
test('a failed Q38 load shows the error without clearing the chat routing choice', async t=>{
  const p=await fixture(t,{native:nativeFixture});
  await open(p);await p.getByRole('menuitemradio',{name:/Remote API/}).click();
  await open(p);await p.getByRole('menuitem',{name:/Q38 test/}).click();
  await p.getByRole('status').waitFor();
  await p.evaluate(()=>window.__finishNative(false));
  assert.equal(await p.getByRole('alert').innerText(),'Model load failed');
  assert.deepEqual(await p.evaluate(()=>window.__nativeSelections),[]);
  await p.getByRole('button',{name:'Choose model: Remote API'}).waitFor();
});
