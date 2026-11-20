/** Supplemental component regressions: actual React/ModelRun, controlled in-memory boundaries, no Host or E1 claim. */
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { chromium } from 'playwright-core';
import { testBrowserExecutable, TEST_BROWSER_LAUNCH_TIMEOUT_MS } from './helpers/browser-executable.mjs';
import { mkdtemp, mkdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const ui = `import React from 'react';
export const Btn=({children,onClick,disabled})=>{if(children==='Load model')window.__runModel=onClick;if(children==='Save settings'||children==='Update settings')window.__saveProfile=onClick;return React.createElement('button',{onClick,disabled},children)};
export const Select=({ariaLabel,value,options,onChange,disabled})=>React.createElement('select',{'aria-label':ariaLabel,value,disabled,onChange:e=>onChange(e.target.value)},options.map(o=>React.createElement('option',{key:o.value,value:o.value},o.label)));
export const Screen=({children,titleDetail,actions})=>React.createElement('main',null,titleDetail,actions,children);
export const ErrorNotice=({message})=>React.createElement('p',{role:'alert'},message);
export const RunImageModel=({library,onSelect})=>React.createElement(React.Fragment,null,React.createElement('output',{'data-image':true},library?.selected??'off'),React.createElement('button',{'data-retry-image':true,onClick:()=>onSelect(library?.selected??'off')},'Retry image selection'));
export const Dialog=({open,children,actions})=>open?React.createElement('section',{role:'dialog'},children,actions):null,FloatingMenu=({children})=>React.createElement('div',null,children);
export const ActionError=()=>null,Icon=()=>null,DemoTest=()=>null,RunExpertEditor=()=>null,RunPreview=()=>null,RunningModels=()=>null,LocalModelImport=()=>null,GpuRoles=()=>null,NativeModels=()=>null,NativeLocalModel=()=>null,ModelsScreen=()=>null;`;
const store = `import {useSyncExternalStore} from 'react';
const listeners=new Set();
export const getState=()=>window.__fixture;
function put(patch){window.__fixture={...window.__fixture,...patch};listeners.forEach(f=>f());}
window.__putHardware=hardware=>put({hardware});
export const refreshHardware=async()=>{window.__refreshes++;};
export const useStore=select=>select(useSyncExternalStore(f=>{listeners.add(f);return()=>listeners.delete(f)},getState));
export function setConfigDraft(configDraft){put({configDraft});}
export function imageSelectionReady(selected){const s=getState();return !s.busy['image:select']&&s.imageModels?.selected===selected&&s.imageSelection.confirmed&&s.imageSelection.intent===selected&&!s.imageSelection.error;}
export function selectImageModel(selected){const request=window.__requests.length;put({imageSelection:{intent:selected,confirmed:false,error:null},busy:{'image:select':true}});return new Promise((resolve,reject)=>window.__requests.push({selected,resolve:()=>{window.__hostSelected=selected;const latest=request===window.__requests.length-1;put({imageModels:{...window.__fixture.imageModels,selected},...(latest?{imageSelection:{intent:selected,confirmed:true,error:null},busy:{'image:select':false}}:{})});resolve()},reject:reason=>{if(request===window.__requests.length-1)put({imageSelection:{intent:selected,confirmed:false,error:reason},busy:{'image:select':false}});reject(reason)}}));}
export const useLifecycle=()=>({allowed:true});
const forbidden=async()=>{throw Error('unexpected model operation')};
export const applyImageModel=async(selected=getState().imageSelection.intent)=>{if(!window.__allowRun)return forbidden();if(!imageSelectionReady(selected))throw getState().imageSelection.error??Error('unconfirmed image selection');window.__applies.push(window.__hostSelected);},loadRuntimeModel=async()=>{if(!window.__allowRun)return forbidden();window.__loads++;},loadVariant=forbidden,restartEngine=forbidden,saveEngine=forbidden;
export const pushToast=()=>{},validateEngine=async()=>{window.__validations++;return window.__holdValidation?new Promise((resolve,reject)=>window.__validationGates.push(Object.assign(resolve,{reject}))):({ok:true,issues:[]})};`;
const stubs = {
    store,
    native: 'export const useNative=()=>null,nativeSettled=()=>true,subscribeNative=()=>()=>{};',
    client: 'export const api={post:async path=>{if(window.__allowRun&&path==="/api/runtime/plan"){window.__plans++;return {allowed:true,memory:{},budget:{}}}throw Error("unexpected API")}};',
    i18n: 'export const tr=value=>value;',
    experience: 'export const contextTokens=value=>value==="recommended"?131072:parseFloat(value)*1024,estimate=()=>({}),execution=()=>({threads:4});',
    'run-memory': 'export const planMemory=()=>null,demoRunExperience=value=>value;',
    features: 'export const IMAGE_GENERATION_VISIBLE=true;',
};
const bundle = await build({
    stdin: { contents: `import React from 'react'; import {createRoot} from 'react-dom/client'; import {ModelRunScreen} from './src/screens/ModelRun'; window.__root=createRoot(document.getElementById('root')); window.__root.render(React.createElement(ModelRunScreen));`, resolveDir: root, loader: 'tsx' },
    bundle: true, write: false, format: 'iife', platform: 'browser', jsx: 'automatic',
    define: { 'process.env.NODE_ENV': '"test"' },
    plugins: [{ name: 'controlled-component-boundaries', setup(b) {
        b.onResolve({ filter: /\/components\/|\.\/Models$/ }, () => ({ path: 'ui', namespace: 'boundary' }));
        b.onResolve({ filter: /\/lib\/(store|native|client|i18n|experience|run-memory|features)$/ }, args => ({ path: args.path.split('/').at(-1), namespace: 'boundary' }));
        b.onLoad({ filter: /.*/, namespace: 'boundary' }, args => ({ contents: args.path === 'ui' ? ui : stubs[args.path], loader: 'jsx', resolveDir: root }));
    } }],
});

const model = (id, quant) => ({ id, name: id, label: id, family: 'Local models', quant, params: '27B', local: { architecture: 'qwen35' }, state: 'installed', contextMax: id === 'q36' ? 131072 : 32768, diskGb: 20, tags: [], kvModes: ['FP16'] });
const hardware = { host: { name: 'Test', platform: 'win32', arch: 'x64' }, cpu: { model: 'CPU', cores: 6, threads: 12, topologyKnown: true, topologySource: 'live' }, processors: [{ id: 'CPU0', socket: 'CPU0', model: 'CPU', cores: 6, threads: 12 }], memory: { totalGb: 32 }, gpus: [], accelerator: { backend: 'cpu', unified: false } };
const unknownHardware = { ...hardware, cpu: { ...hardware.cpu, cores: 12, topologyKnown: false, topologySource: 'unavailable' }, processors: [] };
const qconfig = { schema: 'strata.engine/1', beginner: { contextLength: '128k' }, expert: { 'ctx.length': '128K', 'gpu.cpuMoeLayers': 28, 'runtime.backend': 'Automatic' } };
const engine = { engine: { version: 'test' }, beginner: qconfig.beginner, expert: Object.entries(qconfig.expert).map(([id, value]) => ({ id, value, recommended: id === 'gpu.cpuMoeLayers' ? 0 : value })) };
const computer = JSON.stringify({ host: 'Test', platform: 'win32', arch: 'x64', cpu: 'CPU', cores: 6, ram: 32, processors: [['CPU0', 'CPU', 6]], gpus: [], backend: 'cpu' });
const empty = { schema: 1, profiles: [], defaults: {} };
const saved = { schema: 1, defaults: { [computer+'\nq36']: 'q', [computer+'\nhikari']: 'h' }, profiles: [
    { id: 'q', name: 'Q', variantId: 'q36', hardware: computer, engineVersion: 'test', config: qconfig, imageModel: 'q6' },
    { id: 'h', name: 'H', variantId: 'hikari', hardware: computer, engineVersion: 'test', config: { ...qconfig, beginner: { contextLength: '32k' }, expert: { 'ctx.length': '32K', 'gpu.cpuMoeLayers': 0, 'runtime.backend': 'PrismML' } }, imageModel: 'off' },
] };

let browserPromise;
after(async()=>{if(browserPromise)await browserPromise.then(browser=>browser.close(),()=>{});});
async function fixture(t, id, profiles = empty, draft = null, allowRun = false, hw = hardware) {
    const home = await mkdtemp(join(tmpdir(), 's-e2-'));
    const dirs = Object.fromEntries(['home','local','roaming','workspace','models','components','runtime','tmp'].map(name=>[name,join(home,name)]));
    await Promise.all(Object.values(dirs).map(path=>mkdir(path,{recursive:true})));
    const env = Object.fromEntries(Object.entries(process.env).filter(([key])=>!/^STRATA_|^DSH_|^WEBVIEW2_/i.test(key)));
    Object.assign(env,{LOCALAPPDATA:dirs.local,APPDATA:dirs.roaming,TEMP:dirs.tmp,TMP:dirs.tmp,STRATA_HOME:dirs.home,STRATA_WORKSPACE:dirs.workspace,STRATA_MODELS_DIR:dirs.models,STRATA_COMPONENTS_DIR:dirs.components,STRATA_RUNTIME_DIR:dirs.runtime,STRATA_SETTINGS_FILE:join(home,'settings.json'),STRATA_NO_ACQUIRE:'1',STRATA_NO_BUNDLED_ENGINE:'1',STRATA_NO_REPAIR_ACCESS:'1',STRATA_AGENT:'off'});
    // Windows token discovery needs the real user root; browser/app data remain task-only.
    if (process.env.STRATA_TEST_WINDOWS_PROFILE) { env.USERPROFILE=process.env.STRATA_TEST_WINDOWS_PROFILE; delete env.HOME; }
    browserPromise??=chromium.launch({ executablePath: testBrowserExecutable(), headless: true, env, timeout:TEST_BROWSER_LAUNCH_TIMEOUT_MS,args: ['--disable-gpu','--disable-background-networking','--disable-component-update','--no-first-run'] });
    const browser=await browserPromise,context=await browser.newContext();
    t.diagnostic('isolated browser context; fixture home: '+home);
    t.after(()=>context.close());
    const page = await context.newPage();
    page.setDefaultTimeout(10000);
    const errors=[]; page.on('pageerror',e=>errors.push(e.message));
    await page.route(/^https?:/,route=>route.abort());
    await page.route('http://127.0.0.1/',route=>route.fulfill({contentType:'text/html',body:'<!doctype html><html><body><div id="root"></div></body></html>'}));
    await page.goto('http://127.0.0.1/',{waitUntil:'domcontentloaded',timeout:TEST_BROWSER_LAUNCH_TIMEOUT_MS});
    assert.equal(await page.evaluate(()=>window.isSecureContext),true);
    await page.evaluate(({id,profiles,engine,hardware,models,draft,allowRun})=>{
        localStorage.clear();sessionStorage.clear();localStorage.setItem('strata.run-profiles.v1',JSON.stringify(profiles));
        if(id===null)localStorage.setItem('strata.run-local-variant.v1','hikari');
        if(draft)sessionStorage.setItem('strata.run-variant.v1',id);
        location.hash=id===null?'/models/local':'/models/local?variant='+id;
        window.__requests=[];
        window.__allowRun=allowRun;window.__hostSelected='q6';window.__loads=0;window.__applies=[];window.__plans=0;window.__validationGates=[];window.__validations=0;window.__refreshes=0;
        window.__fixture={engine,hardware,library:{mode:'simulated',loaded:null,current:null,variants:models,disk:{modelsDir:'isolated'}},runtime:null,imageModels:{selected:'q6',applied:'off',generationAvailable:true,variants:[]},imageSelection:{intent:null,confirmed:false,error:null},configDraft:draft,busy:{},connection:allowRun?'live':'offline'};
    },{id,profiles,engine,hardware:hw,draft,allowRun,models:[model('q36','Q4_K_M'),model('hikari','PQ2_0')]});
    await page.addScriptTag({content:bundle.outputFiles[0].text});
    await page.locator('select[aria-label="Model version"]').waitFor();
    t.after(()=>assert.deepEqual(errors,[]));
    return page;
}

test('mounted fresh explicit target without a saved default matches picker restoration, not global Q36 settings', async t => {
    const direct=await fixture(t,'hikari');
    const picked=await fixture(t,'q36');
    await picked.locator('select[aria-label="Model version"]').selectOption('hikari');
    await picked.waitForFunction(()=>new URLSearchParams(location.hash.split('?')[1]).get('variant')==='hikari');
    await picked.waitForFunction(()=>window.__fixture.configDraft!==null);
    const a=await direct.evaluate(()=>JSON.parse(window.__fixture.configDraft??JSON.stringify({beginner:window.__fixture.engine.beginner,expert:Object.fromEntries(window.__fixture.engine.expert.map(p=>[p.id,p.value]))})));
    const b=await picked.evaluate(()=>JSON.parse(window.__fixture.configDraft));
    assert.deepEqual(a,b);
    assert.equal(a.beginner.contextLength,'32k');
    assert.equal(a.expert['gpu.cpuMoeLayers'],0);
    assert.equal(a.expert['runtime.backend'],'PrismML');
});

test('mounted bare-route restoration initializes the durable target while an established own session draft survives', async t => {
    const restored=await fixture(t,null);
    await restored.waitForFunction(()=>new URLSearchParams(location.hash.split('?')[1]).get('variant')==='hikari');
    const config=await restored.evaluate(()=>JSON.parse(window.__fixture.configDraft));
    assert.equal(config.beginner.contextLength,'32k');
    assert.equal(config.expert['gpu.cpuMoeLayers'],0);
    const own={...qconfig,expert:{...qconfig.expert,'gpu.cpuMoeLayers':17}};
    const editing=await fixture(t,'q36',saved,JSON.stringify(own));
    assert.equal(await editing.evaluate(()=>JSON.parse(window.__fixture.configDraft).expert['gpu.cpuMoeLayers']),17);
});

test('mounted rapid Back/Forward binds intended images before delayed replies and ignores stale completions', async t => {
    const page=await fixture(t,'q36',saved);
    await page.waitForFunction(()=>JSON.parse(window.__fixture.configDraft??'{}').beginner?.contextLength==='128k');
    const initial=await page.evaluate(()=>window.__requests.length);
    await page.evaluate(()=>{location.hash='/models/local?variant=hikari';});
    await page.waitForFunction(n=>window.__requests.length>n,initial);
    await page.goBack();
    await page.waitForFunction(n=>window.__requests.length>n+1,initial);
    await page.goForward();
    await page.waitForFunction(n=>window.__requests.length>n+2,initial);
    const transitions=await page.evaluate(n=>window.__requests.slice(n).map(r=>r.selected),initial);
    assert.deepEqual(transitions,['off','q6','off']);
    await page.evaluate(n=>window.__requests[n+2].resolve(),initial);
    await page.evaluate(n=>window.__requests[n+1].resolve(),initial);
    assert.equal(await page.locator('[data-image]').textContent(),'off');
    if(initial) {
        await page.evaluate(()=>window.__requests[0].reject(new Error('obsolete model image failure')));
        await page.evaluate(()=>new Promise(resolve=>queueMicrotask(resolve)));
        assert.equal(await page.getByRole('alert').count(),0);
    }
    await page.evaluate(n=>window.__requests[n].resolve(),initial);
    await page.goBack();
    await page.waitForFunction(n=>window.__requests.length>n+3,initial);
    assert.equal(await page.locator('[data-image]').textContent(),'q6');
});

test('mounted failed off intent stays visible but disables loading and guards direct run admission until recovery', async t => {
    const page=await fixture(t,'hikari',empty,null,true);
    await page.waitForFunction(()=>window.__plans>0&&window.__requests.length===1);
    const load=page.getByRole('button',{name:'Load model',exact:true});
    assert.equal(await load.isDisabled(),true);
    await page.evaluate(()=>window.__requests[0].reject(new Error('Host selection failed: exact source reason')));
    await page.waitForFunction(()=>!window.__fixture.busy['image:select']);
    assert.equal(await page.locator('[data-image]').textContent(),'off');
    const disabled=await load.isDisabled();
    const attempts=await page.evaluate(async()=>{window.__runModel();for(let n=0;n<8;n++)await Promise.resolve();return {loads:window.__loads,applies:window.__applies};});
    assert.deepEqual(attempts,{loads:0,applies:[]},'calling the real handler bypassing DOM disabled must still refuse');
    assert.equal(disabled,true);
    assert.equal(await page.getByRole('alert').textContent(),'Host selection failed: exact source reason');
    await page.locator('[data-retry-image]').click();
    await page.waitForFunction(()=>window.__requests.length===2);
    assert.equal(await load.isDisabled(),true);
    await page.evaluate(()=>window.__requests[1].resolve());
    await page.waitForFunction(()=>!document.querySelector('button')?.disabled&&window.__fixture.imageSelection.confirmed);
    assert.equal(await load.isEnabled(),true);
    assert.equal(await page.getByRole('alert').count(),0);
    await page.evaluate(async()=>{window.__runModel();for(let n=0;n<8;n++)await Promise.resolve();});
    await page.waitForFunction(()=>window.__applies.length===1);
    assert.deepEqual(await page.evaluate(()=>({loads:window.__loads,applies:window.__applies})),{loads:1,applies:['off']});
});

test('mounted run rechecks confirmation after delayed validation before any fake model load', async t => {
    const page=await fixture(t,'hikari',empty,null,true);
    await page.waitForFunction(()=>window.__plans>0&&window.__requests.length===1);
    await page.evaluate(()=>window.__requests[0].resolve());
    await page.waitForFunction(()=>window.__fixture.imageSelection.confirmed);
    await page.evaluate(()=>{window.__holdValidation=true;window.__runModel();});
    await page.waitForFunction(()=>window.__validationGates.length>0);
    await page.locator('[data-retry-image]').click();
    await page.waitForFunction(()=>window.__requests.length===2);
    await page.evaluate(()=>{window.__requests[1].reject(new Error('selection changed during validation'));});
    await page.waitForFunction(()=>!window.__fixture.busy['image:select']);
    await page.evaluate(async()=>{window.__holdValidation=false;for(const resolve of window.__validationGates)resolve({ok:true,issues:[]});for(let n=0;n<8;n++)await Promise.resolve();});
    assert.deepEqual(await page.evaluate(()=>({loads:window.__loads,applies:window.__applies})),{loads:0,applies:[]});
});

async function startProfileSave(page,name){
    await page.getByRole('button',{name:/Saved settings/}).click();
    await page.getByRole('menuitem',{name:/Save current settings|Save as new/}).click();
    await page.getByRole('textbox',{name:'Settings name',exact:true}).fill(name);
    await page.getByRole('button',{name:'Save settings',exact:true}).click();
}
for(const outcome of ['success','invalid','failure'])test(`profile save A ${outcome} after B selection cannot own B dialog, error or pending`,async t=>{
    const page=await fixture(t,'q36');
    await page.waitForFunction(()=>window.__requests.length===1);
    await page.evaluate(()=>window.__requests[0].resolve());
    await page.waitForFunction(()=>window.__fixture.imageSelection.confirmed);
    await page.evaluate(()=>{window.__holdValidation=true});
    await startProfileSave(page,'A saved profile');
    await page.waitForFunction(()=>window.__validationGates.length===1);
    await page.evaluate(()=>{location.hash='/models/local?variant=hikari'});
    await page.waitForFunction(()=>JSON.parse(window.__fixture.configDraft??'{}').beginner?.contextLength==='32k');
    await page.evaluate(()=>window.__requests.at(-1).resolve());
    await page.waitForFunction(()=>window.__fixture.imageSelection.confirmed);
    const picker=page.getByRole('combobox',{name:'Model version',exact:true});
    const free=await picker.isEnabled();
    assert.equal(free,true,'new model must not inherit the previous model pending save');
    await startProfileSave(page,'B pending profile');
    await page.waitForFunction(()=>window.__validationGates.length===2);
    // Finish the old operation only after the current model's draft is visible.
    await page.evaluate(outcome=>{const gate=window.__validationGates[0];outcome==='failure'?gate.reject(Error('obsolete A validation failure')):gate({ok:outcome==='success',issues:outcome==='invalid'?[{level:'error',message:'obsolete A issue'}]:[]})},outcome);
    await page.evaluate(()=>new Promise(r=>setTimeout(r,0)));
    assert.equal(await page.getByRole('alert').count(),0);
    assert.equal(await page.getByRole('dialog').count(),1);
    assert.equal(await page.getByRole('textbox',{name:'Settings name',exact:true}).inputValue(),'B pending profile');
    assert.equal(await page.getByRole('button',{name:'Saving…',exact:true}).isDisabled(),true);
    assert.equal(await picker.inputValue(),'hikari');
    const records=await page.evaluate(()=>JSON.parse(localStorage.getItem('strata.run-profiles.v1')).profiles);
    assert.equal(records.length,outcome==='success'?1:0);
    if(outcome==='success'){assert.equal(records[0].variantId,'q36');assert.equal(records[0].name,'A saved profile');}
    await page.evaluate(()=>window.__validationGates[1]({ok:true,issues:[]}));
    await page.getByRole('dialog').waitFor({state:'hidden'});
    const final=await page.evaluate(()=>JSON.parse(localStorage.getItem('strata.run-profiles.v1')).profiles);
    assert.equal(final.length,outcome==='success'?2:1);assert.equal(final.at(-1).variantId,'hikari');
});

test('late A profile save merges with a newer completed B save and leaves B active',async t=>{
    const page=await fixture(t,'q36');await page.waitForFunction(()=>window.__requests.length===1);await page.evaluate(()=>window.__requests[0].resolve());await page.waitForFunction(()=>window.__fixture.imageSelection.confirmed);
    await page.evaluate(()=>{window.__holdValidation=true});await startProfileSave(page,'A late');await page.waitForFunction(()=>window.__validationGates.length===1);
    await page.evaluate(()=>{location.hash='/models/local?variant=hikari'});await page.waitForFunction(()=>JSON.parse(window.__fixture.configDraft??'{}').beginner?.contextLength==='32k');await page.evaluate(()=>window.__requests.at(-1).resolve());await page.waitForFunction(()=>window.__fixture.imageSelection.confirmed);
    await startProfileSave(page,'B first');await page.waitForFunction(()=>window.__validationGates.length===2);await page.evaluate(()=>window.__validationGates[1]({ok:true,issues:[]}));await page.getByRole('dialog').waitFor({state:'hidden'});
    await page.evaluate(()=>window.__validationGates[0]({ok:true,issues:[]}));await page.waitForFunction(()=>JSON.parse(localStorage.getItem('strata.run-profiles.v1')).profiles.length===2);
    assert.deepEqual(await page.evaluate(()=>JSON.parse(localStorage.getItem('strata.run-profiles.v1')).profiles.map(p=>[p.variantId,p.name])),[['hikari','B first'],['q36','A late']]);
    assert.equal(await page.locator('.run-saved-settings').textContent(),'B first');
});

test('unverified hardware leaves manual settings usable but disables profile provenance writes', async t => {
    const page = await fixture(t, 'q36', saved, null, false, unknownHardware);
    await page.getByRole('button', { name: /Saved settings/ }).click();
    assert.equal(await page.getByRole('menuitem', { name: /Save current settings|Save as new/ }).isDisabled(), true);
    assert.match(await page.locator('main').textContent(), /hardware.*unavailable|reading.*hardware/i);
    assert.equal(await page.getByRole('button', { name: '128K', exact: true }).isEnabled(), true);
    assert.deepEqual(await page.evaluate(() => JSON.parse(localStorage.getItem('strata.run-profiles.v1'))), saved);
});

test('unknown startup defers the verified default and restores it exactly once on verification', async t => {
    const page = await fixture(t, 'q36', saved, null, false, unknownHardware);
    assert.equal(await page.locator('.run-saved-settings').textContent(), 'Saved settings');
    await page.evaluate(hw => window.__putHardware(hw), hardware);
    await page.waitForFunction(() => document.querySelector('.run-saved-settings').textContent === 'Q');
    assert.equal(await page.evaluate(() => JSON.parse(window.__fixture.configDraft).expert['gpu.cpuMoeLayers']), 28);
    await page.getByRole('button', { name: '96K', exact: true }).click();
    await page.evaluate(hw => window.__putHardware({ ...hw, cpu: { ...hw.cpu, topologySource: 'cached' } }), hardware);
    assert.equal(await page.evaluate(() => JSON.parse(window.__fixture.configDraft).beginner.contextLength), '96k');
    assert.equal(await page.locator('.run-saved-settings').textContent(), 'Q');
});

test('edits while identity waits survive verification and model round trips', async t => {
    const page = await fixture(t, 'q36', saved, null, false, unknownHardware);
    await page.getByRole('button', { name: '96K', exact: true }).click();
    await page.getByRole('combobox', { name: 'Model version' }).selectOption('hikari');
    await page.evaluate(hw => window.__putHardware(hw), hardware);
    await page.waitForFunction(() => document.querySelector('.run-saved-settings').textContent === 'H');
    await page.getByRole('combobox', { name: 'Model version' }).selectOption('q36');
    assert.equal(await page.evaluate(() => JSON.parse(window.__fixture.configDraft).beginner.contextLength), '96k');
    assert.equal(await page.locator('.run-saved-settings').textContent(), 'Saved settings');
    await page.getByRole('button', { name: /Saved settings/ }).click();
    assert.equal(await page.getByRole('menuitemradio', { name: /Q/ }).count(), 1, 'exact verified profile remains available explicitly');
});

for (const update of [false, true]) test(`real profile ${update ? 'update/default' : 'save/default'} handler refuses unknown identity before validation`, async t => {
    const page = await fixture(t, 'q36', update ? saved : empty);
    await page.waitForFunction(() => window.__requests.length === 1);
    await page.evaluate(() => window.__requests[0].resolve());
    await page.getByRole('button', { name: /Saved settings/ }).click();
    await page.getByRole('menuitem', { name: update ? /Edit saved settings/ : /Save current settings/ }).click();
    const before = await page.evaluate(() => ({ validations: window.__validations, data: localStorage.getItem('strata.run-profiles.v1') }));
    await page.evaluate(hw => window.__putHardware(hw), unknownHardware);
    await page.evaluate(async () => { window.__saveProfile(); for (let n = 0; n < 10; n++) await Promise.resolve(); });
    assert.deepEqual(await page.evaluate(() => ({ validations: window.__validations, data: localStorage.getItem('strata.run-profiles.v1') })), before);
});

test('cold verification recognises an exact saved session default but preserves a different existing session draft',async t=>{
    const exact=await fixture(t,'q36',saved,JSON.stringify(qconfig),false,unknownHardware);
    await exact.evaluate(hw=>window.__putHardware(hw),hardware);
    await exact.waitForFunction(()=>document.querySelector('.run-saved-settings').textContent==='Q');
    const own={...qconfig,expert:{...qconfig.expert,'gpu.cpuMoeLayers':17}};
    const editing=await fixture(t,'q36',saved,JSON.stringify(own),false,unknownHardware);
    await editing.evaluate(hw=>window.__putHardware(hw),hardware);
    assert.equal(await editing.evaluate(()=>JSON.parse(window.__fixture.configDraft).expert['gpu.cpuMoeLayers']),17);
    assert.equal(await editing.locator('.run-saved-settings').textContent(),'Saved settings');
});

test('unknown profile identity permits manual load and a bounded refresh, preserving old fallback-bound records',async t=>{
    const oldComputer=JSON.stringify({host:'Test',platform:'win32',arch:'x64',cpu:'CPU',cores:12,ram:32,processors:[],gpus:[],backend:'cpu'});
    const old={...saved,profiles:[...saved.profiles,{...saved.profiles[0],id:'old',name:'Old fallback',hardware:oldComputer}],defaults:{...saved.defaults,[oldComputer+'\nq36']:'old'}};
    const p=await fixture(t,'q36',old,null,true,unknownHardware);
    await p.waitForFunction(()=>window.__plans>0&&window.__requests.length===1);await p.evaluate(()=>window.__requests[0].resolve());
    const load=p.getByRole('button',{name:'Load model',exact:true});assert.equal(await load.isEnabled(),true);
    await p.getByRole('button',{name:'Refresh hardware',exact:true}).click();assert.equal(await p.evaluate(()=>window.__refreshes),1);
    await p.evaluate(async()=>{window.__runModel();for(let n=0;n<10;n++)await Promise.resolve();});await p.waitForFunction(()=>window.__loads===1);
    await p.evaluate(hw=>window.__putHardware(hw),hardware);
    assert.equal(await p.locator('.run-saved-settings').textContent(),'Saved settings','manual load keeps its selected draft');
    assert.deepEqual(await p.evaluate(()=>JSON.parse(localStorage.getItem('strata.run-profiles.v1'))),old);
    await p.getByRole('button',{name:/Saved settings/}).click();assert.equal(await p.getByRole('menuitemradio',{name:/Old fallback/}).count(),0);
});
