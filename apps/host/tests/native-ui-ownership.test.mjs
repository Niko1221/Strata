/** Supplemental mounted React component tests, with controlled API promises. No Host or engine E2E. */
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { chromium } from 'playwright-core';
import { testBrowserExecutable, TEST_BROWSER_LAUNCH_TIMEOUT_MS } from './helpers/browser-executable.mjs';
import { mkdtemp, mkdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import {NATIVE_FIELDS} from '../server/native-settings.mjs';

const root = resolve(import.meta.dirname, '..');
const controls = `import React from 'react';
export const Btn=({children,onClick,disabled})=>React.createElement('button',{onClick,disabled},children);
export const Select=({ariaLabel,value,options,onChange,disabled})=>React.createElement('select',{'aria-label':ariaLabel,value,disabled,onChange:e=>onChange(e.target.value)},options.map(o=>React.createElement('option',{key:o.value,value:o.value},o.label)));
export const Screen=({children,actions})=>React.createElement('main',null,actions,children);
export const ErrorNotice=({message,onRetry})=>React.createElement('div',{role:'alert'},message,onRetry?React.createElement('button',{onClick:onRetry},'Try again'):null);
export const Dialog=()=>null,Icon=()=>null,VersionPicker=()=>null,FamilySwitch=()=>null,PcComputer=()=>null,GpuRoles=()=>null,DownloadParts=()=>null,EngineWarning=()=>null,EngineUpdateCard=()=>null,EngineUpdateDialog=()=>null,NativeDeleteDialog=()=>null,NativeSettingsFile=()=>null,NativeTune=()=>null,NativeTuneBadge=()=>null,NativePresets=()=>null,NativeMcpEditor=()=>null,RunImageModel=()=>null;`;
const boundaries = {
    ui: controls,
    native: `import {useSyncExternalStore} from 'react';export * from './src/lib/native.ts';
const listeners=new Set();export const useNative=()=>useSyncExternalStore(f=>{listeners.add(f);return()=>listeners.delete(f)},()=>window.__native);
window.__putNative=value=>{window.__native=value;listeners.forEach(f=>f())};export const refreshNative=async()=>{};
export const nativeAction=async(path,body)=>{window.__actions.push({path,body});if(path==='operations')return {id:'job-A'};};`,
    client: `const request=(method,path,body)=>new Promise((resolve,reject)=>window.__requests.push({method,path,body,resolve,reject}));export const api={get:path=>request('get',path),post:(path,body)=>request('post',path,body),patch:(path,body)=>request('patch',path,body)};`,
    store: `export const go=()=>{},setSettingsSection=()=>{},selectRemoteModel=()=>{},pushToast=()=>{},useStore=select=>select({imageModels:null,hardware:null});`,
    i18n: 'export const tr=value=>value;',
    features: 'export const IMAGE_GENERATION_VISIBLE=false;',
};
const bundle = await build({
    stdin: { contents: `import React from 'react';import {createRoot} from 'react-dom/client';import {NativeSettings} from './src/components/NativeSettings';import {NativeLocalModel} from './src/components/NativeLocalModel';import {NativeModels,NativeProgress} from './src/components/NativeModels';import {nativePollDelay} from './src/lib/native.ts';
window.__root=createRoot(document.getElementById('root'));window.__pollDelay=nativePollDelay;window.__render=(kind,model='A')=>window.__root.render(kind==='settings'?<NativeSettings model={model}/>:kind==='local'?<NativeLocalModel onBack={()=>{}} onAlternative={()=>{}}/>:kind==='models'?<NativeModels onOtherModels={()=>{}}/>:<NativeProgress job={window.__native.jobs[0]}/>);`, resolveDir: root, loader: 'tsx' },
    bundle:true,write:false,format:'iife',platform:'browser',jsx:'automatic',define:{'process.env.NODE_ENV':'"test"'},
    plugins:[{name:'controlled-boundaries',setup(b){
        b.onResolve({filter:/\/lib\/(native|client|store|i18n|features)$/},a=>({path:a.path.split('/').at(-1),namespace:'boundary'}));
        b.onResolve({filter:/\/components\/|^\.\/(ui|layout|VersionPicker|FamilySwitch|PcComputer|GpuRoles|DownloadParts|EngineWarning|EngineUpdateCard|EngineUpdateDialog|NativeDelete|NativeSettingsFile|NativeTune|NativeTuneBadge|NativePresets|NativeMcpEditor|RunImageModel)$/},a=> /\/(NativeSettings|NativeLocalModel|NativeModels|DecimalInput)$/.test(a.path)?undefined:({path:'ui',namespace:'boundary'}));
        b.onLoad({filter:/.*/,namespace:'boundary'},a=>({contents:boundaries[a.path],loader:'jsx',resolveDir:root}));
    }}],
});
const job = (state,summary='Operation A')=>({id:'job-A',kind:'load',model:'A',state,summary,error:state==='error'?'A failed':null,result:null,startedAt:'2026-11-06T00:00:00Z'});
const state = jobs=>({available:true,jobs,instances:[],fields:[],catalog:{families:[],contexts:[32768]},installed:{models:[]},hardware:{ram_gb:32,gpus:[]}});
let browserPromise;
after(async()=>{if(browserPromise)await browserPromise.then(browser=>browser.close(),()=>{});});
async function fixture(t,kind='settings',jobs=[]) {
    const home=await mkdtemp(join(tmpdir(),'s-e3-'));
    const dirs=Object.fromEntries(['home','local','roaming','workspace','models','components','runtime','tmp'].map(n=>[n,join(home,n)]));
    await Promise.all(Object.values(dirs).map(p=>mkdir(p,{recursive:true})));
    const env=Object.fromEntries(Object.entries(process.env).filter(([k])=>!/^STRATA_|^DSH_|^WEBVIEW2_/i.test(k)));
    Object.assign(env,{LOCALAPPDATA:dirs.local,APPDATA:dirs.roaming,TEMP:dirs.tmp,TMP:dirs.tmp,STRATA_HOME:dirs.home,STRATA_WORKSPACE:dirs.workspace,STRATA_MODELS_DIR:dirs.models,STRATA_COMPONENTS_DIR:dirs.components,STRATA_RUNTIME_DIR:dirs.runtime,STRATA_SETTINGS_FILE:join(home,'settings.json'),STRATA_NO_ACQUIRE:'1',STRATA_NO_BUNDLED_ENGINE:'1',STRATA_NO_REPAIR_ACCESS:'1',STRATA_AGENT:'off'});
    if (process.env.STRATA_TEST_WINDOWS_PROFILE) { env.USERPROFILE=process.env.STRATA_TEST_WINDOWS_PROFILE; delete env.HOME; }
    browserPromise??=chromium.launch({executablePath:testBrowserExecutable(),headless:true,env,timeout:TEST_BROWSER_LAUNCH_TIMEOUT_MS,args:['--disable-gpu','--disable-background-networking','--disable-component-update','--no-first-run']});
    const browser=await browserPromise,context=await browser.newContext();
    t.after(()=>context.close());t.diagnostic('isolated browser context; fixture home: '+home);
    const page=await context.newPage();page.setDefaultTimeout(5000);const errors=[];page.on('pageerror',e=>errors.push(e.message));t.after(()=>assert.deepEqual(errors,[]));
    await page.route(/^https?:/,r=>r.abort());
    await page.route('http://127.0.0.1/',r=>r.fulfill({contentType:'text/html',body:'<!doctype html><div id="root"></div>'}));
    await page.goto('http://127.0.0.1/',{waitUntil:'domcontentloaded',timeout:TEST_BROWSER_LAUNCH_TIMEOUT_MS});
    assert.equal(await page.evaluate(()=>window.isSecureContext),true);
    await page.evaluate(s=>{window.__native=s;window.__requests=[];window.__actions=[]},state(jobs));await page.addScriptTag({content:bundle.outputFiles[0].text});await page.evaluate(k=>window.__render(k),kind);
    return page;
}
const values = n=>({values:{sampling:{temperature:n}},mcp:{},fields:[]});

test('memory selector is explicitly experimental and actual storage never follows the selected value',async t=>{
    const p=await fixture(t);
    await p.waitForFunction(()=>window.__requests.length===1);
    const field=NATIVE_FIELDS.find(f=>f.id==='q36_memory');
    await settle(p,0,{values:{q36_memory:'bounded-experts-v1'},mcp:{},fields:[{...field,supported:true,choices:['legacy','bounded-experts-v1']}]});
    await p.locator('details.native-settings').evaluate(el=>el.open=true);
    const selector=p.getByRole('combobox',{name:'Q36 expert working storage'});
    assert.match(await selector.locator('option[value="bounded-experts-v1"]').textContent(),/experimental/);
    assert.equal(await p.getByText('Actual expert storage is not reported yet.',{exact:true}).count(),1);
    const s=state([]);s.instances=[{id:'a',model:'A',status:'ready',phase:'idle',metrics:{live:{state:'idle'},engine:{native_child:{generation:'child-a',process_id:101,admitted:true,alive:true},q36_memory_generation:'child-a',q36_memory_process_id:101,q36_memory_admitted:true,q36_memory_mode:'bounded-experts-v1',q36_memory_scope:'expert-working-storage',q36_memory_total_process_bound:0,q36_memory_cpu_stage_bytes:100,q36_memory_h2d_stage_bytes:200,q36_memory_io_scratch_bytes:300}}}];
    await p.evaluate(s=>window.__putNative(s),s);
    assert.equal(await p.locator('[data-q36-memory-observed="bounded-experts-v1"]').count(),1);
    await selector.selectOption('legacy');
    assert.equal(await p.locator('[data-q36-memory-observed="bounded-experts-v1"]').count(),1,'editing a future choice cannot rewrite the running receipt');
    await p.getByRole('button',{name:'Save settings',exact:true}).click();
    await p.waitForFunction(()=>window.__requests.length===2);
    assert.equal(await p.evaluate(()=>window.__requests[1].body.values.q36_memory),'legacy');
    await p.evaluate(s=>window.__putNative(s),{...s,instances:[{...s.instances[0],status:'error'}]});
    assert.equal(await p.locator('[data-q36-memory-observed]').count(),0,'stopped/error receipts are not current observations');
});
test('service-ready UI requires a current admitted child and hides stale or premature memory',async t=>{
    const p=await fixture(t);await p.waitForFunction(()=>window.__requests.length===1);
    const field=NATIVE_FIELDS.find(f=>f.id==='q36_memory');
    await settle(p,0,{values:{q36_memory:'bounded-experts-v1'},mcp:{},fields:[{...field,supported:true}]});
    await p.locator('details.native-settings').evaluate(el=>el.open=true);
    const child={generation:'child-a',process_id:101,admitted:true,alive:true};
    const info={native_child:child,q36_memory_generation:'child-a',q36_memory_process_id:101,q36_memory_admitted:true,q36_memory_mode:'bounded-experts-v1',q36_memory_scope:'expert-working-storage',q36_memory_total_process_bound:0,q36_memory_cpu_stage_bytes:100,q36_memory_h2d_stage_bytes:200,q36_memory_io_scratch_bytes:300};
    const put=async(engine,phase='idle')=>{const s=state([]);s.instances=[{id:'a',model:'A',status:'ready',phase,metrics:{live:{state:phase},engine}}];await p.evaluate(s=>window.__putNative(s),s);};
    await put(info);assert.equal(await p.locator('[data-q36-memory-observed]').count(),1);
    for(const [engine,phase] of [
        [info,'unloaded'],[info,'loading'],
        [{...info,native_child:{...child,admitted:false}},'idle'],
        [{...info,native_child:{...child,alive:false}},'idle'],
        [{...info,native_child:{...child,generation:'child-b'}},'idle'],
        [{...info,native_child:{...child,process_id:102}},'idle'],
        [{...info,q36_memory_admitted:false},'idle'],
        [{...info,native_child:undefined},'idle'],
    ]){await put(engine,phase);assert.equal(await p.locator('[data-q36-memory-observed]').count(),0,JSON.stringify({engine,phase}));assert.equal(await p.getByText('Actual expert storage is not reported yet.',{exact:true}).count(),1);}
    await put({...info,native_child:{...child,generation:'child-b'},q36_memory_generation:'child-b',q36_memory_cpu_stage_bytes:400});
    assert.equal(await p.locator('[data-q36-memory-observed]').count(),1,'a newly admitted matching child becomes observable');
});
async function settle(page,index,value,error=false){await page.evaluate(({index,value,error})=>{const r=window.__requests[index];error?r.reject(Error(value)):r.resolve(value)}, {index,value,error});await page.evaluate(()=>new Promise(r=>setTimeout(r,0)));}
async function settingsReady(page,index,n){await page.waitForFunction(i=>window.__requests.length>i,index);await settle(page,index,values(n));await page.locator('details.native-settings').evaluate(el=>el.open=true);await page.getByRole('tab',{name:'JSON',exact:true}).click();}

test('settings expansion during a pending GET survives successful settlement',async t=>{
    const p=await fixture(t);const panel=p.locator('details.native-settings');
    await p.waitForFunction(()=>window.__requests.length===1);
    assert.equal(await panel.evaluate(el=>el.open),false);
    assert.equal(await panel.getByText('Reading saved settings…',{exact:true}).count(),1);
    await panel.locator(':scope > summary').click();
    await panel.getByRole('status').waitFor({state:'visible'});
    await settle(p,0,values(0.3));
    assert.equal(await panel.evaluate(el=>el.open),true,'GET success must preserve the user expansion');
    assert.equal(await panel.getByRole('searchbox',{name:'Search native parameters'}).isVisible(),true);
    const temperature=panel.locator('.native-setting-row input').first();
    assert.equal(await temperature.isVisible(),true);assert.equal(await temperature.inputValue(),'0.3');
    await panel.getByRole('tab',{name:'JSON',exact:true}).click();
    assert.equal(JSON.parse(await panel.getByRole('textbox',{name:'Native configuration JSON'}).inputValue()).sampling.temperature,0.3);
});
test('settings stays initially closed and accepts manual collapse after a successful GET',async t=>{
    const p=await fixture(t);const panel=p.locator('details.native-settings');
    await p.waitForFunction(()=>window.__requests.length===1);await settle(p,0,values(0.3));
    assert.equal(await panel.evaluate(el=>el.open),false);
    const summary=panel.locator(':scope > summary');await summary.click();
    assert.equal(await panel.getByRole('searchbox',{name:'Search native parameters'}).isVisible(),true);
    await summary.click();await p.evaluate(s=>window.__putNative(s),state([]));
    assert.equal(await panel.evaluate(el=>el.open),false);
});
for(const collapse of [false,true]) test(`settings load error expands and retry preserves the user's ${collapse?'collapsed':'expanded'} choice`,async t=>{
    const p=await fixture(t);const panel=p.locator('details.native-settings');const summary=panel.locator(':scope > summary');
    await p.waitForFunction(()=>window.__requests.length===1);await settle(p,0,'current load failed',true);
    await panel.getByRole('alert').waitFor({state:'visible'});
    assert.equal(await panel.evaluate(el=>el.open),true);
    assert.equal(await panel.getByRole('alert').isVisible(),true);
    await panel.locator('.native-settings-diagnostic > summary').click();
    assert.equal(await panel.getByText('current load failed',{exact:true}).isVisible(),true);
    await summary.click();await p.evaluate(s=>window.__putNative(s),state([]));
    assert.equal(await panel.evaluate(el=>el.open),false,'an existing error must allow manual collapse');
    await summary.click();await panel.getByRole('button',{name:'Try again',exact:true}).click();
    await p.waitForFunction(()=>window.__requests.length===2);
    assert.equal(await panel.evaluate(el=>el.open),true,'retry must preserve the open panel');
    await panel.getByRole('status').waitFor({state:'visible'});
    if(collapse) await summary.click();
    await settle(p,1,values(0.6));
    assert.equal(await panel.evaluate(el=>el.open),!collapse,'retry success must preserve the current choice');
    assert.equal(await panel.getByRole('alert').count(),0);
    if(collapse) await summary.click();
    assert.equal(await panel.getByRole('searchbox',{name:'Search native parameters'}).isVisible(),true);
});

for(const fail of [false,true]) test(`settings model B ignores model A PATCH ${fail?'failure':'success'} and finally`,async t=>{
    const p=await fixture(t);await settingsReady(p,0,0.1);await p.getByRole('textbox',{name:'Native configuration JSON'}).fill(JSON.stringify({sampling:{temperature:0.2}}));await p.getByRole('button',{name:'Save settings',exact:true}).click();
    await p.waitForFunction(()=>window.__requests.length===2);await p.evaluate(()=>window.__render('settings','B'));await settingsReady(p,2,0.7);
    await p.getByRole('textbox',{name:'Native configuration JSON'}).fill(JSON.stringify({sampling:{temperature:0.8}}));
    // Current model is free to save while A's disk write finishes.
    const save=p.locator('.native-settings-actions button').first();
    const enabled=await save.isEnabled();
    assert.equal(enabled,true);
    await save.click();await p.waitForFunction(()=>window.__requests.length===4);
    await settle(p,1,fail?'obsolete A save failed':values(0.2),fail);
    assert.equal(JSON.parse(await p.getByRole('textbox',{name:'Native configuration JSON'}).inputValue()).sampling.temperature,0.8);
    assert.equal(await p.getByRole('alert').count(),0);assert.equal(await p.getByText(/^Saved\. Load/).count(),0);assert.equal(enabled,true);
    assert.equal(await p.getByRole('button',{name:'Saving…',exact:true}).isDisabled(),true);
    await settle(p,3,values(0.8));assert.equal(await p.getByRole('button',{name:'Save settings',exact:true}).isDisabled(),true);
});

const preview=path=>({path,preview:{file_count:1,total_bytes:1,file_set_status:'plausible',issues:[],candidate_selection:{family:'qwen',model:'B'}}});
for(const terminal of ['error','done']) test(`new file check detaches A ${terminal} result while keeping diagnostics`,async t=>{
    const p=await fixture(t,'local');const input=p.getByPlaceholder('Full path to the folder or the file');await input.fill('A.gguf');await p.getByRole('button',{name:'Check files',exact:true}).click();await settle(p,0,preview('A.gguf'));await p.getByRole('button',{name:'Prepare and load',exact:true}).click();await p.waitForFunction(()=>window.__actions.length===1);
    await p.evaluate(s=>window.__putNative(s),state([job(terminal)]));await p.getByText(terminal==='error'?'A failed':'Operation A',{exact:true}).waitFor();
    await input.fill('B.gguf');await p.getByRole('button',{name:'Check files',exact:true}).click();await settle(p,1,preview('B.gguf'));
    assert.equal(await p.getByText(terminal==='error'?'A failed':'Operation A',{exact:true}).count(),0);assert.equal(await p.evaluate(()=>window.__native.jobs[0].state),terminal);assert.equal(await p.getByRole('button',{name:'Prepare and load',exact:true}).isEnabled(),true);
});
test('running and cancelling remain visible and busy through another file preview until a terminal state',async t=>{
    const p=await fixture(t,'local',[job('running')]);const input=p.getByPlaceholder('Full path to the folder or the file');await input.fill('B.gguf');await p.getByRole('button',{name:'Check files',exact:true}).click();await settle(p,0,preview('B.gguf'));
    const prepare=p.getByRole('button',{name:'Prepare and load',exact:true});assert.equal(await prepare.isDisabled(),true);
    await p.evaluate(s=>window.__putNative(s),state([job('cancelling')]));
    assert.equal(await prepare.isDisabled(),true);assert.equal(await p.locator('.native-progress[data-state="cancelling"]').count(),1);assert.match(await p.locator('.native-progress').textContent(),/Stopping…/);assert.equal(await p.getByRole('button',{name:'Stopping…',exact:true}).isDisabled(),true);
    assert.equal(await p.evaluate(()=>window.__pollDelay(window.__native)),1000);
    for(const terminal of ['cancelled','error','done']){await p.evaluate(s=>window.__putNative(s),state([job(terminal)]));assert.equal(await prepare.isEnabled(),true);assert.equal(await p.locator('.native-progress').count(),0);assert.equal(await p.evaluate(()=>window.__pollDelay(window.__native)),2500);}
});
test('NativeModels keeps Load disabled and progress visible while cancellation is pending',async t=>{
    const p=await fixture(t,'models');
    const ready=state([]);ready.catalog.families=[{family:'qwen',title:'Qwen',images:false,sizes:[{id:'B',model:'B',download_gb:1,ram_needed_gb:1,experts_gb:1}]}];ready.installed.models=[{model:'B',ready:true,context:32768,contextChosen:true,kv:'int8'}];
    await p.evaluate(s=>window.__putNative(s),ready);
    const load=p.getByRole('button',{name:'Load model',exact:true});await load.waitFor();assert.equal(await load.isEnabled(),true);
    await p.evaluate(s=>window.__putNative(s),{...ready,jobs:[job('cancelling')]});
    assert.equal(await p.locator('.native-progress[data-state="cancelling"]').count(),1);assert.equal(await load.isDisabled(),true);
    await p.evaluate(s=>window.__putNative(s),{...ready,jobs:[job('cancelled')]});assert.equal(await load.isEnabled(),true);
});

test('returning to A performs a fresh read and ignores the previous visit save reply',async t=>{
    const p=await fixture(t);await settingsReady(p,0,0.1);await p.getByRole('textbox',{name:'Native configuration JSON'}).fill(JSON.stringify({sampling:{temperature:0.2}}));await p.getByRole('button',{name:'Save settings',exact:true}).click();await p.waitForFunction(()=>window.__requests.length===2);
    await p.evaluate(()=>window.__render('settings','B'));await settingsReady(p,2,0.7);
    await p.evaluate(()=>window.__render('settings','A'));await settingsReady(p,3,0.4);
    assert.deepEqual(await p.evaluate(()=>window.__requests.filter(r=>r.method==='get').map(r=>r.path)),['/api/native/models/A/settings','/api/native/models/B/settings','/api/native/models/A/settings']);
    await settle(p,1,values(0.2));assert.equal(JSON.parse(await p.getByRole('textbox',{name:'Native configuration JSON'}).inputValue()).sampling.temperature,0.4);assert.equal(await p.getByText(/^Saved\. Load/).count(),0);
});

const ordinaryPolicy={id:'q38',ordinaryKv:['fp16','int8','q4_0','k8v4'],note:'Ordinary serving'};
function preparedModel(policy=ordinaryPolicy,kv='int8') {
    const s=state([]);s.catalog.families=[{family:'qwen',title:'Qwen',images:false,sizes:[{id:'B',model:'B',download_gb:1,ram_needed_gb:1,experts_gb:1}]}];
    s.installed.models=[{model:'B',ready:true,context:32768,contextChosen:true,kv}];s.servingPolicies={B:policy};return s;
}

test('DeepSeek remains selected with an unavailable-runtime reason and cannot load or tune',async t=>{
    const p=await fixture(t,'models');
    const s=preparedModel({id:'deepseek41',ordinaryKv:[],note:'Dedicated engine'});
    s.catalog.families[0].family='deepseek41';s.catalog.families[0].title='DeepSeek';
    s.installed.models[0].runtime={available:false,reason:'Dedicated DeepSeek engine is missing.',executionCompatibility:'unverified'};
    await p.evaluate(s=>window.__putNative(s),s);
    assert.equal(await p.getByRole('button',{name:'Load model',exact:true}).isDisabled(),true);
    assert.equal(await p.getByRole('button',{name:'Tune model',exact:true}).isDisabled(),true);
    assert.equal(await p.getByRole('status').filter({hasText:'Dedicated DeepSeek engine is missing.'}).count(),1);
    assert.equal(await p.evaluate(()=>window.__actions.length),0);
    s.installed.models[0].runtime={available:true,reason:null,executionCompatibility:'unverified'};
    await p.evaluate(s=>window.__putNative(s),s);
    assert.equal(await p.getByRole('button',{name:'Load model',exact:true}).isEnabled(),true);
});
test('NativeModels applies changed ordinary KV directly without preparation and shows FP16',async t=>{
    const p=await fixture(t,'models');await p.evaluate(s=>window.__putNative(s),preparedModel());
    await p.getByRole('button',{name:'Expert mode',exact:true}).click();
    const selector=p.getByRole('combobox',{name:'Native KV format'});
    assert.equal(await selector.locator('option[value="fp16"]').count(),1);
    await selector.selectOption('q4_0');await p.getByRole('button',{name:'Load model',exact:true}).click();
    await p.waitForFunction(()=>window.__actions.length>0||window.__requests.some(r=>r.path==='/api/native/plan'));
    const actions=await p.evaluate(()=>window.__actions);
    assert.equal(actions[0]?.body.kind,'load');assert.equal(actions[0]?.body.args.kv,'q4_0');
    assert.equal(await p.evaluate(()=>window.__requests.some(r=>r.path==='/api/native/plan')),false);
});
test('unknown saved ordinary KV needs an explicit choice, not the unchanged-load shortcut',async t=>{
    const p=await fixture(t,'models');await p.evaluate(s=>window.__putNative(s),preparedModel(ordinaryPolicy,null));
    await p.getByRole('button',{name:'Load model',exact:true}).click();
    await p.evaluate(()=>new Promise(r=>setTimeout(r,0)));
    assert.equal(await p.evaluate(()=>window.__actions.length),0);
    assert.match(await p.getByRole('alert').textContent(),/KV|context memory/i);
});
for(const id of ['q36','deepseek41','unknown'])test(`NativeModels does not offer ordinary Q38 KV for ${id}`,async t=>{
    const p=await fixture(t,'models');await p.evaluate(s=>window.__putNative(s),preparedModel({id,ordinaryKv:[],note:'Adapter-specific cache'}));
    await p.getByRole('button',{name:'Expert mode',exact:true}).click();
    assert.equal(await p.getByRole('combobox',{name:'Native KV format'}).count(),0);
});
test('incompatible stored native field stays visible with an actionable clear operation',async t=>{
    const p=await fixture(t);await p.waitForFunction(()=>window.__requests.length===1);
    await settle(p,0,{values:{cache_type_k:'q8_0'},mcp:{},fields:[{id:'cache_type_k',label:'KV cache K format',group:'GPU & memory',type:'enum',choices:[],supported:false,unavailableReason:'Ordinary serving cannot use this format.'}],settingsIssues:[{field:'cache_type_k',message:'Clear cache_type_k before ordinary serving.'}]});
    const panel=p.locator('details.native-settings');await panel.evaluate(el=>el.open=true);
    assert.match(await panel.textContent(),/q8_0/);
    await p.getByRole('button',{name:'Clear KV cache K format',exact:true}).click();
    await p.getByRole('button',{name:'Save settings',exact:true}).click();
    assert.equal(await p.evaluate(()=>window.__requests.at(-1).body.values.cache_type_k),null);
});

test('local Q36 preparation does not inject the automatic ordinary Q38 KV choice',async t=>{
    const p=await fixture(t,'local');const s=preparedModel({id:'q36',ordinaryKv:[],note:'Q36 scalar cache'});s.catalog.families[0].family='qwen36';
    await p.evaluate(s=>window.__putNative(s),s);
    await p.getByPlaceholder('Full path to the folder or the file').fill('owned-fixture.gguf');
    await p.getByRole('button',{name:'Check files',exact:true}).click();
    const found=preview('owned-fixture.gguf');found.preview.candidate_selection={family:'qwen36',model:'B'};await settle(p,0,found);
    await p.getByRole('button',{name:'Prepare and load',exact:true}).click();
    const args=await p.evaluate(()=>window.__actions[0].body.args);assert.equal(args.family,'qwen36');assert.equal('kv' in args,false);
});
