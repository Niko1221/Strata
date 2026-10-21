import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {createRequire} from 'node:module';
import {build} from 'esbuild';

const root=resolve(import.meta.dirname,'..');
function stubFor(file,overrides){
 const source=readFileSync(resolve(root,'src/lib',file+'.ts'),'utf8');
 const names=new Set([...source.matchAll(/export\s+(?:async\s+)?(?:function\*?|const|let|var|class)\s+([A-Za-z_$][\w$]*)/g)].map(m=>m[1]));
 for(const m of source.matchAll(/export\s*\{([^}]*)\}/g))for(const part of m[1].split(','))names.add(part.split(/\s+as\s+/).pop().trim());
 names.delete('');
 return [...names].map(name=>`export const ${name}=${overrides[name]??'()=>{throw Error("Unexpected State Mutation Or Network Call");}'};`).join('\n');
}
const stubs={
 native:stubFor('native',{useNative:'()=>globalThis.__diagnosticNative'}),
 store:stubFor('store',{useStore:'selector=>selector({setup:{app:{version:"0.2.1"}},engine:{engine:{version:"0.1.46"}}})'}),
 client:stubFor('client',{}),
 router:stubFor('router',{}),
 'update-check':stubFor('update-check',{useInstalledVersions:'()=>({appVersion:"0.2.1",engineVersion:"0.1.46"})',useLastUpdateCheck:'()=>({checkedAt:"2026-10-21T00:00:00Z",strata:{status:"current",message:"Up To Date"},engine:{status:"current",message:"Up To Date"}})'}),
};
const compiled=await build({stdin:{contents:`export {NativeOperations} from './src/components/NativeOperations.tsx';export {SettingsUpdates} from './src/components/SettingsUpdates.tsx';export {createElement} from 'react';export {renderToStaticMarkup} from 'react-dom/server';`,resolveDir:root,loader:'tsx'},
 bundle:true,write:false,platform:'node',format:'cjs',jsx:'automatic',logLevel:'silent',loader:{'.svg':'dataurl','.png':'dataurl','.css':'empty'},
 plugins:[{name:'read-only-diagnostic-state',setup(b){b.onResolve({filter:/\/lib\/(native|store|client|router|update-check)$/},args=>({path:args.path.split('/').pop(),namespace:'diagnostic-state'}));b.onLoad({filter:/.*/,namespace:'diagnostic-state'},args=>({contents:stubs[args.path],resolveDir:root}));}}]});
const module={exports:{}};
globalThis.window??=globalThis;globalThis.matchMedia??=()=>({matches:false,addEventListener(){},removeEventListener(){}});
new Function('require','module','exports',compiled.outputFiles[0].text)(createRequire(import.meta.url),module,module.exports);
const ui=module.exports;
const message='The host restarted before completion. Check diagnostics, then retry; completed files are kept.';
const failed={id:'interrupted-app-update',kind:'update-strata',state:'error',summary:'Interrupted operation',error:message,result:null,startedAt:'2026-10-20T11:41:02.916Z'};
const done={id:'later-app-update',kind:'update-strata',state:'done',summary:'Complete',error:null,result:{summary:'Complete'},startedAt:'2026-10-20T23:11:18.883Z',finishedAt:'2026-10-20T23:11:37.606Z'};
function render(component,jobs){
 globalThis.__diagnosticNative={available:true,version:'0.1.46.3',installed:{models:[],engine:{version:'0.1.46'}},instances:[],fields:[],jobs,capabilities:{componentUpdates:true}};
 return ui.renderToStaticMarkup(ui.createElement(ui[component]));
}
for(const component of ['NativeOperations','SettingsUpdates']){
 test(`${component}: later successful App update turns the interrupted warning into retained history`,()=>{
  const jobs=structuredClone([failed,done]),before=JSON.stringify(jobs),html=render(component,jobs);
  assert.doesNotMatch(html,/role="alert"/,'Recovered interruption must not remain a current alert');
  assert.match(html,/Recovered Update History/);
  assert.ok(html.includes(message),'The original interruption remains readable');
  assert.equal(JSON.stringify(jobs),before,'Rendering never changes job state or history');
 });
 for(const [name,success] of [
  ['no completed retry',null],['running retry',{...done,state:'running'}],['cancelled retry',{...done,state:'cancelled'}],
  ['success before interruption',{...done,startedAt:'2026-10-19T00:00:00.000Z'}],['same timestamp',{...done,startedAt:failed.startedAt}],
  ['invalid success time',{...done,startedAt:'unknown'}],['other component',{...done,kind:'update-engine'}],
 ])test(`${component}: ${name} keeps the interrupted warning active`,()=>{
  const html=render(component,[failed,...(success?[success]:[])]);
  assert.match(html,/role="alert"/);assert.ok(html.includes(message));assert.doesNotMatch(html,/Recovered Update History/);
 });
 test(`${component}: an ordinary update failure and a later failure remain active`,()=>{
  const current={...failed,id:'current-failure',startedAt:'2026-10-21T00:00:00.000Z',summary:'Download Failed',error:'Download Failed'};
  const html=render(component,[failed,done,current]);
  assert.match(html,/role="alert"/);assert.match(html,/Download Failed/);assert.match(html,/Recovered Update History/);
  assert.match(render(component,[{...failed,error:'Installer Failed'},done]),/role="alert"/);
 });
 test(`${component}: array order does not change recovery and invalid interruption time remains an alert`,()=>{
  assert.doesNotMatch(render(component,[done,failed]),/role="alert"/);
  assert.match(render(component,[{...failed,startedAt:'unknown'},done]),/role="alert"/);
 });
}
