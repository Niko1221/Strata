// Regressions for authoritative category selection and in-flight responses; no model/GPU.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const copy = (v) => JSON.parse(JSON.stringify(v));
const settle = async () => { for (let i=0; i<12; i++) await new Promise(setImmediate); };
const gate = () => { let resolve; const promise=new Promise(r=>resolve=r); return {promise,resolve}; };

function setup(options={}) {
  const elements=new Map(), intervals=[], calls=[], toasts=[], storage=new Map();
  if (options.last) storage.set('lastAdditionalProvider',options.last);
  function element(id) {
    if (!elements.has(id)) elements.set(id,{value:'',hidden:false,disabled:false,textContent:'',children:[],listeners:{},
      replaceChildren(...children){this.children=children},addEventListener(type,fn){this.listeners[type]=fn},
      click(){this.onclick?.()},scrollIntoView(){},focus(){}});
    return elements.get(id);
  }
  const state={current:options.current||null, native:'coder', switch:{}, fail:false, providerGate:null, postGate:null,
    profiles:[{id:'ollama',name:'Qwen3.5 4B'},{id:'bonsai',name:'Bonsai 2 27B'},{id:'long',name:'Bonsai 長文RAM'}]};
  const models=()=>({models:['original','coder','quality','swift','uncensored'].map(id=>({id,name:id,available:true})),
    current:state.native,can_switch:true,busy:false,loaded:true,switch:copy(state.switch)});
  const providers=()=>({current:state.current,providers:copy(state.profiles)});
  const ctx={window:{},console,document:{createElement:()=>({})},AbortSignal,
    $:element,store:{get:(key,fallback)=>storage.has(key)?storage.get(key):fallback,set:(key,v)=>storage.set(key,v)},
    busy:null,modelSwitching:false,health:{native:options.staleHealth===undefined?true:options.staleHealth,model:'mock'},
    contextRevision:0,contextTimer:null,clearTimeout(){},setInterval(fn){intervals.push(fn)},
    headers(){return {}},toast(...args){toasts.push(args)},scheduleContext(){},showTab(){},loadMcp:async()=>{},
    loadHealth:async()=>{},setBusy(){ctx.window.StrataProviderControls?.sync();ctx.window.StrataNativeModels?.sync()},
    async fetch(url,request={}) {
      const body=request.body?JSON.parse(request.body):null; calls.push({url,body});
      let result,ok=true;
      if (url==='api/local-models') result=models();
      else if (url==='api/providers') {
        result=providers(); if (state.providerGate) {const wait=state.providerGate;state.providerGate=null;await wait.promise;}
      } else if (url==='api/providers/select') {
        if (state.postGate) await state.postGate.promise;
        if (state.fail) {ok=false;result={error:{message:'unavailable'}};} else {state.current=body.id;result={current:body.id};}
      } else if (url==='api/local-models/switch') {
        state.native=body.model;state.switch={id:'op1',status:'starting',target:body.model};result={id:'op1',status:'starting'};
      } else throw new Error('Unexpected request '+url);
      return {ok,status:ok?200:503,json:async()=>result};
    }};
  vm.createContext(ctx);
  for (const name of ['models.js','providers.js']) vm.runInContext(fs.readFileSync(path.join(__dirname,name),'utf8'),ctx);
  return {ctx,state,element,intervals,calls,toasts,async change(id,value){const el=element(id);el.value=value;await el.listeners.change();await settle()}};
}

test('native and additional lists are separate even when health is stale',async()=>{
  const app=setup({staleHealth:false});await settle();
  assert.equal(app.element('model-family-select').value,'native');
  assert.equal(app.element('model-selector').hidden,false);assert.equal(app.element('provider-selector').hidden,true);
  assert.equal(app.element('model-select').value,'coder');assert.equal(app.element('model-select').children.length,5);
  assert.deepEqual(app.element('provider-select').children.map(o=>o.value),['ollama','bonsai','long']);
});
test('switching category remembers the last additional model and returning keeps the native variant',async()=>{
  const app=setup({last:'bonsai'});await settle();
  await app.change('model-family-select','additional');
  assert.equal(app.state.current,'bonsai');assert.equal(app.element('model-family-select').value,'additional');
  assert.equal(app.element('model-selector').hidden,true);assert.equal(app.element('provider-select').value,'bonsai');
  await app.change('provider-select','long');assert.equal(app.state.current,'long');
  await app.change('model-family-select','native');
  assert.equal(app.state.current,null);assert.equal(app.element('model-select').value,'coder');
  await app.change('model-family-select','additional');assert.equal(app.state.current,'long');
});
test('a delayed provider poll cannot replace the confirmed selection',async()=>{
  const app=setup({last:'bonsai'});await settle();
  const delayed=gate();app.state.providerGate=delayed;const poll=app.intervals[1]();await settle();
  await app.change('model-family-select','additional');delayed.resolve();await poll;await settle();
  assert.equal(app.element('model-family-select').value,'additional');assert.equal(app.element('provider-select').value,'bonsai');
});
test('a failed category change restores the actual server selection',async()=>{
  const app=setup();await settle();app.state.fail=true;
  await app.change('model-family-select','additional');
  assert.equal(app.state.current,null);assert.equal(app.element('model-family-select').value,'native');
  assert.equal(app.element('model-selector').hidden,false);assert(app.toasts.some(t=>t[0]==='error'));
});
test('readiness refreshes preserve the existing option nodes and do not reset an open picker',async()=>{
  const app=setup({current:'bonsai'});await settle();
  const options=[...app.element('provider-select').children];
  app.state.profiles.forEach(p=>{p.ready=false;p.checked_at=12345});
  await app.intervals[1]();await settle();
  assert(options.every((node,i)=>node===app.element('provider-select').children[i]));
  assert.equal(app.element('provider-select').value,'bonsai');
});
test('all selectors lock during provider changes and while generating',async()=>{
  const app=setup();await settle();const delayed=gate();app.state.postGate=delayed;
  const change=app.change('model-family-select','additional');await settle();
  for (const id of ['model-family-select','model-select','provider-select']) assert.equal(app.element(id).disabled,true);
  delayed.resolve();await change;app.ctx.busy={};app.ctx.setBusy(true);
  assert.equal(app.element('model-family-select').disabled,true);assert.equal(app.element('provider-select').disabled,true);
});
test('native variant changes stay selected until the same operation finishes',async()=>{
  const app=setup();await settle();await app.change('model-select','original');
  assert.equal(app.element('model-family-select').value,'native');assert.equal(app.element('model-select').value,'original');
  assert.equal(app.ctx.modelSwitching,true);assert.equal(app.element('model-family-select').disabled,true);
  app.state.switch.status='ready';await app.intervals[0]();await settle();
  assert.equal(app.ctx.modelSwitching,false);assert.equal(app.element('model-family-select').disabled,false);
  assert.equal(app.element('model-select').value,'original');
});
