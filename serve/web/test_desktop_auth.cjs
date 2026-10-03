const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const code = fs.readFileSync(__dirname + '/app.js', 'utf8').split('// ------------------------------------------------------------------ server access')[1].split('let health =')[0];
function setup(desktop) {
  const elements = Object.fromEntries(['api-key', 'api-key-save', 'api-key-field', 'desktop-key-note', 'settings-storage-note'].map(id => [id, {value:'', hidden:false, textContent:''}]));
  const stored = {apikey:'synthetic-local-key'};
  const context = vm.createContext({window:{__STRATA_DESKTOP__:desktop}, $:id=>elements[id],
    store:{get:(key,fallback)=>stored[key]??fallback, set:(key,value)=>{stored[key]=value;}},
    toast:()=>{}, historyReady:true, initializeHistory:()=>{throw new Error('Unexpected init');}});
  vm.runInContext(code, context);
  return {context,elements,stored,headers:()=>JSON.parse(vm.runInContext('JSON.stringify(headers(true))', context)),save:()=>vm.runInContext('saveApiKey()', context)};
}
test('browser clients retain bearer authentication and key saving',()=>{
  const state=setup(false);
  assert.equal(state.headers().Authorization,'Bearer synthetic-local-key');
  state.elements['api-key'].value=' replacement-test-key ';
  state.save();
  assert.equal(state.stored.apikey,'replacement-test-key');
});
test('desktop auth ignores browser keys and never copies them into fields or new browser storage',()=>{
  const state=setup(true);
  assert.equal(state.headers().Authorization,undefined);
  assert.equal(state.headers()['Content-Type'],'application/json');
  assert.equal(state.elements['api-key'].value,'');
  assert.equal(state.elements['api-key-field'].hidden,true);
  assert.equal(state.elements['desktop-key-note'].hidden,false);
  state.elements['api-key'].value='must-not-be-saved'; state.save();
  assert.equal(state.stored.apikey,'synthetic-local-key');
});
