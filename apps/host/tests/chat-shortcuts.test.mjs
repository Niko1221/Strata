import { fileURLToPath } from 'node:url';
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { createRequire } from 'node:module';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';
const root=fileURLToPath(new URL('..',import.meta.url));
const { outputFiles } = await build({ stdin: { contents: "export * from './src/lib/commands'; export {tr,LANGUAGES,setTranslationLanguage} from './src/lib/i18n';", resolveDir: root }, bundle: true, write: false, platform: 'node', format: 'cjs', plugins: [{ name: 'test-store', setup(build) { build.onResolve({ filter: /^\.\/store$/ }, () => ({ path: 'store', namespace: 'test-store' })); build.onLoad({ filter: /.*/, namespace: 'test-store' }, () => ({ contents: `
export const getState = () => testStore.state;
export const selectSession = id => { testStore.state.activeSession = id; testStore.calls.push(['select',id]); };
export const go = (...args) => testStore.calls.push(['go',...args]);
export const exportConversation=()=>{},newSession=()=>({id:'new'}),pushToast=()=>{},restartEngine=()=>{},setFocus=()=>{},setNavOpen=()=>{},setPalette=()=>{},setShortcuts=()=>{},setTheme=()=>{},toggleFocus=()=>{},toggleRail=()=>{},unloadModel=()=>{};`, loader: 'js' })); } }] });
function harness(overrides={}) {
  const testStore={state:{theme:'light',sessions:Array.from({length:8},(_,i)=>({id:String(i),title:'Chat '+i,messages:[],updatedAt:`2026-10-${String(10-i).padStart(2,'0')}`})),activeSession:'7'},calls:[]};
  const module={exports:{}};
  new Function('require','module','exports','testStore','localStorage','navigator','window',outputFiles[0].text)(createRequire(import.meta.url),module,module.exports,testStore,{getItem:()=>JSON.stringify(overrides)},{platform:'MacIntel'},{location:{hash:'#/chat/7'}});
  const press=(key, extra={})=> {const event={key,code:'',metaKey:true,ctrlKey:false,altKey:false,shiftKey:false,target:{matches:()=>false},preventDefault(){this.prevented=true;},...extra};module.exports.handleAppShortcut(event);return event;};
  return {...module.exports,testStore,press};
}
test('numbered shortcuts switch to the same current-plus-five slots shown in the sidebar',()=>{
  const h=harness();
  assert.equal(h.press('2').prevented,true);
  assert.deepEqual(h.testStore.calls,[['select','1'],['go','chat','1']]);
  h.testStore.calls=[];
  h.press('1',{metaKey:false,ctrlKey:true});
  assert.deepEqual(h.testStore.calls,[['select','0'],['go','chat','0']]);
});
test('Option digit navigation stays distinct from numbered chats on macOS',()=>{
  const h=harness();h.press('¡',{altKey:true,code:'Digit1'});
  assert.deepEqual(h.testStore.calls,[['go','chat']]);
});
test('saved custom shortcuts dispatch after reload and recording suppresses app actions',()=>{
  const h=harness({'go:chat':{key:'c',mod:true,alt:false,shift:true}});
  h.press('C',{shiftKey:true});assert.deepEqual(h.testStore.calls,[['go','chat']]);
  h.testStore.state.shortcutsOpen=true;h.testStore.calls=[];h.press('1');assert.deepEqual(h.testStore.calls,[]);
});

// Read the composer's literal descriptors rather than copying its hint strings into this test.
// Registry-generated hints below come from the real getSlashCommands() implementation.
const chatSource=ts.createSourceFile('Chat.tsx',await readFile(new URL('../src/screens/Chat.tsx',import.meta.url),'utf8'),ts.ScriptTarget.Latest,true,ts.ScriptKind.TSX);
const composerCommands=[];
function collectComposerCommands(node){
  if(ts.isObjectLiteralExpression(node)){
    const properties=new Map(node.properties.filter(ts.isPropertyAssignment).map(p=>[p.name.getText(chatSource),p.initializer]));
    const label=properties.get('label'),hint=properties.get('hint');
    if(label&&ts.isStringLiteral(label)&&label.text.startsWith('/')){
      assert.ok(hint&&ts.isStringLiteral(hint),`${label.text} needs a literal, catalogued hint`);
      composerCommands.push({label:label.text,hint:hint.text});
    }
  }
  ts.forEachChild(node,collectComposerCommands);
}
collectComposerCommands(chatSource);
const catalogs=Object.fromEntries(await Promise.all(harness().LANGUAGES.map(async({value})=>{
  const parts=await Promise.all(['','tune-','tools-'].map(async prefix=>JSON.parse(await readFile(new URL(`../src/locales/${prefix}${value}.json`,import.meta.url),'utf8'))));
  return[value,Object.assign({},...parts)];
})));

test('every composer and registry slash hint has an exact translation in every language',()=>{
  const h=harness();
  h.testStore.state.sessions.find(s=>s.id===h.testStore.state.activeSession).messages.push({role:'user',content:'hello'});
  const commands=[...composerCommands,...h.getSlashCommands()];
  assert.ok(commands.some(c=>c.label==='/plan')&&commands.some(c=>c.label==='/export'));
  assert.equal(new Set(commands.map(c=>c.label)).size,commands.length);
  const ids=new Map(Object.entries(catalogs.en).map(([id,text])=>[text,id]));
  assert.deepEqual(commands.filter(c=>!ids.has(c.hint)).map(c=>c.label),[], 'missing exact catalog entries');
  for(const {value:language} of h.LANGUAGES){
    h.setTranslationLanguage(language);
    for(const command of commands){
      const translated=catalogs[language][ids.get(command.hint)];
      assert.ok(translated?.trim(),`${language}: ${command.label}`);
      assert.equal(h.tr(command.hint),translated,`${language}: ${command.label}`);
      if(language!=='en')assert.notEqual(translated,command.hint,`${language}: ${command.label} still uses English`);
      assert.equal(h.tr(command.label),command.label,'slash tokens remain stable');
    }
  }
  h.setTranslationLanguage('zh');
  const expected={
    '/tune':'打开模型设置','/serve':'打开 API 服务','/model':'打开模型','/skills':'打开技能',
    '/plan':'让助手先写一份计划；经你批准后才会执行',
    '/research':'搜索网络、阅读网页并附上来源回答',
    '/new':'开始新对话，并保留当前对话',
  };
  for(const [label,hint] of Object.entries(expected))assert.equal(h.tr(commands.find(c=>c.label===label).hint),hint,label);
});

test('localized slash commands keep their routes and export availability',()=>{
  const h=harness();
  for(const {value:language} of h.LANGUAGES){
    h.setTranslationLanguage(language);
    const commands=h.getSlashCommands();
    assert.ok(!commands.some(c=>c.label==='/export'),'empty conversations cannot be exported');
    for(const [label,route] of Object.entries({'/tune':'models','/serve':'serving','/model':'models','/monitor':'monitor','/tasks':'tasks','/skills':'skills'})){
      h.testStore.calls=[];
      commands.find(c=>c.label===label).run();
      assert.deepEqual(h.testStore.calls,[['go',route]],`${language}: ${label}`);
    }
  }
  h.testStore.state.sessions.find(s=>s.id===h.testStore.state.activeSession).messages.push({role:'user',content:'hello'});
  assert.ok(h.getSlashCommands().some(c=>c.label==='/export'));
});
