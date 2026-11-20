/** Actual shell/Monitor components and translation catalogs, isolated from polling and private data. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {build} from 'esbuild';

globalThis.__monitorUI = {};
globalThis.window = {location: {origin: 'http://127.0.0.1:12345'}};
globalThis.localStorage = {getItem: () => null};
const boundary = {
  store: `const c=globalThis.__monitorUI; export const useStore=fn=>fn(c.store),liveChatIds=()=>c.working;
    export const bootstrap=()=>{c.retries++;return Promise.resolve()};
    export const bootOnce=()=>{},setNavOpen=()=>{},setFocus=()=>{},addWorkspace=()=>{},deleteSession=()=>{},deleteWorkspaceSessions=()=>{},moveSession=()=>{},newSession=()=>{},toggleArchive=()=>{},go=()=>{},isUnread=()=>false,pickWorkspace=()=>{},pushToast=()=>{},renameSession=()=>{},removeWorkspace=()=>{},renameWorkspace=()=>{},selectSession=()=>{},setActiveWorkspace=()=>{},togglePin=()=>{},toggleRail=()=>{},setTheme=()=>{};`,
  native: `export const useNative=()=>globalThis.__monitorUI.native;export const nativeRouting=()=>null;`,
  router: `export const useLocation=()=>({route:'monitor',query:new URLSearchParams()}),href=()=> '#/models',closeSettings=()=>{};`,
  ui: `import React from 'react'; export const Btn=props=>{globalThis.__monitorUI.buttons.push(props);return React.createElement('button',{disabled:props.disabled},props.children)};
    export const Dialog=()=>null,Icon=()=>null,IconBtn=()=>null,Toasts=()=>null,Select=()=>null,Bar=()=>null;`,
  Brand: `export const Brand=()=>null;`,
  layout: `import React from 'react';export const Screen=props=>React.createElement('main',null,React.createElement('h1',null,props.title),props.children);`,
  charts: `export const Gauge=()=>null,ReferenceSpark=()=>null;`,
  responsive: `export const useMedia=()=>false;`,
  focus: `export const useSurfaceFocus=()=>{};`,
  commands: `export const MOD='Ctrl',handleAppShortcut=()=>{},runCommand=()=>{};`,
  'shortcut-bindings': `export const useBindings=()=>{},bindingFor=()=>null,bindingKeys=()=>[];`,
  'use-copy': `export const useCopy=()=>({copied:false,copy:()=>{}});`,
  'app-mode': `export const useAppMode=()=>({canQuit:false});`,
  'kept-state': `import {useState} from 'react';export const useKept=(_key,initial)=>useState(initial);`,
  money: `export const useMoney=()=>({format:String});`,
  client: `export const api={get:async()=>({}),post:async()=>({})};`,
  snapshot: `export const monitorPng=async()=>'';`,
};
const empty = {WorkspaceFiles:'WorkspaceFiles',TaskTree:'TaskTree',FloatingMenu:'FloatingMenu',QuitStrata:'QuitStrataDialog',HakuaLlamaChat:'HakuaLlamaChat',SnapshotDialog:'SnapshotDialog',CommandPalette:'CommandPalette',Shortcuts:'ShortcutsDialog',FolderPathDialog:'FolderPathDialog',Setup:'SetupScreen',Chat:'ChatScreen',ModelRun:'ModelRunScreen',Serving:'ServingScreen',Tasks:'TasksScreen',Skills:'SkillsScreen',Settings:'SettingsScreen'};
for (const [file, name] of Object.entries(empty)) boundary[file] = `export const ${name}=()=>null;`;
const compiled = await build({
  stdin: {contents: `import React from 'react';import {renderToStaticMarkup} from 'react-dom/server';
    import {FocusBar,ConnectionBar} from './src/app/App';import {Sidebar} from './src/app/Sidebar';import {MonitorScreen} from './src/screens/Monitor';
    export {setTranslationLanguage,tr,LANGUAGES} from './src/lib/i18n';
    export const render=name=>renderToStaticMarkup(React.createElement({FocusBar,ConnectionBar,Sidebar,MonitorScreen}[name],{containerRef:{current:null}}));`, resolveDir: process.cwd(), loader: 'tsx'},
  bundle: true, write: false, format: 'cjs', platform: 'node', jsx: 'automatic', external: ['react','react-dom','react-dom/server'],
  plugins: [{name: 'controlled-ui-boundaries', setup(b) {
    b.onResolve({filter: /^\.\.?\//}, args => {
      const name=args.path.split('/').at(-1);
      if(boundary[name])return {path:name,namespace:'boundary'};
    });
    b.onLoad({filter: /.*/,namespace:'boundary'}, args => ({contents:boundary[args.path],loader:'js',resolveDir:process.cwd()}));
    // Expose private components for tests without changing or duplicating their implementation.
    b.onLoad({filter: /[/\\]app[/\\]App\.tsx$/}, args => ({contents:readFileSync(args.path,'utf8')+'\nexport {FocusBar,ConnectionBar};',loader:'tsx'}));
  }}],
});
const module={exports:{}};
new Function('require','module','exports',compiled.outputFiles[0].text)(createRequire(import.meta.url),module,module.exports);
const {render,tr,setTranslationLanguage,LANGUAGES}=module.exports;
const text=html=>html.replace(/<[^>]*>/g,'').replace(/&#x27;/g,"'").replace(/&amp;/g,'&');
function setup(phase='unloaded',status='ready',error=null){
  Object.assign(__monitorUI,{retries:0,buttons:[],working:['fictional-away'],native:{defaultId:'one',instances:[{id:'one',status,phase,error,label:'Fictional Swift',model:'swift',gpus:[],metrics:{hardware:{cpu:12,ram_used:8*1024**3,ram_total:32*1024**3},requests:[]}}]},store:{sessions:[{id:'fictional-away',title:'Garden notebook',messages:[]}],workspaces:[],liveChats:1,library:{loaded:null,variants:[]},monitor:null,hardware:null,connection:'offline'}});
  setTranslationLanguage('en');
}
const chip=html=>/<button[^>]*class="enginechip"[\s\S]*?<\/button>/.exec(html)?.[0]??'';
test('unloaded model wins over working-chat chip without changing the ongoing chat',()=>{
  setup(); const before=structuredClone(__monitorUI.store),working=[...__monitorUI.working];
  const html=chip(render('Sidebar'));
  assert.match(text(html),/No model loaded/);assert.doesNotMatch(text(html),/Working|Ready/);
  assert.match(html,/data-state="off"/);assert.deepEqual(__monitorUI.store,before);assert.deepEqual(__monitorUI.working,working);
});
test('FocusBar and Monitor show no model loaded, not ready or an error',()=>{
  setup(); const focus=render('FocusBar'),monitor=render('MonitorScreen');
  assert.match(text(focus),/No model loaded/);assert.doesNotMatch(text(focus),/ready|engine error/);assert.match(focus,/data-state="idle"/);
  assert.match(text(monitor),/No model loaded/);assert.doesNotMatch(monitor,/data-phase="error"/);
  const speed=/<section[^>]*aria-label="Speed"[\s\S]*?<\/section>/.exec(monitor)?.[0]??'';
  assert.match(text(speed),/Unavailable/);assert.doesNotMatch(text(speed),/Measured/);
  assert.match(text(monitor),/12/, 'PC sensor reading remains visible');
});
test('actual errors and unknown phases outrank workflow activity across the shell',()=>{
  for(const [phase,status,error,label] of [['idle','error','Fictional load failure','Error'],['unavailable','ready',null,'Status unavailable'],['future-phase','ready',null,'Status unavailable']]){
    setup(phase,status,error);assert.match(text(chip(render('Sidebar'))),new RegExp(label));assert.doesNotMatch(text(chip(render('Sidebar'))),/Working/);
    assert.ok(text(render('FocusBar')).includes(tr(label)));
  }
});
test('loaded and active phases retain ready/busy color and actual workflow context',()=>{
  for(const [phase,color] of [['idle','ready'],['reading','busy'],['thinking','busy'],['generating','busy'],['loading','busy']]){
    setup(phase);assert.match(chip(render('Sidebar')),new RegExp(`data-state="${color}"`));
    if(phase==='generating')assert.match(text(chip(render('Sidebar'))),/Working/);
  }
});
test('connection banner describes app/live updates, uses real retry handler and hides only when live',async()=>{
  setup();const html=text(render('ConnectionBar'));
  assert.match(html,/Live app updates are interrupted/);assert.doesNotMatch(html,/local engine|engine host|its engine/);
  assert.match(html,/Retained readings may be stale/);
  __monitorUI.buttons[0].onClick();await Promise.resolve();assert.equal(__monitorUI.retries,1);
  __monitorUI.store.connection='connecting';assert.match(text(render('ConnectionBar')),/Connecting to the app and live updates/);
  __monitorUI.store.connection='live';assert.equal(render('ConnectionBar'),'');
});
test('all nine catalogs translate unloaded and connection states through production components',async()=>{
  for(const {value:language} of LANGUAGES){
    setup();await setTranslationLanguage(language);
    assert.ok(text(chip(render('Sidebar'))).includes(tr('No model loaded')));
    assert.ok(text(render('MonitorScreen')).includes(tr('No model loaded')));
    for(const key of ['Live app updates are interrupted.','Connecting to the app and live updates…','. Retained readings may be stale. This connection carries app data and live updates.','App data could not be refreshed','Retry the app connection.']){
      const translated=tr(key);assert.ok(translated);if(language!=='en')assert.notEqual(translated,key,`${language}: ${key}`);
    }
    for(const [phase,status] of [['idle','Ready'],['reading','Reading prompt'],['thinking','Thinking'],['generating','Generating'],['unloading','Unloading…'],['unavailable','Status unavailable'],['error','Error']]){
      __monitorUI.native.instances[0].phase=phase;
      assert.ok(text(render('FocusBar')).includes(tr(status)),`${language}: ${phase}`);
    }
  }
});
