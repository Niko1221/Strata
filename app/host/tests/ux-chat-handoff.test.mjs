/** Runs actual ChatScreen submit closures and MessageView's early user return. No browser/GUI claim. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,mkdtempSync,rmSync} from 'node:fs';
import {build} from 'esbuild';
import {dirname,resolve,join} from 'node:path';
import {tmpdir} from 'node:os';
import {fileURLToPath} from 'node:url';
import {createChatStore} from '../server/chat-store.mjs';
import {installFakeHostFetch} from './helpers/fake-host-chats.mjs';
import {component,button,flush} from './helpers/ux-component.mjs';

async function chat({lifecycle=false}={}){
  const file=fileURLToPath(new URL('../src/screens/Chat.tsx',import.meta.url)),source=readFileSync(file,'utf8');
  const key='chat'+Math.random().toString(36).slice(2),c={states:[],cursor:0,sent:[],store:{sessions:[{id:'a',messages:[],preferences:{agent:false,useTools:false}}],parked:{},quotes:{},activeSession:'a',activeWorkspace:'root',streaming:false,connection:'live',library:{loaded:'hikari',variants:[{id:'hikari',label:'Existing Hikari'}],mode:'real'},remoteModel:null,providers:{connections:[]}}};globalThis[key]=c;
  const storage=()=>{const map=new Map();return {get length(){return map.size;},key:i=>[...map.keys()][i]??null,getItem:k=>map.get(k)??null,setItem:(k,v)=>map.set(k,String(v)),removeItem:k=>map.delete(k)};};
  Object.defineProperty(globalThis,'localStorage',{configurable:true,value:storage()});Object.defineProperty(globalThis,'sessionStorage',{configurable:true,value:storage()});
  c.lifecycle=lifecycle;c.effects=[];
  let fake,dir;
  if(lifecycle){
   globalThis.window=new EventTarget();globalThis.document=Object.assign(new EventTarget(),{hasFocus:()=>true,visibilityState:'visible',documentElement:{dataset:{},style:{setProperty(){}},classList:{toggle(){},add(){},remove(){}}}});
   globalThis.matchMedia=()=>({matches:false,addEventListener(){},removeEventListener(){},addListener(){},removeListener(){}});
   dir=mkdtempSync(join(tmpdir(),'strata-first-search-'));fake=installFakeHostFetch(createChatStore({dir}));
   const bundled=await build({entryPoints:[fileURLToPath(new URL('../src/lib/store.ts',import.meta.url))],bundle:true,format:'esm',platform:'node',write:false,logLevel:'silent',alias:{react:fileURLToPath(new URL('./helpers/react-stub.mjs',import.meta.url))}});
   c.actualStore=await import('data:text/javascript;base64,'+Buffer.from(bundled.outputFiles[0].text+'\n// '+key).toString('base64'));
   c.store.sessions=[];c.store.activeSession=null;
  }
  const imports=new Map();for(const match of source.matchAll(/import\s*\{([\s\S]*?)\}\s*from\s*['"]([^'"]+)['"]/g)){const names=match[1].split(',').map(s=>s.trim()).filter(s=>s&&!s.startsWith('type '));imports.set(match[2],[...new Set([...(imports.get(match[2])??[]),...names])]);}
  const custom={
    useStore:'fn=>fn(c.store)',sendMessage:'async(text,options)=>{c.sent.push({text,options});c.store.sessions[0].messages.push({id:"sent",role:"user",content:text,at:new Date().toISOString(),webSources:options?.webSearch?.sources.map(({url,title})=>({url,title}))});}',
    newSession:'(...args)=>{const s=c.actualStore.newSession(...args);c.store.sessions=c.actualStore.getState().sessions;c.store.activeSession=s.id;return s;}',
    useLocation:'()=>({route:"chat",param:c.store.activeSession})',useNative:'()=>({instances:[]})',useChatPhotos:'()=>[]',usePlan:'()=>null',usePlanQuiet:'()=>null',useLiveStatus:'()=>null',messageArtifacts:'()=>[]',
    tr:'v=>v',localeTag:'()=>"en"',relTime:'()=>"now"',timeSkipPrompt:'()=>null',noModelNotice:'()=>null',previewWidth:'()=>400',DEFAULT_PREVIEW_FRACTION:'0.5',PREVIEW_SPLIT_KEY:'"preview"',KEYS_UP:'[]',PILL_DISTANCE:'100',getSlashCommands:'()=>[]',
  };
  const compiled=await build({stdin:{contents:source+'\nexport { MessageView };',resolveDir:dirname(file),sourcefile:file,loader:'tsx'},bundle:true,write:false,format:'esm',platform:'node',jsx:'automatic',logLevel:'silent',plugins:[{name:'chat-boundaries',setup(b){b.onResolve({filter:/.*/},args=>args.path==='../lib/quotes'?{path:resolve(dirname(file),'../lib/quotes.ts')}:{path:args.path,namespace:'stub'});b.onLoad({filter:/.*/,namespace:'stub'},args=>{
    if(args.path==='react/jsx-runtime')return {contents:'export const Fragment="Fragment";export const jsx=(type,props)=>({type,props});export const jsxs=jsx;'};
    if(args.path==='react')return {contents:`const c=globalThis[${JSON.stringify(key)}];export const Suspense=({children})=>children,memo=f=>f,useMemo=f=>f(),useCallback=f=>f;export function useEffect(fn,deps){if(!c.lifecycle)return;const i=c.cursor++,before=c.states[i];if(!before||!deps||deps.some((v,n)=>v!==before.deps[n])){before?.cleanup?.();c.states[i]={deps};c.effects.push(()=>c.states[i].cleanup=fn());}}export function useState(initial){const i=c.cursor++;if(!(i in c.states))c.states[i]=typeof initial==='function'?initial():initial;return [c.states[i],v=>{const next=typeof v==='function'?v(c.states[i]):v;if(!Object.is(next,c.states[i]))c.dirty=true;c.states[i]=next;}];}export const useRef=v=>useState({current:v})[0];`};
    return {contents:`const c=globalThis[${JSON.stringify(key)}];`+(imports.get(args.path)??[]).map(name=>`export const ${name}=${custom[name]??(/^[A-Z]/.test(name)?JSON.stringify(name):'()=>null')};`).join('\n')};
  });}}]});
  const module=await import('data:text/javascript;base64,'+Buffer.from(compiled.outputFiles[0].text).toString('base64'));
  return {c,module,render(){let tree;for(let n=0;n<10;n++){c.cursor=0;c.dirty=false;tree=module.ChatScreen();for(const fn of c.effects.splice(0))fn();if(!c.dirty||!c.lifecycle)return tree;}throw new Error('Render did not settle');},async dispose(){for(const row of c.states)row?.cleanup?.();if(c.actualStore)await c.actualStore.flushSessions();fake?.restore();if(dir)rmSync(dir,{recursive:true,force:true});delete globalThis[key];}};
}
function nodes(node,predicate){if(!node||typeof node!=='object')return [];if(Array.isArray(node))return node.flatMap(n=>nodes(n,predicate));return [...(predicate(node)?[node]:[]),...Object.values(node.props??{}).flatMap(v=>nodes(v,predicate))];}
const composer=tree=>nodes(tree,n=>n.type?.name==='Composer')[0];
for(const typed of ['Keep this unsent question','/research topic'])test(`first chat preserves ${typed} through search cancel, close and reopen while query stays independent`,async t=>{
 const h=await chat({lifecycle:true});t.after(()=>h.dispose());let tree=h.render();assert.equal(composer(tree).props.session,null);
 composer(tree).props.setDraft(typed);tree=h.render();if(typed.startsWith('/research'))composer(tree).props.onSubmit();else composer(tree).props.onSearch();
 tree=h.render();const expected=typed.startsWith('/research')?'topic':typed;
 assert.equal(composer(tree).props.draft,expected,'real hydration must retain the typed draft');
 const id=composer(tree).props.session.id;assert.equal(sessionStorage.getItem('strata.chat-draft.'+id),expected);assert.equal(h.c.sent.length,0);
 const posts=[],search=await component('WebSearchDialog.tsx',{get:async()=>({available:true}),post:async(...args)=>posts.push(args)});
 for(const close of ['Cancel','close']){
  const props=nodes(tree,n=>n.type==='WebSearchDialog')[0].props;search.render('WebSearchDialog',props);await flush();let dialog=search.render('WebSearchDialog',props);
  const input=nodes(dialog,n=>n.type==='input')[0];assert.equal(input.props.value,'');input.props.onChange({target:{value:'Independent query'}});dialog=search.render('WebSearchDialog',props);
  if(close==='Cancel')button(dialog,'Cancel').props.onClick();else dialog.props.onClose();
  tree=h.render();assert.equal(composer(tree).props.draft,expected);assert.equal(nodes(tree,n=>n.type==='WebSearchDialog')[0].props.open,false);
  search.render('WebSearchDialog',nodes(tree,n=>n.type==='WebSearchDialog')[0].props);
  composer(tree).props.onSearch();tree=h.render();assert.equal(composer(tree).props.session.id,id);assert.equal(composer(tree).props.draft,expected);
 }
 assert.equal(posts.length,0);assert.equal(h.c.store.sessions.length,1);assert.equal(h.c.actualStore.getState().sessions[0].messages.length,0);
});
test('external typed research opens explicit search once, then selected results send an ordinary question to the same model',async()=>{
  const h=await chat();let tree=h.render();composer(tree).props.setDraft('/research topic');tree=h.render();composer(tree).props.onSubmit();
  tree=h.render();let dialog=nodes(tree,n=>n.type==='WebSearchDialog')[0];assert.equal(dialog.props.open,true);assert.equal(h.c.sent.length,0);
  const result={resultId:'record',sources:[{url:'https://example.com/source',title:'Source',snippet:'Snippet'}],expiresAt:Date.now()+60000};dialog.props.onUse(result);dialog.props.onClose();
  tree=h.render();composer(tree).props.onSubmit();
  assert.equal(h.c.sent.length,1);assert.equal(h.c.sent[0].text,'topic');assert.equal(h.c.sent[0].options.webSearch.resultId,'record');assert.equal(h.c.store.library.loaded,'hikari');assert.equal(h.c.store.remoteModel,null);
});
test('sent user message renders source URLs and snippet-only disclosure through its actual early-return branch',async()=>{
  const h=await chat();let tree=h.render();composer(tree).props.setDraft('Question');tree=h.render();composer(tree).props.onSearch();tree=h.render();const dialog=nodes(tree,n=>n.type==='WebSearchDialog')[0];
  dialog.props.onUse({resultId:'record',sources:[{url:'https://example.com/source',title:'Source',snippet:'Snippet'}],expiresAt:Date.now()+60000});dialog.props.onClose();tree=h.render();composer(tree).props.onSubmit();
  const message=h.c.store.sessions[0].messages.at(-1);assert.equal(message.role,'user');h.c.cursor=0;tree=h.module.MessageView({message,isLast:true});
  assert.equal(nodes(tree,n=>n.type==='a'&&n.props.href==='https://example.com/source').length,1);
  assert.equal(nodes(tree,n=>n.type==='p'&&n.props.children==='Search references · snippets only').length,1);
});
test('supervised agent research retains the current main-line route',async()=>{
  const h=await chat();h.c.store.engine={engine:{running:true}};h.c.store.sessions[0].preferences.agent=true;
  let tree=h.render();composer(tree).props.setDraft('/research topic');tree=h.render();composer(tree).props.onSubmit();
  assert.equal(h.c.sent.length,1);assert.equal(h.c.sent[0].text,'/research topic');
  assert.equal(nodes(h.render(),n=>n.type==='WebSearchDialog')[0].props.open,false);
});
test('selected references coexist with current quote composition and are consumed once',async()=>{
  const h=await chat();h.c.store.quotes.a=[{id:'q',text:'Quoted passage',messageId:'earlier',role:'assistant'}];
  let tree=h.render();composer(tree).props.setDraft('Question');tree=h.render();composer(tree).props.onSearch();
  const dialog=nodes(h.render(),n=>n.type==='WebSearchDialog')[0];
  dialog.props.onUse({resultId:'record',sources:[],expiresAt:Date.now()+60000});dialog.props.onClose();
  composer(h.render()).props.onSubmit();
  assert.equal(h.c.sent[0].text,'Quoting:\n「Quoted passage」\n→ Reply: Question');assert.equal(h.c.sent[0].options.webSearch.resultId,'record');
  tree=h.render();composer(tree).props.setDraft('Next');composer(h.render()).props.onSubmit();
  assert.equal(h.c.sent[1].options.webSearch,undefined);
});
