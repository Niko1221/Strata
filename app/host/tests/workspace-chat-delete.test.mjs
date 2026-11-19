/** Bulk sidebar deletion over the real writer and disk store; only fictional chats. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {build} from 'esbuild';
import {fileURLToPath} from 'node:url';
import {mkdtempSync, rmSync} from 'node:fs';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {createChatStore, realIo} from '../server/chat-store.mjs';
import {installFakeHostFetch} from './helpers/fake-host-chats.mjs';

const here = path => fileURLToPath(new URL(path, import.meta.url));
const workspace = {id:'a',name:'Notes',path:'C:/fictional/notes',aliases:['C:/fictional/old-notes']};
const chat = (id, path=workspace.path, archived=false) => ({id,title:id,workspace:path,archived,
  createdAt:'2026-11-10T00:00:00Z',updatedAt:'2026-11-10T00:00:00Z',messages:[{id:id+'-m',role:'user',content:'Fictional note'}]});

async function fixture(t, sessions, io=realIo) {
  const dir=mkdtempSync(join(tmpdir(),'strata-bulk-chats-')),disk=createChatStore({dir,io});
  for(const session of sessions)await disk.save(session.id,JSON.stringify(session));
  const fake=installFakeHostFetch(disk),memory=new Map();
  globalThis.window=globalThis;
  globalThis.document={addEventListener(){},hasFocus:()=>true,visibilityState:'visible',documentElement:{dataset:{},style:{setProperty(){}},classList:{toggle(){},add(){},remove(){}}}};
  globalThis.localStorage={getItem:k=>memory.get(k)??null,setItem:(k,v)=>memory.set(k,v),removeItem:k=>memory.delete(k),key:i=>[...memory.keys()][i],get length(){return memory.size;}};
  globalThis.matchMedia=()=>({matches:false,addEventListener(){},removeEventListener(){},addListener(){},removeListener(){}});
  globalThis.__bulkStreams=[];
  const compiled=await build({entryPoints:[here('../src/lib/store.ts')],bundle:true,write:false,format:'esm',platform:'node',
    alias:{react:here('./helpers/react-stub.mjs')},plugins:[{name:'controlled-client',setup(b){
      b.onResolve({filter:/\/client$/},()=>({path:'client',namespace:'bulk'}));
      b.onLoad({filter:/.*/,namespace:'bulk'},()=>({contents:`
        export class ApiError extends Error {}
        export const api={get:async(path)=>{if(/^\\/api\\/workspace(?:\\?|$)/.test(path))return {path:${JSON.stringify(workspace.path)},entries:[]};throw Error('unrelated route');},post:async(path)=>path==='/api/workspaces'?${JSON.stringify(workspace)}:{},put:async()=>({}),patch:async()=>({}),del:async()=>({})};
        export const onEvent=()=>()=>{};
        export async function streamPost(path,body,onFrame){await new Promise(release=>globalThis.__bulkStreams.push({body,onFrame,release}));}` }));
    }}]});
  const store=await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text+'\n//# sourceURL=workspace-delete-fixture.mjs').toString('base64')}`);
  await store.loadChatsFromHost();await store.addWorkspace(workspace.path);
  t.after(async()=>{await store.flushSessions();for(const toast of store.getState().toasts)store.dismissToast(toast.id);fake.restore();rmSync(dir,{recursive:true,force:true});});
  return {store,disk,fake};
}

test('deletes every confirmed chat including archived/alias rows beyond the visible20, but keeps other folders and new chats',async t=>{
  const sessions=Array.from({length:25},(_,n)=>chat('s'+n));sessions.push(chat('archived',workspace.path,true),chat('alias',workspace.aliases[0]),chat('other','D:/fictional/other'));
  const {store,disk}=await fixture(t,sessions);
  const confirmed=sessions.filter(s=>s.id!=='other').map(s=>s.id);
  const added=store.newSession();await store.flushSessions();
  assert.equal(await store.deleteWorkspaceSessions(workspace.id,confirmed),true);
  assert.deepEqual(disk.list().chats.map(s=>s.id).sort(),[added.id,'other'].sort());
  assert.deepEqual(store.getState().sessions.map(s=>s.id).sort(),[added.id,'other'].sort());
  assert.equal(createChatStore({dir:join(disk.root,'..')}).has('s24'),false,'deletion persists after restart');
});

test('a refused deletion stays visible and retrying does not remove another folder',async t=>{
  let refuse=true;
  const io={...realIo,async unlink(path){if(refuse&&path.endsWith('blocked.json'))throw Object.assign(Error('Fictional permission refusal'),{code:'EACCES'});return realIo.unlink(path);}};
  const {store,disk}=await fixture(t,[chat('blocked'),chat('deleted'),chat('other','D:/fictional/other')],io);
  assert.equal(await store.deleteWorkspaceSessions(workspace.id,['blocked','deleted']),false);
  assert.equal(disk.has('blocked'),true);assert.equal(disk.has('deleted'),false);assert.equal(disk.has('other'),true);
  assert.equal(store.getState().sessions.some(s=>s.id==='blocked'),true);
  refuse=false;await store.flushSessions();
  assert.equal(disk.has('blocked'),true,'a background save must not retry a failed deletion after the chat was restored');
});

test('a chat that starts answering after confirmation blocks the batch without deleting any chat',async t=>{
  const {store,disk}=await fixture(t,[chat('live'),chat('idle')]);
  store.selectSession('live');const turn=store.sendMessage('A fictional question');
  try{
    for(let n=0;n<100&&!globalThis.__bulkStreams.length;n++)await new Promise(r=>setTimeout(r,5));
    assert.equal(globalThis.__bulkStreams.length,1);
    assert.equal(await store.deleteWorkspaceSessions(workspace.id,['live','idle']),false);
    assert.equal(disk.has('live'),true);assert.equal(disk.has('idle'),true);
  }finally{for(const stream of globalThis.__bulkStreams){stream.onFrame({type:'done',tokensIn:1,tokensOut:0});stream.release();}await turn;}
});
