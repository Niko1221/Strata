const test = require('node:test');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const legacy = require('./history.js');
const {createStore, ACTIVE, PENDING, SNAPSHOT} = require('./disk_history.js');
const clone = (v) => JSON.parse(JSON.stringify(v));
const empty = {through: 0, summary: '', count: 0};
const message = (text, role = 'user') => ({role, text});
function fixture(seed = [{id:'first', messages:[message('existing')]}], saved = {}) {
  const storageData = new Map(Object.entries(saved)), chats = new Map(), receipts = new Map(), requests = [];
  let counter = 0, loseReply = false, badMigration = false;
  const storage = {getItem:k=>storageData.get(k)??null, setItem:(k,v)=>storageData.set(k,v), removeItem:k=>storageData.delete(k)};
  const hash = async raw => crypto.createHash('sha256').update(raw).digest('hex');
  const insert = c => chats.set(c.id, {title:c.messages[0]?.text || 'new', createdAt:10, updatedAt:10, revision:1,
    memory:empty, draft:'', draftAttachments:[], ...clone(c)});
  seed.forEach(insert);
  const meta = c => ({id:c.id,title:c.title,createdAt:c.createdAt,updatedAt:c.updatedAt,revision:c.revision,messageCount:c.messages.length});
  const fail = (status, message) => { const e=new Error(message); e.status=status; throw e; };
  async function request(path, body, options = {}) {
    requests.push({path, body:body===undefined?undefined:clone(body), options});
    const u = new URL('http://local/'+path), id=u.searchParams.get('id');
    if (u.pathname==='/api/history/migrate') {
      const book=JSON.parse(body); book.chats.forEach(insert);
      return {saved:true, source_sha256:badMigration?'bad':await hash(body), source_count:book.chats.length,
        mapped_count:book.chats.length,active:book.active,backup:'migration.json.gz'};
    }
    if (u.pathname==='/api/history') {
      const q=u.searchParams.get('q')||'', all=[...chats.values()].filter(c=>!q || c.title.includes(q));
      return {chats:all.map(meta),total:all.length,hasMore:false};
    }
    if(u.pathname==='/api/history/recover') {
      const source=chats.get(body.id), same=JSON.stringify(source.messages.slice(body.offset))===JSON.stringify(body.messages) && source.draft===body.draft;
      let c=source, separate=false;
      if(source.revision!==body.revision && !same) {
        const id='recovered-'+body.recoveryId; insert({...source,id,messages:source.messages.slice(0,body.offset)});c=chats.get(id);separate=true;
      }
      c.messages=[...c.messages.slice(0,body.offset),...clone(body.messages)];c.memory=clone(body.memory);c.draft=body.draft;c.draftAttachments=clone(body.draftAttachments);c.revision++;
      return {saved:true,chat:meta(c),offset:body.offset,recoveredSeparately:separate};
    }
    if (u.pathname==='/api/history/chat' && body!==undefined) {
      if(receipts.has(body.writeId)) return {...clone(receipts.get(body.writeId)),replayed:true};
      let c=chats.get(body.id);
      if(c && c.revision!==body.revision) fail(409,'conflict');
      if(!c) { insert({id:body.id,messages:[],revision:0}); c=chats.get(body.id); }
      const written=body.messages?.length||0;
      if(body.start!==undefined) c.messages=[...c.messages.slice(0,body.start),...clone(body.messages)];
      c.memory=clone(body.memory||empty); c.draft=body.draft||''; c.draftAttachments=clone(body.draftAttachments||[]);
      c.title=c.messages[0]?.text||'new'; c.revision++; c.updatedAt++;
      const result={saved:true,chat:meta(c),writtenMessages:written}; receipts.set(body.writeId,clone(result));
      if(loseReply) { loseReply=false; fail(503,'lost reply'); }
      return result;
    }
    if(['/api/history/chat','/api/history/messages'].includes(u.pathname)) {
      const c=chats.get(id); if(!c) fail(404,'missing');
      const limit=+(u.searchParams.get('limit')||40), end=u.searchParams.has('end')?+u.searchParams.get('end'):c.messages.length,
        start=u.searchParams.has('start')?+u.searchParams.get('start'):Math.max(0,end-limit);
      return {...meta(c),offset:start,messages:clone(c.messages.slice(start,start+limit)),memory:clone(c.memory),draft:c.draft,draftAttachments:clone(c.draftAttachments)};
    }
    if(u.pathname==='/api/history/recall') return {hits:[]};
    if(u.pathname==='/api/history/import') return {saved:true,added:0};
    throw new Error('Unexpected path '+path);
  }
  const open=()=>createStore({storage,request,legacy,hash,makeId:()=>`generated-${++counter}`});
  return {open,requests,chats,storageData,lose:()=>{loseReply=true;},badMigration:()=>{badMigration=true;}};
}

test('startup fetches metadata and only the selected recent window',async()=>{
  const records=Array.from({length:120},(_,i)=>message('record '+i));
  const f=fixture([{id:'first',messages:records},{id:'other',messages:[message('private other')]}]), s=f.open();
  await s.init();
  assert.equal(s.current().offset,80); assert.equal(s.current().messages.length,40);
  assert.equal(f.requests.filter(r=>r.path.includes('id=other')).length,0);
  assert.equal(s.list().chats[0].messages,undefined);
});
test('draft-only saves and unchanged saves avoid writing transcripts',async()=>{
  const f=fixture(),s=f.open(); await s.init(); const c=s.current();
  await s.save(c.messages,c.memory,'draft');
  const writes=f.requests.filter(r=>r.body?.writeId);
  assert.equal(writes.length,1); assert.equal(writes[0].body.messages,undefined);
  await s.save(c.messages,c.memory,'draft');
  assert.equal(f.requests.filter(r=>r.body?.writeId).length,1);
});
test('an append writes only new messages and leaves older/other chats intact',async()=>{
  const original=Array.from({length:120},(_,i)=>message('record '+i));
  const f=fixture([{id:'first',messages:original},{id:'other',messages:[message('untouched')]}]),s=f.open(); await s.init();
  const c=s.current(); await s.save([...c.messages,message('new'),message('answer','assistant')],empty,'');
  const body=f.requests.find(r=>r.body?.writeId).body;
  assert.equal(body.start,120); assert.equal(body.messages.length,2);
  assert.deepEqual(f.chats.get('first').messages.slice(0,120),original);
  assert.equal(f.chats.get('other').messages[0].text,'untouched');
});
test('migration retires browser copies only after checksummed disk acknowledgement',async()=>{
  const book={version:1,active:'old',chats:[{id:'old',messages:[message('old data')],memory:empty,draft:'unsent'}]};
  const f=fixture([],{[legacy.KEY]:JSON.stringify(book),'strata.chat':'legacy raw'}),s=f.open(); await s.init();
  assert.equal(s.current().id,'old'); assert.equal(s.current().draft,'unsent');
  assert.equal(f.storageData.has(legacy.KEY),false);
  assert.equal(f.storageData.has('strata.chat'),false);
  const migration=f.requests.find(r=>r.path==='api/history/migrate');
  assert.equal(JSON.parse(migration.body).legacyStorage.chat,'legacy raw');
  const g=fixture([],{[legacy.KEY]:JSON.stringify(book)});g.badMigration();
  await assert.rejects(g.open().init());assert.equal(g.storageData.has(legacy.KEY),true);
});
test('a lost save reply is recovered using the same write id before the next edit',async()=>{
  const f=fixture(),s=f.open();await s.init();const c=s.current();f.lose();
  const records=[...c.messages,message('new')];
  await assert.rejects(s.save(records,empty,'draft'));
  assert.ok(f.storageData.has(PENDING+c.id));
  await s.save(records,empty,'next draft');
  assert.equal(f.chats.get(c.id).messages.length,2);assert.equal(f.chats.get(c.id).draft,'next draft');
  assert.equal(f.storageData.has(PENDING+c.id),false);
});
test('stale writes preserve the server chat and the pending recovery record',async()=>{
  const f=fixture(),s=f.open();await s.init();const c=s.current();f.chats.get(c.id).revision++;
  await assert.rejects(s.save([...c.messages,message('unsaved')],empty,''),{status:409});
  assert.equal(f.chats.get(c.id).messages.length,1);assert.ok(f.storageData.has(PENDING+c.id));
});
test('older messages load in bounded pages, and the retained window can be trimmed again',async()=>{
  const records=Array.from({length:120},(_,i)=>message('record '+i));
  const f=fixture([{id:'first',messages:records}]),s=f.open();await s.init();
  const c=await s.older();assert.equal(c.offset,40);assert.deepEqual(c.messages,records.slice(40));
  const tail=s.trim();assert.equal(tail.offset,80);assert.deepEqual(tail.messages,records.slice(80));
  assert.ok(f.requests.some(r=>r.path.includes('end=80&limit=40')));
});
test('reloading selects the last-opened chat and does not leave all-chat browser copies',async()=>{
  const f=fixture([{id:'first',messages:[message('one')]},{id:'second',messages:[message('two')]}]),s=f.open();
  await s.init();await s.select('second');assert.equal(f.storageData.get(ACTIVE),'second');
  const next=f.open();await next.init();assert.equal(next.current().id,'second');
  assert.equal(f.storageData.has(legacy.KEY),false);
});
test('the latest dirty window is staged synchronously and can be restored after page closure',async()=>{
  const f=fixture(),s=f.open();await s.init();const c=s.current();
  const saving=s.save([...c.messages,message('received answer','assistant')],empty,'last keystrokes');
  const raw=f.storageData.get(SNAPSHOT+c.id);assert.ok(raw,'snapshot must exist before the first await');
  await saving;
  const restored=fixture([{id:'first',messages:[message('existing')]}],{[ACTIVE]:'first',[SNAPSHOT+'first']:raw});
  const next=restored.open();await next.init();
  assert.equal(next.current().draft,'last keystrokes');assert.equal(next.current().messages[1].text,'received answer');
  assert.equal(restored.storageData.has(SNAPSHOT+'first'),false);
});
test('a conflicting staged draft is restored separately and leaves the other tabs conversation intact',async()=>{
  const snapshot={id:'first',revision:1,recoveryId:'recovery-test',offset:0,messages:[message('my original')],memory:empty,draft:'unsaved',draftAttachments:[]};
  const f=fixture([{id:'first',messages:[message('other tab update')],revision:2}],{[ACTIVE]:'first',[SNAPSHOT+'first']:JSON.stringify(snapshot)});
  const s=f.open();await s.init();assert.equal(s.current().recoveredSeparately,true);
  assert.equal(f.chats.get('first').messages[0].text,'other tab update');assert.equal(s.current().draft,'unsaved');
});
