"use strict";
const test = require("node:test"), assert = require("node:assert/strict");
const library = require("../serve/web/chat-library.js");
const context = require("../serve/web/chat-context.js");
test("full backup roundtrip preserves original branches, drafts, media and compacted state", () => {
  const session = library.fresh("one", [{role: "user", text: "original", images: [{url: "data:image/png;base64,abc"}], files: [{name: "exact.txt", text: "REVIEW-7421"}]}], {summary: "brief", through: 1});
  session.branches.push({id: "alternate", title: "Alternate", messages: [{role: "assistant", text: "other branch", reasoning: "thoughts"}], context: null});
  session.source = {format: "llama", conversation: {id: "legacy"}, messages: [{original: true}]};
  session.branches[0].draft = {text: "unsent", attachments: []};
  const restored = library.fromBackup(JSON.parse(JSON.stringify(library.backup([session]))));
  assert.deepEqual(restored, [session]); restored[0].branches[0].messages[0].text = "edited";
  assert.equal(session.branches[0].messages[0].text, "original");
});
test("invalid archive/branch/context fails instead of importing a partial archive", () => {
  const good = library.fresh("one", [{role: "user", text: "hello"}]);
  for (const bad of [null, {...good, activeBranchId: "missing"}, {...good, branches: [...good.branches, good.branches[0]]},
    {...good, branches: [{...good.branches[0], context: {summary: "wrong", through: 2}}]},
    {...good, branches: [{...good.branches[0], messages: [{role: "unknown", text: "no"}]}]}]) assert.throws(() => library.validate(bad));
  assert.throws(() => library.fromBackup({format: "other", version: 1, conversations: [good]}));
});
test("archive adapter awaits committed server writes and propagates storage errors", async () => {
  const calls = [], session = library.fresh("one");
  const db = await library.open(async (path, body) => { calls.push({path, body}); if (path.endsWith("save")) throw new Error("quota"); return {}; });
  await assert.rejects(db.save(session, true), /quota/);
  assert.equal(calls[1].body.activate, true); assert.deepEqual(session.branches[0].messages, []);
  await db.seedLegacy([], null); await db.add([session]);
  assert.equal(calls[2].path, "v1/chats/seed"); assert.equal(calls[3].path, "v1/chats/import");
});
test("async storage admission rejects before the first compaction request", async () => {
  let summaries = 0;
  await assert.rejects(context.prepare({archive: [{role: "user", text: "old"}, {role: "assistant", text: "old reply"}, {role: "user", text: "recent"}, {role: "user", text: "now"}], force: true,
    wire: m => [{role: m.role, content: m.text}], count: async () => ({input_tokens: 10, max_context: 1024, effective_max_tokens: 128}),
    summarize: async () => { summaries++; return "summary"; }, beforeCompact: async () => { throw new Error("storage refused"); }}), /storage refused/);
  assert.equal(summaries, 0);
});

test("legacy browser import uses readonly transactions and retains original stores", async () => {
  const records={conversations:[{id:"first",name:"Source title"},{id:"second",name:"Other title"}],
    messages:[{convId:"first",id:"one",content:"source"},{convId:"second",id:"two",content:"other"}]};
  const before=JSON.stringify(records), modes=[];
  let closed=false;
  const db={objectStoreNames:{contains:name=>Object.hasOwn(records,name)},
    transaction(stores,mode){
      modes.push(mode);assert.deepEqual(stores,["conversations","messages"]);assert.equal(mode,"readonly");
      let completed=0;
      const tx={objectStore:name=>({getAll(){
        const req={result:records[name]};
        queueMicrotask(()=>{req.onsuccess();if(++completed===2)queueMicrotask(()=>tx.oncomplete());});
        return req;
      }})};return tx;
    },close(){closed=true;}};
  const opened=[];
  const factory={databases:async()=>[{name:"LlamaUi"}],open(name){
    opened.push(name);const req={result:db};queueMicrotask(()=>req.onsuccess());return req;
  }};
  const imported=await library.legacy(factory);
  assert.equal(imported.length,2);assert.equal(imported[0].source,"LlamaUi");
  assert.equal(imported[0].data.messages[0].id,"one");assert.deepEqual(modes,["readonly"]);
  assert.deepEqual(opened,["LlamaUi"]);assert.equal(closed,true);assert.equal(JSON.stringify(records),before);
});

test("legacy import aborts creating absent databases when enumeration is unavailable", async () => {
  const attempted=[],aborted=[];
  const factory={open(name){
    attempted.push(name);const req={error:Error("Aborted absent database"),transaction:{abort(){aborted.push(name);queueMicrotask(()=>req.onerror());}}};
    queueMicrotask(()=>req.onupgradeneeded());return req;
  }};
  assert.deepEqual(await library.legacy(factory),[]);
  assert.deepEqual(attempted,["LlamaUi","LlamacppWebui"]);assert.deepEqual(aborted,attempted);
});
