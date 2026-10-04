"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const Context = require("../serve/web/chat-context.js");
const source = fs.readFileSync(require.resolve("../serve/web/app.js"), "utf8");
function section(start, end) {
  const a=source.indexOf(start), b=source.indexOf(end,a);
  assert.ok(a>=0 && b>a, `Missing source section ${start}`);
  return source.slice(a,b);
}
function chat(id="first") {return {id,title:"New chat",activeBranchId:"main",branches:[{id:"main",messages:[],context:null,draft:{}}]};}
function harness() {
  const elements={}, notices=[], readers=[], stored=new Map();
  const element=()=>({value:"",style:{},dataset:{},disabled:false,innerHTML:"", children:[],
    appendChild(x){this.children.push(x);}, append(){}, setAttribute(){},
    setCustomValidity(text){this.validity=text;},reportValidity(){},select(){},showModal(){this.open=true;},close(){this.open=false;}});
  class Reader {constructor(){readers.push(this);} readAsDataURL(){} readAsText(){} complete(result){this.result=result;this.onload();}}
  const ctx=vm.createContext({Promise,JSON,Date,FileReader:Reader,StrataChatContext:Context,
    crypto:{randomUUID:()=>"synthetic-new"},$:id=>elements[id] ||= element(),
    localStorage:{setItem:(key,value)=>stored.set(key,value)},store:{set:(key,value)=>stored.set("strata."+key,JSON.stringify(value))},
    document:{createElement:element},icon:()=>"", toast:(...args)=>notices.push(args),
    autosize(){},renderChat(){},renderHistory(){},renderAttachments(){},compactStatus(){},setBusy(){},
    headers:()=>({}),fetch(){throw Error("Unexpected network request");},
    contextRequest(){throw Error("Unexpected model request");}});
  vm.runInContext(`let messages=[], chatContext=null, attachments=[], currentChat=null, library=null;
    let chatEpoch=0, fallbackStorageWarned=false, chatList=[], busy=null, historyBusy=false, legacyRouteResolved=true;
    let health={images:true,model:'synthetic'}, settings={thinking:'none',max:'32',mcp:false},mcpInfo={tools:0},autoCompact=true;`,ctx);
  vm.runInContext(section("function currentBranch()","function renderHistory()")+
    section("function useChat(chat)","async function initializeHistory()")+
    section("async function historyAction(action)",'$("chat-select").onchange')+
    section('$("rename-btn").onclick',"async function importChats(")+
    section('$("new-btn").onclick','$("export-btn").onclick')+
    section("const TEXT_EXT","// a file's text in the message")+
    section("async function prepareChatContext(","// An answer that used MCP tools")+
    section('$("restore-btn").onclick','$("input").addEventListener("keydown"'),ctx);
  return {ctx,elements,notices,readers,stored,run:code=>vm.runInContext(code,ctx)};
}
async function flush(){for(let i=0;i<12;i++)await Promise.resolve();}

test("optional archive preserves browser chat persistence and New chat/Undo", async()=>{
  const h=harness();h.run("messages=[{role:'user',text:'keep original'}]");
  assert.equal(await h.run("saveChat()"),true);
  assert.equal(JSON.parse(h.stored.get("strata.chat"))[0].text,"keep original");
  await h.elements["new-btn"].onclick();
  assert.equal(h.run("messages.length"),0);
  assert.equal(JSON.parse(h.stored.get("strata.chat")).length,0);
  h.notices.at(-1)[4].run();
  assert.equal(h.run("messages[0].text"),"keep original");
});

test("fallback quota failure prevents compaction, retains stored source and full current chat",async()=>{
  const h=harness();h.run("messages=Array.from({length:8},(_,i)=>({role:i%2?'assistant':'user',text:'original '+i}))");
  await h.run("saveChat()");const before=h.stored.get("strata.chat-archive");
  h.ctx.localStorage.setItem=()=>{throw Error("QuotaExceededError");};
  let summaries=0;
  h.ctx.wireMessage=m=>[{role:m.role,content:m.text}];
  h.ctx.contextRequest=async(path)=>{
    if(path.includes("completions")){summaries++;throw Error("Summary must not run");}
    return {input_tokens:950,max_context:1000,effective_max_tokens:32,context_slack:8};
  };
  await assert.rejects(h.run("prepareChatContext(null,true)"),/compaction was not started/);
  assert.equal(summaries,0);assert.equal(h.run("messages.length"),8);assert.equal(h.run("chatContext"),null);
  assert.equal(h.stored.get("strata.chat-archive"),before);
  assert.match(h.notices[0][1],/page only/);
  h.run('$("input").value="unsent"; attachments=[{kind:"image",name:"keep.png",url:"data:image/png;base64,AA"}]');
  await h.elements["new-btn"].onclick();
  assert.equal(h.run("messages.length"),8);assert.equal(h.run("attachments.length"),1);assert.equal(h.elements.input.value,"unsent");
});

test("late save cannot replace a newly selected chat and draft copy is detached",async()=>{
  const h=harness();h.ctx.first=chat();h.run("useChat(first)");
  let finish,snapshot;h.ctx.adapter={save:copy=>{snapshot=copy;return new Promise(resolve=>{finish=()=>resolve(copy);});}};
  h.run('library=adapter;attachments=[{kind:"image",name:"old.png",url:"data:image/png;base64,AA"}]');
  const saving=h.run("saveChat()");h.run('attachments[0].name="changed.png"');
  assert.equal(snapshot.branches[0].draft.attachments[0].name,"old.png");
  h.ctx.next=chat("second");h.run("useChat(next)");finish();await saving;
  assert.equal(h.run("currentChat.id"),"second");assert.equal(h.run("attachments.length"),0);
});

test("old image and text read callbacks cannot enter another branch draft",()=>{
  for(const type of ["image/png","text/plain"]){
    const h=harness();h.ctx.files=[{name:type==='image/png'?"old.png":"old.txt",type,size:20}];
    h.run("addFiles(files)");h.ctx.next=chat("second");h.run("useChat(next)");
    h.readers[0].complete(type==='image/png'?"data:image/png;base64,AA":"old text");
    assert.equal(h.run("attachments.length"),0);
  }
});

test("rename retains branches/source and manually selected New chat title",async()=>{
  const h=harness();const record=chat();record.source={raw:"original graph"};record.branches.push({id:"other",messages:[{role:"user",text:"inactive"}]});
  h.ctx.record=record;h.run("useChat(record)");
  const writes=[];h.ctx.adapter={save:async copy=>{writes.push(JSON.parse(JSON.stringify(copy)));return copy;}};
  h.run("library=adapter");h.ctx.refreshHistory=async()=>{};
  h.elements["rename-title"].value=" New chat ";h.elements["rename-form"].onsubmit({preventDefault(){}});await flush();
  assert.equal(h.run("currentChat.titleManuallySet"),true);assert.equal(writes.at(-1).branches[1].messages[0].text,"inactive");
  assert.equal(writes.at(-1).source.raw,"original graph");
  h.run("delete currentChat.source;messages=[{role:'user',text:'must not rename automatically'}]");await h.run("saveChat()");
  assert.equal(writes.at(-1).title,"New chat");
});

test("rename conflict and failed restore retain title and original compacted state",async()=>{
  const h=harness();h.ctx.record=chat();h.run("useChat(record);chatContext={summary:'retained',through:0}");
  h.ctx.adapter={save:async()=>{throw Error("newer archive retained");}};h.run("library=adapter");
  h.elements["rename-title"].value="Different title";h.elements["rename-form"].onsubmit({preventDefault(){}});await flush();
  assert.equal(h.run("currentChat.title"),"New chat");
  await h.elements["restore-btn"].onclick();assert.equal(h.run("chatContext.summary"),"retained");
});

test("history rendering creates images only for local data, retaining remote URLs as inert labels",()=>{
  const h=harness(), created=[];
  h.ctx.document.createElement=tag=>{const element={tag,dataset:{},children:[],appendChild(child){this.children.push(child);},append(){}};created.push(element);return element;};
  h.ctx.timeStr=()=>"synthetic time";
  h.run(section("function safeImage(url)","function compactStatus(")+section("function msgEl(m, i)","// One MCP tool call"));
  h.ctx.message={role:"user",text:"saved",time:1,images:[{name:"remote.png",url:"https://example.invalid/private.png"},
    {name:"local.png",url:"data:image/png;base64,AA"}]};
  h.run("msgEl(message,0)");
  assert.deepEqual(created.filter(el=>el.tag==="img").map(el=>el.src),["data:image/png;base64,AA"]);
  assert.equal(h.ctx.message.images[0].url,"https://example.invalid/private.png");
});
