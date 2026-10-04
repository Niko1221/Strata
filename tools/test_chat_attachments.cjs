"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const source = fs.readFileSync(require.resolve("../serve/web/app.js"), "utf8");
const section = (start, end) => source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
function harness() {
  const elements = {}, readers = [], saved = [], notices = [];
  const element = () => ({children: [], value: "", attrs: {}, appendChild(x) { this.children.push(x); },
    setAttribute(k, v) { this.attrs[k] = v; }, set innerHTML(v) { this.children = []; }, get innerHTML() { return ""; }});
  class Reader {
    constructor() { readers.push(this); }
    readAsDataURL(file) { if (file.throwRead) throw Error("read failed"); }
    readAsText(file) { this.readAsDataURL(file); }
    complete(result) { this.result = result; return this.onload(); }
    fail() { this.onerror(); }
  }
  const ctx = vm.createContext({Promise, Date, AbortController, FileReader: Reader, document: {createElement: element},
    $: id => elements[id] ||= element(), icon: () => "", toast: (...args) => notices.push(args),
    autosize() {}, renderChat() {}, saveChat() {},
    localStorage: {setItem: (key, value) => { saved.push(JSON.parse(value)); }},
    store: {}, saved, headers: () => ({"Content-Type":"application/json"}),
    fetch() { throw Error("Network must not be used by these tests"); }});
  vm.runInContext(`let attachments = [], attachmentEpoch = 0, pendingAttachmentReads = 0;
    const attachmentExtractions = new Set();
    let documentExtractionQueue = Promise.resolve();
    let busy = null, draftSaveWarned = false;
    let messages = [], health = {images:true};
    function wireMessage(m) { messages=[m]; return apiMessages(); }`, ctx);
  vm.runInContext(section("function saveDraft()", "let busy =") +
    section("function updateSend()", "async function send()") +
    section("async function send()", '$("composer").onsubmit') +
    section("const TEXT_EXT", '$("attach-btn").onclick') +
    section("function apiMessages()", "function setBusy(on)") +
    section('$("new-btn").onclick', '$("export-btn").onclick') +
    'function newChat() { $("new-btn").onclick(); }', ctx);
  return {ctx, elements, readers, saved, notices, run: code => vm.runInContext(code, ctx)};
}
const image = name => ({name, type:"image/png", size:100});
test("reading blocks immediate send, releases on completion, and saves unsent image draft", async () => {
  const h = harness(); h.ctx.files = [image("screen.png")];
  h.run('$("input").value="question"; addFiles(files)');
  assert.equal(h.elements["send-btn"].disabled, true);
  assert.match(h.elements["attachment-status"].textContent, /Reading 1 file/);
  await h.run("send()"); assert.equal(h.run("messages.length"), 0);
  h.readers[0].complete("data:image/png;base64,AA");

  assert.equal(h.elements["send-btn"].disabled, false);
  assert.equal(h.saved[0].attachments[0].name, "screen.png");
  const chip = h.elements.attachments.children[0];
  assert.equal(chip.children[0].src, "data:image/png;base64,AA");
  assert.equal(chip.children[2].attrs["aria-label"], "Remove screen.png");
  chip.children[2].onclick();
  assert.equal(h.saved.at(-1).attachments.length, 0);
});
test("New chat ignores old callbacks without decrementing new reads", () => {
  const h = harness(); h.ctx.files = [image("old.png")]; h.run("addFiles(files)");
  h.run("newChat()"); h.ctx.files = [image("new.png")]; h.run("addFiles(files)");
  h.readers[0].complete("old");
  assert.equal(h.run("attachments.length"), 0); assert.equal(h.run("pendingAttachmentReads"), 1);
  h.readers[1].complete("new");
  assert.equal(h.run("attachments[0].name"), "new.png");
});
test("errors, aborts and synchronous read failures release pending state once", () => {
  for (const failure of ["error", "abort", "throw"]) {
    const h = harness(); h.ctx.files = [{...image("bad.png"), throwRead:failure === "throw"}]; h.run("addFiles(files)");
    if (failure === "error") { h.readers[0].fail(); h.readers[0].fail(); }
    if (failure === "abort") h.readers[0].onabort();
    assert.equal(h.run("pendingAttachmentReads"), 0); assert.equal(h.elements["send-btn"].disabled, false);
    assert.equal(h.notices.length, 1); assert.match(h.notices[0][1], /could not be read/);
  }
});
test("clipboard prefers files, falls back to image items and excludes text/null", () => {
  const h = harness(); let calls = 0; const img = image("paste.png");
  h.ctx.clipboard = {files:[img],items:[{kind:"file",type:"image/png",getAsFile() { calls++; return img; }}]};
  assert.equal(h.run("clipboardImages(clipboard).length"), 1); assert.equal(calls, 0);
  h.ctx.clipboard.files = [];
  h.ctx.clipboard.items.push({kind:"string",type:"text/plain"}, {kind:"file",type:"image/png",getAsFile:() => null});
  assert.equal(h.run("clipboardImages(clipboard).length"), 1); assert.equal(calls, 1);
});
test("image-only message preserves native wire and binary text rejection releases reads", async () => {
  const h = harness(); h.ctx.msg = {role:"user",text:"",images:[{url:"data:image/png;base64,AA"}]};
  const wire = h.run("wireMessage(msg)");
  assert.equal(wire[0].content[1].type, "image_url");
  assert.equal(wire[0].content[1].image_url.url, "data:image/png;base64,AA");
  h.ctx.files = [{name:"code.py",type:"",size:20}]; h.run("addFiles(files)");
  h.readers[0].complete("a\u0000b");
  assert.equal(h.run("attachments.length"), 0); assert.equal(h.run("pendingAttachmentReads"), 0);
});
test("DOCX extraction sends base64 only and blocks send until backend text is ready", async () => {
  const h = harness(); let complete, outgoing;
  h.ctx.fetch = async (path, options) => {
    outgoing = {path, options}; return new Promise(resolve => { complete = resolve; });
  };
  h.ctx.files = [{name:"report.docx",type:"application/octet-stream",size:1000}]; h.run("addFiles(files)");
  const reading = h.readers[0].complete("data:application/octet-stream;base64,QUJD");
  await Promise.resolve();
  assert.equal(h.run("pendingAttachmentReads"), 1); await h.run("send()");
  assert.equal(h.run("messages.length"), 0);
  assert.equal(outgoing.path, "v1/files/extract");
  assert.deepEqual(JSON.parse(outgoing.options.body), {name:"report.docx",data:"QUJD"});
  complete({ok:true,json:async () => ({text:"Document content",format:"DOCX",warnings:[],truncated:true})});
  await reading;
  assert.equal(h.run("pendingAttachmentReads"), 0);
  assert.equal(h.run("attachments[0].text"), "Document content");
  assert.match(h.elements.attachments.children[0].title, /DOCX/);
  assert.match(h.notices[0][2], /content was omitted/);
  assert.doesNotMatch(h.notices[0][2], /512 KB/);
});
test("extraction errors release pending state; New chat cancels and ignores late success", async () => {
  const h = harness(); h.ctx.files = [{name:"sheet.xlsx",type:"",size:100}];
  h.ctx.fetch = async () => ({ok:false,json:async () => ({error:{message:"Encrypted document"}})});
  h.run("addFiles(files)"); await h.readers[0].complete("data:application/octet-stream;base64,AA");
  assert.equal(h.run("pendingAttachmentReads"), 0); assert.equal(h.run("attachments.length"), 0);
  assert.equal(h.notices[0][2], "Encrypted document");
  let release, signal;
  h.ctx.fetch = async (_, options) => { signal=options.signal; return new Promise(resolve => {release=resolve;}); };
  h.run("addFiles(files)"); const reading=h.readers[1].complete("data:application/octet-stream;base64,AA");
  await Promise.resolve();
  h.run("newChat()");
  assert.equal(signal.aborted, true);
  release({ok:true,json:async () => ({text:"Old result",warnings:[]})}); await reading;
  assert.equal(h.run("attachments.length"), 0); assert.equal(h.run("pendingAttachmentReads"), 0);
});
test("scanned PDF pages become native images; disabled vision warns without attaching pages", async () => {
  for (const vision of [true,false]) {
    const h=harness(); h.run(`health.images=${vision}`);
    h.ctx.files=[{name:"scan.pdf",type:"application/pdf",size:100}];
    h.ctx.fetch=async () => ({ok:true,json:async () => ({text:"",format:"PDF",warnings:[],images:[{name:"Page 1",url:"data:image/png;base64,AA"}]})});
    h.run("addFiles(files)"); await h.readers[0].complete("data:application/pdf;base64,AA");
    assert.equal(h.run("attachments.length"), vision ? 1 : 0);
    if(vision) assert.equal(h.run("wireMessage({role:'user',text:'',images:attachments})[0].content[1].type"),"image_url");
    else assert.match(h.notices[0][2], /pictures disabled/);
    assert.equal(h.run("pendingAttachmentReads"),0);
  }
});
test("document batches serialize extraction, continue after failure and remain pending until all finish", async () => {
  const h=harness(); const requests=[]; let active=0,maxActive=0;
  h.ctx.files=["first.pdf","broken.docx","third.xlsx","fourth.pdf"].map(name=>({name,type:"",size:100}));
  h.ctx.fetch=async (_,options)=>{
    active++; maxActive=Math.max(maxActive,active);
    return new Promise(resolve=>requests.push({name:JSON.parse(options.body).name, complete(ok){
      active--; resolve({ok,json:async()=>ok?{text:this.name,warnings:[]}:{error:{message:"Invalid document"}}});
    }}));
  };
  h.run("addFiles(files)"); const reads=h.readers.map(r=>r.complete("data:application/octet-stream;base64,AA"));
  await Promise.resolve();
  assert.equal(requests.length,1); assert.equal(h.run("pendingAttachmentReads"),4);
  for(let i=0;i<4;i++) {
    assert.equal(requests[i].name,h.ctx.files[i].name);
    requests[i].complete(i!==1); await reads[i];
    assert.equal(h.run("pendingAttachmentReads"),3-i);
    if(i<3) { await Promise.resolve(); assert.equal(h.elements["send-btn"].disabled,true); }
  }

  assert.equal(maxActive,1); assert.equal(h.run("attachments.length"),3);
  assert.equal(h.elements["send-btn"].disabled,false); assert.equal(h.notices.length,1);
});
test("New chat skips queued old documents and lets the new draft extract independently", async () => {
  const h=harness(); const requests=[];
  h.ctx.fetch=async (_,options)=>new Promise(resolve=>requests.push({name:JSON.parse(options.body).name,signal:options.signal,resolve}));
  h.ctx.files=["old1.pdf","old2.pdf","old3.pdf"].map(name=>({name,type:"",size:100}));
  h.run("addFiles(files)"); const oldReads=h.readers.map(r=>r.complete("data:application/pdf;base64,AA"));
  await Promise.resolve(); assert.equal(requests.length,1);
  h.run("newChat()");
  assert.equal(requests[0].signal.aborted,true);
  h.ctx.files=[{name:"new.docx",type:"",size:100}]; h.run("addFiles(files)");
  const newRead=h.readers[3].complete("data:application/octet-stream;base64,AA"); await Promise.resolve();
  assert.equal(requests.length,2); assert.equal(requests[1].name,"new.docx");
  requests[1].resolve({ok:true,json:async()=>({text:"New draft",warnings:[]})}); await newRead;
  requests[0].resolve({ok:true,json:async()=>({text:"Stale",warnings:[]})}); await Promise.all(oldReads);
  assert.equal(requests.length,2); assert.equal(h.run("attachments.length"),1);
  assert.equal(h.run("attachments[0].name"),"new.docx"); assert.equal(h.run("pendingAttachmentReads"),0);
});

test("completed draft reload restores local attachments and text without saving it again", () => {
  const h=harness(); h.ctx.files=[image("screen.png")]; h.run('$("input").value="review this"; addFiles(files)');
  h.readers[0].complete("data:image/png;base64,AA");
  const copy=h.saved.at(-1), reloaded=harness();
  reloaded.ctx.store.get=(key,fallback)=>key === "draft" ? copy : fallback;
  reloaded.run('{ ' + section('const draft = store.get', 'let attachmentEpoch').replace('let attachments =', 'attachments =') + section('$("input").value = typeof draft.text', 'const startQuestion') + ' }');
  assert.equal(reloaded.elements.input.value,"review this");
  assert.equal(reloaded.run("attachments[0].url"),"data:image/png;base64,AA");
  assert.equal(reloaded.saved.length,0);
  reloaded.run("newChat()");
  assert.equal(reloaded.saved.at(-1).attachments.length,0);
  assert.equal(reloaded.saved.at(-1).text,"");
});

test("storage quota failure preserves old stored draft and current attachments, and warns", () => {
  const h=harness(); h.ctx.files=[image("first.png")];
  h.run('$("input").value="first question"; addFiles(files)');
  h.readers[0].complete("data:image/png;base64,FIRST");
  const oldDraft=JSON.stringify(h.saved.at(-1));
  h.ctx.localStorage.setItem=()=>{throw Error("QuotaExceededError");};
  h.ctx.files=[image("second.png")];
  h.run('$("input").value="second question"; addFiles(files)');
  h.readers[1].complete("data:image/png;base64,SECOND");
  assert.equal(h.run("attachments.length"),2);
  assert.equal(h.elements.input.value,"second question");
  assert.equal(JSON.stringify(h.saved.at(-1)),oldDraft);
  assert.equal(h.notices.length,1);
  assert.match(h.notices[0][1],/page only/);
  assert.match(h.notices[0][2],/reloading may restore an older draft/);
  h.run("saveDraft()"); assert.equal(h.notices.length,1);
  h.ctx.localStorage.setItem=(_,value)=>h.saved.push(JSON.parse(value));
  h.run("saveDraft()"); assert.equal(h.saved.at(-1).attachments.length,2);
  h.ctx.localStorage.setItem=()=>{throw Error("QuotaExceededError");};
  h.run("saveDraft()"); assert.equal(h.notices.length,2);
});