// What the chat list does, run against the real app.js with a stub DOM and a stub IndexedDB.
// Run with: node serve/test_chat_sidebar.js   (the Python test runs it too and skips when node is missing)
//
// The stub is faithful only where the code under test depends on it: the result of a read is on the request and
// not on the store; a write the browser refuses fires abort on the transaction; a blocked open never resolves;
// and a database already at version 1 does not run onupgradeneeded. Requests answer on a later tick, so the code
// cannot be right by answering too early.
const assert = require("assert");
const fs = require("fs");
const path = require("path");

// ---------------------------------------------------------------- the stub DOM
function makeEl() {
  const el = {
    dataset: {}, attrs: {}, children: [], value: "", hidden: false, className: "", textContent: "",
    classList: {add() {}, remove() {}, toggle() {}}, style: {},
    scrollHeight: 0, scrollTop: 0, clientHeight: 0,
    addEventListener(t, f) { (this._l = this._l || {})[t] = f; },
    appendChild(c) { this.children.push(c); return c; },
    querySelectorAll(sel) { return this.children.filter((c) => c.className.includes(sel.slice(1))); },
    // the page reads a node it built with innerHTML; the stub hands back a place to write into
    querySelector(sel) {
      const found = this.children.find((c) => c.className.split(" ").includes(sel.slice(1)));
      if (found) return found;
      const n = makeEl(); n.className = sel.slice(1); n._parent = this; this.children.push(n); return n;
    },
    setAttribute(k, v) { this.attrs[k] = v; }, getAttribute(k) { return this.attrs[k]; },
    focus() {}, select() {},
    remove() { if (this._parent) this._parent.children = this._parent.children.filter((c) => c !== this); },
    replaceWith(next) { const p = this._parent; if (p) p.children = p.children.map((c) => (c === this ? next : c)); },
    closest(sel) { let n = this; while (n) { if (n.className.includes(sel.slice(1))) return n; n = n._parent; } return null; },
  };
  Object.defineProperty(el, "innerHTML", {get: () => el._html || "", set: (v) => { el._html = v; el.children = []; }});
  return el;
}

// ---------------------------------------------------------------- the stub IndexedDB
function makeIdb(hooks, seed) {
  const data = Object.assign({}, seed);          // the records, keyed by id
  const stores = {};                             // the store is only there if the upgrade handler made it
  const calls = {upgradeneeded: 0, createObjectStore: 0};
  let current = null;                            // the transaction a request belongs to
  const store = {
    transaction: null,
    put(value) {
      const req = {result: null, error: null, transaction: current, onsuccess: null, onerror: null};
      queueMicrotask(() => {
        if (hooks.quota) { req.error = {name: "QuotaExceededError"}; req.onerror && req.onerror(); return; }
        if (hooks.abort) return;                       // the transaction aborts, the request says nothing
        data[value.id] = value; req.result = value; req.onsuccess && req.onsuccess();
      });
      return req;
    },
    delete(key) {
      const req = {result: null, error: null, transaction: current, onsuccess: null, onerror: null};
      queueMicrotask(() => {
        if (hooks.quota) { req.error = {name: "QuotaExceededError"}; req.onerror && req.onerror(); return; }
        if (hooks.abort) return;                       // the transaction aborts, the request says nothing
        delete data[key]; req.result = true; req.onsuccess && req.onsuccess();
      });
      return req;
    },
    getAll() {
      const req = {result: null, error: null, transaction: current, onsuccess: null, onerror: null};
      queueMicrotask(() => {
        if (hooks.abort) return;                       // the transaction aborts, the request says nothing
        req.result = Object.values(data); req.onsuccess && req.onsuccess();
      });
      return req;
    },
  };
  const transaction = () => {
    const t = {error: null, oncomplete: null, onerror: null, onabort: null, objectStore: () => store};
    current = t;
    store.transaction = t;
    queueMicrotask(() => {
      if (hooks.quota || hooks.abort) { t.error = {name: hooks.quota ? "QuotaExceededError" : hooks.abort}; t.onabort && t.onabort(); }
      else if (t.oncomplete) t.oncomplete();
    });
    return t;
  };
  // a version change is what creates the store, so a database already at version 1 may not have it
  const db = {objectStoreNames: {contains: (n) => n in stores}, createObjectStore: (n) => { calls.createObjectStore++; stores[n] = store; },
             transaction};
  return {open: () => {
    if (hooks.throws) throw new Error("private mode");
    const r = {result: db, error: null, onupgradeneeded: null, onsuccess: null, onerror: null, onblocked: null};
    queueMicrotask(() => {
      if (hooks.blocked) { r.onblocked && r.onblocked(); return; }
      if (hooks.newDatabase !== false) { calls.upgradeneeded++; r.onupgradeneeded && r.onupgradeneeded(); }
      r.result = db; r.onsuccess && r.onsuccess();
    });
    return r;
  }, records: data, calls};
}

// ---------------------------------------------------------------- run app.js as the page would
function run(hooks, stored, seed) {
  const byId = {};
  const document = {getElementById: (id) => (byId[id] = byId[id] || makeEl()),
                    createElement: () => makeEl(), documentElement: {dataset: {}},
                    addEventListener() {}, querySelectorAll: () => []};
  const idb = makeIdb(hooks, seed);
  const env = {
    document, localStorage: {getItem: (k) => (k in stored ? stored[k] : null), setItem: (k, v) => { stored[k] = v; }},
    indexedDB: idb, location: {hash: "", pathname: "/", origin: "http://127.0.0.1"},
    history: {replaceState() {}}, matchMedia: () => ({matches: false, addEventListener() {}}),
    fetch: async () => ({ok: true, json: async () => ({}), text: async () => ""}),
    setTimeout: () => 0, requestAnimationFrame: () => 0, navigator: {clipboard: {writeText: async () => {}}},
    innerWidth: 1200, window: {addEventListener() {}}, queueMicrotask,
  };
  const names = Object.keys(env);
  const source = fs.readFileSync(path.join(__dirname, "web", "app.js"), "utf8") +
    "\nreturn {get historyOn() { return historyOn; }, get chats() { return chats; }, get chat() { return chat; }, " +
    "get messages() { return messages; }, get db() { return db; }, chatTitle, setHistory, startChat, openChat, " +
    "deleteChat, renameChat, saveChat, renderSidebar, setQuery: (v) => { sidebarQuery = v; renderSidebar(); }, el: document.getElementById, row: (cls) => document.getElementById(\"sidebar-list\").children.find((c) => c.className.includes(cls))};";
  return {s: new Function(...names, source)(...names.map((n) => env[n])), stored, records: idb.records, calls: idb.calls};
}

const wait = () => new Promise((r) => setTimeout(r, 30));
const rows = (s) => s.el("sidebar-list").children.filter((c) => c.className.split(" ").includes("chat-item")).length;
const groups = (s) => s.el("sidebar-list").children.filter((c) => c.className.split(" ").includes("sidebar-group")).map((c) => c.textContent);
// the warning is a promise to the user, so its kind and its title are part of what is checked here
const warns = (s) => s.el("toasts").children.filter((t) => t.className.split(" ").includes("st-toast--warn"));
const warnTitle = (s) => warns(s).map((t) => t.querySelector(".st-toast__title").textContent).join(" | ");
const warnText = (s) => warns(s).map((t) => t.querySelector(".t-text").textContent).join(" | ");
const firstText = (c) => (c.messages[0] || {}).text || "(empty)";
const msg = (text) => ({role: "user", text, time: Date.now()});

(async () => {
  const done = [];

  // 1. the switch off: the store is never opened, and the single chat is the page from the last release
  const off0 = run({}, {});
  await wait();
  assert.strictEqual(off0.s.historyOn, false, "off is the default");
  assert.strictEqual(off0.s.db, null, "off never opens the store");
  off0.s.messages.push(msg("the first conversation"));
  off0.s.saveChat();
  assert.strictEqual(JSON.parse(off0.stored["strata.chat"])[0].text, "the first conversation");
  assert.deepStrictEqual(Object.keys(off0.stored), ["strata.chat"], "off writes only the single chat");
  done.push("off: the store is not opened, only strata.chat is written");

  // 2. a page load with four records: the list is what the read returned
  const seed = {};
  for (let i = 0; i < 4; i++) seed["c" + i] = {id: "c" + i, title: "chat " + i, time: Date.now() - i * 3600000, messages: [msg("chat " + i)]};
  const four = run({}, {"strata.history": "true", "strata.chatId": JSON.stringify("c1")}, seed);
  await wait();
  assert.strictEqual(four.s.chats.length, 4, "all four records are read");
  assert.strictEqual(rows(four.s), 4, "the list shows them");
  assert.strictEqual(four.s.chatTitle(four.s.chat), "chat 1", "the open one is the one stored as open");
  done.push("read: 4 records in the store, 4 rows in the list");

  // 3. his steps on #1787: off, a conversation, switch on, New chat, reload - the new chat stays empty
  const b = run({}, {});
  await wait();
  b.s.messages.push(msg("the first conversation"));
  b.s.saveChat();
  b.s.setHistory(true, true, true);           // flipped in this session: the chat on the screen becomes the record
  await wait();
  assert.strictEqual(b.s.chats.length, 1, "the chat on the screen became the first record");
  b.s.startChat();
  await wait();
  assert.strictEqual(b.s.chat.messages.length, 0, "the new chat is empty");
  const again = run({}, JSON.parse(JSON.stringify(b.stored)),
                    b.s.chats.reduce((o, c) => (o[c.id] = JSON.parse(JSON.stringify(c)), o), {}));
  await wait();
  assert.strictEqual(again.s.chats.length, 2, "both chats are in the list after the reload");
  assert.strictEqual(again.s.chat.messages.length, 0, "the new chat is still empty after the reload");
  assert.strictEqual(firstText(again.s.chats[0]), "the first conversation", "the migrated record kept the conversation");
  done.push("new chat: stays empty after a reload, the previous one is kept");

  // 4. a browser that only had the single chat gets it as the first record
  const mig = run({}, {"strata.history": "true", "strata.chat": JSON.stringify([msg("old chat")])});
  await wait();
  assert.strictEqual(mig.s.chats.length, 1, "the single chat became the first record");
  assert.strictEqual(firstText(mig.s.chats[0]), "old chat");
  done.push("migration: strata.chat becomes the first record");

  // 5. rename and delete go to the record
  const id = again.s.chats[0].id;
  again.s.chats[0].title = "renamed";
  again.s.saveChat();
  await wait();
  assert.strictEqual(again.records[id].title, "renamed", "the name is in the record");
  again.s.deleteChat(again.s.chats[0]);
  await wait();
  assert.ok(!(id in again.records), "the record is gone");
  done.push("rename and delete: the record changes");

  // 6. private mode: the list lives in memory, but a delete still has to show
  const mem = run({throws: true}, {"strata.history": "true"});
  await wait();
  assert.strictEqual(mem.s.db, null, "the store is not available");
  assert.strictEqual(warns(mem.s).length, 1, "one warning");
  assert.ok(warnTitle(mem.s).includes("Chat list not saved"), "the warning is about the list");
  assert.ok(warnText(mem.s).includes("memory"), "the user is told the list lives in memory");
  mem.s.chats.push({id: "x1", title: "a", time: Date.now(), messages: [msg("hi")]});
  mem.s.chats.push({id: "x2", title: "b", time: Date.now(), messages: [msg("yo")]});
  mem.s.renderSidebar();
  assert.strictEqual(rows(mem.s), 3, "the three chats show");
  mem.s.deleteChat(mem.s.chats[1]);
  await wait();
  assert.strictEqual(rows(mem.s), 2, "the row goes away, so the list re-rendered");
  const memRow = mem.s.el("sidebar-list").children.find((c) => c.className.split(" ").includes("chat-item"));
  mem.s.renameChat(memRow, mem.s.chats[0]);
  const memField = memRow.children.find((c) => c.className.split(" ").includes("chat-item__input"));
  memField.value = "renamed in memory";
  memField._l["keydown"]({key: "Enter", preventDefault() {}});
  await wait();
  assert.ok(mem.s.el("sidebar-list").children.some((c) => (c.innerHTML || "").includes("renamed in memory")),
            "the rename shows in memory-only mode too");
  done.push("memory only: a rename and a delete still update the list");

  // 7. a fresh database: the store exists because app.js made it in the upgrade handler
  const fresh = run({}, {"strata.history": "true"});
  await wait();
  assert.strictEqual(fresh.calls.upgradeneeded, 1, "a fresh database runs onupgradeneeded");
  assert.strictEqual(fresh.calls.createObjectStore, 1, "the upgrade handler created the chats store");
  assert.ok(fresh.s.db, "the store is there");
  done.push("fresh database: the upgrade handler creates the chats store");

  // 8. a database already at version 1 does not run onupgradeneeded, so the store may not be there
  const wrong = run({newDatabase: false}, {"strata.history": "true"});
  await wait();
  assert.strictEqual(wrong.s.db, null, "a missing store is the memory-only path, not a throw");
  assert.strictEqual(wrong.calls.createObjectStore, 0, "nothing was created, and nothing was read from it");
  done.push("missing store: memory only, no unhandled rejection");

  // 9. a blocked open never resolves on its own
  const blocked = run({blocked: true}, {"strata.history": "true"});
  await wait();
  assert.strictEqual(blocked.s.db, null, "the blocked open resolves to the memory-only path");
  assert.ok(warnText(blocked.s).includes("memory"), "the user is told the list lives in memory");
  done.push("blocked open: it resolves to memory only instead of hanging");

  // 10. a write the browser refuses warns on the event that actually fires
  const full = run({quota: true}, {"strata.history": "true"});
  await wait();
  full.s.messages.push(msg("a three megabyte answer"));
  full.s.saveChat();
  await wait();
  assert.ok(warnText(full.s).includes("cannot keep more"), "the quota warning is shown on abort");
  full.s.saveChat();
  await wait();
  assert.strictEqual(warns(full.s).length, 1, "it warns once, not on every write");
  done.push("quota: the warning fires on the abort event");

  // 11. flipping the switch off stores the chat you were in, so a reload shows it
  const off = run({}, {"strata.history": "true", "strata.chat": JSON.stringify([msg("the OLD single chat")])});
  await wait();
  off.s.chats[0].messages.push(msg("work done in the sidebar"));
  off.s.saveChat();
  off.s.setHistory(false, true, true);
  assert.deepStrictEqual(JSON.parse(off.stored["strata.chat"]).map((m) => m.text),
                         ["the OLD single chat", "work done in the sidebar"], "a reload shows the chat you were in");
  done.push("off: the single chat is written when the switch is flipped");

  // 12. pictures are stored as names only
  const pic = run({}, {"strata.history": "true"});
  await wait();
  pic.s.chats[0].messages.push({role: "user", text: "the photo", time: Date.now(),
                                images: [{name: "photo.png", url: "data:image/png;base64,AAA"}]});
  pic.s.saveChat();
  await wait();
  assert.deepStrictEqual(pic.records[pic.s.chat.id].messages[0].images, [{name: "photo.png"}],
                         "no data URL in the record");
  done.push("pictures: names only");

  // 13. the search covers the title and the message text
  const search = run({}, {"strata.history": "true"}, {
    a: {id: "a", title: "the garden", time: Date.now(), messages: [msg("watering the tomatoes")]},
    b: {id: "b", title: "a chat about code", time: Date.now(), messages: [msg("nothing here")]},
    c: {id: "c", title: "tomatoes again", time: Date.now(), messages: [msg("a different question")]}});
  await wait();
  assert.strictEqual(rows(search.s), 3, "all three chats show");
  search.s.setQuery("tomatoes");
  assert.strictEqual(rows(search.s), 2, "two match: one by title, one by its message text");
  search.s.setQuery("code");
  assert.strictEqual(rows(search.s), 1, "one by title only");
  search.s.setQuery("nothing here");
  assert.strictEqual(rows(search.s), 1, "one by its message text");
  done.push("search: titles and message text");

  // 14. the day groups
  const day = 86400000;
  const grouped = run({}, {"strata.history": "true"}, {
    today: {id: "today", title: "today", time: Date.now(), messages: [msg("a")]},
    yesterday: {id: "yesterday", title: "yesterday", time: Date.now() - 1.5 * day, messages: [msg("b")]},
    week: {id: "week", title: "week", time: Date.now() - 5 * day, messages: [msg("c")]},
    older: {id: "older", title: "older", time: Date.now() - 40 * day, messages: [msg("d")]}});
  await wait();
  assert.deepStrictEqual(groups(grouped.s), ["Today", "Yesterday", "Previous 7 days", "Older"], "the day groups");

  // the group names come out newest first, and a chat inside seven days is in the same group
  grouped.s.chats[0].time = Date.now() - 2 * day;
  grouped.s.renderSidebar();
  assert.deepStrictEqual(groups(grouped.s), ["Yesterday", "Previous 7 days", "Older"], "the group follows the chat");
  done.push("day groups: Today, Yesterday, Previous 7 days, Older");

  // 15. rename through the field: Enter keeps the name, Escape gives it up
  const row = search.s.el("sidebar-list").children.find((c) => c.className.split(" ").includes("chat-item"));
  const target = search.s.chats.find((c) => c.id === "b");
  search.s.renameChat(row, target);
  const field = row.children.find((c) => c.className.split(" ").includes("chat-item__input"));
  assert.ok(field, "the row turned into a field");
  field.value = "renamed by the field";
  field._l["keydown"]({key: "Enter", preventDefault() {}});
  await wait();
  assert.strictEqual(search.records["b"].title, "renamed by the field", "Enter keeps the name");
  search.s.renameChat(search.s.el("sidebar-list").children.find((c) => c.className.split(" ").includes("chat-item")), target);
  const field2 = search.s.el("sidebar-list").children.find((c) => c.className.split(" ").includes("chat-item")).children.find((c) => c.className.split(" ").includes("chat-item__input"));
  field2.value = "given up";
  field2._l["keydown"]({key: "Escape", preventDefault() {}});
  await wait();
  assert.strictEqual(search.records["b"].title, "renamed by the field", "Escape gives the name up");
  done.push("rename: Enter keeps, Escape gives up");

  // 16. delete with Undo: the record comes back, and pressing Undo twice does not put it in twice
  const undo = run({}, {"strata.history": "true"}, {a: {id: "a", title: "the one", time: Date.now(), messages: [msg("hi")]}});
  await wait();
  undo.s.deleteChat(undo.s.chats[0]);
  await wait();
  assert.strictEqual(undo.s.chats.length, 1, "the open chat was replaced by a new one");
  assert.ok(!(("a") in undo.records), "the record is gone");
  const button = undo.s.el("toasts").children[0].children.find((c) => c.className.split(" ").includes("st-btn"));
  assert.ok(button, "the toast offers Undo");
  button.onclick();
  await wait();
  assert.strictEqual(undo.records["a"].title, "the one", "the record is back");
  assert.strictEqual(undo.s.chats.length, 2, "the chat you were in is put back next to the new one");
  button.onclick();
  await wait();
  assert.strictEqual(undo.s.chats.length, 2, "the second Undo does not put it in twice");
  done.push("delete with Undo: the record comes back, twice does not double");

  // 17. a read transaction that aborts still answers, so the page does not hang
  const aborted = run({abort: "DataError"}, {"strata.history": "true"},
                      {a: {id: "a", title: "the one", time: Date.now(), messages: [msg("hi")]}});
  await wait();
  assert.strictEqual(rows(aborted.s), 1, "the read answered empty instead of never resolving");
  assert.ok(!aborted.s.chats.some((c) => c.id === "a"), "so the record is not in the list");
  assert.ok(aborted.s.chat, "and the page still rendered a chat");
  done.push("read that aborts: it answers empty instead of never resolving");

  // 18. a transaction that aborts for another reason does not say the storage is full
  const bad = run({abort: "DataError"}, {"strata.history": "true"});
  await wait();
  bad.s.messages.push(msg("a chat that cannot be stored"));
  bad.s.saveChat();
  await wait();
  assert.ok(warnText(bad.s).includes("could not be saved"), "the message fits the error");
  assert.ok(!warnText(bad.s).includes("cannot keep more"), "and not the quota message");
  done.push("abort for another reason: the message is not the quota one");

  // 19. a chat that is not in the list is not the last one
  const stray = run({}, {"strata.history": "true"}, {
    a: {id: "a", title: "first", time: Date.now(), messages: [msg("one")]},
    b: {id: "b", title: "second", time: Date.now(), messages: [msg("two")]}});
  await wait();
  stray.s.deleteChat({id: "z", title: "not in the list", time: Date.now(), messages: [msg("x")]});
  await wait();
  assert.strictEqual(stray.s.chats.length, 2, "nothing was removed");
  assert.ok("a" in stray.records && "b" in stray.records, "both records are still there");
  done.push("delete: a chat the list does not hold removes nothing");

  console.log(done.map((d) => "ok - " + d).join("\n"));
})();
