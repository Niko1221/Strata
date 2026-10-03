const test = require("node:test");
const assert = require("node:assert/strict");
const {createStore, KEY} = require("./history.js");
const empty = {through: 0, summary: "", count: 0};
function fixture(seed = {}) {
  const data = new Map(Object.entries(seed).map(([key, value]) => [key, JSON.stringify(value)]));
  let id = 0, time = 100;
  const storage = {getItem: (key) => data.get(key) ?? null, setItem: (key, value) => data.set(key, value)};
  const options = {makeId: () => `chat-${++id}`, now: () => ++time};
  return {storage, options, data, open: () => createStore(storage, options)};
}
const question = (text, time = 20) => ({role: "user", text, time});
test("imports the existing conversation and its summary without deleting the original", () => {
  const messages = [question("旧チャット"), {role: "assistant", text: "回答", time: 30}];
  const memory = {through: 2, summary: "以前の要約", count: 1};
  const f = fixture({"strata.chat": messages, "strata.compaction": memory}), store = f.open();
  assert.deepEqual(store.current().messages, messages);
  assert.deepEqual(store.current().memory, memory);
  assert.equal(store.current().createdAt, 20);
  assert.equal(store.current().updatedAt, 30);
  assert.equal(store.persist(), true);
  assert.equal(f.open().list().length, 1);
  assert.deepEqual(f.open().current().memory, memory);
  assert.deepEqual(JSON.parse(f.data.get("strata.chat")), messages);
});
test("new chats preserve transcripts, summaries and drafts independently across reloads", () => {
  const f = fixture(), store = f.open(), first = store.current().id;
  const summary = {through: 1, summary: "Auroraの要約", count: 2};
  store.save([question("Auroraの予定")], summary, "未送信の入力");
  store.start(); const second = store.current().id;
  store.save([question("別のプロジェクト")], empty);
  const reloaded = f.open();
  assert.equal(reloaded.current().id, second);
  assert.equal(reloaded.list().length, 2);
  assert.equal(reloaded.select(first), true);
  assert.deepEqual(reloaded.current().memory, summary);
  assert.equal(reloaded.current().draft, "未送信の入力");
  assert.equal(reloaded.current().messages[0].text, "Auroraの予定");
  reloaded.select(second);
  assert.deepEqual(reloaded.current().memory, empty);
  assert.equal(reloaded.current().draft, "");
});
test("search finds older messages and viewing does not reorder the list", () => {
  const store = fixture().open(), first = store.current().id;
  store.save([question("First"), {role: "assistant", text: "uniquely searchable response", time: 30}], empty);
  store.start(); const second = store.current().id;
  store.save([question("Second")], empty);
  store.select(first); const c = store.current();
  store.save(c.messages, c.memory);
  assert.equal(store.list()[0].id, second);
  assert.equal(store.list("SEARCHABLE")[0].id, first);
  assert.deepEqual(store.list("no match"), []);
});
test("quota failure leaves the last saved value intact and retains new chats in memory", () => {
  const f = fixture(), store = f.open(), first = store.current().id;
  store.save([question("saved")], empty);
  const saved = f.data.get(KEY);
  f.storage.setItem = () => { throw new Error("QuotaExceededError"); };
  assert.equal(store.start(), false);
  const second = store.current().id;
  assert.equal(store.save([question("unsaved")], empty), false);
  assert.equal(f.data.get(KEY), saved);
  assert.equal(store.select(first), true);
  assert.equal(store.current().messages[0].text, "saved");
  store.select(second);
  assert.equal(store.current().messages[0].text, "unsaved");
});
test("another tab's new conversation survives writes from an older tab", () => {
  const f = fixture(), first = f.open();
  first.save([question("First tab")], empty);
  const second = f.open(); second.start();
  second.save([question("Second tab")], empty);
  first.save([question("Updated first tab")], empty);
  assert.deepEqual(f.open().list().map((c) => c.title).sort(), ["Second tab", "Updated first tab"]);
});
test("text files survive reload; images remain available when switching in the same page", () => {
  const f = fixture(), store = f.open(), first = store.current().id;
  const message = {...question("attachments"), files: [{name: "notes.txt", text: "full file text"}],
    images: [{name: "image.png", url: "data:image/png;base64,example"}]};
  store.save([message], empty); store.start(); store.select(first);
  assert.equal(store.current().messages[0].images[0].url, message.images[0].url);
  assert.equal(f.open().current().messages[0].files[0].text, "full file text");
  assert.equal(f.open().current().messages[0].images[0].url, undefined);
});
test("a JSON backup restores conversations without replacing or duplicating existing chats", () => {
  const source = fixture().open();
  source.save([question("Backed up chat")], {through: 1, summary: "Saved summary", count: 1}, "draft");
  const target = fixture().open(); target.save([question("Existing chat")], empty);
  const result = target.importData(source.exportData());
  assert.equal(result.added, 1); assert.equal(result.saved, true);
  assert.equal(target.list().length, 2);
  assert.equal(target.current().messages[0].text, "Existing chat");
  assert.equal(target.list("Backed up")[0].memory.summary, "Saved summary");
  assert.equal(target.importData(source.exportData()).added, 0);
  assert.equal(target.importData(target.exportData()).added, 0);
});
test("invalid backup contents are rejected before adding any conversations", () => {
  const f = fixture(), store = f.open(); store.save([question("Existing")], empty);
  const before = f.data.get(KEY);
  assert.throws(() => store.importData({version: 1, chats: [
    {messages: [question("valid")]}, {messages: [{role: "user", text: "invalid", files: {}}]},
  ]}));
  assert.equal(store.list().length, 1);
  assert.equal(f.data.get(KEY), before);
});
