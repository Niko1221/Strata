// Browser-only conversations. One atomic storage write keeps each transcript and its summary together.
(function (root) {
  "use strict";
  const KEY = "strata.conversations.v1";
  const clone = (value) => JSON.parse(JSON.stringify(value));
  const emptyMemory = () => ({through: 0, summary: "", count: 0});
  function title(messages) {
    const first = messages.find((m) => m.role === "user");
    const text = first && (first.text || first.files?.[0]?.name || first.images?.[0]?.name);
    return text ? Array.from(text.trim().replace(/\s+/g, " ")).slice(0, 60).join("") || "New chat" : "New chat";
  }
  function decode(value) {
    if (!value || value.version !== 1 || !Array.isArray(value.chats)) return null;
    const seen = new Set();
    const chats = value.chats.filter((c) => c && typeof c.id === "string" && !seen.has(c.id) &&
      Array.isArray(c.messages) && (seen.add(c.id), true));
    return chats.length ? {version: 1, active: chats.some((c) => c.id === value.active) ? value.active : chats[0].id, chats} : null;
  }
  function createStore(storage, options = {}) {
    const now = options.now || Date.now;
    const makeId = options.makeId || (() => root.crypto.randomUUID());
    function read(key, fallback) {
      try { const raw = storage.getItem(key); return raw === null ? fallback : JSON.parse(raw); }
      catch (e) { return fallback; }
    }
    function blank(messages = [], memory = emptyMemory()) {
      const time = now();
      return {id: makeId(), title: title(messages), createdAt: time, updatedAt: time, messages, memory, draft: ""};
    }
    let book = decode(read(KEY, null));
    if (!book) {
      const legacy = read("strata.chat", []);
      const chat = blank(Array.isArray(legacy) ? legacy : [], read("strata.compaction", null) || emptyMemory());
      const times = chat.messages.map((m) => m.time).filter(Number.isFinite);
      if (times.length) {
        chat.createdAt = times.reduce((a, b) => Math.min(a, b));
        chat.updatedAt = times.reduce((a, b) => Math.max(a, b));
      }
      book = {version: 1, active: chat.id, chats: [chat]};
    }
    // Merge new conversations saved by another tab, without replacing this tab's active conversation.
    function refresh() {
      const latest = decode(read(KEY, null));
      if (!latest) return;
      for (const chat of latest.chats) {
        const index = book.chats.findIndex((c) => c.id === chat.id);
        if (index < 0) book.chats.push(chat);
        else if (chat.id !== book.active && chat.updatedAt > book.chats[index].updatedAt) book.chats[index] = chat;
      }
    }
    function persist() {
      const saved = {...book, chats: book.chats.map((c) => ({...c, messages: c.messages.map((m) => ({...m,
        // Preserve text attachments; large image payloads stay in memory, as in the original chat.
        images: (m.images || []).map((i) => ({name: i.name})),
      }))}))};
      try { storage.setItem(KEY, JSON.stringify(saved)); return true; }
      catch (e) { return false; } // Keep the in-memory conversations and the last saved value intact.
    }
    function current() { return clone(book.chats.find((c) => c.id === book.active)); }
    function save(messages, memory, draft = "") {
      refresh();
      const c = book.chats.find((c) => c.id === book.active);
      if (JSON.stringify([c.messages, c.memory]) !== JSON.stringify([messages, memory])) c.updatedAt = now();
      Object.assign(c, {messages: clone(messages), memory: clone(memory), draft, title: title(messages)});
      return persist();
    }
    function start() {
      const c = blank();
      book.chats.push(c); book.active = c.id;
      return persist();
    }
    function select(id) {
      refresh();
      if (!book.chats.some((c) => c.id === id)) return false;
      book.active = id;
      persist();
      return true;
    }
    function list(query = "") {
      const needle = query.trim().toLocaleLowerCase();
      return book.chats.filter((c) => (c.id === book.active || c.messages.length || c.draft) && (!needle ||
        [c.title, c.memory?.summary, c.draft, ...c.messages.map((m) => m.text)].join("\n").toLocaleLowerCase().includes(needle)))
        .sort((a, b) => b.updatedAt - a.updatedAt);
    }
    function exportData() {
      return {...clone(book), chats: book.chats.map((c) => ({...clone(c), messages: c.messages.map((m) => ({...clone(m),
        images: (m.images || []).map((i) => ({name: i.name})),
      }))}))};
    }
    function importData(value) {
      const fingerprint = (c) => JSON.stringify([c.messages.map((m) => ({...m,
        images: (m.images || []).map((i) => ({name: i.name})),
      })), c.memory, c.draft || ""]);
      const validMessage = (m) => m && ["user", "assistant"].includes(m.role) && typeof m.text === "string" &&
        (m.reasoning == null || typeof m.reasoning === "string") &&
        (m.files == null || Array.isArray(m.files) && m.files.every((f) => f && typeof f.name === "string" &&
          (f.text == null || typeof f.text === "string"))) &&
        (m.images == null || Array.isArray(m.images) && m.images.every((i) => i &&
          (i.name == null || typeof i.name === "string"))) &&
        (m.tools == null || Array.isArray(m.tools) && m.tools.every((t) => t && typeof t === "object" &&
          (t.result == null || typeof t.result === "string")));
      if (!value || value.version !== 1 || !Array.isArray(value.chats) || !value.chats.length ||
          value.chats.some((c) => !c || !Array.isArray(c.messages) || c.messages.some((m) => !m ||
            !validMessage(m)))) {
        throw new Error("Choose a Strata history JSON file.");
      }
      refresh();
      let added = 0;
      for (const source of value.chats) {
        const c = blank(clone(source.messages), clone(source.memory || emptyMemory()));
        c.messages.forEach((m) => { m.images = (Array.isArray(m.images) ? m.images : []).map((i) => ({name: i.name})); });
        c.draft = typeof source.draft === "string" ? source.draft : "";
        if (book.chats.some((existing) => fingerprint(existing) === fingerprint(c))) continue;
        if (Number.isFinite(source.createdAt)) c.createdAt = source.createdAt;
        if (Number.isFinite(source.updatedAt)) c.updatedAt = source.updatedAt;
        book.chats.push(c); added++;
      }
      return {added, saved: persist()};
    }
    return {current, save, start, select, list, refresh, persist, exportData, importData};
  }
  const api = {createStore, title, KEY};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.StrataHistory = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
