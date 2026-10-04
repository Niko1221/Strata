// Strata owns its archive. Import only copies; legacy databases are never modified.
(function (root) {
  "use strict";
  const clone = value => JSON.parse(JSON.stringify(value));
  function validate(session) {
    if (!session || typeof session.id !== "string" || !session.id || typeof session.title !== "string" ||
        !Array.isArray(session.branches) || !session.branches.length) throw new Error("Invalid chat archive.");
    const ids = new Set();
    for (const branch of session.branches) {
      if (!branch || typeof branch.id !== "string" || ids.has(branch.id) || !Array.isArray(branch.messages)) throw new Error("Invalid chat branches.");
      ids.add(branch.id);
      for (const m of branch.messages) {
        if (!m || !["user", "assistant", "system", "tool", "developer"].includes(m.role) || typeof m.text !== "string") throw new Error("Invalid archived message.");
        if (m.wire != null && (!Array.isArray(m.wire) || m.wire.some(w => !w || !["user", "assistant", "system", "tool", "developer"].includes(w.role)))) throw new Error("Invalid archived model messages.");
      }
      const context = branch.context;
      if (context != null && (typeof context.summary !== "string" || !context.summary || !Number.isInteger(context.through) || context.through < 0 || context.through > branch.messages.length)) throw new Error("Invalid archived context.");
    }
    if (!ids.has(session.activeBranchId)) throw new Error("The selected branch is missing from the archive.");
    return clone(session);
  }
  function fresh(id, messages = [], context = null) {
    const first = messages.find(m => m.role === "user" && m.text);
    return {id, title: first ? first.text.slice(0, 64) : "New chat", updated: Date.now(), activeBranchId: "main",
      branches: [{id: "main", title: "Main", messages: clone(messages), context: clone(context)}]};
  }
  function backup(sessions) {
    return {format: "strata-chat-archive", version: 1, conversations: sessions.map(validate)};
  }
  function fromBackup(value) {
    if (!value || value.format !== "strata-chat-archive" || value.version !== 1 || !Array.isArray(value.conversations)) throw new Error("Unsupported Strata backup format.");
    return value.conversations.map(validate);
  }
  function request(req) {
    return new Promise((resolve, reject) => { req.onsuccess = () => resolve(req.result); req.onerror = () => reject(req.error); });
  }
  function finished(tx) {
    return new Promise((resolve, reject) => { tx.oncomplete = resolve; tx.onerror = tx.onabort = () => reject(tx.error || new Error("Chat storage transaction failed.")); });
  }
  async function open(transport) {
    if (typeof transport !== "function") throw new Error("Chat archive service is unavailable.");
    await transport("v1/chats");
    return {
      readAll: () => transport("v1/chats"),
      save: (session, activate = false) => transport("v1/chats/save", {session: validate(session), activate}),
      seedLegacy: (messages, context) => transport("v1/chats/seed", {messages, context}),
      add: sessions => transport("v1/chats/import", {sessions: sessions.map(validate)}),
      close() {},
    };
  }
  async function legacy(factory = root.indexedDB) {
    if (!factory) throw new Error("This browser does not provide legacy chat storage. Import an exported JSON or JSONL file instead.");
    const known = ["LlamaUi", "LlamacppWebui"];
    const names = factory.databases ? new Set((await factory.databases()).map(x => x.name)) : null;
    const result = [];
    for (const name of known) {
      if (names && !names.has(name)) continue;
      const req = factory.open(name);
      let absent = false;
      req.onupgradeneeded = () => { absent = true; req.transaction.abort(); };
      let db;
      try { db = await request(req); } catch (error) { if (absent) continue; throw error; }
      try {
        if (!db.objectStoreNames.contains("conversations") || !db.objectStoreNames.contains("messages")) throw new Error(`${name}: unsupported chat database. Use file export instead.`);
        const tx = db.transaction(["conversations", "messages"], "readonly"), done = finished(tx);
        const [conversations, messages] = await Promise.all([request(tx.objectStore("conversations").getAll()), request(tx.objectStore("messages").getAll())]);
        await done;
        for (const conv of conversations) result.push({source: name, data: {conv, messages: messages.filter(m => m.convId === conv.id)}});
      } finally { db.close(); }
    }
    return result;
  }
  const api = {validate, fresh, backup, fromBackup, open, legacy};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.StrataChatLibrary = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
