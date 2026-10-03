// Selected-chat windows on the browser, gzip records on the PC. v1 is migration-only.
(function (root) {
  "use strict";
  const ACTIVE = "strata.history.active.v2", PENDING = "strata.history.pending.v2.", SNAPSHOT = "strata.history.draft.v2.";
  const clone = (v) => JSON.parse(JSON.stringify(v));
  const emptyMemory = () => ({through: 0, summary: "", count: 0});
  const metadata = (c) => ({id: c.id, title: c.title, createdAt: c.createdAt, updatedAt: c.updatedAt,
    revision: c.revision, messageCount: c.messageCount, empty: c.empty});

  function createStore(options) {
    const storage = options.storage, makeId = options.makeId || (() => root.crypto.randomUUID());
    const request = options.request;
    let active = null, rows = [], catalogTotal = 0, hasMore = false, query = "", queue = Promise.resolve(), catalogRevision = 0;
    function read(key) { try { return storage.getItem(key); } catch (e) { return null; } }
    function write(key, value) { try { storage.setItem(key, value); return true; } catch (e) { return false; } }
    function remove(key) { try { storage.removeItem(key); } catch (e) { /* The disk copy is already committed. */ } }
    function updateRow(c) {
      const index = rows.findIndex((r) => r.id === c.id);
      if (index >= 0) rows[index] = metadata(c);
      else if (!query) { rows.push(metadata(c)); catalogTotal++; }
      rows.sort((a, b) => b.updatedAt - a.updatedAt);
    }
    async function refresh(nextQuery = query, more = false) {
      const ticket = ++catalogRevision;
      const result = await request(`api/history?q=${encodeURIComponent(nextQuery)}&offset=${more ? rows.length : 0}&limit=50`);
      if (ticket !== catalogRevision) return list();
      query = nextQuery;
      rows = more ? [...rows, ...result.chats.filter((c) => !rows.some((r) => r.id === c.id))] : result.chats;
      catalogTotal = result.total; hasMore = result.hasMore;
      return list();
    }
    function list() { return {chats: rows.map(metadata), total: catalogTotal, hasMore}; }
    async function recover(id, restoreSnapshot = false) {
      const raw = read(PENDING + id);
      let recovered = null;
      if (raw) {
        let pending;
        try { pending = JSON.parse(raw); } catch (e) { throw new Error("Could not read the unsaved draft. The browser data has been preserved."); }
        try {
          const result = await request("api/history/chat", pending.body);
          remove(PENDING + id);
          recovered = {...pending.snapshot, ...result.chat};
        } catch (e) { if (!restoreSnapshot || e.status !== 409 || !read(SNAPSHOT + id)) throw e; }
      }
      if (restoreSnapshot && read(SNAPSHOT + id)) {
        const snapshot = JSON.parse(read(SNAPSHOT + id));
        const result = await request("api/history/recover", snapshot);
        remove(SNAPSHOT + id); remove(PENDING + id);
        recovered = {...snapshot, ...result.chat, offset: result.offset, recoveredSeparately: result.recoveredSeparately};
      }
      return recovered;
    }
    async function select(id) {
      await flush();
      const recovered = await recover(id, true);
      if (recovered) id = recovered.id;
      const page = await request(`api/history/chat?id=${encodeURIComponent(id)}&limit=40`);
      active = {...page, recoveredSeparately: recovered?.recoveredSeparately, baseline: clone(page)};
      write(ACTIVE, id);
      return current();
    }
    function current() { return active ? clone({...active, baseline: undefined}) : null; }
    async function start() {
      await flush();
      const id = makeId();
      await request("api/history/chat", {id, revision: 0, writeId: makeId(), total: 0, start: 0, messages: [],
        memory: emptyMemory(), draft: "", draftAttachments: []});
      await select(id);
      updateRow(active);
      return current();
    }
    async function init() {
      let migrated = null;
      const legacy = options.legacy;
      const legacyRaw = legacy && (read(legacy.KEY) || read("strata.chat"));
      if (legacyRaw) {
        let book;
        try { book = JSON.parse(read(legacy.KEY) || "null"); } catch (e) { throw new Error("Could not read the old history. The original browser data has been preserved."); }
        if (!book) book = legacy.createStore(storage).exportData();
        // Preserve obsolete keys inside the durable backup before retiring their browser copies.
        const raw = JSON.stringify({...book, legacyStorage: {chat: read("strata.chat"), compaction: read("strata.compaction")}});
        const result = await request("api/history/migrate", raw);
        const hash = options.hash || (async (text) => Array.from(new Uint8Array(await root.crypto.subtle.digest("SHA-256", new TextEncoder().encode(text))))
          .map((n) => n.toString(16).padStart(2, "0")).join(""));
        if (!result.saved || result.source_sha256 !== await hash(raw) || result.source_count !== book.chats.length ||
            result.mapped_count !== book.chats.length || !result.backup) throw new Error("Could not verify history migration. The original browser data has been preserved.");
        migrated = result.active;
        // Verify the chosen disk transcript is readable before deleting the legacy browser copies.
        if (migrated) await request(`api/history/chat?id=${encodeURIComponent(migrated)}&limit=1`);
        if (migrated) write(ACTIVE, migrated);
        remove(legacy.KEY); remove("strata.chat"); remove("strata.compaction");
      }
      await refresh("");
      const id = migrated || read(ACTIVE) || rows[0]?.id;
      if (id) {
        try { return await select(id); }
        catch (e) { if (e.status !== 404) throw e; }
      }
      return rows.length ? select(rows[0].id) : start();
    }
    function save(messages, memory, draft, draftAttachments = []) {
      if (!active) return Promise.reject(new Error("History is still loading."));
      const target = active;
      const snapshot = {offset: target.offset, messages: clone(messages), memory: clone(memory), draft, draftAttachments: clone(draftAttachments)};
      const snapshotKey = SNAPSHOT + target.id;
      const snapshotRaw = JSON.stringify({id: target.id, revision: target.revision, recoveryId: makeId(), ...snapshot});
      // Stage synchronously, including the last keystrokes before pagehide.
      write(snapshotKey, snapshotRaw);
      const run = async () => {
        const recovered = await recover(target.id);
        if (recovered) { target.baseline = recovered; target.revision = recovered.revision; }
        const before = target.baseline;
        let changed = 0;
        while (changed < Math.min(before.messages.length, snapshot.messages.length) &&
          JSON.stringify(before.messages[changed]) === JSON.stringify(snapshot.messages[changed])) changed++;
        const sameMessages = changed === before.messages.length && changed === snapshot.messages.length;
        if (sameMessages && JSON.stringify([before.memory, before.draft, before.draftAttachments || []]) ===
            JSON.stringify([snapshot.memory, snapshot.draft, snapshot.draftAttachments])) {
          if (read(snapshotKey) === snapshotRaw) remove(snapshotKey);
          return true;
        }
        const body = {id: target.id, revision: target.revision, writeId: makeId(), total: snapshot.offset + snapshot.messages.length,
          memory: snapshot.memory, draft: snapshot.draft, draftAttachments: snapshot.draftAttachments};
        if (!sameMessages) { body.start = snapshot.offset + changed; body.messages = snapshot.messages.slice(changed); }
        // A single dirty chat can be recovered after an interrupted save. No all-chat browser copy.
        const durable = write(PENDING + target.id, JSON.stringify({body, snapshot}));
        let result;
        try { result = await request("api/history/chat", body); }
        catch (e) { e.pendingSaved = durable; throw e; }
        target.baseline = clone(snapshot);
        Object.assign(target, snapshot, result.chat);
        remove(PENDING + target.id);
        if (read(snapshotKey) === snapshotRaw) remove(snapshotKey);
        updateRow(target);
        return true;
      };
      const pending = queue.then(run);
      queue = pending.catch(() => {});
      return pending;
    }
    async function flush() { await queue; }
    async function older() {
      await flush();
      if (!active || active.offset === 0) return current();
      const before = active.offset;
      const page = await request(`api/history/chat?id=${encodeURIComponent(active.id)}&end=${before}&limit=40`);
      if (page.revision !== active.revision) throw new Error("Another window updated this conversation. Save your changes, then reopen it.");
      active.messages = [...page.messages, ...active.baseline.messages]; active.offset = page.offset;
      active.baseline = clone({...active, baseline: undefined});
      return current();
    }
    async function page(start, limit = 20) {
      await flush();
      return request(`api/history/messages?id=${encodeURIComponent(active.id)}&start=${start}&limit=${limit}`);
    }
    function trim() {
      if (!active) return current();
      const records = active.baseline.messages;
      let keep = 0, bytes = 0;
      for (let i = records.length - 1; i >= 0 && keep < 40; i--) {
        const size = new TextEncoder().encode(JSON.stringify(records[i])).length;
        if (keep && bytes + size > 4 * 1024 * 1024) break;
        bytes += size; keep++;
      }
      const drop = records.length - keep;
      if (!drop) return current();
      active.messages = active.baseline.messages.slice(drop); active.offset += drop;
      active.baseline = clone({...active, baseline: undefined});
      return current();
    }
    async function recall(text, through, signal) {
      return request(`api/history/recall?q=${encodeURIComponent(text.slice(0, 4096))}&id=${encodeURIComponent(active.id)}&through=${through}`, undefined, {signal});
    }
    async function importData(raw, gzip = false) {
      await flush();
      const result = await request("api/history/import", raw, {gzip});
      await refresh("");
      return result;
    }
    return {init, current, start, select, save, flush, refresh, list, older, page, trim, recall, importData,
      info: () => active ? metadata(active) : null,
      stats: () => request("api/history/stats")};
  }
  const api = {createStore, ACTIVE, PENDING, SNAPSHOT};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.StrataDiskHistory = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
