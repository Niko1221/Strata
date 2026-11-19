// The Python server owns identity and persistence. This page only holds consent
// and a transcript scoped to the currently entered profile key.
"use strict";

function createMemoryTranscripts(ordinary) {
  let normal = ordinary;
  let memory = [];
  let selected = false;
  let key = "";
  let version = 0;
  const freshSession = () => globalThis.crypto?.randomUUID?.() ?? null;
  let normalSession = freshSession(), memorySession = freshSession();
  return {
    active: () => selected ? memory : normal,
    version: () => version,
    cacheSession: () => selected ? memorySession : normalSession,
    resetCacheSession: () => { if (selected) memorySession = freshSession(); else normalSession = freshSession(); },
    select: enabled => {
      if (enabled !== selected) version++;
      selected = enabled;
      return selected ? memory : normal;
    },
    setKey: next => {
      if (next !== key) { memory = []; memorySession = freshSession(); version++; }
      key = next;
      return memory;
    },
    replaceActive: next => {
      if (selected) { memory = next; memorySession = freshSession(); }
      else { normal = next; normalSession = freshSession(); }
      version++;
      return next;
    },
  };
}

function persistApiKey(storage, key, memoryMode) {
  return memoryMode || storage.set("apikey", key);
}

function prepareMemoryRequest(base, {key, entered, text, attachments, read, storeSummary, crypto}) {
  if (!read && !storeSummary) throw new Error("Select a memory permission for this turn.");
  if (!key || !entered) throw new Error("Enter your own memory profile key for this visit.");
  if (attachments.length || !text || text.length > 16384) {
    throw new Error("Memory accepts text only, up to 16,384 characters.");
  }
  if (!crypto?.getRandomValues) throw new Error("This browser cannot create a secure memory turn ID.");
  const turnId = Array.from(crypto.getRandomValues(new Uint8Array(16)), b => b.toString(16).padStart(2, "0")).join("");
  const {strata_mcp: _unused, ...payload} = base;
  payload.strata_memory = {read: !!read, store_summary: !!storeSummary, turn_id: turnId};
  return payload;
}

const StrataMemoryChat = Object.freeze({createMemoryTranscripts, persistApiKey, prepareMemoryRequest});
if (typeof module === "object" && module.exports) module.exports = StrataMemoryChat;
else globalThis.StrataMemoryChat = StrataMemoryChat;
