/**
 * Helper agents (the Harness's subagents) as the rows the chat already has.
 *
 * A helper is a child session of the chat's Harness session. This module is the pure part: it folds one child's
 * `session/follow` items into
 *   - an `agent` frame (the shape `apps/host/src/lib/agents.ts updateAgent` merges into the chat's existing rows:
 *     id, childSessionId, name, description, status, durationMs, tokens, updatedAt), and
 *   - tool cards, but only for a helper's request that needs the person (an approval). Everything else a helper does
 *     stays in its row; the transcript is not filled with the work of four helpers at once.
 *
 * A helper's approval card is the same card a direct request gets (state `proposed`, answered with the same
 * Approve/Deny route). Its id is `<childSessionId>:<callId>`, which is also the key the bridge keeps the pending
 * question under, so the answer finds the helper that asked and no other chat can answer it.
 *
 * Wording is for the person reading the row, not for a log: no session ids, no tool internals.
 */
import { closeSync, existsSync, fsyncSync, mkdirSync, openSync, readFileSync, rmSync, writeSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { atomicWriteFileSync } from '../atomic-file.mjs';
import { friendlyFailure, parseArguments, presentTool, resultText, toolScope, toolVerb } from './events.mjs';

const clip = (value, max) => {
  const text = String(value ?? '').replace(/\s+/g, ' ').trim();
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
};

/**
 * One bounded transcript per helper. The interface shows these entries, and a tool call is one
 * readable row, never raw JSON. The same entries are saved beside the helper rows, so a restart
 * keeps the conversation instead of leaving only a summary.
 */
export const MAX_TRANSCRIPT_ENTRIES = 500;
export const MAX_TRANSCRIPT_CHARS = 512 * 1024;
export const MAX_TRANSCRIPT_ENTRY_CHARS = 64 * 1024;
/** How many finished helpers keep a transcript in this host process (older ones are dropped first). */
export const MAX_TRANSCRIPTS = 100;
/** A JSONL file is rewritten once it grows past this; old lines are not kept forever. */
const MAX_STORED_BYTES = 2 * 1024 * 1024;
const STORE_VERSION = 1;
const STORE_FLUSH_MS = 250;
export const TASK_MISSING = 'The main assistant did not include a task description.';
export const SENT_NOTICE = 'Sent the result to the main assistant.';
export const CONTINUE_NOTICE = 'Continuing the task.';
const HIDDEN_TOOLS = new Set(['send_message', 'interrupt_agent', 'list_agents']);

const messageText = content => (Array.isArray(content) ? content : [])
  .filter(part => part?.type === 'text')
  .map(part => part?.text ?? '')
  .join('')
  .trim();

/** The helper's conversation, not the Harness's own context snapshots (time, skills, working folder, notices). */
const conversationUser = data => {
  const source = data?.source ?? {};
  if (source.form) return false;
  return !['working-directory', 'session-reference', 'user-approval', 'user-question-reply'].includes(source.kind);
};

const looksLikeJson = value => /^\s*[\[{]/.test(String(value ?? ''));
const safeTarget = value => {
  const text = String(value ?? '').trim();
  return !text || looksLikeJson(text) ? '' : clip(text, 240);
};

const toolLine = card => {
  const target = safeTarget(card.target ?? card.preview);
  return clip(`${card.verb || 'Used a tool'}${target ? `: ${target}` : ''}`, 240);
};

const safeName = value => String(value ?? '').replace(/[^A-Za-z0-9._-]/g, '_').slice(0, 120) || 'unknown';

const boundedArgs = args => {
  try {
    const text = JSON.stringify(args ?? {});
    if (text.length <= 8_000) return args ?? {};
    return { truncated: true, preview: `${text.slice(0, 7_999)}…` };
  } catch {
    return {};
  }
};

const normalizeEntry = raw => {
  if (!raw || typeof raw !== 'object') return null;
  const role = ['assistant', 'user', 'tool', 'system'].includes(raw.role) ? raw.role : 'user';
  const kind = ['message', 'task', 'tool', 'notice'].includes(raw.kind) ? raw.kind : role === 'tool' ? 'tool' : role === 'system' ? 'notice' : 'message';
  const author = ['assistant', 'user', 'parent', 'system', 'tool'].includes(raw.author) ? raw.author : role === 'assistant' ? 'assistant' : role === 'tool' ? 'tool' : role === 'system' ? 'system' : 'user';
  const text = String(raw.text ?? '');
  const at = Number.isFinite(raw.at) ? raw.at : 0;
  const id = String(raw.id ?? '').trim() || `e${at.toString(36)}`;
  const entry = { id, role, kind, author, text, at };
  if (raw.open === true) entry.open = true;
  if (raw.partial === true) entry.partial = true;
  if (typeof raw.sourceId === 'string' && raw.sourceId) entry.sourceId = raw.sourceId;
  if (raw.tool && typeof raw.tool === 'object') entry.tool = raw.tool;
  return entry;
};

/**
 * The conversation of one helper, bounded in both directions. Entries carry their author and
 * kind so the page can tell the main assistant, the person, the helper and a tool apart.
 * Streaming text grows the open assistant entry, and the committed message replaces it so a
 * socket gap neither loses nor doubles words.
 */
export class ChildTranscript {
  constructor({ maxEntries = MAX_TRANSCRIPT_ENTRIES, maxChars = MAX_TRANSCRIPT_CHARS, now = Date.now, onChange = null } = {}) {
    this.maxEntries = maxEntries;
    this.maxChars = maxChars;
    this.now = now;
    this.onChange = onChange;
    this.entries = [];
    this.chars = 0;
    this.counter = 0;
    this.revision = 0;
    this.taskEntry = null;
    /** Highest session event sequence this transcript consumed, so a replayed snapshot is not written twice. */
    this.lastSeq = -1;
  }

  get hasTask() {
    return this.taskEntry !== null;
  }

  #touch() {
    this.revision += 1;
    this.onChange?.(this);
  }

  #entryMax() {
    return Math.max(1, Math.min(MAX_TRANSCRIPT_ENTRY_CHARS, this.maxChars));
  }

  #removeAt(index) {
    const [removed] = this.entries.splice(index, 1);
    if (!removed) return;
    this.chars -= removed.text.length;
    if (removed === this.taskEntry) this.taskEntry = null;
  }

  #trim() {
    while (this.entries.length > this.maxEntries) {
      // Keep the task card: it is the one line that explains what this conversation is.
      this.#removeAt(this.entries[0]?.kind === 'task' && this.entries.length > 1 ? 1 : 0);
    }
    while (this.chars > this.maxChars && this.entries.length > 1) {
      this.#removeAt(this.entries[0]?.kind === 'task' && this.entries.length > 1 ? 1 : 0);
    }
    if (this.chars > this.maxChars && this.entries.length === 1) {
      const entry = this.entries[0];
      const text = entry.text.slice(0, this.maxChars);
      this.chars = text.length;
      entry.text = text;
    }
  }

  #nextId() {
    this.counter += 1;
    return `e${this.counter.toString(36)}`;
  }

  #make({ role, kind, author, text, at, ...extra }) {
    const max = this.#entryMax();
    let value = String(text ?? '');
    if (value.length > max) value = `${value.slice(0, max - 1)}…`;
    return {
      id: this.#nextId(),
      role,
      kind,
      author,
      text: value,
      at: Number.isFinite(at) ? at : this.now(),
      ...extra,
    };
  }

  #append(data) {
    if (!String(data.text ?? '')) return null;
    const entry = this.#make(data);
    this.entries.push(entry);
    this.chars += entry.text.length;
    this.#trim();
    this.#touch();
    return this.entries.includes(entry) ? entry : null;
  }

  #setText(entry, value) {
    if (!entry || !this.entries.includes(entry)) return;
    let text = String(value ?? '');
    const max = this.#entryMax();
    if (text.length > max) text = `${text.slice(0, max - 1)}…`;
    this.chars += text.length - entry.text.length;
    entry.text = text;
    this.#trim();
    this.#touch();
  }

  #closeOpen(at) {
    const last = this.entries[this.entries.length - 1];
    if (!last || last.role !== 'assistant' || !last.open) return;
    last.open = false;
    last.at = Number.isFinite(at) ? at : last.at;
    if (!last.text.trim()) this.#removeAt(this.entries.length - 1);
    this.#touch();
  }

  /** The task the main assistant handed over. Replaces the raw spawn prompt, which never enters the timeline. */
  task(value, at = this.now()) {
    const text = String(value ?? '').trim() || TASK_MISSING;
    if (this.taskEntry) {
      if (this.taskEntry.text !== text) this.#setText(this.taskEntry, text);
      this.taskEntry.at = Number.isFinite(at) ? at : this.taskEntry.at;
      return this.taskEntry;
    }
    const entry = this.#make({ role: 'user', kind: 'task', author: 'parent', text, at });
    this.entries.unshift(entry);
    this.chars += entry.text.length;
    this.taskEntry = entry;
    this.#trim();
    this.#touch();
    return entry;
  }

  /** Text that arrives while the helper is writing: one growing assistant entry. */
  stream(value, at = this.now()) {
    const text = String(value ?? '');
    if (!text) return;
    const last = this.entries[this.entries.length - 1];
    if (last?.role === 'assistant' && last.open) {
      last.at = Number.isFinite(at) ? at : last.at;
      this.#setText(last, last.text + text);
      return;
    }
    this.#append({ role: 'assistant', kind: 'message', author: 'assistant', text, at, open: true });
  }

  /** The helper's committed assistant message: its own words, replacing what the live stream carried. */
  message(value, at = this.now()) {
    const text = String(value ?? '').trim();
    const openAt = this.entries.findLastIndex(entry => entry.role === 'assistant' && entry.open);
    if (openAt >= 0) {
      const open = this.entries[openAt];
      open.open = false;
      open.at = Number.isFinite(at) ? at : open.at;
      if (text) this.#setText(open, text);
      else this.#removeAt(openAt);
      this.#touch();
      return;
    }
    const last = this.entries[this.entries.length - 1];
    // The committed message can arrive twice when two readers follow the helper (the parent turn's
    // stream and the panel watcher). The same words within a few seconds are one message, not two.
    if (text && last?.role === 'assistant' && last.text === text && Math.abs((Number.isFinite(at) ? at : 0) - last.at) < 10_000) {
      last.at = Number.isFinite(at) ? at : last.at;
      last.partial = false;
      this.#touch();
      return;
    }
    if (text) this.#append({ role: 'assistant', kind: 'message', author: 'assistant', text, at });
  }

  /** A message the person sent directly to this helper. */
  user(value, at = this.now(), sourceId = null) {
    const text = String(value ?? '').trim();
    if (!text) return null;
    // A message sent while the helper is still writing does not end the stream: the committed
    // answer for that stream can arrive after this line and must replace the growing entry above it.
    const last = this.entries[this.entries.length - 1];
    // The host writes the sent line before the Harness echoes it back: keep one line.
    if (last?.role === 'user' && last.echo && last.text === text) {
      last.echo = false;
      last.at = Number.isFinite(at) ? at : last.at;
      if (sourceId) last.sourceId = sourceId;
      this.#touch();
      return last;
    }
    return this.#append({ role: 'user', kind: 'message', author: 'user', text, at, ...(sourceId ? { sourceId } : {}) });
  }

  /** A tool call with a readable line; raw arguments stay inside `tool` for the expanded view. */
  tool(card, at = this.now()) {
    this.#closeOpen(at);
    const source = typeof card === 'string' ? { verb: card } : (card ?? {});
    const tool = {
      ...source,
      target: safeTarget(source.target ?? source.preview),
      preview: safeTarget(source.preview),
      state: source.state || 'running',
      args: boundedArgs(source.args),
    };
    return this.#append({ role: 'tool', kind: 'tool', author: 'tool', text: toolLine(tool), at, tool });
  }

  /** Update the existing tool row when its result or approval state arrives. */
  updateTool(entryOrId, patch = {}) {
    const entry = typeof entryOrId === 'string'
      ? this.entries.find(item => item.id === entryOrId)
      : entryOrId;
    if (!entry?.tool) return null;
    const tool = {
      ...entry.tool,
      ...patch,
      target: safeTarget(patch.target ?? entry.tool.target),
      preview: safeTarget(patch.preview ?? entry.tool.preview),
    };
    entry.tool = tool;
    entry.text = toolLine(tool);
    this.#touch();
    return entry;
  }

  /** A tool that was still running when the turn ended has no result, and must not spin forever. */
  finishTools(at = this.now()) {
    let changed = false;
    for (const entry of this.entries) {
      if (!entry.tool || (entry.tool.state !== 'running' && entry.tool.state !== 'proposed')) continue;
      entry.tool = {
        ...entry.tool,
        state: 'interrupted',
        verb: entry.tool.name ? toolVerb(entry.tool.name, 'running') : entry.tool.verb,
        result: entry.tool.result || 'Stopped before it finished.',
        durationMs: Number.isFinite(entry.tool.durationMs) ? entry.tool.durationMs : Math.max(0, at - entry.at),
      };
      entry.text = toolLine(entry.tool);
      changed = true;
    }
    if (changed) this.#touch();
  }

  /** A short line from the host, such as "Sent the result to the main assistant." */
  notice(text, at = this.now()) {
    const value = String(text ?? '').trim();
    if (!value) return null;
    const last = this.entries[this.entries.length - 1];
    if (last?.role === 'system' && last.text === value && Math.abs((Number.isFinite(at) ? at : this.now()) - last.at) < 2000) return last;
    return this.#append({ role: 'system', kind: 'notice', author: 'system', text: value, at });
  }

  /** A line the host wrote for a message it just sent; the Harness echo clears the mark. */
  beginUser(value, at = this.now(), sourceId = null) {
    return this.#append({
      role: 'user', kind: 'message', author: 'user', text: String(value ?? '').trim(), at,
      echo: true, ...(sourceId ? { sourceId } : {}),
    });
  }

  cancelUser(entry) {
    const at = this.entries.indexOf(entry);
    if (at >= 0) {
      this.#removeAt(at);
      this.#touch();
    }
  }

  close(at = this.now()) {
    this.#closeOpen(at);
  }

  setLastSeq(seq) {
    if (!Number.isInteger(seq) || seq <= this.lastSeq) return false;
    this.lastSeq = seq;
    this.#touch();
    return true;
  }

  /** Load saved entries. Open text becomes partial, and running tools become stopped. */
  load(entries, seq = -1) {
    const seen = new Set();
    const loaded = [];
    for (const raw of Array.isArray(entries) ? entries : []) {
      const entry = normalizeEntry(raw);
      if (!entry || seen.has(entry.id)) continue;
      seen.add(entry.id);
      if (entry.open) {
        entry.open = false;
        entry.partial = true;
      }
      if (entry.tool && (entry.tool.state === 'running' || entry.tool.state === 'proposed')) {
        entry.tool = {
          ...entry.tool,
          state: 'interrupted',
          verb: entry.tool.name ? toolVerb(entry.tool.name, 'running') : entry.tool.verb,
          result: entry.tool.result || 'Stopped before it finished.',
        };
      } else if (entry.tool?.name && (entry.tool.state === 'done' || entry.tool.state === 'error' || entry.tool.state === 'denied')) {
        // Older saved rows kept the "Reading…" wording after they finished; show the past tense on load too.
        entry.tool = { ...entry.tool, verb: toolVerb(entry.tool.name, entry.tool.state === 'done' ? 'done' : 'error') };
        entry.text = toolLine(entry.tool);
      }
      loaded.push(entry);
    }
    const taskAt = loaded.findIndex(entry => entry.kind === 'task');
    if (taskAt > 0) loaded.unshift(...loaded.splice(taskAt, 1));
    // Older stores can hold the streamed copy and the committed copy of one answer. Merge them on
    // load so reopening a saved helper shows one message, not the same words twice.
    const merged = [];
    for (const entry of loaded) {
      const previous = merged[merged.length - 1];
      const duplicate = entry.role === 'assistant' && previous?.role === 'assistant'
        && !entry.tool && !previous.tool && entry.text === previous.text
        && Math.abs(entry.at - previous.at) < 10_000;
      if (duplicate) {
        previous.at = Math.max(previous.at, entry.at);
        previous.partial = previous.partial === true || entry.partial === true;
        continue;
      }
      merged.push(entry);
    }
    this.entries = merged;
    this.chars = merged.reduce((total, entry) => total + entry.text.length, 0);
    this.lastSeq = Number.isInteger(seq) ? seq : -1;
    this.counter = merged.reduce((max, entry) => {
      const match = /^e([0-9a-z]+)$/.exec(entry.id);
      return match ? Math.max(max, parseInt(match[1], 36) || 0) : max;
    }, 0);
    this.taskEntry = merged.find(entry => entry.kind === 'task') ?? null;
    this.revision = 1;
    this.#trim();
    return this;
  }

  /** Backward-compatible three-field view for older callers. */
  view() {
    return this.entries.map(({ role, text, at }) => ({ role, text, at }));
  }

  /** What the page reads: authors, kinds and structured tool rows, without internal bookkeeping. */
  viewRich() {
    return this.entries.map(entry => ({
      id: entry.id,
      role: entry.role,
      kind: entry.kind,
      author: entry.author,
      text: entry.text,
      at: entry.at,
      ...(entry.open ? { open: true } : {}),
      ...(entry.partial ? { partial: true } : {}),
      ...(entry.sourceId ? { sourceId: entry.sourceId } : {}),
      ...(entry.tool ? { tool: { ...entry.tool } } : {}),
    }));
  }
}

/**
 * Transcripts of the helpers this host has seen. The memory copy stays bounded, and the JSONL copy
 * beside the helper rows survives a restart. Each line is a put, a delete or the event high-water
 * mark; a partial last line after a crash is ignored. Large rewrites use the same atomic layer as
 * the rest of the host, so a failed save never replaces the last good copy.
 */
export function createChildTranscriptStore({ maxTranscripts = MAX_TRANSCRIPTS, now = Date.now, dir = null, maxEntries = MAX_TRANSCRIPT_ENTRIES, maxChars = MAX_TRANSCRIPT_CHARS, log = () => {} } = {}) {
  const items = new Map();
  const keyOf = (chatId, childId) => `${chatId}\u0000${childId}`;
  const fileOf = (chatId, childId) => dir ? join(dir, safeName(chatId), `${safeName(childId)}.jsonl`) : null;
  const headerOf = item => JSON.stringify({ v: STORE_VERSION, chat: item.chatId, child: item.childId });

  function readStored(file, chatId, childId) {
    if (!file) return null;
    const read = candidate => {
      try { return readFileSync(candidate, 'utf8'); } catch { return null; }
    };
    let text = read(file);
    if (!text) text = read(`${file}.bak`);
    if (!text) return null;
    const entries = new Map();
    let seq = -1;
    let sawHeader = false;
    for (const line of text.split(/\r?\n/)) {
      if (!line.trim()) continue;
      let record;
      try { record = JSON.parse(line); } catch { continue; }
      if (!record || record.v !== STORE_VERSION) continue;
      if (record.chat || record.child) {
        sawHeader = true;
        // The file is already scoped to one chat and one helper; a mismatched header is not read.
        if (String(record.chat) !== String(chatId) || String(record.child) !== String(childId)) return null;
        continue;
      }
      if (record.op === 'put') {
        const entry = normalizeEntry(record.entry);
        if (entry) entries.set(entry.id, entry);
        if (Number.isInteger(record.seq)) seq = Math.max(seq, record.seq);
      } else if (record.op === 'del') {
        if (typeof record.id === 'string') entries.delete(record.id);
        if (Number.isInteger(record.seq)) seq = Math.max(seq, record.seq);
      } else if (record.op === 'seq' && Number.isInteger(record.seq)) {
        seq = Math.max(seq, record.seq);
      }
    }
    return { entries: [...entries.values()], seq, sawHeader };
  }

  function makeItem(chatId, childId, transcript = null) {
    const file = fileOf(chatId, childId);
    const item = {
      chatId, childId, file, at: now(), timer: null, dirty: false,
      transcript: transcript ?? new ChildTranscript({ maxEntries, maxChars, now }),
      written: new Map(),
      lastSeq: -1,
    };
    item.transcript.onChange = () => schedule(item);
    if (!transcript && file) {
      const stored = readStored(file, chatId, childId);
      if (stored) {
        item.transcript.load(stored.entries, stored.seq);
        item.lastSeq = stored.seq;
        for (const entry of item.transcript.viewRich()) item.written.set(entry.id, JSON.stringify(entry));
      }
    }
    return item;
  }

  function evict() {
    while (items.size > maxTranscripts) {
      const [key, item] = items.entries().next().value;
      flushItem(item);
      items.delete(key);
    }
  }

  function ensure(chatId, childId, { create = true } = {}) {
    const key = keyOf(chatId, childId);
    const known = items.get(key);
    if (known) {
      known.at = now();
      items.delete(key);
      items.set(key, known);
      return known;
    }
    const file = fileOf(chatId, childId);
    if (!create && (!file || !existsSync(file))) return null;
    if (!create && !file) return null;
    const item = makeItem(chatId, childId);
    items.set(key, item);
    evict();
    return item;
  }

  function appendText(file, text) {
    mkdirSync(dirname(file), { recursive: true });
    let fd;
    try {
      fd = openSync(file, 'a');
      const bytes = Buffer.from(text, 'utf8');
      let offset = 0;
      while (offset < bytes.length) {
        const written = writeSync(fd, bytes, offset, bytes.length - offset);
        if (!(written > 0)) throw new Error(`Could not append ${file}.`);
        offset += written;
      }
      fsyncSync(fd);
    } finally {
      if (fd !== undefined) try { closeSync(fd); } catch { /* already closed */ }
    }
  }

  function flushItem(item) {
    if (!item || !item.dirty || !item.file) return true;
    const entries = item.transcript.viewRich();
    const current = new Map(entries.map(entry => [entry.id, JSON.stringify(entry)]));
    const lines = [];
    const nextWritten = new Map();
    for (const entry of entries) {
      const serialized = current.get(entry.id);
      nextWritten.set(entry.id, serialized);
      if (item.written.get(entry.id) !== serialized) {
        lines.push(JSON.stringify({ v: STORE_VERSION, op: 'put', seq: item.transcript.lastSeq, entry }));
      }
    }
    for (const id of item.written.keys()) {
      if (!current.has(id)) lines.push(JSON.stringify({ v: STORE_VERSION, op: 'del', id, seq: item.transcript.lastSeq }));
    }
    if (item.lastSeq !== item.transcript.lastSeq) {
      lines.push(JSON.stringify({ v: STORE_VERSION, op: 'seq', seq: item.transcript.lastSeq }));
    }
    if (!lines.length) {
      item.dirty = false;
      return true;
    }
    const body = `${lines.join('\n')}\n`;
    try {
      let size = 0;
      try { size = existsSync(item.file) ? readFileSync(item.file).length : 0; } catch { size = 0; }
      const rewrite = size + Buffer.byteLength(body) > MAX_STORED_BYTES;
      if (rewrite) {
        const snapshot = [
          headerOf(item),
          ...entries.map(entry => JSON.stringify({ v: STORE_VERSION, op: 'put', seq: item.transcript.lastSeq, entry })),
          JSON.stringify({ v: STORE_VERSION, op: 'seq', seq: item.transcript.lastSeq }),
        ].join('\n') + '\n';
        atomicWriteFileSync(item.file, snapshot);
      } else {
        const prefix = size === 0 ? `${headerOf(item)}\n` : '';
        appendText(item.file, `${prefix}${body}`);
      }
      item.written = nextWritten;
      item.lastSeq = item.transcript.lastSeq;
      item.dirty = false;
      return true;
    } catch (error) {
      item.dirty = true;
      log(`agent · helper transcript could not be saved: ${error.message}`);
      return false;
    }
  }

  function schedule(item) {
    if (!item?.file) return;
    item.dirty = true;
    if (item.timer) return;
    item.timer = setTimeout(() => {
      item.timer = null;
      flushItem(item);
    }, STORE_FLUSH_MS);
    item.timer.unref?.();
  }

  function acquire(chatId, childId, transcript = null) {
    const item = ensure(chatId, childId, { create: true });
    if (transcript && item.transcript !== transcript) {
      // Pure callers may bring their own transcript. Keep it, and let the store own its saving.
      item.transcript = transcript;
      item.transcript.onChange = () => schedule(item);
    }
    return item.transcript;
  }

  function get(chatId, childId) {
    return items.get(keyOf(chatId, childId))?.transcript ?? null;
  }

  function view(chatId, childId) {
    const item = ensure(chatId, childId, { create: false });
    return item ? item.transcript.viewRich() : null;
  }

  function revision(chatId, childId) {
    const item = ensure(chatId, childId, { create: false });
    return item ? item.transcript.revision : 0;
  }

  function has(chatId, childId) {
    if (items.has(keyOf(chatId, childId))) return true;
    const file = fileOf(chatId, childId);
    return !!file && existsSync(file);
  }

  function flush(chatId, childId) {
    const item = items.get(keyOf(chatId, childId));
    if (!item) return true;
    if (item.timer) { clearTimeout(item.timer); item.timer = null; }
    return flushItem(item);
  }

  function flushAll() {
    let ok = true;
    for (const item of items.values()) {
      if (item.timer) { clearTimeout(item.timer); item.timer = null; }
      ok = flushItem(item) && ok;
    }
    return ok;
  }

  function forget(chatId) {
    for (const [key, item] of [...items]) {
      if (item.chatId !== chatId) continue;
      flushItem(item);
      if (item.timer) clearTimeout(item.timer);
      items.delete(key);
    }
    if (dir) {
      try { rmSync(join(dir, safeName(chatId)), { recursive: true, force: true }); } catch { /* best effort */ }
    }
  }

  return {
    acquire, get, view, revision, has, flush, flushAll, forget,
    clear: () => { flushAll(); items.clear(); },
    get size() { return items.size; },
    _debug: () => ({ items, dir }),
  };
}

/**
 * True when this session event is newer than the transcript's high-water mark. A reopened follow
 * stream starts with a snapshot of the whole session; without this the transcript would repeat itself.
 */
export function advanceEvent(transcript, event) {
  const seq = event?.seq;
  if (!Number.isInteger(seq)) return true;
  if (seq <= transcript.lastSeq) return false;
  transcript.setLastSeq(seq);
  return true;
}

/** What a helper is doing, in words (shown as the row's description while it works). */
export function activityOf(name, args = {}) {
  const scope = toolScope(name);
  if (scope === 'shell') return 'Running a command';
  if (scope === 'write') return 'Changing files';
  if (name === 'web_fetch' || name === 'web_search') return 'Looking something up online';
  if (name === 'read' || name === 'glob' || name === 'grep' || name === 'read_image') return 'Reading files';
  if (name === 'subagent' || name === 'subagent_fork') return 'Handing work to another helper';
  if (scope === 'read') return 'Looking through the project';
  return 'Using a tool';
}

export class ChildProjector {
  /**
   * @param {{id:string, name?:string, task?:string, mode?:'manual'|'plan'|'bypass'|null, now?:()=>number, transcript?:ChildTranscript}} options
   */
  constructor({ id, name = 'Helper', task = '', mode = null, now = () => Date.now(), transcript = null }) {
    this.id = id;
    this.name = clip(name, 60) || 'Helper';
    this.taskText = String(task ?? '').replace(/\s+/g, ' ').trim().slice(0, 4_000);
    this.task = clip(this.taskText, 140);
    /** The chat mode in force when this helper started; it never follows the parent's later switches. */
    this.mode = ['manual', 'plan', 'bypass'].includes(mode) ? mode : null;
    this.now = now;
    this.transcript = transcript ?? new ChildTranscript({ now });
    this.startedAt = now();
    this.status = 'queued';
    this.tokens = 0;
    this.sawUsage = false;
    this.activity = null;
    this.closing = '';              // the helper's last words, shown once it finished
    this.failure = null;
    this.endedAt = null;
    this.tools = new Map();         // callId -> { card, calledAt, asked, entryId }
    this.asked = new Map();         // approval id -> callId
    this.internalRequests = new Set();
    this.firstUser = this.transcript.hasTask;
    this.lastEmit = 0;
    if (!this.transcript.hasTask && this.transcript.entries.length && this.taskText) {
      this.transcript.task(this.taskText, this.startedAt);
      this.firstUser = true;
    }
  }

  /** The spawn call carries the human task separately from the Harness's appended guidance. */
  setTask(value) {
    const text = String(value ?? '').replace(/\s+/g, ' ').trim().slice(0, 4_000);
    if (!text) return;
    this.taskText = text;
    this.task = clip(text, 140);
    if (this.transcript.hasTask || this.firstUser) this.transcript.task(text, this.startedAt);
  }

  /** A continue request is a host action, not something the person typed into this conversation. */
  markInternal(requestId) {
    if (typeof requestId === 'string' && requestId) this.internalRequests.add(requestId);
  }

  /** The row as the UI merges it. */
  agent() {
    const at = this.now();
    const done = this.status === 'completed' || this.status === 'failed' || this.status === 'cancelled' || this.status === 'interrupted';
    let description = this.task;
    if (this.status === 'queued') description = this.task ? `Waiting for its turn. ${this.task}` : 'Waiting for its turn.';
    if (this.status === 'working' && this.activity) description = this.activity;
    if (this.status === 'completed') description = this.closing ? clip(this.closing, 140) : 'Finished.';
    if (this.status === 'failed') description = this.failure ?? 'Stopped because something went wrong.';
    if (this.status === 'cancelled') description = 'Stopped.';
    return {
      id: this.id, childSessionId: this.id, name: this.name, description, status: this.status,
      ...(this.mode ? { mode: this.mode } : {}),
      durationMs: Math.max(0, (done ? this.endedAt : at) - this.startedAt), tokens: this.tokens, updatedAt: at,
    };
  }

  /**
   * Make this row final before its stream is closed. A helper that the run is done with must not keep a
   * "working" row: the row is the only thing the page has, and a stream that closes without a terminal row
   * leaves it running forever (#4218).
   */
  seal(status = 'interrupted') {
    if (this.finished) return false;
    this.endedAt = this.now();
    this.transcript.close(this.endedAt);
    if (status === 'failed') this.failure = this.failure ?? 'Stopped because something went wrong.';
    this.status = status;
    return true;
  }

  /**
   * The person stopped this helper on purpose. The Harness ends such a turn as aborted (the same
   * reason a parent's Stop gives); for this row that means "stopped, can continue", not "cancelled".
   * A helper that failed or finished on its own keeps its own answer.
   */
  markStopped() {
    if (this.finished && this.status !== 'cancelled') return false;
    this.endedAt = this.endedAt ?? this.now();
    this.transcript.close(this.endedAt);
    this.status = 'interrupted';
    return true;
  }

  #cardId(callId) { return `${this.id}:${callId}`; }

  #card(callId, patch, at = this.now()) {
    const entry = this.tools.get(callId);
    if (!entry) return null;
    entry.card = { ...entry.card, ...patch };
    if (patch.state) entry.card.verb = toolVerb(entry.card.name, patch.state === 'done' ? 'done' : patch.state === 'error' || patch.state === 'denied' ? 'error' : 'running');
    if (entry.entryId) this.transcript.updateTool(entry.entryId, entry.card);
    return { type: 'agent-tool', tool: entry.card };
  }

  /**
   * Fold one follow item. Returns `{ frames, row }`: card frames to send now, and whether the row changed
   * (the caller decides when to send it, so a stream of usage chunks does not flood the page).
   */
  consume(item) {
    const frames = [];
    let row = false;
    const set = (status) => { if (this.status !== status) { this.status = status; row = true; } };

    if (item.type === 'assistant-stream') {
      const f = item.frame;
      if (f.type === 'chunk' && f.chunk?.type === 'usage' && f.chunk.usage) {
        this.sawUsage = true;
        this.tokens += (Number(f.chunk.usage.inputTokens) || 0) + (Number(f.chunk.usage.outputTokens) || 0);
        row = true;
      } else if (f.type === 'chunk' && f.chunk?.type === 'text-delta') {
        this.transcript.stream(f.chunk.text, this.now());
        if (this.status === 'queued') set('working');
      } else if (f.type === 'chunk' && f.chunk?.type === 'reasoning-delta') {
        if (this.status === 'queued') set('working');
      }
      return { frames, row };
    }
    if (item.type !== 'event') return { frames, row };
    const e = item.event;
    const d = e.data ?? {};
    const at = Number.isFinite(e.time) ? e.time : this.now();
    switch (e.type) {
      case 'user/message': {
        if (!conversationUser(d)) break;
        const text = messageText(d.content);
        if (!text) break;
        const rpcId = typeof d.source?.rpcId === 'string' ? d.source.rpcId : null;
        if (rpcId && this.internalRequests.has(rpcId)) {
          this.internalRequests.delete(rpcId);
          this.transcript.notice(CONTINUE_NOTICE, at);
          break;
        }
        if (!this.firstUser) {
          // The first user message in a child session is the spawn prompt. It also carries the
          // Harness's "your parent agent id" guidance, so the timeline starts with the task card
          // taken from the parent's structured call, never with that raw text.
          this.firstUser = true;
          this.transcript.task(this.taskText || TASK_MISSING, at);
          break;
        }
        this.transcript.user(text, at, rpcId);
        break;
      }
      case 'step/start':
      case 'turn/start':
        // A request was sent. Whether the engine has room for it yet is told by its first word: until then the helper is waiting for its turn.
        // A helper that had finished and got a new message starts over as waiting.
        if (this.finished) { this.endedAt = null; this.failure = null; set('queued'); }
        break;
      case 'assistant/message': {
        const text = (d.message?.content ?? []).filter(p => p.type === 'text').map(p => p.text).join('').trim();
        if (text) this.closing = text;
        this.transcript.message(text, at);
        break;
      }
      case 'tool/call': {
        const args = parseArguments(d.arguments);
        if (d.name === 'send_message') {
          this.transcript.notice(SENT_NOTICE, at);
          break;
        }
        if (HIDDEN_TOOLS.has(d.name)) break;
        const base = presentTool({ callId: this.#cardId(d.callId), name: d.name, args });
        const card = {
          ...base,
          target: safeTarget(base.target ?? base.preview),
          preview: safeTarget(base.preview),
          state: 'running',
          agentName: this.name,
        };
        const entry = this.transcript.tool(card, at);
        this.tools.set(d.callId, { card, calledAt: e.time, asked: false, entryId: entry?.id ?? null });
        this.activity = activityOf(d.name, args);
        set('working');
        row = true;
        break;
      }
      case 'approval/asked': {
        if (!d.callId || !this.tools.has(d.callId)) break;
        this.asked.set(d.id, d.callId);
        const entry = this.tools.get(d.callId);
        entry.asked = true;
        const frame = this.#card(d.callId, { state: 'proposed', effect: clip(`${this.name} asks: ${d.reason ?? 'to use a tool'}`, 400) });
        if (frame) frames.push(frame);
        break;
      }
      case 'approval/decided': {
        const callId = this.asked.get(d.id);
        const entry = callId ? this.tools.get(callId) : null;
        if (!entry) break;
        const frame = d.outcome === 'allowed-once'
          ? this.#card(callId, { state: 'running', approved: true })
          : this.#card(callId, { state: 'denied', result: d.outcome === 'rejected' ? 'Denied. Nothing ran.' : 'Cancelled. Nothing ran.' });
        if (frame) frames.push(frame);
        break;
      }
      case 'tool/result': {
        const callId = d.message?.toolCallId;
        const entry = this.tools.get(callId);
        if (!entry) break;
        const wasDenied = entry.card.state === 'denied';
        const nextState = wasDenied ? 'denied' : d.message?.isError === true ? 'error' : 'done';
        entry.card = {
          ...entry.card,
          state: nextState,
          verb: toolVerb(entry.card.name, nextState === 'done' ? 'done' : 'error'),
          result: wasDenied ? entry.card.result : resultText(d.message),
          durationMs: Math.max(0, (e.time ?? 0) - entry.calledAt),
        };
        if (entry.entryId) this.transcript.updateTool(entry.entryId, entry.card);
        if (entry.asked) frames.push({ type: 'agent-tool', tool: entry.card });
        this.activity = null;
        row = true;
        break;
      }
      case 'turn/end': {
        const reason = d.reason ?? { kind: 'completed' };
        // The event's own clock: a replayed snapshot must not make an old end look like a new one.
        this.endedAt = at;
        this.transcript.close(at);
        this.transcript.finishTools(at);
        if (reason.kind === 'aborted') set('cancelled');
        else if (reason.kind === 'error') { this.failure = clip(friendlyFailure(reason.error), 140); set('failed'); }
        else if (reason.kind === 'blocked' || reason.kind === 'interrupted') { this.failure = 'Stopped before it finished.'; set('failed'); }
        else set('completed');
        // A request still waiting when its helper ended can no longer be answered.
        for (const [callId, entry] of this.tools) {
          if (entry.asked && entry.card.state === 'proposed') {
            const frame = this.#card(callId, { state: 'denied', result: 'Cancelled. Nothing ran.' });
            if (frame) frames.push(frame);
          }
        }
        row = true;
        break;
      }
      default:
        break;
    }
    return { frames, row };
  }

  get finished() { return this.status === 'completed' || this.status === 'failed' || this.status === 'cancelled' || this.status === 'interrupted'; }
}
