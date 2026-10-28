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
import { friendlyFailure, parseArguments, presentTool, resultText, toolScope } from './events.mjs';

const clip = (value, max) => {
  const text = String(value ?? '').replace(/\s+/g, ' ').trim();
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
};

/** One bounded transcript per helper. The interface shows these lines; a tool call is one line, never raw JSON. */
export const MAX_TRANSCRIPT_ENTRIES = 200;
export const MAX_TRANSCRIPT_CHARS = 64 * 1024;
export const MAX_TRANSCRIPT_ENTRY_CHARS = 8 * 1024;
/** How many finished helpers keep a transcript in this host process (older ones are dropped first). */
export const MAX_TRANSCRIPTS = 100;

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

const toolLine = card => clip(`${card.verb}${card.preview ? `: ${card.preview}` : ''}`, 240);

/**
 * The conversation of one helper, kept in this host process only and bounded in both directions:
 * at most 200 lines and 64 KB of text. The oldest lines go first. A line is exactly
 * `{ role, text, at }`; streaming text grows the open assistant line, and the final assistant
 * message replaces it so a socket gap neither loses nor doubles words.
 */
export class ChildTranscript {
  constructor({ maxEntries = MAX_TRANSCRIPT_ENTRIES, maxChars = MAX_TRANSCRIPT_CHARS, now = Date.now } = {}) {
    this.maxEntries = maxEntries;
    this.maxChars = maxChars;
    this.now = now;
    this.entries = [];
    this.chars = 0;
    /** Highest session event sequence this transcript consumed, so a replayed snapshot is not written twice. */
    this.lastSeq = -1;
  }

  #entryMax() {
    return Math.max(1, Math.min(MAX_TRANSCRIPT_ENTRY_CHARS, this.maxChars));
  }

  #trim() {
    while (this.entries.length > this.maxEntries) this.#removeAt(0);
    while (this.chars > this.maxChars && this.entries.length > 1) this.#removeAt(0);
    // One line is clipped to MAX_TRANSCRIPT_ENTRY_CHARS (well below maxChars), so it cannot break the bound.
    if (this.chars > this.maxChars && this.entries.length === 1) {
      const entry = this.entries[0];
      const text = entry.text.slice(0, this.maxChars);
      this.chars = text.length;
      entry.text = text;
    }
  }

  #removeAt(index) {
    const [removed] = this.entries.splice(index, 1);
    if (removed) this.chars -= removed.text.length;
  }

  #setText(entry, value) {
    let text = String(value ?? '');
    const max = this.#entryMax();
    if (text.length > max) text = `${text.slice(0, max - 1)}…`;
    this.chars += text.length - entry.text.length;
    entry.text = text;
    this.#trim();
  }

  #append(role, value, at, extra = {}) {
    let text = String(value ?? '');
    if (!text) return null;
    const max = this.#entryMax();
    if (text.length > max) text = `${text.slice(0, max - 1)}…`;
    const entry = { role, text, at: Number.isFinite(at) ? at : this.now(), ...extra };
    this.entries.push(entry);
    this.chars += text.length;
    this.#trim();
    return this.entries.includes(entry) ? entry : null;
  }

  #closeOpen(at) {
    const last = this.entries[this.entries.length - 1];
    if (!last || last.role !== 'assistant' || !last.open) return;
    last.open = false;
    last.at = Number.isFinite(at) ? at : last.at;
    if (!last.text.trim()) this.#removeAt(this.entries.length - 1);
  }

  /** Text that arrives while the helper is writing: one growing assistant line. */
  stream(value, at = this.now()) {
    const text = String(value ?? '');
    if (!text) return;
    const last = this.entries[this.entries.length - 1];
    if (last?.role === 'assistant' && last.open) {
      last.at = Number.isFinite(at) ? at : last.at;
      this.#setText(last, last.text + text);
      return;
    }
    this.#append('assistant', text, at, { open: true });
  }

  /** The helper's committed assistant message: its own words, replacing what the live stream carried. */
  message(value, at = this.now()) {
    const text = String(value ?? '').trim();
    const last = this.entries[this.entries.length - 1];
    if (last?.role === 'assistant' && last.open) {
      this.#setText(last, text);
      last.open = false;
      last.at = Number.isFinite(at) ? at : last.at;
      if (!last.text.trim()) this.#removeAt(this.entries.length - 1);
      return;
    }
    this.#append('assistant', text, at);
  }

  /** A message the person sent to this helper. */
  user(value, at = this.now()) {
    const text = String(value ?? '').trim();
    if (!text) return;
    this.#closeOpen(at);
    const last = this.entries[this.entries.length - 1];
    // The host writes the sent line before the Harness echoes it back as user/message: keep one line.
    if (last?.role === 'user' && last.echo && last.text === text) {
      last.echo = false;
      last.at = Number.isFinite(at) ? at : last.at;
      return;
    }
    this.#append('user', text, at);
  }

  /** A tool call, compressed by presentTool's own words (never raw JSON). */
  tool(value, at = this.now()) {
    this.#closeOpen(at);
    this.#append('tool', value, at);
  }

  /** A line the host wrote for a message it just sent; the Harness echo clears the mark. */
  beginUser(value, at = this.now()) {
    const entry = this.#append('user', String(value ?? '').trim(), at, { echo: true });
    return entry;
  }

  cancelUser(entry) {
    const at = this.entries.indexOf(entry);
    if (at >= 0) this.#removeAt(at);
  }

  close(at = this.now()) {
    this.#closeOpen(at);
  }

  /** What the API returns: only the three fields the page needs, without the bookkeeping flags. */
  view() {
    return this.entries.map(({ role, text, at }) => ({ role, text, at }));
  }
}

/**
 * Transcripts of every helper the host has seen in this run. Finished helpers stay readable until the
 * process ends; the oldest transcript is dropped once `maxTranscripts` is reached, so memory is bounded.
 */
export function createChildTranscriptStore({ maxTranscripts = MAX_TRANSCRIPTS, now = Date.now } = {}) {
  const items = new Map();
  const keyOf = (chatId, childId) => `${chatId}\u0000${childId}`;

  function acquire(chatId, childId, transcript = null) {
    const key = keyOf(chatId, childId);
    const known = items.get(key);
    if (known) {
      known.at = now();
      items.delete(key);
      items.set(key, known);
      return known.transcript;
    }
    const made = transcript ?? new ChildTranscript({ now });
    items.set(key, { chatId, childId, transcript: made, at: now() });
    while (items.size > maxTranscripts) items.delete(items.keys().next().value);
    return made;
  }

  function get(chatId, childId) {
    const known = items.get(keyOf(chatId, childId));
    return known ? known.transcript : null;
  }

  function view(chatId, childId) {
    return get(chatId, childId)?.view() ?? null;
  }

  function has(chatId, childId) {
    return items.has(keyOf(chatId, childId));
  }

  function forget(chatId) {
    for (const [key, item] of items) if (item.chatId === chatId) items.delete(key);
  }

  return {
    acquire, get, view, has, forget,
    clear: () => items.clear(),
    get size() { return items.size; },
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
  transcript.lastSeq = seq;
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
   * @param {{id:string, name?:string, task?:string, now?:()=>number, transcript?:ChildTranscript}} options
   */
  constructor({ id, name = 'Helper', task = '', now = () => Date.now(), transcript = null }) {
    this.id = id;
    this.name = clip(name, 60) || 'Helper';
    this.task = clip(task, 140);
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
    this.tools = new Map();         // callId -> { card, calledAt, asked }
    this.asked = new Map();         // approval id -> callId
    this.lastEmit = 0;
  }

  /** The row as the UI merges it. */
  agent() {
    const at = this.now();
    const done = this.status === 'completed' || this.status === 'failed' || this.status === 'cancelled';
    let description = this.task;
    if (this.status === 'queued') description = this.task ? `Waiting for its turn. ${this.task}` : 'Waiting for its turn.';
    if (this.status === 'working' && this.activity) description = this.activity;
    if (this.status === 'completed') description = this.closing ? clip(this.closing, 140) : 'Finished.';
    if (this.status === 'failed') description = this.failure ?? 'Stopped because something went wrong.';
    if (this.status === 'cancelled') description = 'Stopped.';
    return {
      id: this.id, childSessionId: this.id, name: this.name, description, status: this.status,
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

  #card(callId, patch) {
    const entry = this.tools.get(callId);
    if (!entry) return null;
    entry.card = { ...entry.card, ...patch };
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
        if (text) this.transcript.user(text, at);
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
        const card = { ...presentTool({ callId: this.#cardId(d.callId), name: d.name, args }), state: 'running', agentName: this.name };
        this.transcript.tool(toolLine(card), at);
        this.tools.set(d.callId, { card, calledAt: e.time, asked: false });
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
        if (entry.asked) {
          const wasDenied = entry.card.state === 'denied';
          const frame = this.#card(callId, {
            state: wasDenied ? 'denied' : d.message?.isError === true ? 'error' : 'done',
            result: wasDenied ? entry.card.result : resultText(d.message),
            durationMs: Math.max(0, (e.time ?? 0) - entry.calledAt),
          });
          if (frame) frames.push(frame);
        }
        this.activity = null;
        row = true;
        break;
      }
      case 'turn/end': {
        const reason = d.reason ?? { kind: 'completed' };
        // The event's own clock: a replayed snapshot must not make an old end look like a new one.
        this.endedAt = at;
        this.transcript.close(at);
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

  get finished() { return this.status === 'completed' || this.status === 'failed' || this.status === 'cancelled'; }
}
