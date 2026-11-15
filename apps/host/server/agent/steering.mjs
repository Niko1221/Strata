/** Durable receipts for text accepted into one running Harness session. */
import { existsSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { randomUUID } from 'node:crypto';

export const SOFT_CUT_PATCH = 'strata-harness-soft-cut-v2';

export const STEERING_UNAVAILABLE = 'The parent agent is not accepting this steering instruction. Your correction is kept.';

export function isSteeringDelivery(item, event, sessionId, requestId) {
  return !!item && item.sessionId === sessionId && item.turnId === requestId
    && Number.isSafeInteger(item.parentTurn) && event.type === 'user/message'
    && event.data?.source?.rpcId === item.id && event.data.source.steeringTurn === item.parentTurn;
}

/** Fold only the parent session's durable lifecycle, including late follow snapshots. */
export class ParentSteering {
  seq = -1;
  turn = null;
  closed = -1;
  observe(event) {
    if (!Number.isSafeInteger(event.seq) || event.seq <= this.seq) return;
    this.seq = event.seq;
    const n = event.data?.turn;
    if (!Number.isSafeInteger(n)) return;
    if (event.type === 'turn/end') {
      this.closed = Math.max(this.closed, n);
      if (this.turn !== null && this.turn <= n) this.turn = null;
    } else if (['turn/start', 'step/start'].includes(event.type) && n > this.closed && (this.turn === null || n >= this.turn)) this.turn = n;
  }
  reject(turn) {
    this.closed = Math.max(this.closed, turn);
    if (this.turn === turn) this.turn = null;
  }
}

export class SteeringStore {
  constructor(file, changed = () => {}) {
    this.file = file;
    this.changed = changed;
    this.items = [];
    if (existsSync(file)) {
      this.items = JSON.parse(readFileSync(file, 'utf8'));
      if (!Array.isArray(this.items)) throw new Error('The saved steering receipts are invalid.');
    }
  }

  list(chatId) { return this.items.filter(item => item.chatId === chatId); }
  get(chatId, id) { return this.items.find(item => item.chatId === chatId && item.id === id); }

  add({ chatId, sessionId, turnId, parentTurn, id, text }) {
    const existing = this.get(chatId, id);
    if (existing) {
      if (existing.text !== text) throw new Error('This steering message ID already belongs to different text.');
      return existing;
    }
    if (this.list(chatId).filter(item => item.state === 'pending').length >= 20) throw new Error('Twenty steering messages are already waiting.');
    const item = { chatId, sessionId, turnId, parentTurn, id, text, state: 'pending', accepted: false, revision: 0 };
    this.save(chatId, [...this.items, item]);
    return item;
  }

  update(chatId, id, patch) {
    const item = this.get(chatId, id);
    if (!item || item.state === 'delivered') return item;
    const next = { ...item, ...patch, revision: (item.revision ?? 0) + 1 };
    this.save(chatId, this.items.map(value => value === item ? next : value));
    return next;
  }

  save(chatId, items) {
    const temporary = `${this.file}.${randomUUID()}.tmp`;
    writeFileSync(temporary, JSON.stringify(items), { mode: 0o600 });
    renameSync(temporary, this.file);
    this.items = items;
    this.changed(chatId, this.list(chatId));
  }
}
