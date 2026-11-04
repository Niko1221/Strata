/** Durable receipts for text accepted into one running Harness session. */
import { existsSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { randomUUID } from 'node:crypto';

export const SOFT_CUT_PATCH = 'strata-harness-soft-cut-v1';

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

  add({ chatId, sessionId, turnId, id, text }) {
    const existing = this.get(chatId, id);
    if (existing) {
      if (existing.text !== text) throw new Error('This steering message ID already belongs to different text.');
      return existing;
    }
    if (this.list(chatId).filter(item => item.state === 'pending').length >= 20) throw new Error('Twenty steering messages are already waiting.');
    const item = { chatId, sessionId, turnId, id, text, state: 'pending', accepted: false, revision: 0 };
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
