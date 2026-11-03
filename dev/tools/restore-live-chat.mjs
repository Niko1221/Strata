#!/usr/bin/env node
/**
 * Strata Dev rescue: prepares a Live chat list with one chat put back, in a COPY of the Live window's Local Storage under Documents.
 * Never touches Live's own folder: the person swaps the prepared folder in by hand (see the printed steps).
 *
 *   node dev/tools/restore-live-chat.mjs --backup <pre-restore folder with "Local Storage"> --chats <rescued-chats-full-*.json>
 *                                        --id <chat id> --work <work folder under Documents> [--keep 2000]
 *
 * Chrome opens the copy as profile "Default" (it also replays the LevelDB journal, so the newest state is read), a page on the
 * app's origin (http://127.0.0.1:47831, served by the browser itself, no server) changes "strata.sessions.v1":
 *   - in the chats already there, every user message longer than 20,000 characters (an attached file) keeps its first --keep
 *     characters and a note where the full text is;
 *   - the chat --id from --chats is added (unless the list has it), with its original id, so the agent session it belongs to fits.
 * Prints only counts and sizes. Writes the result to <work>\staging\Local Storage.
 */
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join } from 'node:path';

const arg = n => { const i = process.argv.indexOf(`--${n}`); return i > 0 ? process.argv[i + 1] : null; };
const DEEP = process.argv.includes('--deep');
const BACKUP = arg('backup'), CHATS = arg('chats'), ID = arg('id'), WORK = arg('work'), KEEP = Number(arg('keep') ?? 2000);
if (!BACKUP || !CHATS || !ID || !WORK) { console.error('--backup, --chats, --id, --work are required'); process.exit(1); }
const require = createRequire(new URL('../../apps/host/package.json', import.meta.url));
const { chromium } = require('playwright-core');

// --from-id: the chat's id in --chats when it differs (the Dev copy "<id>-full"); it goes into the list as --id.
const FROM = arg('from-id') ?? ID;
const chat = JSON.parse(readFileSync(CHATS, 'utf8')).find(c => c.id === FROM);
if (chat) { chat.id = ID; chat.title = String(chat.title).replace(/ (vollständig)$/, ''); }
if (!chat) { console.error(`${FROM} is not in ${CHATS}`); process.exit(1); }
delete chat.rescued;

const profile = join(WORK, 'profile');
rmSync(WORK, { recursive: true, force: true });
mkdirSync(join(profile, 'Default'), { recursive: true });
cpSync(join(BACKUP, 'Local Storage'), join(profile, 'Default', 'Local Storage'), { recursive: true });

const note = `[bei der Wiederherstellung am ${new Date().toLocaleDateString('de-DE')} gekürzt; der volle Text steht in der Sicherung ${BACKUP}]`;
const context = await chromium.launchPersistentContext(profile, { channel: 'chrome', headless: true });
let result;
try {
  const page = context.pages()[0] ?? await context.newPage();
  await page.route('http://127.0.0.1:47831/**', route => route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><title>restore</title>' }));
  await page.goto('http://127.0.0.1:47831/');
  result = await page.evaluate(({ chat, keep, note, deep }) => {
    const KEY = 'strata.sessions.v1';
    const raw = localStorage.getItem(KEY);
    if (raw === null) return { error: 'strata.sessions.v1 not found (wrong origin or unreadable storage)' };
    let list; try { list = JSON.parse(raw); } catch (e) { return { error: `not JSON: ${e.message}` }; }
    const before = raw.length, perChat = [];
    const shorten = (v, n) => (typeof v === 'string' && v.length > n + 200 ? `${v.slice(0, n)}\n\n${note}` : v);
    for (const s of list) {
      let cut = 0, saved = 0;
      const was = JSON.stringify(s).length;
      for (const m of s.messages ?? []) {
        if (m.role === 'user' && typeof m.content === 'string' && m.content.length > 20000) { m.content = `${m.content.slice(0, keep)}\n\n${note}`; cut++; }
        // --deep: the assistant's reasoning and the tool cards' long texts too (the chat and its answer texts stay).
        if (deep && m.role === 'assistant') {
          if (typeof m.reasoning === 'string' && m.reasoning.length > 1200) { m.reasoning = shorten(m.reasoning, 1000); cut++; }
          for (const t of m.tools ?? []) {
            for (const k of ['result', 'preview', 'detail']) if (typeof t[k] === 'string' && t[k].length > 1200) { t[k] = shorten(t[k], 1000); cut++; }
            if (t.args && typeof t.args === 'object') for (const [k, v] of Object.entries(t.args)) if (typeof v === 'string' && v.length > 1200) { t.args[k] = shorten(v, 1000); cut++; }
          }
        }
      }
      saved = was - JSON.stringify(s).length;
      perChat.push({ updated: s.updatedAt, workspace: s.workspace, cut, saved });
    }
    const had = list.some(s => s.id === chat.id);
    if (!had) list.push(chat);
    const text = JSON.stringify(list);
    localStorage.setItem(KEY, text);
    const check = JSON.parse(localStorage.getItem(KEY));
    return { before, after: text.length, chats: check.length, hadChat: had, added: check.some(s => s.id === chat.id), perChat, keys: Object.keys(localStorage).length };
  }, { chat, keep: KEEP, note, deep: DEEP });
} finally {
  await context.close();
}
if (result?.error) { console.error(result.error); process.exit(1); }
const staging = join(WORK, 'staging', 'Local Storage');
mkdirSync(join(WORK, 'staging'), { recursive: true });
cpSync(join(profile, 'Default', 'Local Storage'), staging, { recursive: true });
console.log(JSON.stringify({ ...result, staging, stagingExists: existsSync(join(staging, 'leveldb', 'CURRENT')) }, null, 1));
