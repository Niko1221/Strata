#!/usr/bin/env node
/**
 * Strata Dev: rebuild chats from the agent's transcripts (Harness sessions) when the app's chat list lost them.
 *
 *   node dev/tools/rescue-chats.mjs --agent <copy of .strata\data\agent> --storage <copy of the window's Local Storage\leveldb>
 *                                   --workspaces <workspaces.json> [--out <folder>]
 *
 * Reads only the copies it is given. Writes two files into --out (default %LOCALAPPDATA%\StrataDev\rescue):
 *   rescued-chats-full-<time>.json    every rebuilt chat, texts as they were
 *   rescued-chats-import-<time>.json  the same chats in the app's format (strata.sessions.v1), long texts shortened so the list
 *                                     stays under the app's limit for its saved chat list (~3.5 million characters)
 * and prints, per workspace, how many chats it rebuilt and their titles (never their contents).
 *
 * A chat is a top-level agent session (helpers/subagents are left out). Its messages are what the person typed (source "user")
 * and the text parts of the assistant's replies, one reply per turn; reasoning and tool calls are left out. Chats the app's list
 * still has (same chat id) are not rebuilt again.
 */
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join, basename } from 'node:path';
import { createZstdDecompress, zstdDecompressSync } from 'node:zlib';

const arg = name => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : null; };
const AGENT = arg('agent'), STORAGE = arg('storage'), WORKSPACES = arg('workspaces');
const OUT = arg('out') ?? join(process.env.LOCALAPPDATA ?? '.', 'StrataDev', 'rescue');
const LIMIT = 3_000_000;            // characters for the import file (the app keeps ~3.5M; room for the chats it still has)
if (!AGENT || !existsSync(join(AGENT, 'sessions'))) { console.error('--agent: a copy of .strata\\data\\agent with sessions\\ inside'); process.exit(1); }

/* ------------------------------------------------------------------------------------------- zstd, all frames --- */
async function readZst(file) {
  const buf = readFileSync(file);
  const viaStream = await new Promise(res => { const chunks = []; const d = createZstdDecompress(); d.on('data', c => chunks.push(c)); d.on('end', () => res(Buffer.concat(chunks))); d.on('error', () => res(null)); d.end(buf); });
  const frames = []; let start = 0;
  for (let i = 4; i <= buf.length - 4; i++) if (buf.readUInt32LE(i) === 0xFD2FB528) { frames.push(buf.subarray(start, i)); start = i; }
  frames.push(buf.subarray(start));
  const parts = []; let pending = null;
  for (const f of frames) { const chunk = pending ? Buffer.concat([pending, f]) : f; try { parts.push(zstdDecompressSync(chunk)); pending = null; } catch { pending = chunk; } }
  const viaFrames = Buffer.concat(parts);
  return (viaStream && viaStream.length >= viaFrames.length ? viaStream : viaFrames).toString('utf8');
}

/* ------------------------------------------------------------- the chat ids the window's list still has (leveldb) --- */
function varint(b, p) { let r = 0, s = 0, x; do { x = b[p++]; r += (x & 0x7f) * 2 ** s; s += 7; } while (x & 0x80); return [r, p]; }
function snappy(src) {
  let [len, p] = varint(src, 0); const out = Buffer.alloc(len); let o = 0;
  while (p < src.length) {
    const tag = src[p++], kind = tag & 3;
    if (kind === 0) { let n = tag >> 2; if (n >= 60) { const k = n - 59; n = 0; for (let i = 0; i < k; i++) n |= src[p++] << (8 * i); } n += 1; src.copy(out, o, p, p + n); p += n; o += n; }
    else { let n, off; if (kind === 1) { n = ((tag >> 2) & 7) + 4; off = ((tag >> 5) << 8) | src[p++]; } else if (kind === 2) { n = (tag >> 2) + 1; off = src[p] | (src[p + 1] << 8); p += 2; } else { n = (tag >> 2) + 1; off = src.readUInt32LE(p); p += 4; } for (let i = 0; i < n; i++) { out[o] = out[o - off]; o++; } }
  }
  return out;
}
function blockEntries(block) {
  const restarts = block.readUInt32LE(block.length - 4), end = block.length - 4 - restarts * 4; const out = []; let p = 0, last = Buffer.alloc(0);
  while (p < end) { let sh, un, vl; [sh, p] = varint(block, p); [un, p] = varint(block, p); [vl, p] = varint(block, p); const key = Buffer.concat([last.subarray(0, sh), block.subarray(p, p + un)]); p += un; out.push({ key, value: block.subarray(p, p + vl) }); p += vl; last = key; }
  return out;
}
function tableEntries(buf) {
  const f = buf.subarray(buf.length - 48); let p = 0, a, b, io, is; [a, p] = varint(f, p); [b, p] = varint(f, p); [io, p] = varint(f, p); [is, p] = varint(f, p);
  const read = (off, size) => buf[off + size] === 1 ? snappy(buf.subarray(off, off + size)) : buf.subarray(off, off + size);
  return blockEntries(read(io, is)).flatMap(({ value }) => { let q = 0, off, size; [off, q] = varint(value, q); [size, q] = varint(value, q); return blockEntries(read(off, size)); });
}
function existingChats(dir) {
  if (!dir || !existsSync(dir)) return [];
  let best = null;
  for (const name of readdirSync(dir).filter(n => n.endsWith('.ldb'))) {
    for (const e of tableEntries(readFileSync(join(dir, name)))) {
      if (!e.key.toString('latin1').includes('strata.sessions.v1')) continue;
      const tag = e.key.readBigUInt64LE(e.key.length - 8); if ((tag & 0xffn) !== 1n) continue;
      if (!best || (tag >> 8n) > best.seq) best = { seq: tag >> 8n, value: e.value };
    }
  }
  if (!best) return [];
  const text = best.value[0] === 0 ? best.value.subarray(1).toString('utf16le') : best.value.subarray(1).toString('latin1');
  try { return JSON.parse(text); } catch { return []; }
}

/* -------------------------------------------------------------------------------------------------- rebuild --- */
const iso = ms => new Date(ms).toISOString();
const textOf = content => (Array.isArray(content) ? content : []).filter(c => c?.type === 'text' && typeof c.text === 'string').map(c => c.text).join('\n');
const imagesIn = content => (Array.isArray(content) ? content : []).filter(c => c?.type === 'image').length;
const withoutDevNote = t => t.replace(/\n\n\[Strata Dev: in this chat load only these skills:[^\]]*\]\s*$/, '');

const workspaces = WORKSPACES && existsSync(WORKSPACES) ? JSON.parse(readFileSync(WORKSPACES, 'utf8')) : [];
const workspaceFor = cwd => {
  const c = String(cwd ?? '').toLowerCase().replace(/[\\/]+$/, '');
  const hit = workspaces.filter(w => { const p = String(w.path).toLowerCase().replace(/[\\/]+$/, ''); return c === p || c.startsWith(`${p}\\`) || c.startsWith(`${p}/`); }).sort((a, b) => b.path.length - a.path.length)[0];
  return hit ? { path: hit.path, known: true } : { path: cwd, known: false };
};

const map = existsSync(join(AGENT, 'sessions.json')) ? JSON.parse(readFileSync(join(AGENT, 'sessions.json'), 'utf8')) : {};
const chatOf = new Map(Object.entries(map).map(([chatId, v]) => [v.sessionId, { chatId, cwd: v.cwd }]));
const existing = existingChats(STORAGE);
const existingIds = new Set(existing.map(s => s.id));

const rebuilt = [], skipped = { subagent: 0, empty: 0, existing: 0 };
for (const w of readdirSync(join(AGENT, 'sessions'))) {
  for (const s of readdirSync(join(AGENT, 'sessions', w))) {
    const file = join(AGENT, 'sessions', w, s, 'session.v4.jsonl.zstd');
    if (!existsSync(file)) continue;
    const records = (await readZst(file)).split('\n').filter(Boolean).map(l => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean).sort((a, b) => (a.seq ?? -1) - (b.seq ?? -1));
    const head = records.find(r => r.type === 'session');
    if (!head) continue;
    if (head.origin === 'subagent' || head.parentSession) { skipped.subagent++; continue; }
    const link = chatOf.get(head.id);
    if (link && existingIds.has(link.chatId)) { skipped.existing++; continue; }
    const messages = []; let reply = null, title = null, tokensIn = 0, tokensOut = 0, last = head.createdAt;
    const flush = () => { if (reply && reply.content.trim()) messages.push(reply); reply = null; };
    for (const r of records) {
      if (typeof r.time === 'number') last = Math.max(last, r.time);
      if (r.type === 'session/title' && typeof r.data?.title === 'string') title = r.data.title;
      if (r.type === 'user/message' && r.data?.source?.kind === 'user') {
        flush();
        const text = withoutDevNote(textOf(r.data.content)), pics = imagesIn(r.data.content);
        if (text.trim() || pics) messages.push({ id: `rescued-${r.data.id ?? r.seq}`, role: 'user', content: `${text}${pics ? `${text ? '\n\n' : ''}[${pics} Bild(er) – nicht wiederhergestellt]` : ''}`, at: iso(r.time) });
      }
      if (r.type === 'assistant/message') {
        const text = textOf(r.data?.message?.content);
        tokensIn += r.data?.usage?.inputTokens ?? 0; tokensOut += r.data?.usage?.outputTokens ?? 0;
        if (text.trim()) { reply ??= { id: `rescued-${r.data?.message?.id ?? r.seq}`, role: 'assistant', content: '', at: iso(r.time) }; reply.content = reply.content ? `${reply.content}\n\n${text}` : text; reply.at = iso(r.time); }
      }
      if (r.type === 'turn/end') flush();
    }
    flush();
    if (!messages.some(m => m.role === 'user')) { skipped.empty++; continue; }
    const ws = workspaceFor(link?.cwd ?? head.cwd);
    // A compacted chat starts with the summary the agent wrote ("Earlier in this conversation, ..."): not a title.
    const summary = /^Earlier in this conversation\b/;
    const firstUser = (messages.find(m => m.role === 'user' && !summary.test(m.content)) ?? messages.find(m => m.role === 'user')).content.split('\n')[0].trim();
    rebuilt.push({
      id: link?.chatId && !existingIds.has(link.chatId) ? link.chatId : `s_rescued_${head.id.replace(/[^a-z0-9]/gi, '').slice(-12)}`,
      title: (title || firstUser || 'Wiederhergestellter Chat').slice(0, 120),
      defaultTitle: false,
      createdAt: iso(head.createdAt), updatedAt: iso(last), lastReadAt: iso(last),
      messages, workspace: ws.path, tokensIn, tokensOut, preferences: {}, statsAvailable: false,
      rescued: { harnessSession: head.id, cwd: head.cwd, workspaceKnown: ws.known },
    });
  }
}
rebuilt.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));

/* -------------------------------------------------------------------- the import copy: under the app's limit --- */
const cut = (s, max) => (s.length > max ? `${s.slice(0, max)}\n\n[beim Wiederherstellen gekürzt: ${s.length.toLocaleString('de-DE')} Zeichen; der volle Text steht in rescued-chats-full-*.json]` : s);
let perMessage = 40_000, imported;
for (;;) {
  imported = rebuilt.map(c => ({ ...c, messages: c.messages.map(m => ({ ...m, content: cut(m.content, perMessage) })) }));
  if (JSON.stringify(imported).length <= LIMIT || perMessage <= 1000) break;
  perMessage = Math.floor(perMessage * 0.7);
}

mkdirSync(OUT, { recursive: true });
const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\..+/, '').replace('T', '-');
const fullFile = join(OUT, `rescued-chats-full-${stamp}.json`), importFile = join(OUT, `rescued-chats-import-${stamp}.json`);
writeFileSync(fullFile, JSON.stringify(rebuilt, null, 1));
writeFileSync(importFile, JSON.stringify(imported));

/* -------------------------------------------------------------------------------- report: counts and titles --- */
const byWs = new Map();
for (const c of rebuilt) { const k = `${c.workspace}${c.rescued.workspaceKnown ? '' : '  (nicht in der Workspace-Liste)'}`; if (!byWs.has(k)) byWs.set(k, []); byWs.get(k).push(c); }
console.log(`Rekonstruiert: ${rebuilt.length} Chats (übersprungen: ${skipped.existing} schon in der Liste, ${skipped.subagent} Helfer, ${skipped.empty} ohne eigene Nachricht)`);
for (const [ws, list] of byWs) {
  console.log(`\n${ws}: ${list.length}`);
  for (const c of list) console.log(`  - ${c.updatedAt.slice(0, 10)}  ${c.messages.length} Nachr.  ${c.title.split('\n')[0].slice(0, 70)}`);
}
const full = JSON.stringify(rebuilt).length, imp = JSON.stringify(imported).length;
const shortened = imported.reduce((n, c) => n + c.messages.filter(m => m.content.includes('[beim Wiederherstellen gekürzt')).length, 0);
console.log(`\nVoll: ${full.toLocaleString('de-DE')} Zeichen -> ${basename(fullFile)}`);
console.log(`Import: ${imp.toLocaleString('de-DE')} Zeichen (Grenze ${LIMIT.toLocaleString('de-DE')}), ${shortened} Nachricht(en) auf ${perMessage.toLocaleString('de-DE')} Zeichen gekürzt -> ${basename(importFile)}`);
console.log(`Ordner: ${OUT}`);
