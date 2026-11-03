#!/usr/bin/env node
/**
 * Strata Dev: one chat, as complete as the agent transcript allows.
 *
 *   node dev/tools/rescue-one-chat.mjs --session <...\sessions\<ws>\<session-id>> --chat <chat id> --title <title> --workspace <path>
 *                                      --attachments <...\agent\home\attachments> --out <folder>
 *
 * Writes into --out:
 *   <name>.md                                   readable: every message, the assistant's texts, each tool step in one line
 *   rescued-chats-import-<digits>.json          the chat in the app's format for Strata Dev's Rescue tab: texts, tool cards, reasoning
 *                                               (long tool output and reasoning shortened), pictures as names; a new id so the chat
 *                                               loaded before stays and this one shows beside it ("vollständig")
 * Prints only counts. Reads only the copies it is given.
 */
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { createZstdDecompress, zstdDecompressSync } from 'node:zlib';

const arg = n => { const i = process.argv.indexOf(`--${n}`); return i > 0 ? process.argv[i + 1] : null; };
const SESSION = arg('session'), CHAT = arg('chat'), TITLE = arg('title') ?? 'Wiederhergestellter Chat', WORKSPACE = arg('workspace'), ATTACH = arg('attachments'), OUT = arg('out'), NAME = arg('name') ?? 'chat';
if (!SESSION || !CHAT || !OUT) { console.error('--session, --chat and --out are required'); process.exit(1); }

async function readZst(file) {
  const buf = readFileSync(file);
  const viaStream = await new Promise(res => { const c = []; const d = createZstdDecompress(); d.on('data', x => c.push(x)); d.on('end', () => res(Buffer.concat(c))); d.on('error', () => res(null)); d.end(buf); });
  const frames = []; let start = 0;
  for (let i = 4; i <= buf.length - 4; i++) if (buf.readUInt32LE(i) === 0xFD2FB528) { frames.push(buf.subarray(start, i)); start = i; }
  frames.push(buf.subarray(start));
  const parts = []; let pending = null;
  for (const f of frames) { const chunk = pending ? Buffer.concat([pending, f]) : f; try { parts.push(zstdDecompressSync(chunk)); pending = null; } catch { pending = chunk; } }
  const viaFrames = Buffer.concat(parts);
  return (viaStream && viaStream.length >= viaFrames.length ? viaStream : viaFrames).toString('utf8');
}

const iso = ms => new Date(ms).toISOString();
const local = ms => new Date(ms).toLocaleString('de-DE');
const cut = (s, n) => { s = String(s ?? ''); return s.length > n ? `${s.slice(0, n)} … [gekürzt, ${s.length.toLocaleString('de-DE')} Zeichen]` : s; };
const textOf = c => (Array.isArray(c) ? c : []).filter(p => p?.type === 'text').map(p => p.text).join('\n');
const resultText = c => (Array.isArray(c) ? c : []).map(p => p?.type === 'text' ? p.text : p?.type ? `[${p.type}]` : '').join('\n');
const parseArgs = a => { try { return typeof a === 'string' ? JSON.parse(a) : a ?? {}; } catch { return { raw: String(a) }; } };
const targetOf = (name, a) => a.command ?? a.file_path ?? a.path ?? a.pattern ?? a.url ?? a.query ?? a.description ?? '';
const scopeOf = name => (/^(write|edit|job_kill)$/.test(name) ? 'write' : /^(pwsh|bash)$/.test(name) ? 'shell' : /^mcp__/.test(name) ? 'connected' : 'read');

const sessionFile = existsSync(join(SESSION, 'session.v4.jsonl.zstd')) ? join(SESSION, 'session.v4.jsonl.zstd') : SESSION;
const recs = (await readZst(sessionFile)).split('\n').filter(Boolean).map(l => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean).sort((a, b) => (a.seq ?? -1) - (b.seq ?? -1));
const attachments = ATTACH && existsSync(ATTACH) ? readdirSync(ATTACH, { recursive: true }).map(String) : [];

const counts = { user: 0, replies: 0, toolCalls: 0, toolErrors: 0, reasoningParts: 0, images: 0, imagesFound: 0, compactions: 0 };
const md = [`# ${TITLE.split('\n')[0]}`, '', `Workspace: ${WORKSPACE ?? ''}  ·  Chat-ID: ${CHAT}  ·  wiederhergestellt aus dem Agent-Verlauf ${sessionFile.split(/[\\/]/).slice(-2, -1)[0] ?? ''}`, ''];
const messages = []; let reply = null; const calls = new Map();
const flush = () => { if (reply && (reply.content.trim() || reply.tools.length || reply.reasoning)) { if (!reply.reasoning) delete reply.reasoning; messages.push(reply); } reply = null; };
const ensureReply = r => (reply ??= { id: `rescued-full-${r.seq}`, role: 'assistant', content: '', at: iso(r.time), tools: [], reasoning: '' });

for (const r of recs) {
  if (r.type === 'user/message' && r.data?.source?.kind === 'compact-checkpoint') { counts.compactions++; md.push(`> _(Verlauf hier vom Agenten verdichtet)_`, ''); continue; }
  if (r.type === 'user/message' && r.data?.source?.kind === 'user') {
    flush(); counts.user++;
    const pics = (r.data.content ?? []).filter(p => p?.type === 'image').map(p => p.attachment ?? {});
    counts.images += pics.length;
    const picNotes = pics.map(p => { const hit = attachments.find(a => p.attachmentId && a.includes(p.attachmentId)); if (hit) counts.imagesFound++; return `[Bild: ${p.name ?? p.attachmentId ?? 'ohne Namen'}${hit ? ` – Datei: attachments\\${hit}` : ' – Datei nicht gefunden'}]`; });
    const text = [textOf(r.data.content), ...picNotes].filter(Boolean).join('\n\n');
    messages.push({ id: `rescued-full-${r.data.id ?? r.seq}`, role: 'user', content: cut(text, 60_000), at: iso(r.time) });
    md.push(`## Du · ${local(r.time)}`, '', text, '');
  }
  if (r.type === 'assistant/message') {
    for (const p of r.data?.message?.content ?? []) {
      if (p.type === 'text' && p.text?.trim()) { const x = ensureReply(r); x.content = x.content ? `${x.content}\n\n${p.text}` : p.text; x.at = iso(r.time); md.push(`## Assistent · ${local(r.time)}`, '', p.text, ''); }
      if (p.type === 'reasoning') { counts.reasoningParts++; const x = ensureReply(r); x.reasoning = cut(`${x.reasoning}${x.reasoning ? '\n\n' : ''}${p.text ?? p.reasoning ?? ''}`, 4000); }
    }
  }
  if (r.type === 'tool/call') {
    counts.toolCalls++;
    const a = parseArgs(r.data?.arguments), name = String(r.data?.name ?? 'tool');
    const card = { id: r.data?.callId ?? `call-${r.seq}`, name, args: Object.fromEntries(Object.entries(a).map(([k, v]) => [k, typeof v === 'string' ? cut(v, 2000) : v])), preview: cut(String(targetOf(name, a)), 300), effect: '', scope: scopeOf(name), target: cut(String(targetOf(name, a)), 300), state: 'done', live: false };
    ensureReply(r).tools.push(card); calls.set(card.id, { card, time: r.time });
  }
  if (r.type === 'tool/result') {
    const id = r.data?.message?.toolCallId ?? r.data?.message?.source?.callId, c = calls.get(id);
    const err = !!r.data?.message?.isError; if (err) counts.toolErrors++;
    if (c) { c.card.state = err ? 'error' : 'done'; c.card.result = cut(resultText(r.data?.message?.content), 1500); c.card.durationMs = Math.max(0, r.time - c.time); md.push(`- \`${c.card.name}\` ${cut(c.card.target, 160).replace(/\n/g, ' ')}${err ? ' — **Fehler**' : ''}`); }
  }
  if (r.type === 'turn/end') { if (reply) counts.replies++; flush(); md.push(''); }
}
if (reply) counts.replies++;
flush();

mkdirSync(OUT, { recursive: true });
const mdFile = join(OUT, `${NAME}.md`);
writeFileSync(mdFile, md.join('\n'));
const head = recs.find(r => r.type === 'session');
const last = recs.reduce((m, r) => Math.max(m, r.time ?? 0), head?.createdAt ?? 0);
let chat = { id: `${CHAT}-full`, title: `${TITLE.split('\n')[0].slice(0, 100)} (vollständig)`, defaultTitle: false, createdAt: iso(head?.createdAt ?? last), updatedAt: iso(last), lastReadAt: iso(last), messages, workspace: WORKSPACE ?? head?.cwd ?? '', tokensIn: 0, tokensOut: 0, preferences: {}, statsAvailable: false, rescued: { harnessSession: head?.id, detail: 'full' } };
// Under the app's limit beside the 72 chats loaded before (~1.2M characters): shorten tool output and reasoning further if needed.
for (const [res, rea] of [[1500, 4000], [600, 1500], [200, 400], [0, 0]]) {
  chat = { ...chat, messages: messages.map(m => m.role !== 'assistant' ? m : { ...m, ...(m.reasoning !== undefined ? { reasoning: rea ? cut(m.reasoning, rea) : undefined } : {}), tools: m.tools.map(t => ({ ...t, result: res ? cut(t.result ?? '', res) : undefined })) }) };
  if (JSON.stringify(chat).length <= 1_800_000) break;
}
const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\..+/, '').replace('T', '');
const importFile = join(OUT, `rescued-chats-import-${stamp}.json`);
writeFileSync(importFile, JSON.stringify([chat]));
console.log(JSON.stringify({ ...counts, mdFile, mdChars: md.join('\n').length, importFile, importChars: JSON.stringify([chat]).length }, null, 1));
