#!/usr/bin/env node
/**
 * AGENT-LOST end to end: a long single-file write through the real agent bridge, the real Harness and a real engine.
 *
 *   STRATA_HARNESS_DIR=<harness folder> node scripts/agent-lost-e2e.mjs [pagoda|long|cut ...]
 *
 * Scenarios (default: all)
 *   pagoda  the owner's own prompt ("Write me a 100 voxel pagoda in one html file with garden."), thinking Medium, bypass mode
 *   long    a deterministic long write: one file of N numbered lines (E2E_LINES, default 1800, about 22K tokens), thinking Off
 *   raw     the model server alone: a reply cut inside a tool call, as the client receives it
 *   cut     the same long write with the Harness told a small window (E2E_CUT_WINDOW, default 24000), so the reply budget is
 *           about 7K tokens and the reply is cut inside the tool call: the turn must say so in words and the agent must carry on
 *
 * A model server that is already running is used when E2E_BASE_URL is set (nothing is started or stopped). Otherwise this
 * starts serve/server.py as its own child with the engine named below and stops it by PID.
 *
 * Environment: E2E_ENGINE (strata executable), E2E_PACK, E2E_GGUF1, E2E_GGUF2, E2E_MTP, E2E_PROFILE, E2E_LIB (library folder),
 * E2E_PORT (18590), E2E_WINDOW (65536), E2E_OUT (result folder), E2E_PYTHON.
 *
 * It prints one table and writes <E2E_OUT>/result.json with the frames summary of each scenario.
 */
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';

const REPO = join(import.meta.dirname, '..', '..', '..');
const env = (name, fallback) => process.env[name] ?? fallback;
const HOME = process.env.HOME ?? '';
const PORT = Number(env('E2E_PORT', '18590'));
const WINDOW = Number(env('E2E_WINDOW', '65536'));
const EXISTING = process.env.E2E_BASE_URL ?? '';
const PYTHON = env('E2E_PYTHON', 'python3');
const OUT = env('E2E_OUT', mkdtempSync(join(tmpdir(), 'strata-agent-lost-')));
const LINES = Number(env('E2E_LINES', '1400'));
const wanted = process.argv.slice(2);
const want = id => wanted.length === 0 || wanted.includes(id);
mkdirSync(OUT, { recursive: true });

if (!process.env.STRATA_HARNESS_DIR) { console.error('STRATA_HARNESS_DIR is not set'); process.exit(2); }
const sleep = ms => new Promise(r => setTimeout(r, ms));

let server = null, bridge = null;
const finish = async (code) => {
  try { await bridge?.close(); } catch { /* going away */ }
  if (server?.pid) { try { process.kill(-server.pid, 'SIGTERM'); } catch { try { process.kill(server.pid, 'SIGTERM'); } catch { /* gone */ } } }
  await sleep(1500);
  console.log(`\nresult folder: ${OUT}`);
  process.exit(code);
};
process.on('SIGINT', () => finish(130));

if (!EXISTING) {
  const pack = env('E2E_PACK', join(HOME, '026q/pack-iq3s'));
  const g1 = env('E2E_GGUF1', join(HOME, 'models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf'));
  const g2 = env('E2E_GGUF2', join(HOME, 'models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf'));
  const mtp = env('E2E_MTP', join(HOME, 'pr1426/mtp-rt'));
  const profile = env('E2E_PROFILE', join(HOME, 'bwm/cnt3/data/expert-profile.bin'));
  const exe = env('E2E_ENGINE', join(HOME, 'agentlost/eng/strata'));
  const lib = env('E2E_LIB', join(HOME, 'agentlost/eng'));
  for (const f of [pack, g1, g2, mtp, exe]) if (!existsSync(f)) { console.error(`missing: ${f}`); process.exit(2); }
  const config = join(OUT, 'server.json');
  writeFileSync(config, JSON.stringify({
    exe, args: ['--pack', pack, '--native', g1, '--ple-gguf', g2, ...(existsSync(profile) ? ['--expert-profile', profile] : []),
      '--prefill', 'auto', '--spec-min-p', '0.5', '--mtp', mtp, '--max-context', String(WINDOW), '--kv', 'int8', '--expert-cache', 'auto', '--spec', '4',
      ...(process.env.E2E_ENGINE_ARGS ? process.env.E2E_ENGINE_ARGS.split(' ').filter(Boolean) : [])],
    cwd: REPO, tokenizer: join(pack, 'tokenizer'), model_name: 'qwen3.8-flash-next-iq3_s', log: join(OUT, 'engine.log'), lib_dirs: [lib], port: PORT, gpu: 0,
    fit_max_tokens: true,
  }, null, 2));
  server = spawn(PYTHON, ['serve/server.py', '--engine', 'strata', '--config', config, '--port', String(PORT), '--no-api-key', '--ready-file', join(OUT, 'ready.json')], {
    cwd: REPO, env: { ...process.env, LD_LIBRARY_PATH: `${lib}:${process.env.LD_LIBRARY_PATH ?? ''}`, STRATA_DATA_DIR: join(OUT, 'data'), PYTHONPATH: `${join(REPO, 'tools')}:${REPO}` }, stdio: ['ignore', 'pipe', 'pipe'], detached: true,
  });
  const sink = [];
  server.stdout.on('data', d => sink.push(String(d)));
  server.stderr.on('data', d => sink.push(String(d)));
  server.on('exit', code => { sink.push(`[server exited ${code}]`); server = null; writeFileSync(join(OUT, 'server.log'), sink.join('')); });
}

const base = EXISTING || `http://127.0.0.1:${PORT}`;
let ready = false;
for (let i = 0; i < 360 && !ready; i++) {
  if (!EXISTING && !server) break;
  try { ready = (await fetch(`${base}/health`, { signal: AbortSignal.timeout(2000) })).ok; } catch { /* still loading */ }
  if (!ready) await sleep(2000);
}
if (!ready) { console.error('the engine did not become ready'); await finish(4); }
const model = (await (await fetch(`${base}/v1/models`)).json()).data[0].id;
console.log(`engine ready: ${model}`);

const rows = [];
const note = (id, ok, detail) => { rows.push({ id, ok, detail }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${id}  ${detail}`); };

/** The bridge for a window as the host would tell it (the engine's own window stays WINDOW). */
function makeBridge(window, dir) {
  mkdirSync(join(OUT, dir), { recursive: true });
  return createAgentBridge({
    dataDir: join(OUT, dir), env: process.env, startTimeoutMs: 180_000, log: l => console.log('   bridge:', l),
    localInstances: () => [{ id: 'e2e', label: 'iq3_s', baseUrl: `${base}/v1`, modelId: model, context: window, slots: 1, key: '' }], connections: () => [],
  });
}
bridge = makeBridge(WINDOW, 'agent');

/** One agent turn; every frame is kept in a compact timeline so the live data path can be read afterwards. */
async function turn(name, message, { thinking = 'Medium', mode = 'bypass', chatId = name } = {}) {
  const workspace = join(OUT, 'ws-' + name);
  mkdirSync(workspace, { recursive: true });
  const frames = [];
  const t0 = Date.now();
  await bridge.runTurn({ chatId, message, workspacePath: workspace, mode, thinking, onFrame: f => frames.push({ at: Date.now() - t0, f }) });
  const seconds = (Date.now() - t0) / 1000;
  const byType = {};
  for (const { f } of frames) byType[f.type] = (byType[f.type] ?? 0) + 1;
  const done = frames.map(x => x.f).filter(f => f.type === 'done').at(-1) ?? null;
  const error = frames.map(x => x.f).filter(f => f.type === 'error').at(-1) ?? null;
  const prepare = frames.map(x => x.f).filter(f => f.type === 'status' && (f.phase === 'preparing' || f.code === 'preparing-tool'));
  const cards = frames.map(x => x.f).filter(f => f.type === 'agent-tool').map(f => f.tool);
  const text = frames.map(x => x.f).filter(f => f.type === 'delta').map(f => f.text).join('');
  const files = readdirSync(workspace).map(n => ({ name: n, bytes: statSync(join(workspace, n)).size }));
  const cutNotes = frames.map(x => x.f).filter(f => f.type === 'delta' && f.noteReason === 'cut').map(f => f.text.trim());
  const drafts = frames.map(x => x.f).filter(f => f.type === 'agent-tool-draft').map(f => f.draft);
  const writes = drafts.filter(d => d.name === 'write' && d.bytes > 0);
  const draftStats = { frames: drafts.length, writeFrames: writes.length, lastWrite: writes.filter(d => !d.done).at(-1) ? { target: writes.filter(d => !d.done).at(-1).target, bytes: writes.filter(d => !d.done).at(-1).bytes, lines: writes.filter(d => !d.done).at(-1).lines } : null, previewSample: writes.filter(d => !d.done).at(-1)?.preview.slice(-80) ?? null };
  const summary = { name, seconds, cutNotes, draftStats, byType, done, error, files, cards: cards.map(c => ({ name: c.name, state: c.state, target: c.target })), prepareStatuses: prepare.length, firstPrepare: prepare[0] ?? null, lastPrepare: prepare.at(-1) ?? null, text: text.slice(-1200) };
  writeFileSync(join(OUT, `turn-${name}.json`), JSON.stringify({ summary, frames: frames.slice(0, 4000) }, null, 1));
  return { ...summary, frames, workspace };
}

const longPrompt = `Create the file big.txt in the current folder with the write tool, in ONE call. It must hold exactly ${LINES} lines. Line N is "N: " followed by one short true sentence about a different city of the world (a new city on every line, no repeats, no pattern). Write every line out yourself, do not use a script or a loop, and do not shorten it.`;

if (want('raw')) {
  // What the model server itself sends when a reply is cut inside a tool call (no Harness): the stream of tool_calls pieces, then finish_reason "length".
  const body = { model, stream: true, max_tokens: Number(env('E2E_RAW_CAP', '700')), temperature: 0.6,
    chat_template_kwargs: { enable_thinking: false },
    tools: [{ type: 'function', function: { name: 'write', description: 'Write a file.', parameters: { type: 'object', properties: { file_path: { type: 'string' }, content: { type: 'string' } }, required: ['file_path', 'content'] } } }],
    messages: [{ role: 'user', content: longPrompt }] };
  const res = await fetch(`${base}/v1/chat/completions`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  const text = await res.text();
  const events = text.split(String.fromCharCode(10)).filter(l => l.startsWith('data: {')).map(l => JSON.parse(l.slice(6)));
  let args = '', name = '', content = '', finish = null, pieces = 0, ids = new Set();
  for (const e of events) for (const c of e.choices ?? []) {
    for (const tc of c.delta?.tool_calls ?? []) { pieces += 1; if (tc.id) ids.add(tc.id); if (tc.function?.name) name = tc.function.name; args += tc.function?.arguments ?? ''; }
    content += c.delta?.content ?? '';
    if (c.finish_reason) finish = c.finish_reason;
  }
  const usage = events.map(e => e.usage).filter(Boolean).at(-1);
  writeFileSync(join(OUT, 'raw-sse.txt'), text);
  note('raw: a reply cut inside a tool call reaches the client as pieces of the call and finish_reason length', finish === 'length' && name === 'write' && pieces > 5, `finish ${finish}; tool ${name}; ${pieces} call pieces, ${args.length} argument characters, closed JSON ${(() => { try { JSON.parse(args); return true; } catch { return false; } })()}; plain content ${JSON.stringify(content.slice(0, 80))}; ids ${ids.size}; usage ${JSON.stringify(usage)}`);
}

if (want('pagoda')) {
  const t = await turn('pagoda', 'Write me a 100 voxel pagoda in one html file with garden.');
  const html = t.files.find(f => /\.html?$/i.test(f.name));
  console.log(`   pagoda: ${t.seconds.toFixed(0)} s, frames ${JSON.stringify(t.byType)}, done ${JSON.stringify(t.done && { finishReason: t.done.finishReason, tokensOut: t.done.tokensOut })}, files ${JSON.stringify(t.files)}`);
  note('pagoda: a file was written or the turn says why not', !!html || /cut|too long|ran out|limit/i.test(t.text), `file ${html ? html.name + ' ' + html.bytes + ' B' : 'none'}; finish ${t.done?.finishReason}; text tail: ${JSON.stringify(t.text.slice(-240))}`);
}
if (want('long')) {
  const t = await turn('long', longPrompt, { thinking: 'Off' });
  const f = t.files.find(x => x.name === 'big.txt');
  const lines = f ? readFileSync(join(t.workspace, 'big.txt'), 'utf8').split(String.fromCharCode(10)).filter(Boolean).length : 0;
  console.log(`   long: ${t.seconds.toFixed(0)} s, frames ${JSON.stringify(t.byType)}, done ${JSON.stringify(t.done && { finishReason: t.done.finishReason, tokensOut: t.done.tokensOut })}, draft ${JSON.stringify(t.draftStats)}, cut notes ${JSON.stringify(t.cutNotes)}`);
  note(`long: big.txt written (${LINES} lines asked)`, lines >= LINES * 0.95 || t.cutNotes.length > 0 && !!f, `file ${f ? f.bytes + ' B, ' + lines + ' lines' : 'none'}; finish ${t.done?.finishReason}; tokens out ${t.done?.tokensOut}; cards ${JSON.stringify(t.cards.filter(c => c.state !== 'running').map(c => c.name))}; cut notes ${t.cutNotes.length}; draft frames ${t.draftStats.frames}`);
  note('long: the turn did not end silently', !!f || t.cutNotes.length > 0 || t.text.trim().length > 0, `text tail ${JSON.stringify(t.text.slice(-160))}`);
}

if (want('cut')) {
  // A small window told to the Harness (E2E_CUT_WINDOW, default 24000; the engine's own stays WINDOW) makes the reply budget about 7K tokens (raised to
  // 12K by the fix), so the first reply is cut inside the tool call, as the pagoda's was at 19,660.
  await bridge.close();
  bridge = makeBridge(Number(env('E2E_CUT_WINDOW', '24000')), 'agent-cut');
  const t = await turn('cut', longPrompt, { thinking: 'Off' });
  const f = t.files.find(x => x.name === 'big.txt');
  const lines = f ? readFileSync(join(t.workspace, 'big.txt'), 'utf8').split(String.fromCharCode(10)).filter(Boolean).length : 0;
  console.log(`   cut: ${t.seconds.toFixed(0)} s, frames ${JSON.stringify(t.byType)}, done ${JSON.stringify(t.done && { finishReason: t.done.finishReason })}, files ${JSON.stringify(t.files)}, cut notes ${JSON.stringify(t.cutNotes)}, draft ${JSON.stringify(t.draftStats)}`);
  note('cut: the chat said the reply was cut, in words', t.cutNotes.length > 0 && t.cutNotes.every(n => /length limit/.test(n)), JSON.stringify(t.cutNotes));
  note('cut: the agent went on and the file exists', !!f, `file ${f ? f.bytes + ' B, ' + lines + ' lines' : 'none'}; finish ${t.done?.finishReason}`);
}

writeFileSync(join(OUT, 'result.json'), JSON.stringify(rows, null, 2));
console.log('\n' + rows.map(r => `${r.ok ? 'PASS' : 'FAIL'}  ${r.id}`).join('\n'));
await finish(rows.some(r => !r.ok) ? 1 : 0);
