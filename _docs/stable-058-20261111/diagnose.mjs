/** Private synthetic Harness request/lifecycle diagnostics; no live app profile is opened. */
import { mkdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { createAgentBridge } from '../../app/host/server/agent/agent-bridge.mjs';
import { startScriptedWorker } from '../../app/host/tests/helpers/scripted-worker.mjs';

const output = 'H:/Strata/releases/stable-058-20261111-codex';
const attempt = process.argv[2] ?? 'r1';
if (!/^r[1-9][0-9]*$/.test(attempt)) throw new Error('Explicit diagnosis attempt required');
const dir = join(output, 'diagnose-' + attempt);
mkdirSync(dir);
const ws = join(dir, 'workspace'); mkdirSync(ws);
const worker = await startScriptedWorker({ window: 32768, charsPerToken: 3 });
const logs = [];
const bridge = createAgentBridge({ dataDir: dir, localInstances: () => [{ id: 'i', label: 'S', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 2 }], connections: () => [],
  env: { ...process.env, STRATA_HARNESS_DIR: resolve(import.meta.dirname, '../../apps/desktop/src-tauri/resources/harness') }, startTimeoutMs: 90_000, log: line => logs.push(line) });
const outputRows = [];
try {
  for (let i = 0; i < 3; i++) {
    const frames = [];
    const result = await bridge.runTurn({ chatId: 'c', message: `SCRIPT:echo r${i} ${'q=0x1f ok;'.repeat(700)}`, workspacePath: ws, mode: 'bypass', onFrame: f => frames.push(f) });
    outputRows.push({ i, result, running: bridge.isRunning('c'), frames });
    console.log(JSON.stringify({ i, result, running: bridge.isRunning('c'), answer: frames.filter(f => f.type === 'delta').map(f => f.text.slice(0, 100)).join(''), terminal: frames.at(-1) }));
  }
  const frames = [];
  await bridge.compactCommand({ chatId: 'c', workspacePath: ws, mode: 'bypass', choice: null, onFrame: f => frames.push(f) });
  outputRows.push({ compact: frames });
  console.log(JSON.stringify({ compact: frames }));
} finally {
  writeFileSync(join(dir, 'requests.json'), JSON.stringify(worker.requests, null, 2));
  writeFileSync(join(dir, 'frames.json'), JSON.stringify(outputRows, null, 2));
  writeFileSync(join(dir, 'logs.json'), JSON.stringify(logs, null, 2));
  console.log(JSON.stringify(worker.requests.map(row => { const body = JSON.parse(row.raw); return { directive: row.directive, summary: row.summary, refused: row.refused, promptChars: row.promptChars, messages: body.messages.map(m => ({ role: m.role, chars: JSON.stringify(m.content).length, text: JSON.stringify(m.content).slice(0, 160) })) }; })));
  await bridge.close(); await worker.close();
}
