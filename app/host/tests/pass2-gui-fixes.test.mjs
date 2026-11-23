/** Pass 2 GUI regression tests: #6680 Hy4-Flash reasoning_effort clamp, #6668 QuestionCard usability, #6661 shutdown chat-save drain. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import net from 'node:net';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { buildProfile, connectionReasoningEfforts, effortOf, HY4_LOCAL_EFFORTS, INKLING_LOCAL_EFFORTS, LOCAL_EFFORTS } from '../server/agent/profile.mjs';
import { clampHy4Effort, shapeBody } from '../server/agent/llm-shape.mjs';
import { parseQuestions, buildAnswer } from '../server/agent/user-questions.mjs';
import { createChatStore, drainChatStore, realIo } from '../server/chat-store.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
const delay = ms => new Promise(r => setTimeout(r, ms));
async function freePort() {
  const s = net.createServer();
  await new Promise(r => s.listen(0, '127.0.0.1', r));
  const p = s.address().port;
  await new Promise(r => s.close(r));
  return p;
}
const bundleTs = async rel => {
  const built = await build({
    entryPoints: [fileURLToPath(new URL(rel, import.meta.url))],
    bundle: true, write: false, platform: 'node', format: 'esm', loader: { '.json': 'json' },
  });
  return import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
};

test('#6680 Hy4-Flash effort clamping never emits xhigh and Inkling-2-Mini thinks like the other local models (0.2.21)', () => {
  assert.equal(clampHy4Effort('xhigh'), 'max');
  assert.equal(clampHy4Effort('max'), 'max');
  assert.equal(clampHy4Effort('high'), 'high');
  assert.equal(clampHy4Effort('medium'), 'high');
  assert.equal(clampHy4Effort('low'), 'high');
  assert.equal(clampHy4Effort('none'), 'no_think');
  assert.equal(clampHy4Effort('off'), 'no_think');
  assert.equal(effortOf('xhigh'), 'max');

  const shapedTop = shapeBody({ model: 'hy4-flash-sq-64', reasoning_effort: 'xhigh', messages: [] }, { nativeAgent: true });
  assert.equal(shapedTop.reasoning_effort, 'max');

  const shapedCtk = shapeBody({ model: 'hy4-flash-sq-64', chat_template_kwargs: { enable_thinking: true, reasoning_effort: 'xhigh' }, messages: [] });
  assert.equal(shapedCtk.chat_template_kwargs.reasoning_effort, 'max');

  const shapedRenamed = shapeBody({ model: 'my-custom-hy4', chat_template_kwargs: { enable_thinking: true, reasoning_effort: 'xhigh' }, messages: [] }, { hy4: true });
  assert.equal(shapedRenamed.chat_template_kwargs.reasoning_effort, 'max');

  const built = buildProfile({
    local: [
      { id: 'h-renamed', modelId: 'custom-sq-64', arch: 'hy4flash', label: 'My Hy4', baseUrl: 'http://127.0.0.1:9101/v1', context: 131072 },
      { id: 'ink-renamed', modelId: 'custom-ink', arch: 'inkling2_loop', label: 'My Inkling', baseUrl: 'http://127.0.0.1:9102/v1', context: 32768 },
    ],
    connections: [],
  });
  const rows = JSON.parse(built.patch.slice(built.patch.indexOf('[')));
  const providers = rows.find(e => e.id === 'llm-pi-ai').config.providers;
  assert.deepEqual(providers['strata-local'].models[0].reasoningEfforts, HY4_LOCAL_EFFORTS);
  assert.equal(providers['strata-local-ink-renamed'].models[0].reasoning, true);
  assert.deepEqual(providers['strata-local-ink-renamed'].models[0].reasoningEfforts, INKLING_LOCAL_EFFORTS);
  assert.equal(INKLING_LOCAL_EFFORTS.max, 'max');
  assert.equal(INKLING_LOCAL_EFFORTS.off, null);
  assert.equal(providers['strata-local-ink-renamed'].compat.thinkingFormat, 'chat-template');

  assert.deepEqual(connectionReasoningEfforts({ protocol: 'openai-chat', reasoningFormat: 'strata' }, { id: 'hy4-flash-sq-64', reasoning: true }), HY4_LOCAL_EFFORTS);
  assert.equal(connectionReasoningEfforts({ protocol: 'openai-chat', reasoningFormat: 'strata' }, { id: 'inkling-2-mini-sq-20', reasoning: true }), LOCAL_EFFORTS);
});

test('#6668 QuestionCard data layer supports >4 questions, string/numeric multiSelect, duplicate ids/labels, and combined option+note', async () => {
  const { readAsk, answerComplete, answeredCount, toAnswers } = await bundleTs('../src/lib/agent-ask.ts');
  const six = Array.from({ length: 6 }, (_, i) => ({
    id: `q${i + 1}`,
    question: `Question ${i + 1}?`,
    multiSelect: i % 2 === 1 ? 'true' : false,
    options: [{ label: 'Alpha' }, { label: 'Beta' }, { label: 'Alpha' }],
  }));
  const parsed = parseQuestions(six);
  assert.equal(parsed.length, 6, 'supports >4 (up to 6) questions cleanly');
  assert.equal(parsed[1].multiSelect, true, 'string "true" multiSelect normalized');
  assert.deepEqual(parsed[0].options.map(o => o.label), ['Alpha', 'Beta'], 'duplicate option labels deduplicated');

  const ask = readAsk({ callId: 'call-6', questions: [...parsed, { id: 'q1', question: 'Duplicate id?', multiSelect: 1, options: [{ label: 'Yes' }] }] });
  assert.equal(ask.questions.length, 7);
  assert.equal(ask.questions[6].id, 'q1-2', 'duplicate question ids deduplicated');
  assert.equal(ask.questions[6].multiSelect, true);

  const draft = { q1: { selected: ['Alpha'], custom: 'with extra note' } };
  assert.equal(answeredCount(parsed, draft), 1);
  assert.equal(answerComplete(parsed, draft), false);
  assert.deepEqual(toAnswers([parsed[0]], draft), [{ id: 'q1', selected: ['Alpha'], custom: 'with extra note' }]);
  assert.deepEqual(buildAnswer([parsed[0]], [{ id: 'q1', selected: ['Alpha'], custom: 'with extra note' }]), {
    answers: [{ id: 'q1', selected: ['Alpha'], custom: 'with extra note' }],
  });

  const cardSrc = readFileSync(new URL('../src/components/QuestionCard.tsx', import.meta.url), 'utf8');
  assert.match(cardSrc, /stopSession\(sessionId\)/, 'QuestionCard has a real Stop action');
  assert.match(cardSrc, /Select all that apply/, 'multi-select helper text present');
  assert.match(cardSrc, /askq__indicator--radio/, 'radio affordance present');
  assert.match(cardSrc, /askq__indicator--check/, 'checkbox affordance present');
  const chatSrc = readFileSync(new URL('../src/screens/Chat.tsx', import.meta.url), 'utf8');
  assert.match(chatSrc, /data-awaiting-answer=\{awaitingQuestion \|\| undefined\}/, 'main composer dims/disables while a QuestionCard is active');
});

test('#6661 drainChatStore waits for queued and in-flight chat saves and bounds hangs', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'strata-drain-unit-'));
  try {
    let writeDone = false;
    const store = createChatStore({
      dir,
      io: {
        ...realIo,
        async writeFile(path, text) {
          await delay(120);
          await realIo.writeFile(path, text);
          if (path.includes('s_slow.json')) writeDone = true;
        },
      },
    });
    const inFlight = new Set();
    const p = (async () => {
      await delay(40);
      await store.save('s_slow', JSON.stringify({ id: 's_slow', title: 'Slow', messages: [{ id: 'm1', role: 'user', content: 'saved before exit' }] }));
    })();
    inFlight.add(p);
    p.finally(() => inFlight.delete(p));

    const res = await drainChatStore(store, { inFlight, timeoutMs: 2000 });
    assert.equal(res.drained, true);
    assert.equal(writeDone, true);
    const saved = JSON.parse(await store.read('s_slow'));
    assert.equal(saved.session.messages[0].content, 'saved before exit');

    // Bounded timeout: a stuck save never hangs shutdown forever
    const stuck = createChatStore({
      dir,
      io: { ...realIo, writeFile: () => new Promise(() => {}) },
    });
    void stuck.save('s_hung', JSON.stringify({ id: 's_hung', title: 'Hung', messages: [] })).catch(() => {});
    const t0 = Date.now();
    const resTimeout = await drainChatStore(stuck, { timeoutMs: 120 });
    assert.equal(resTimeout.drained, false);
    assert.ok(Date.now() - t0 < 1000, 'drain respected bounded timeout');
  } finally {
    await rm(dir, { recursive: true, force: true }).catch(() => {});
  }
});

test('#6661 real host shutdown drains an in-flight chat save before exiting', { timeout: 30000 }, async () => {
  const home = await mkdtemp(join(tmpdir(), 'strata-drain-host-'));
  const port = await freePort();
  const token = 'a'.repeat(40);
  const env = {
    ...process.env,
    STRATA_HOME: home,
    STRATA_MODELS_DIR: join(home, 'models'),
    STRATA_PORT: String(port),
    STRATA_SHARE_PORT: String(await freePort()),
    STRATA_TUNNEL_PORT: String(await freePort()),
    STRATA_REAL_ENGINE: '1',
    STRATA_SHUTDOWN_TOKEN: token,
    STRATA_SHUTDOWN_CHAT_DRAIN_MS: '4000',
    STRATA_SETTINGS_FILE: join(home, 'app-settings.json'),
    STRATA_COMPONENTS_DIR: join(home, 'components'),
    LOCALAPPDATA: join(home, 'Local'),
    APPDATA: join(home, 'Roaming'),
    XDG_DATA_HOME: join(home, 'xdg-data'),
    XDG_STATE_HOME: join(home, 'xdg-state'),
    XDG_CONFIG_HOME: join(home, 'xdg-config'),
  };
  const host = spawn(process.execPath, ['server/index.mjs'], { cwd: root, stdio: 'ignore', env, windowsHide: true });
  const exited = new Promise(r => host.once('exit', (code, signal) => r({ code, signal })));
  try {
    const base = `http://127.0.0.1:${port}`;
    let up = false;
    for (let i = 0; i < 150 && !up; i++) {
      try { up = (await fetch(`${base}/api/health`)).ok; } catch { await delay(100); }
    }
    assert.ok(up, 'host started');

    // Start a large (~4 MB) chat save over a streaming HTTP request, wait briefly so the host
    // has accepted it into its in-flight chat-save set, then trigger clean shutdown before finishing the body.
    const { default: http } = await import('node:http');
    const payload = JSON.stringify({
      id: 's_drain',
      title: 'Drain test',
      createdAt: '2026-11-22T12:00:00.000Z',
      updatedAt: '2026-11-22T12:00:01.000Z',
      messages: [{ id: 'm1', role: 'assistant', content: 'Z'.repeat(4_000_000) }],
    });
    const half = Math.floor(payload.length / 2);
    const savePromise = new Promise((resolveSave, rejectSave) => {
      const req = http.request({
        host: '127.0.0.1',
        port,
        path: '/api/chats/s_drain',
        method: 'PUT',
        headers: {
          'x-strata-client': 'ui',
          'x-strata-window': 'w1',
          'content-type': 'application/json',
          'content-length': Buffer.byteLength(payload),
        },
      }, res => {
        let body = '';
        res.on('data', c => { body += c; });
        res.on('end', () => resolveSave({ status: res.statusCode, body }));
      });
      req.on('error', rejectSave);
      req.write(payload.slice(0, half));
      // Hold the second half for 80 ms so /api/control/shutdown arrives mid-upload.
      setTimeout(() => req.end(payload.slice(half)), 80);
    });
    await delay(30);
    const stopRes = await fetch(`${base}/api/control/shutdown`, {
      method: 'POST',
      headers: { 'x-strata-shutdown': token },
    });
    const saveRes = await savePromise;
    assert.equal(stopRes.status, 202);
    assert.equal(saveRes.status, 200, 'in-flight chat save completed cleanly during shutdown');
    const exitInfo = await exited;
    assert.equal(exitInfo.code, 0, 'host exited cleanly');

    const disk = JSON.parse(await readFile(join(home, 'data', 'chats', 's_drain.json'), 'utf8'));
    assert.equal(disk.rev, 1);
    assert.equal(disk.session.id, 's_drain');
    assert.equal(disk.session.messages[0].content.length, 4_000_000);
  } finally {
    if (host.exitCode === null) { host.kill('SIGKILL'); await exited; }
    await rm(home, { recursive: true, force: true }).catch(() => {});
  }
});
