// Real Host, Harness, tool executor, private files and next-turn skill loading.
// Only model responses are scripted; no model-quality claim follows.
import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { mkdtemp, mkdir, writeFile, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { host } from './helpers/reference-host.mjs';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';

const skip = !process.env.STRATA_HARNESS_DIR || !resolveHarness({ env: process.env }).ok ? 'Built Harness is required' : false;
const until = async fn => { for (let i = 0; i < 150; i++) { const value = await fn(); if (value) return value; await delay(200); } throw new Error('Learning integration condition timed out'); };

test('completed real tools → bounded tool-free review → private skill → later Harness use; local controls and undo persist', { skip, timeout: 150000 }, async t => {
  const worker = await startScriptedWorker({ chunkDelayMs: 5 });
  const reviews = [];
  const adapter = http.createServer(async (req, res) => {
    if (req.url.endsWith('/models')) { res.setHeader('content-type', 'application/json'); res.end(JSON.stringify({ data: [{ id: 'scripted-model', name: 'Fixture model' }] })); return; }
    if (req.url === '/props') { res.setHeader('content-type', 'application/json'); res.end('{}'); return; }
    const parts = []; for await (const part of req) parts.push(part);
    const body = JSON.parse(Buffer.concat(parts).toString());
    const prompt = body.messages?.filter(row => row.role === 'user').at(-1)?.content ?? '';
    if (typeof prompt === 'string' && prompt.includes('Observed workflow:\n')) {
      reviews.push(body);
      const packet = JSON.parse(prompt.split('Observed workflow:\n')[1]);
      const value = { learn: true, name: 'fixture-validation', title: 'Fixture validation', description: 'Use when checking the validation fixture.',
        triggers: ['fixture validation', 'check validation fixture'], steps: packet.tools.slice(0, 3).map((tool, i) => ({ toolId: tool.id, note: ['Inspect the first source.', 'Inspect the second source.', 'Check the remaining fixture inputs.'][i] })), pitfalls: ['Check current inputs before repeating the procedure.'] };
      res.writeHead(200, { 'content-type': 'text/event-stream' });
      res.end('data: ' + JSON.stringify({ choices: [{ index: 0, delta: { content: JSON.stringify(value) }, finish_reason: 'stop' }] }) + '\n\ndata: [DONE]\n\n');
      return;
    }
    const response = await fetch(worker.url + '/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
    res.writeHead(response.status, { 'content-type': 'text/event-stream' });
    for await (const chunk of response.body) res.write(chunk);
    res.end();
  });
  await new Promise(resolve => adapter.listen(0, '127.0.0.1', resolve));
  const home = await mkdtemp(join(tmpdir(), 'strata-learning-api-'));
  const workspace = join(home, 'work'); await mkdir(workspace);
  for (const name of ['first.txt', 'second.txt', 'third.txt']) await writeFile(join(workspace, name), 'PRIVATE_FILE_TEXT_MUST_NOT_ENTER_REVIEW\n');
  let h = await host(home, { env: { STRATA_HARNESS_DIR: process.env.STRATA_HARNESS_DIR, STRATA_AGENT: 'on' } });
  t.after(async () => {
    await fetch(h.base + '/api/app/quit', { method: 'POST', headers: { origin: h.base, 'x-strata-client': 'learning-test' } }).catch(() => {});
    await h.stop(); adapter.closeAllConnections(); await new Promise(resolve => adapter.close(resolve)); await worker.close();
    // Keep the isolated artifact directory for diagnostic evidence.
  });
  const request = async (path, method = 'GET', body, origin = h.base) => {
    const response = await fetch(h.base + path, { method, headers: { origin, 'content-type': 'application/json' }, ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
    return { status: response.status, data: await response.json() };
  };
  assert.equal((await request('/api/skills/learning')).data.enabled, false);
  assert.equal((await request('/api/skills/learning', 'PUT', { enabled: true }, 'https://other.example')).status, 403);
  assert.equal((await request('/api/skills/learning', 'GET', undefined, 'https://other.example')).status, 403);
  assert.equal((await request('/api/skills/learning', 'PUT', { enabled: true })).status, 200);
  const saved = await request('/api/model-providers', 'POST', { kind: 'custom', name: 'Learning fixture', baseUrl: `http://127.0.0.1:${adapter.address().port}/v1`, protocol: 'openai-chat', models: [{ id: 'scripted-model', name: 'Fixture model', vision: false, reasoning: false, contextMax: 32768 }] });
  assert.equal(saved.status, 200);
  const remoteModel = { connectionId: saved.data.connections.find(row => row.name === 'Learning fixture').id, modelId: 'scripted-model' };
  const run = async (message, sessionId = 'learning-fixture', workspaceId) => {
    const response = await fetch(h.base + '/api/chat/stream', { method: 'POST', headers: { origin: h.base, 'content-type': 'application/json' }, body: JSON.stringify({ sessionId, message, mode: 'bypass', remoteModel, workspaceId }) });
    assert.equal(response.status, 200);
    const text = await response.text();
    return text.trim().split('\n').filter(Boolean).map(line => JSON.parse(line));
  };
  const frames = await run('Perform fixture validation. SCRIPT:multi ' + JSON.stringify({ calls: ['first.txt', 'second.txt', 'third.txt'].map(file => ['read', { file_path: file }]), say: 'Fixture inputs checked.' }));
  assert.equal(frames.at(-1).type, 'done');
  assert.equal(new Set(frames.filter(frame => frame.type === 'agent-tool' && frame.tool.state === 'done').map(frame => frame.tool.id)).size, 3);
  await until(async () => (await request('/api/skills/learning')).data.last?.phase === 'saved');
  assert.equal(reviews.length, 1); assert.equal(reviews[0].tools, undefined);
  assert.equal(reviews[0].model, remoteModel.modelId);
  assert.equal(JSON.stringify(reviews[0]).includes('PRIVATE_FILE_TEXT_MUST_NOT_ENTER_REVIEW'), false);
  const file = join(home, 'data/skills/learned-fixture-validation/SKILL.md');
  assert.match(await readFile(file, 'utf8'), /Inspect the first source/);
  const list = (await request('/api/skills')).data, learned = list.find(row => row.id === 'my-learned-fixture-validation');
  assert.equal(learned.learning.revision, 1); assert.equal(learned.enabled, true);
  await until(async () => {
    const next = await run('SCRIPT:call skill {"name":"learned-fixture-validation"}');
    return next.some(frame => frame.type === 'agent-tool' && frame.tool.name === 'skill' && frame.tool.state === 'done' && /Inspect the first source/.test(frame.tool.result));
  });
  assert.equal(reviews.length, 1, 'Loading one saved skill is not another eligible workflow');
  assert.equal((await request('/api/skills/learning', 'PUT', { autoApply: true })).data.autoApply, true);
  const automatic = await run('Perform fixture validation. SCRIPT:echo Applied automatically.', 'automatic-fixture');
  assert.ok(automatic.some(frame => frame.type === 'skills-used' && frame.skills.some(skill => skill.name === 'Fixture validation')));
  assert.equal(automatic.some(frame => frame.type === 'agent-tool' && frame.tool.name === 'skill'), false);
  assert.match(worker.requests.at(-1).raw, /Inspect the first source/);
  const otherFolder = join(home, 'other-work'); await mkdir(otherFolder);
  const scope = (await request('/api/workspaces', 'POST', { path: otherFolder })).data.id;
  await request('/api/skills', 'POST', { name: 'Scoped check', body: 'SCOPED_WORKFLOW_MARKER', enabled: true, onlyIn: scope, triggers: ['scoped validation'] });
  await run('scoped validation SCRIPT:echo Wrong workspace.', 'scope-outside');
  assert.doesNotMatch(worker.requests.at(-1).raw, /SCOPED_WORKFLOW_MARKER/);
  const scoped = await run('scoped validation SCRIPT:echo Correct workspace.', 'scope-inside', scope);
  assert.ok(scoped.some(frame => frame.type === 'skills-used' && frame.skills.some(skill => skill.name === 'Scoped check')));
  assert.match(worker.requests.at(-1).raw, /SCOPED_WORKFLOW_MARKER/);
  const goalFrames = await run('/goal fixture validation SCRIPT:goalstep', 'auto-goal');
  assert.ok(goalFrames.some(frame => frame.type === 'skills-used' && frame.skills.some(skill => skill.name === 'Fixture validation')));
  assert.ok(goalFrames.filter(frame => frame.type === 'goal' && frame.goal).every(frame => !frame.goal.objective.includes('strata-workflow-skills')));
  assert.equal((await request('/api/skills/learning/undo', 'POST', {})).status, 200);
  assert.equal((await request('/api/skills')).data.find(row => row.id === learned.id).enabled, false);
  const afterUndo = await run('fixture validation SCRIPT:echo Disabled skill stays off.', 'auto-disabled');
  assert.equal(afterUndo.some(frame => frame.type === 'skills-used' && frame.skills.some(skill => skill.name === 'Fixture validation')), false);
  await fetch(h.base + '/api/app/quit', { method: 'POST', headers: { origin: h.base, 'x-strata-client': 'learning-test' } }).catch(() => {});
  await h.stop();
  h = await host(home);
  assert.equal((await request('/api/skills/learning')).data.enabled, true);
  assert.equal((await request('/api/skills/learning')).data.autoApply, true);
  assert.equal((await request('/api/skills')).data.find(row => row.id === learned.id).enabled, false);
});
