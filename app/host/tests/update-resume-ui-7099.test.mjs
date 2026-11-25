import './helpers/isolate-profile.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';

/**
 * #7099, the second trigger: the chat toolbar's "Pause for update" button (and /pause-for-update) calls the same prepareForUpdate and had no way back.
 * The toolbar button now turns into "Resume agents" while the host is paused; /resume-agents does the same; the host refuses a resume while an update runs.
 */

const src = p => readFileSync(new URL(p, import.meta.url), 'utf8');
const langs = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he'];
const catalogs = Object.fromEntries(langs.map(l => [l, JSON.parse(src(`../src/locales/${l}.json`))]));

test('the agent status says whether the host is paused, so the toolbar can show Resume (also after a reload)', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-a7099-ui-'));
  const b = createAgentBridge({ dataDir: dir, connections: () => [], localInstances: () => [] });
  try {
    assert.equal(b.status().paused, false);
    await b.prepareForUpdate();
    assert.equal(b.status().paused, true);
    await b.resumeAfterUpdate();
    assert.equal(b.status().paused, false);
  } finally { await b.close(); rmSync(dir, { recursive: true, force: true }); }
});

test('a second resume has nothing to lift', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-a7099-ui-'));
  const b = createAgentBridge({ dataDir: dir, connections: () => [], localInstances: () => [] });
  try {
    await b.prepareForUpdate();
    assert.equal(b.status().paused, true);
    assert.deepEqual(await b.resumeAfterUpdate(), { resumed: true });
    assert.deepEqual(await b.resumeAfterUpdate(), { resumed: false }, 'a second resume has nothing to lift');
  } finally { await b.close(); rmSync(dir, { recursive: true, force: true }); }
});

test('the host route resumes only when no "Update Strata" job is running; the page offers it as a button and as /resume-agents', () => {
  const index = src('../server/index.mjs');
  const route = index.slice(index.indexOf("path === '/api/chat/resume-after-update'"));
  assert.match(route.slice(0, 600), /j\.kind === 'update-strata' && j\.state === 'running'\)\) return error\(res, 409/);
  assert.match(route.slice(0, 700), /agent\.resumeAfterUpdate\(\)/);
  const chat = src('../src/screens/Chat.tsx'), bar = src('../src/components/ChatToolbar.tsx');
  assert.match(chat, /api\.post\('\/api\/chat\/resume-after-update'/);
  assert.match(chat, /text === '\/resume-agents'/);
  assert.match(chat, /id: 'resume-agents', label: '\/resume-agents', hint: 'Let the paused agents work again'/);
  assert.match(chat, /onResumeAfterUpdate=\{resumeAfterUpdate\}/);
  assert.match(bar, /\/api\/agent\/status/);
  assert.match(bar, /tr\(paused \? 'Resume agents' : 'Pause for update'\)/);
  assert.match(bar, /if \(paused\) \{ if \(await onResumeAfterUpdate\(\)\) setPaused\(false\); \}/);
});

test('every new sentence has a key and a real translation in all nine other languages', () => {
  const sentences = ['Resume agents', 'Let the paused agents work again', 'Agent work resumed.', 'Could not resume agent work'];
  for (const text of sentences) {
    const id = Object.entries(catalogs.en).find(([, v]) => v === text)?.[0];
    assert.ok(id, `"${text}" has a key`);
    for (const l of langs.slice(1)) assert.ok(catalogs[l][id] && catalogs[l][id] !== text, `${l}: "${text}"`);
  }
});
