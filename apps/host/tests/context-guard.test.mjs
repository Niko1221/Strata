// Strata Dev "contextGuard" (dev.config.json): early summaries so a local chat never fills up. Off: everything as Live.
import test from 'node:test';
import assert from 'node:assert/strict';
import { contextGuardOf, devSettings, loadDevConfig, setActiveDevSettings } from '../server/dev-mode.mjs';
import { guardCompactAfterTurn, guardSummarize } from '../server/agent/context-fit.mjs';
import { buildProfile, compactionHeadroom, localOutputBudget } from '../server/agent/profile.mjs';

const W = 65536;
const ON = { enabled: true, midTaskAt: 0.55, compactAfterTurnAt: 0.55, compactBeforeSendAt: 0.60 };
const local = [{ id: 'a', modelId: 'm', baseUrl: 'http://x', context: W }];
const thresholdOf = (w, midTaskAt) => { const out = localOutputBudget(w); return Math.floor(Math.min(w * 0.8, w - out - compactionHeadroom(w, out, midTaskAt))); };
const headroomIn = profile => Number(/"headroomTokens": (\d+)/.exec(profile.patch)[1]);

test('switch off or absent: headroom, threshold and profile are unchanged', () => {
  for (const raw of [undefined, null, false, { enabled: false }, { enabled: false, midTaskAt: 0.5 }, {}]) assert.equal(contextGuardOf(raw).settings, null);
  const out = localOutputBudget(W);
  assert.equal(compactionHeadroom(W, out), 5734);
  assert.equal(compactionHeadroom(W, out, null), compactionHeadroom(W, out));
  assert.equal(thresholdOf(W, null), 40142);
  setActiveDevSettings(null);
  const live = buildProfile({ local, connections: [] });
  setActiveDevSettings(devSettings({ contextGuard: { enabled: false } }));
  assert.equal(buildProfile({ local, connections: [] }).patch, live.patch);
  setActiveDevSettings(null);
});

test('switch on, 64K window: the Harness threshold is about 55 percent', () => {
  setActiveDevSettings(devSettings({ contextGuard: ON }));
  try {
    const head = headroomIn(buildProfile({ local, connections: [] }));
    assert.equal(head, compactionHeadroom(W, localOutputBudget(W), 0.55));
    const threshold = W - localOutputBudget(W) - head;
    assert.ok(Math.abs(threshold / W - 0.55) < 0.001, `threshold ${threshold}`);
    assert.equal(Math.round(100 * thresholdOf(W, 0.55) / W), 55);
  } finally { setActiveDevSettings(null); }
});

test('a short message: 85 percent full is summarised first, 40 percent is sent', () => {
  const threshold = thresholdOf(W, 0.55);
  const ask = pct => guardSummarize({ window: W, pressure: Math.round(W * pct), messageTokens: 50, threshold, beforeSendAt: 0.6 });
  assert.equal(ask(0.85), true);
  assert.equal(ask(0.40), false);
  assert.equal(ask(0.60), true);
  assert.equal(guardSummarize({ window: W, pressure: 100, messageTokens: 50, threshold, beforeSendAt: 0.6 }), false, 'an almost empty chat has nothing to summarise');
  // below 60 percent, but the message and the reply room would carry it past the threshold
  assert.equal(guardSummarize({ window: W, pressure: threshold - 100, messageTokens: 900, threshold, beforeSendAt: 0.99 }), true);
});

test('end of a turn: 75 percent compacts once, 40 percent does not', () => {
  let calls = 0;
  for (const pct of [0.75, 0.40]) if (guardCompactAfterTurn({ window: W, tokens: Math.round(W * pct), afterTurnAt: 0.55 })) calls++;
  assert.equal(calls, 1);
  assert.equal(guardCompactAfterTurn({ window: 0, tokens: 10, afterTurnAt: 0.55 }), false);
});

test('a broken value in dev.config.json: the guard is off, a warning, no throw', () => {
  for (const bad of [{ ...ON, midTaskAt: 0.95 }, { ...ON, compactAfterTurnAt: 0.1 }, { ...ON, compactBeforeSendAt: '0.6' }, { enabled: true }, { ...ON, midTaskAt: NaN }, 'yes', [1], { enabled: 'true' }]) {
    const r = contextGuardOf(bad);
    assert.equal(r.settings, null, JSON.stringify(bad));
    assert.equal(r.warnings.length, 1);
    assert.match(r.warnings[0], /contextGuard/);
  }
  const loaded = loadDevConfig({ file: 'x.json', read: () => JSON.stringify({ contextGuard: { ...ON, midTaskAt: 2 } }) });
  assert.match(loaded.warnings.join(' '), /contextGuard\.midTaskAt/);
  assert.equal(devSettings(loaded.config).contextGuard, null);
  assert.deepEqual(contextGuardOf(ON).settings, ON);
});

test('Settings page: saving writes only contextGuard, checks every value first, keeps the rest of the file', async () => {
  const { saveContextGuard, contextGuardView } = await import('../server/dev-mode.mjs');
  const files = { f: JSON.stringify({ version: 1, badge: { label: 'X' }, hooks: { timeoutMs: 100 } }) };
  const io = { read: f => { if (!(f in files)) throw Object.assign(new Error('no'), { code: 'ENOENT' }); return files[f]; }, write: (f, t) => { files[f] = t; }, rename: (a, b) => { files[b] = files[a]; delete files[a]; } };
  const saved = saveContextGuard('f', { ...ON, midTaskAt: 0.5 }, io);
  assert.deepEqual(saved, { ...ON, midTaskAt: 0.5 });
  const after = JSON.parse(files.f);
  assert.deepEqual(after.contextGuard, saved); assert.equal(after.badge.label, 'X'); assert.equal(after.hooks.timeoutMs, 100);
  const before = files.f;
  for (const bad of [{ ...ON, midTaskAt: 0.95 }, { ...ON, compactAfterTurnAt: 'x' }, { midTaskAt: 0.5 }, { ...ON, enabled: 'yes' }, null]) assert.throws(() => saveContextGuard('f', bad, io));
  assert.equal(files.f, before, 'a refused value changes nothing');
  assert.throws(() => saveContextGuard('f', ON, { ...io, read: () => '{ broken' }), /fix it by hand/);
  assert.deepEqual(contextGuardView({ enabled: false, midTaskAt: 0.5 }), { enabled: false, midTaskAt: 0.5, compactAfterTurnAt: 0.55, compactBeforeSendAt: 0.6 });
  assert.equal(contextGuardView(undefined).enabled, false);
});
