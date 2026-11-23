import test from 'node:test';
import assert from 'node:assert/strict';
import { engineProgressMark, engineKeepsTurnAlive, ENGINE_QUIET_MS } from '../server/agent/engine-progress.mjs';

const inst = live => ({ metrics: { live } });
const MIN = 60_000;

test('a model that is thinking, with no word shown, still moves the mark (#6794)', () => {
  const a = engineProgressMark(inst({ state: 'generating', phase: 'thinking', prompt_read: null, generated: 100 }));
  const b = engineProgressMark(inst({ state: 'generating', phase: 'thinking', prompt_read: null, generated: 101 }));
  assert.ok(a && b && a !== b);
  assert.equal(engineProgressMark(inst({ state: 'idle' })), null);
  assert.equal(engineProgressMark({ metrics: null }), null);
});

test('the engine keeps the turn alive while the mark moves, even long after the Harness last spoke (#6794)', () => {
  const t0 = 1_000_000;
  const turn = { heard: t0, lastActivity: t0, engineMark: null };
  // 60 minutes of silent thinking: every look the count is higher
  for (let m = 1; m <= 60; m++) {
    const now = t0 + m * MIN;
    assert.equal(engineKeepsTurnAlive(turn, `generating|${m}`, { now, graceMs: 2 * 60 * MIN }), true, `minute ${m}`);
    assert.equal(turn.lastActivity, now);
  }
});

test('a busy engine whose mark has not moved yet (one slow prompt chunk) keeps the turn alive for the quiet allowance, then no more (#6794)', () => {
  const t0 = 5_000_000;
  const turn = { heard: t0, lastActivity: t0, engineMark: null };
  const opts = { graceMs: 2 * 60 * MIN };
  assert.equal(engineKeepsTurnAlive(turn, 'reading|0', { ...opts, now: t0 + MIN }), true);
  const last = turn.lastActivity;
  assert.equal(engineKeepsTurnAlive(turn, 'reading|0', { ...opts, now: t0 + 10 * MIN }), true);      // 9 minutes quiet: a slow chunk
  assert.equal(turn.lastActivity, last, 'a quiet look is not new activity');
  assert.equal(engineKeepsTurnAlive(turn, 'reading|0', { ...opts, now: t0 + MIN + ENGINE_QUIET_MS + 1 }), false);   // 20+ minutes quiet: not counted any more
});

test('an idle or silent engine counts for nothing, and so does one once the Harness has been silent past the grace (#3623)', () => {
  const t0 = 9_000_000;
  const turn = { heard: t0, lastActivity: t0, engineMark: 'reading|5' };
  assert.equal(engineKeepsTurnAlive(turn, null, { now: t0 + MIN, graceMs: 30 * MIN }), false);
  assert.equal(engineKeepsTurnAlive(turn, 'reading|6', { now: t0 + 31 * MIN, graceMs: 30 * MIN }), false);
  assert.equal(turn.engineMark, 'reading|5', 'nothing is recorded outside the grace');
});
