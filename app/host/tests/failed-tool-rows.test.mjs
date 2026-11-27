// #7184: a failed tool row stayed on the page after the same call worked later in the turn, and every failed call was a red row the person could not act
// on. The host now marks a failed card `superseded` when the SAME tool with the SAME arguments worked later in the turn, and the page does not draw it;
// "Hide failed steps" (Settings > General) hides every failed card. The owner decides the default; it is one constant (features.ts), and today it is "show".
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';
import { TurnProjector, callSignature } from '../server/agent/events.mjs';

const ev = (type, data, seq = 1, time = 1000) => ({ type: 'event', event: { type, seq, time, data } });
const call = (p, callId, name, args, time) => p.consume(ev('tool/call', { callId, name, arguments: JSON.stringify(args) }, 1, time));
const fail = (p, callId, time, text = 'Error: fetch failed') => p.consume(ev('tool/result', { message: { toolCallId: callId, isError: true, content: [{ type: 'text', text }] } }, 2, time));
const ok = (p, callId, time) => p.consume(ev('tool/result', { message: { toolCallId: callId, content: [{ type: 'text', text: 'the page' }] } }, 2, time));
const cards = frames => frames.filter(f => f.type === 'agent-tool').map(f => f.tool);

test('the same call that failed and then worked leaves no failed row behind', () => {
  const p = new TurnProjector({ startedAt: 0 });
  call(p, 'a', 'web_fetch', { url: 'https://example.org/x' }, 1000);
  const failed = cards(fail(p, 'a', 1100));
  assert.equal(failed[0].state, 'error');
  assert.ok(!failed[0].superseded);
  call(p, 'b', 'web_fetch', { url: 'https://example.org/x' }, 1200);
  const after = cards(ok(p, 'b', 1300));
  const second = after.find(c => c.id === 'b'), first = after.find(c => c.id === 'a');
  assert.equal(second.state, 'done');
  assert.equal(first.state, 'error', 'the card keeps its truth: it did fail');
  assert.equal(first.superseded, true, 'and is marked, so the page does not draw it');
});

test('the order of the argument keys does not matter, and a different argument does', () => {
  assert.equal(callSignature({ name: 'grep', args: { pattern: 'x', path: 'a' } }), callSignature({ name: 'grep', args: { path: 'a', pattern: 'x' } }));
  assert.notEqual(callSignature({ name: 'grep', args: { pattern: 'x' } }), callSignature({ name: 'grep', args: { pattern: 'y' } }));
  assert.notEqual(callSignature({ name: 'grep', args: {} }), callSignature({ name: 'glob', args: {} }));
  const p = new TurnProjector({ startedAt: 0 });
  call(p, 'a', 'web_fetch', { url: 'https://example.org/x' }, 1000); fail(p, 'a', 1100);
  call(p, 'b', 'web_fetch', { url: 'https://example.org/other' }, 1200);
  assert.ok(!cards(ok(p, 'b', 1300)).some(c => c.id === 'a'), 'another address is another step: the failed row stays');
  call(p, 'c', 'web_search', { url: 'https://example.org/x' }, 1400);
  assert.ok(!cards(ok(p, 'c', 1500)).some(c => c.id === 'a'), 'another tool with the same words is another step');
});

test('a failed call is not marked by a later call that also failed, nor by one that is denied, nor is a call that worked marked', () => {
  const p = new TurnProjector({ startedAt: 0 });
  call(p, 'a', 'bash', { command: 'make' }, 1000); fail(p, 'a', 1100, 'Error: exit 2');
  call(p, 'b', 'bash', { command: 'make' }, 1200);
  assert.ok(!cards(fail(p, 'b', 1300, 'Error: exit 2')).some(c => c.superseded));
  call(p, 'c', 'bash', { command: 'make' }, 1400);
  const done = cards(ok(p, 'c', 1500));
  assert.deepEqual(done.filter(c => c.superseded).map(c => c.id).sort(), ['a', 'b'], 'both earlier failures of the identical call are marked');
  assert.ok(!done.find(c => c.id === 'c').superseded);
});

test('a card is marked once; a success that came BEFORE the failure marks nothing', () => {
  const p = new TurnProjector({ startedAt: 0 });
  call(p, 'a', 'read', { file_path: '/w/a.txt' }, 1000); ok(p, 'a', 1100);
  call(p, 'b', 'read', { file_path: '/w/a.txt' }, 1200); const failedLater = cards(fail(p, 'b', 1300));
  assert.equal(failedLater.length, 1);
  assert.ok(!failedLater[0].superseded, 'a call that worked first does not make a later failure untrue');
  call(p, 'c', 'read', { file_path: '/w/a.txt' }, 1400); const first = cards(ok(p, 'c', 1500));
  assert.equal(first.filter(c => c.superseded).length, 1);
  call(p, 'd', 'read', { file_path: '/w/a.txt' }, 1600);
  assert.equal(cards(ok(p, 'd', 1700)).filter(c => c.superseded).length, 0, 'no second frame for a card that is already marked');
});

const { code } = await transform(await readFile(new URL('../src/lib/tool-cards.ts', import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' });
const { shownToolCard } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

test('the page: a superseded failure is never drawn; "Hide failed steps" hides every failure; questions and refusals stay', () => {
  const failed = { state: 'error' }, old = { state: 'error', superseded: true }, ask = { state: 'proposed' }, denied = { state: 'denied' }, done = { state: 'done' };
  for (const grouped of [false, true]) {
    assert.equal(shownToolCard(failed, grouped, false), true, 'by default a failed step is shown');
    assert.equal(shownToolCard(old, grouped, false), false);
    assert.equal(shownToolCard(failed, grouped, true), false);
    assert.equal(shownToolCard(old, grouped, true), false);
    for (const t of [ask, denied]) { assert.equal(shownToolCard(t, grouped, false), true); assert.equal(shownToolCard(t, grouped, true), true); }
  }
  assert.equal(shownToolCard(done, false, true), true);
  assert.equal(shownToolCard(done, true, true), false, 'a reply with activity lines keeps only the cards that need the person');
});

test('the setting: default "show" is one constant; the choice is saved only once made; a toggle and its words exist in every language', async () => {
  const features = await readFile(new URL('../src/lib/features.ts', import.meta.url), 'utf8');
  assert.match(features, /export const HIDE_FAILED_TOOLS_BY_DEFAULT = false;/);
  const store = await readFile(new URL('../src/lib/store.ts', import.meta.url), 'utf8');
  assert.match(store, /hideFailedTools: prefs\.hideFailedTools \?\? HIDE_FAILED_TOOLS_BY_DEFAULT,/);
  assert.match(store, /hideFailedTools: hideFailedToolsChosen \? state\.hideFailedTools : undefined,/);
  assert.match(store, /export function setHideFailedTools\(hideFailedTools: boolean\) \{\n  hideFailedToolsChosen = true;/);
  const chat = await readFile(new URL('../src/screens/Chat.tsx', import.meta.url), 'utf8');
  assert.match(chat, /shownToolCard\(t, grouped, hideFailedTools\)/);
  const settings = await readFile(new URL('../src/screens/Settings.tsx', import.meta.url), 'utf8');
  assert.match(settings, /label: 'Hide failed steps'[^\n]*<HideFailedToolsSetting \/>/);
  for (const lang of ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he']) {
    const catalog = JSON.parse(await readFile(new URL(`../src/locales/${lang}.json`, import.meta.url), 'utf8'));
    assert.ok(catalog.hideFailedToolsLabel && catalog.hideFailedToolsHint, `${lang} has both strings`);
  }
  const en = JSON.parse(await readFile(new URL('../src/locales/en.json', import.meta.url), 'utf8'));
  assert.ok(settings.includes(`hint: '${en.hideFailedToolsHint}'`), 'the row uses the catalog sentence word for word');
});
