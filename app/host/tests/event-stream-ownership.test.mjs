/** Controlled EventSource callbacks exercise the production subscription, including closed sources. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {build} from 'esbuild';

const compiled = await build({entryPoints: ['src/lib/client.ts'], bundle: true, write: false, format: 'esm', platform: 'node'});
const {onEvent} = await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);

function fixture(t) {
  const sources = [], statuses = [], frames = [];
  class Source {
    listeners = new Map();
    closed = false;
    constructor(url) { this.url = url; sources.push(this); }
    addEventListener(name, fn) { this.listeners.set(name, fn); }
    close() { this.closed = true; }
    // Deliberately deliver already queued callbacks even after close().
    emit(name, data = {}) { this.listeners.get(name)?.({data: JSON.stringify(data)}); }
  }
  t.mock.method(globalThis, 'setInterval', fn => { fixture.tick = fn; return 1; });
  t.mock.method(globalThis, 'clearInterval', () => {});
  t.mock.method(Date, 'now', () => fixture.now);
  const previous = globalThis.EventSource;
  globalThis.EventSource = Source;
  fixture.now = 0;
  const stop = onEvent(frame => frames.push(frame), status => statuses.push(status));
  t.after(() => { stop(); globalThis.EventSource = previous; });
  return {sources, statuses, frames, stop, advance(ms) { fixture.now += ms; fixture.tick(); }};
}

test('old error after replacement open cannot mark the current stream offline', t => {
  const f = fixture(t);
  f.sources[0].emit('open');
  f.advance(10001);
  assert.equal(f.sources[0].closed, true);
  f.sources[1].emit('open');
  f.sources[0].emit('error');
  assert.equal(f.statuses.at(-1), 'live');
});

test('closed-source frames cannot overwrite data or keep a silent replacement alive', t => {
  const f = fixture(t);
  f.advance(10001);
  fixture.now += 9000;
  f.sources[0].emit('monitor', {phase: 'error'});
  f.sources[0].emit('message', {stale: true});
  assert.deepEqual(f.frames, []);
  f.advance(1001);
  assert.equal(f.sources.length, 3, 'replacement silence still expires at the original limit');
});

test('unsubscribe revokes every callback and a queued watchdog tick', t => {
  const f = fixture(t);
  f.stop();
  for (const name of ['open', 'error', 'monitor', 'message']) f.sources[0].emit(name);
  f.advance(20000);
  assert.deepEqual(f.statuses, []);
  assert.deepEqual(f.frames, []);
  assert.equal(f.sources.length, 1);
});

test('current source still delivers named/default frames and reports real errors', t => {
  const f = fixture(t), source = f.sources[0];
  source.emit('open'); source.emit('native', {available: true}); source.emit('message', {ok: true}); source.emit('error');
  assert.deepEqual(f.statuses, ['live', 'offline']);
  assert.deepEqual(f.frames.map(frame => frame.event), ['native', 'message']);
});
