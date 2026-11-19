import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { setTimeout as delay } from 'node:timers/promises';
import { build } from 'esbuild';
import { localSilenceLimit, createAgentBridge } from '../server/agent/agent-bridge.mjs';

test('local silence configuration is opt-in, integral and bounded to one day', () => {
  for (const fallback of [1_800_000, 8000]) {
    for (const value of [undefined, null, '']) assert.equal(localSilenceLimit(value, fallback), fallback);
    for (const value of ['1', '10000', '3600000', '86400000']) assert.equal(localSilenceLimit(value, fallback), Number(value));
    for (const value of ['0', '-1', 'Infinity', 'NaN', '1.5', '86400001', '10ms', ' 1000 ', '1e3', '9007199254740993']) {
      assert.throws(() => localSilenceLimit(value, fallback), /STRATA_LOCAL_STALL_MS/);
    }
  }
});

test('invalid bridge configuration fails before writing agent state', t => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-local-limit-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  assert.throws(() => createAgentBridge({ dataDir: dir, env: { STRATA_LOCAL_STALL_MS: 'bad' }, localInstances: () => [], connections: () => [] }), /STRATA_LOCAL_STALL_MS/);
  assert.equal(existsSync(join(dir, 'agent')), false);
});

// Exercise runTurn's real interval and cancellation path without an installed
// Harness or model. Only the Harness process and gateway transport are replaced.
const bridgePath = fileURLToPath(new URL('../server/agent/agent-bridge.mjs', import.meta.url));
const stubs = {
  'harness-runtime.mjs': `export const resolveHarness = () => ({ok:true, dir:'fixture', version:'fixture'});
    export const stopStaleAgent = () => null; export const redact = text => text;
    export class HarnessProcess { running = false; async start() { this.running = true; return {}; } async stop() { this.running = false; } }`,
  'gateway-client.mjs': `export class GatewayClient {
    constructor() { this.f = globalThis.__silenceFixture; }
    async connect() {} close() {}
    openStream(name, args, options) {
      if (name === 'session/follow') {
        this.f.follow = options.onItem;
        queueMicrotask(() => options.onItem({type:'snapshot',cursor:0,records:[]}));
      }
      return {close(){}};
    }
    async rpc(name, args) {
      this.f.calls.push(name);
      if (name === 'session/create') return {sessionId:'fixture-session'};
      if (name === 'session/projections') return {values:{permissions:{currentValue:'strata-bypass'}}};
      if (name === 'session/prompt') {
        this.f.follow({type:'event',event:{seq:1,type:'user/message',data:{source:{rpcId:args.request.requestId}}}});
        if (this.f.tool) this.f.follow({type:'event',event:{seq:2,type:'tool/call',data:{callId:'tool-1',name:'read',arguments:{file_path:'fixture'}}}});
        this.f.prompted = true;
      }
      return {};
    }
  }`,
};
const result = await build({
  entryPoints: [bridgePath], bundle: true, write: false, format: 'esm', platform: 'node',
  define: { 'import.meta.url': JSON.stringify(pathToFileURL(bridgePath).href) },
  plugins: [{ name: 'fixture-transport', setup(b) {
    b.onResolve({ filter: /^\.\// }, args => {
      const name = args.path.slice(2);
      return name in stubs ? { path: name, namespace: 'fixture' } : { path: pathToFileURL(resolve(args.resolveDir, args.path)).href, external: true };
    });
    b.onLoad({ filter: /.*/, namespace: 'fixture' }, args => ({ contents: stubs[args.path] }));
  } }],
});
const bridgeModule = await import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);

for (const scenario of [
  { name: 'unset preserves the thirty-minute default', env: {}, before: 1_799_999, after: 1_800_001 },
  { name: 'empty preserves an explicit local fallback', env: { STRATA_LOCAL_STALL_MS: '' }, localStallMs: 8000, before: 7999, after: 8001 },
  { name: 'override replaces the explicit local fallback', env: { STRATA_LOCAL_STALL_MS: '10000' }, localStallMs: 2500, before: 9999, after: 10001 },
  { name: 'override does not change remote silence', env: { STRATA_LOCAL_STALL_MS: '10000' }, remote: true, before: 499, after: 501 },
  { name: 'override does not change running-tool silence', env: { STRATA_LOCAL_STALL_MS: '10000' }, tool: true, before: 5999, after: 6001 },
]) {
  test(`real watchdog: ${scenario.name}`, { timeout: 10_000 }, async t => {
    // Production deliberately unrefs its watchdog. The fixture has no gateway
    // socket, so retain an event-loop handle until the real interval fires.
    const keepAlive = setTimeout(() => {}, 8000);
    t.after(() => clearTimeout(keepAlive));
    const dir = mkdtempSync(join(tmpdir(), 'strata-watchdog-limit-'));
    const fixture = { calls: [], prompted: false, tool: scenario.tool };
    globalThis.__silenceFixture = fixture;
    let now = 1_000_000;
    t.mock.method(Date, 'now', () => now);
    const bridge = bridgeModule.createAgentBridge({ dataDir: dir, env: { ...process.env, STRATA_LOCAL_STALL_MS: undefined, ...scenario.env },
      localInstances: () => scenario.remote ? [] : [{ id: 'local', modelId: 'fixture', baseUrl: 'http://127.0.0.1:1/v1', context: 32768 }],
      connections: () => scenario.remote ? [{ id: 'remote', name: 'fixture', protocol: 'openai-chat', baseUrl: 'https://fixture.invalid/v1', models: [{ id: 'fixture', name: 'fixture' }] }] : [],
      stallMs: 500, stallToolMs: 6000, ...(scenario.localStallMs == null ? {} : { localStallMs: scenario.localStallMs }),
      engineGraceMs: 0,
    });
    t.after(async () => { await bridge.close(); rmSync(dir, { recursive: true, force: true }); delete globalThis.__silenceFixture; });
    const frames = [];
    const pending = bridge.runTurn({ chatId: 'test', message: 'quiet', mode: 'bypass',
      choice: scenario.remote ? { connectionId: 'remote', modelId: 'fixture' } : null, onFrame: f => frames.push(f) });
    // Observe rejection immediately rather than leaving a failed startup unhandled.
    let failure; pending.catch(e => { failure = e; });
    for (let tries = 0; !fixture.prompted && !failure && tries < 100; tries++) await delay(5);
    if (failure) throw failure;
    assert.equal(fixture.prompted, true, 'turn reached the actual watchdog');
    if (scenario.tool) assert.ok(frames.some(f => f.type === 'agent-tool' && f.tool.state === 'running'), 'running-tool branch is active');
    now = 1_000_000 + scenario.before;
    await delay(250);
    assert.equal(fixture.calls.includes('session/cancel'), false, 'no early cancellation');
    now = 1_000_000 + scenario.after;
    await pending;
    assert.equal(fixture.calls.filter(name => name === 'session/cancel').length, 1);
    assert.equal(frames.at(-1).code, 'AGENT_STOPPED');
    assert.match(frames.at(-1).message, /stopped answering/);
    assert.equal(bridge.status().activeTurns, 0);
  });
}
