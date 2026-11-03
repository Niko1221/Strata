// Strata Dev (docs/DEV_BUILD.md): the backend hook system (server/dev-hooks.mjs), its log buffer and routes (dev-log.mjs, dev-api.mjs),
// the agent plugin row and the dev.config.json overrides (agent/profile.mjs, dev-mode.mjs), and the Windows port check (ports.mjs).
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { EventEmitter } from 'node:events';
import { createDevHooks, pythonEvents, withTimeout } from '../server/dev-hooks.mjs';
import * as devLog from '../server/dev-log.mjs';
import { debugReport, engineEntries, mergedLogs, requestRows } from '../server/dev-api.mjs';
import { activeDevSettings, devChatMode, devSettings, devToolsSettings, setActiveDevSettings } from '../server/dev-mode.mjs';
import { buildProfile } from '../server/agent/profile.mjs';
import { parseExcludedPortRanges, reservedPortMessage, windowsReservedRange } from '../server/ports.mjs';
import { apply as applyToolHooks } from '../server/agent/plugins/strata-dev-hooks.mjs';

const dirWith = files => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-dev-hooks-'));
  for (const [name, text] of Object.entries(files)) writeFileSync(join(dir, name), text);
  return dir;
};

/** A response stand-in: write/end like node:http's, collecting what was written. */
function fakeResponse() {
  const res = new EventEmitter();
  res.out = '';
  res.write = chunk => { res.out += String(chunk); return true; };
  res.end = chunk => { if (chunk) res.out += String(chunk); res.ended = true; return res; };
  res.writeHead = (status, headers) => { res.status = status; res.headers = headers; };
  return res;
}

test('hooks: before_request can replace the body, a throwing hook is switched off and the rest still runs', async () => {
  devLog.resetDevLog(); devLog.enableDevLog();
  const dir = dirWith({
    'a-change.mjs': `export function before_request(body) { return { ...body, message: body.message + ' (changed)' }; }`,
    'b-throws.mjs': `export function before_request() { throw new Error('broken on purpose'); }`,
    'c-sees.mjs': `export const seen = []; export function before_request(body) { globalThis.__devHookSeen = body.message; }`,
    '_skipped.mjs': `export function before_request() { throw new Error('never loaded'); }`,
    'notes.txt': 'not a hook',
  });
  try {
    const hooks = createDevHooks({ dir });
    const list = await hooks.load();
    assert.deepEqual(list.map(h => h.file), ['a-change.mjs', 'b-throws.mjs', 'c-sees.mjs']);
    const out = await hooks.run('before_request', { message: 'hi' });
    assert.equal(out.message, 'hi (changed)');
    assert.equal(globalThis.__devHookSeen, 'hi (changed)', 'the hook after the broken one still ran');
    const broken = hooks.list().find(h => h.file === 'b-throws.mjs');
    assert.equal(broken.enabled, false);
    assert.match(broken.error, /broken on purpose/);
    assert.ok(devLog.entries({ sources: ['hooks'] }).some(e => e.level === 'error' && /b-throws\.mjs switched off/.test(e.line)));
    // Switched on again from the panel, it runs (and fails) once more.
    assert.equal(hooks.setEnabled('b-throws.mjs', true), true);
    assert.equal(hooks.list().find(h => h.file === 'b-throws.mjs').error, null);
    assert.equal(hooks.setEnabled('missing.mjs', true), false);
  } finally { rmSync(dir, { recursive: true, force: true }); delete globalThis.__devHookSeen; }
});

test('hooks: one that takes too long is switched off after its time limit; a file that does not load is off with its error', async () => {
  devLog.resetDevLog(); devLog.enableDevLog();
  const dir = dirWith({
    'slow.mjs': `export function after_response() { return new Promise(r => setTimeout(r, 2000)); }`,
    'syntax.mjs': `export function before_request( {`,
  });
  try {
    const hooks = createDevHooks({ dir, settings: { timeoutMs: 60 } });
    await hooks.load();
    assert.equal(hooks.list().find(h => h.file === 'syntax.mjs').enabled, false);
    assert.match(hooks.list().find(h => h.file === 'syntax.mjs').error, /^load:/);
    const started = Date.now();
    await hooks.run('after_response', { text: 'x' });
    assert.ok(Date.now() - started < 1500, 'the request did not wait for the slow hook');
    const slow = hooks.list().find(h => h.file === 'slow.mjs');
    assert.equal(slow.enabled, false);
    assert.match(slow.error, /took longer than 60 ms/);
    assert.equal(slow.stats.after_response.calls, 1);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('hooks: config "disabled" keeps a file off; Reload picks up an edited file', async () => {
  const dir = dirWith({ 'one.mjs': `export function before_request(b) { return { ...b, v: 1 }; }`, 'two.mjs': `export function before_request(b) { return { ...b, two: true }; }` });
  try {
    const hooks = createDevHooks({ dir, settings: { disabled: ['two.mjs'] } });
    await hooks.load();
    assert.deepEqual(await hooks.run('before_request', {}), { v: 1 });
    writeFileSync(join(dir, 'one.mjs'), `export function before_request(b) { return { ...b, v: 2 }; }`);
    await hooks.load();
    assert.deepEqual(await hooks.run('before_request', {}), { v: 2 });
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('the chat stream: on_token may change a frame, after_response sees the answer, the trace links the request id', async () => {
  const dir = dirWith({
    'upper.mjs': `export function on_token(frame) { return frame.type === 'delta' ? { ...frame, text: frame.text.toUpperCase() } : frame; }
                  export function after_response(r) { globalThis.__devHookAnswer = r; }`,
  });
  try {
    const hooks = createDevHooks({ dir });
    await hooks.load();
    const res = fakeResponse();
    const body = { message: 'hi', sessionId: 's1' };
    const trace = await hooks.chatStream({}, res, body);
    res.write(`${JSON.stringify({ type: 'start', requestId: 'req-1' })}\n`);
    res.write(`${JSON.stringify({ type: 'delta', text: 'hello ' })}\n${JSON.stringify({ type: 'delta', text: 'world' })}\n`);
    res.write('partial line without newline passes as is');
    res.end();
    await new Promise(r => setTimeout(r, 20));
    assert.match(res.out, /"text":"HELLO "/);
    assert.match(res.out, /partial line without newline passes as is$/);
    assert.equal(trace.requestId, 'req-1');
    assert.equal(globalThis.__devHookAnswer.text, 'HELLO WORLD');
    assert.equal(globalThis.__devHookAnswer.sessionId, 's1');
    assert.ok(trace.hooks.find(h => h.event === 'on_token').calls >= 2);
    const rows = requestRows({ requests: [{ id: 'req-1', at: new Date().toISOString(), endpoint: '/chat/completions', state: 'done', durationMs: 12, tokensIn: 3, tokensOut: 2 }], traces: hooks.traces() });
    assert.equal(rows.length, 1);
    assert.equal(rows[0].tokensOut, 2);
    assert.ok(rows[0].hooks.length >= 2);
  } finally { rmSync(dir, { recursive: true, force: true }); delete globalThis.__devHookAnswer; }
});

test('the chat stream: an on_token hook over its budget is switched off and the frame goes out unchanged', async () => {
  const dir = dirWith({ 'busy.mjs': `export function on_token(f) { const end = Date.now() + 30; while (Date.now() < end) {} return { ...f, text: 'X' }; }` });
  try {
    const hooks = createDevHooks({ dir, settings: { tokenBudgetMs: 5 } });
    await hooks.load();
    const res = fakeResponse();
    await hooks.chatStream({}, res, { message: 'm' });
    res.write(`${JSON.stringify({ type: 'delta', text: 'a' })}\n`);
    res.write(`${JSON.stringify({ type: 'delta', text: 'b' })}\n`);
    assert.equal(hooks.list()[0].enabled, false);
    assert.match(hooks.list()[0].error, /budget 5 ms/);
    assert.match(res.out, /"text":"a"/);
    assert.match(res.out, /"text":"b"/);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('on_model_load runs once per model that becomes ready', async () => {
  const dir = dirWith({ 'm.mjs': `export function on_model_load(m) { (globalThis.__devModels ??= []).push(m.id); }` });
  try {
    const hooks = createDevHooks({ dir });
    await hooks.load();
    hooks.observeModels({ instances: [{ id: 'a', status: 'loading' }] });
    hooks.observeModels({ instances: [{ id: 'a', status: 'ready', port: 1 }] });
    hooks.observeModels({ instances: [{ id: 'a', status: 'ready', port: 1 }, { id: 'b', status: 'ready' }] });
    hooks.observeModels({ instances: [] });
    hooks.observeModels({ instances: [{ id: 'a', status: 'ready' }] });
    await new Promise(r => setTimeout(r, 20));
    assert.deepEqual(globalThis.__devModels, ['a', 'b', 'a']);
  } finally { rmSync(dir, { recursive: true, force: true }); delete globalThis.__devModels; }
});

test('routes: /dev/<name>/<path> answers JSON, text or a status; a failing route is a 500, an unknown one falls through', async () => {
  const dir = dirWith({ 'api.mjs': `export const name = 'tools';
    export const routes = { 'GET /ping': ({ query }) => ({ pong: query.x }), 'POST /echo': ({ body }) => ({ status: 201, json: body }), 'GET /text': () => 'plain', 'GET /fail': () => { throw new Error('nope'); }, 'bad key': () => 1 };` });
  try {
    const hooks = createDevHooks({ dir });
    await hooks.load();
    assert.deepEqual(hooks.list()[0].routes, ['GET /dev/tools/ping', 'POST /dev/tools/echo', 'GET /dev/tools/text', 'GET /dev/tools/fail']);
    const call = async (method, path, body) => {
      const res = fakeResponse();
      const url = new URL(`http://x${path}`);
      const handled = await hooks.route({ headers: {} }, res, url.pathname, method, url, async () => body);
      return { handled, res };
    };
    let r = await call('GET', '/dev/tools/ping?x=1');
    assert.equal(r.handled, true); assert.equal(r.res.status, 200); assert.deepEqual(JSON.parse(r.res.out), { pong: '1' });
    r = await call('POST', '/dev/tools/echo', { a: 1 });
    assert.equal(r.res.status, 201); assert.deepEqual(JSON.parse(r.res.out), { a: 1 });
    r = await call('GET', '/dev/tools/text');
    assert.equal(r.res.out, 'plain');
    r = await call('GET', '/dev/tools/fail');
    assert.equal(r.res.status, 500); assert.match(r.res.out, /nope/);
    assert.equal((await call('GET', '/dev/tools/none')).handled, false);
    assert.equal((await call('GET', '/dev/other/ping')).handled, false);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('Python hooks: events from the header line, JSON in and out, a failure switches the hook off', async (t) => {
  assert.deepEqual(pythonEvents('# strata-hooks: before_request, on_token, on_tool_call\nprint(1)'), ['before_request', 'on_tool_call']);
  assert.deepEqual(pythonEvents('import json'), ['before_request', 'after_response']);
  const python = process.platform === 'win32' ? 'python' : 'python3';
  if (spawnSync(python, ['--version']).status !== 0) { t.skip('no Python on PATH'); return; }
  const dir = dirWith({
    'tag.py': `# strata-hooks: before_request\nimport json, sys\nd = json.load(sys.stdin)\np = d["payload"]\np["tag"] = "py"\nprint(json.dumps({"payload": p}))\n`,
    'bad.py': `# strata-hooks: before_request\nprint("not json")\n`,
  });
  try {
    const hooks = createDevHooks({ dir, python: { command: python, args: [] }, settings: { timeoutMs: 15000 } });
    await hooks.load();
    const out = await hooks.run('before_request', { message: 'm' });
    assert.equal(out.tag, 'py');
    const bad = hooks.list().find(h => h.file === 'bad.py');
    assert.equal(bad.enabled, false);
    assert.match(bad.error, /not JSON/);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('on_tool_call: the agent plugin refuses a call a hook denies and lets the others through; a broken hook never blocks', async () => {
  const dir = dirWith({
    'deny.mjs': `export function on_tool_call(call) { if (call.name === 'bash' && /rm -rf/.test(call.arguments.command)) return { deny: 'not this' }; }`,
    'broken.mjs': `export function on_tool_call() { throw new Error('x'); }`,
  });
  const quiet = console.error, quietLog = console.log;
  console.error = () => {}; console.log = () => {};
  try {
    let handler;
    applyToolHooks({ on: (event, fn) => { assert.equal(event, 'tools/pre-execute'); handler = fn; } }, { dir, timeoutMs: 500 });
    const next = async () => ({ kind: 'allow' });
    assert.deepEqual(await handler({ name: 'bash', arguments: { command: 'ls' } }, next), { kind: 'allow' });
    const refused = await handler({ name: 'bash', arguments: { command: 'rm -rf /' } }, next);
    assert.equal(refused.kind, 'deny');
    assert.match(refused.reason, /not this \(Strata Dev hook deny\.mjs\)/);
    // No dir in the config (outside Dev the row is never written): the plugin does nothing.
    let registered = false;
    applyToolHooks({ on: () => { registered = true; } }, {});
    assert.equal(registered, false);
  } finally { console.error = quiet; console.log = quietLog; rmSync(dir, { recursive: true, force: true }); }
});

test('withTimeout never rejects', async () => {
  assert.deepEqual(await withTimeout(() => 5, 50), { value: 5 });
  assert.match((await withTimeout(() => { throw new Error('e'); }, 50)).error.message, /e/);
  assert.equal((await withTimeout(() => new Promise(() => {}), 20)).error.code, 'timeout');
});

test('the log buffer: off by default (Live), per source, filterable, with the last error and its detail', () => {
  devLog.resetDevLog();
  devLog.push('host', 'ignored while off');
  assert.deepEqual(devLog.entries(), []);
  devLog.enableDevLog({ verbose: true });
  assert.equal(devLog.devLogVerbose(), true);
  devLog.push('host', 'one\r\ntwo\n\n');
  devLog.push('ui', 'Boom\n  at x\n  at y', 'error');
  devLog.push('harness', 'agent line');
  assert.deepEqual(devLog.entries({ sources: ['host'] }).map(e => e.line), ['one', 'two']);
  assert.equal(devLog.lastErrorEntry().line, 'Boom');
  assert.match(devLog.lastErrorEntry().detail, /at x\n {2}at y/);
  const engine = engineEntries({ supervisorLogs: [{ at: '2000-01-01T00:00:00.000Z', stream: 'stderr', line: 'engine said' }], nativeLogs: [{ at: '2000-01-01T00:00:01.000Z', instanceId: 'i1', line: 'Traceback here' }] });
  const merged = mergedLogs({ engine });
  assert.equal(merged[0].line, 'engine said', 'merged by time');
  assert.equal(merged[1].level, 'error');
  assert.deepEqual(mergedLogs({ sources: ['harness'], engine }).map(e => e.line), ['agent line']);
  assert.deepEqual(mergedLogs({ q: 'TWO', engine }).map(e => e.line), ['two']);
  const report = debugReport({ summary: { configFile: 'c.json', warnings: [], verbose: true }, hooks: [], requests: [{ at: 'now', endpoint: '/x', error: 'bad', durationMs: 3 }], logs: merged });
  assert.match(report, /## Last error\n.*Boom\n {2}at x/);
  assert.match(report, /\/x\s+3 ms: bad/);
  devLog.resetDevLog();
});

test('dev.config.json settings: wrong values are left out, tokens must be custom properties with a plain value', () => {
  const s = devSettings({
    logging: { verbose: true },
    tokens: { '--st-accent': ' #e0218a ', color: 'red', '--st-bad': 'x;}body{', '--st-num': 3 },
    ui: { elements: { 'sidebar.monitor': { hidden: true, label: '', order: 'x' }, 'x': 3 }, extensions: { a: { enabled: false }, b: {} } },
    hooks: { timeoutMs: 5, tokenBudgetMs: 20, disabled: ['a.mjs', 3] },
    overrides: { webSearch: { enabled: false, provider: 'google', withoutAsking: true }, loop: { maxRuns: 0 }, folder: { enabled: false, defaultPath: ' C:\\work ' }, permissions: { defaultMode: 'bypass', allowedModes: ['plan', 'nope'] } },
  });
  assert.equal(s.verbose, true);
  assert.deepEqual(s.tokens, { '--st-accent': '#e0218a' });
  assert.deepEqual(s.ui, { elements: { 'sidebar.monitor': { hidden: true } }, extensions: { a: { enabled: false } } });
  assert.deepEqual(s.hooks, { timeoutMs: 2000, tokenBudgetMs: 20, disabled: ['a.mjs'] });
  assert.deepEqual(s.overrides.webSearch, { enabled: false, withoutAsking: true });
  assert.deepEqual(s.overrides.loop, {});
  assert.deepEqual(s.overrides.folder, { enabled: false, defaultPath: 'C:\\work' });
  assert.deepEqual(s.overrides.permissions, { allowedModes: ['plan'] }, 'a default mode outside the allowed ones is dropped');
  assert.deepEqual(devSettings({}).overrides, { webSearch: {}, loop: {}, folder: {}, permissions: {} });
});

test('the chat mode: unchanged outside Dev; in Dev a mode that is not allowed becomes the default', () => {
  assert.equal(devChatMode('bypass', null), 'bypass');
  assert.equal(devChatMode(null, null), null);
  const dev = devSettings({ overrides: { permissions: { defaultMode: 'plan', allowedModes: ['manual', 'plan'] } } });
  assert.equal(devChatMode('bypass', dev), 'plan');
  assert.equal(devChatMode('manual', dev), 'manual');
  assert.equal(devChatMode(null, dev), 'plan');
});

test('the agent profile: outside Dev no dev plugin and no override; in Dev the on_tool_call plugin, the web search override and the default mode', () => {
  const local = [{ id: 'a', modelId: 'm', baseUrl: 'http://127.0.0.1:9/v1', context: 32768 }];
  setActiveDevSettings(null);
  assert.equal(activeDevSettings(), null);
  const live = buildProfile({ local });
  assert.doesNotMatch(live.patch, /strata-dev-hooks/);
  assert.match(live.patch, /"defaultPreset": "strata-manual"/);
  assert.match(live.patch, /strata-web-search/);
  try {
    setActiveDevSettings({ ...devSettings({ overrides: { webSearch: { enabled: false }, permissions: { defaultMode: 'plan' } } }), backendDir: 'C:\\dev\\extensions\\backend', python: { command: 'python', args: [] } });
    const dev = buildProfile({ local });
    assert.match(dev.patch, /"id": "strata-dev-hooks"/);
    assert.match(dev.patch, /strata-dev-hooks\.mjs/);
    assert.match(dev.patch, /"defaultPreset": "strata-plan"/);
    assert.doesNotMatch(dev.patch, /strata-web-search/, 'web search switched off by the override');
    assert.ok(dev.patch.indexOf('"strata-gate"') < dev.patch.indexOf('"strata-dev-hooks"'), 'the dev hooks come after the gate');
  } finally { setActiveDevSettings(null); }
  const tools = { off: new Set(['search']), search: { provider: 'builtin', key: '' }, webWithoutAsking: false };
  const on = devToolsSettings(tools, devSettings({ overrides: { webSearch: { enabled: true, provider: 'exa', withoutAsking: true } } }));
  assert.equal(on.off.has('search'), false);
  assert.equal(on.search.provider, 'exa');
  assert.equal(on.webWithoutAsking, true);
  assert.equal(tools.off.has('search'), true, 'the saved settings are not changed');
  assert.equal(devToolsSettings(tools, devSettings({})), tools);
});

test('Windows reserved port ranges: parsed from netsh, named in the Dev message, nothing on other systems', () => {
  const text = `\r\nProtocol tcp Port Exclusion Ranges\r\n\r\nStart Port    End Port\r\n----------    --------\r\n      5357        5357\r\n      8780        8879\r\n     50000       50059     *\r\n\r\n* - Administered port exclusions.\r\n`;
  const ranges = parseExcludedPortRanges(text);
  assert.deepEqual(ranges, [{ start: 5357, end: 5357, admin: false }, { start: 8780, end: 8879, admin: false }, { start: 50000, end: 50059, admin: true }]);
  const run = () => ({ status: 0, stdout: text });
  assert.deepEqual(windowsReservedRange(8792, { platform: 'win32', run }), { start: 8780, end: 8879, admin: false });
  assert.equal(windowsReservedRange(8081, { platform: 'win32', run }), null);
  assert.equal(windowsReservedRange(8792, { platform: 'linux', run }), null);
  assert.equal(windowsReservedRange(8792, { platform: 'win32', run: () => ({ status: 1, stdout: '' }) }), null);
  assert.match(reservedPortMessage(8792, { start: 8780, end: 8879 }), /Port 8792 ist von Windows reserviert \(Bereich 8780-8879/);
});
