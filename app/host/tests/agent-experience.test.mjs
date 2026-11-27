import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import {
  apply, caseFromSuccess, caseStore, commandKeys, errorLines, failedShell, name, noteText, recall, STORE_FILE, toolCalls,
} from '../server/agent/plugins/strata-experience.mjs';
import { buildProfile } from '../server/agent/profile.mjs';

const T0 = Date.UTC(2026, 10, 24);
const event = (type, seq, data) => ({ type, seq, time: T0 + seq * 1000, data });
const call = (seq, id, tool, args) => event('tool/call', seq, { callId: id, name: tool, arguments: JSON.stringify(args) });
const result = (seq, id, text, isError = false) => event('tool/result', seq, { message: { toolCallId: id, isError, content: [{ type: 'text', text }] } });
const FAIL = 'Traceback (most recent call last):\n  File "D:\\work\\tools\\check_conf.py", line 12, in <module>\nKeyError: setting \'owner\' missing in D:\\work\\conf\\app.json\n[exit code: 1]';

/** One failure of check_conf.py, a read and an edit of the config, nothing else yet. */
const failedThenEdited = [
  call(1, 'r0', 'read', { file_path: 'D:\\work\\conf\\app.json' }),
  result(2, 'r0', '{"name":"x"}'),
  call(3, 's1', 'pwsh', { command: 'python tools\\check_conf.py --strict' }),
  result(4, 's1', FAIL),
  call(5, 'e1', 'edit', { file_path: 'D:\\work\\conf\\app.json', old_string: 'x', new_string: 'y' }),
  result(6, 'e1', 'ok'),
];

test('command keys name the script or tool, never the interpreter or the whole line', () => {
  assert.deepEqual([...commandKeys('& "C:\\Python\\python.exe" F:\\repo\\tools\\check_odr.py --all')], ['check_odr.py']);
  assert.deepEqual([...commandKeys('python -m pytest tests/unit -q')], ['python -m pytest']);
  assert.deepEqual([...commandKeys('pytest tests/unit -q')], ['pytest']);
  assert.deepEqual([...commandKeys('cmake --build build --target app')], ['cmake --build']);
  assert.equal(commandKeys('Get-ChildItem').size, 0);
  assert.equal(commandKeys(undefined).size, 0);
});

test('error lines drop what changes between runs and keep the cause', () => {
  const lines = errorLines(FAIL);
  assert.equal(lines[0], "keyerror: setting 'owner' missing in app.json");
  assert(lines.includes('traceback (most recent call last)'));
  assert(!lines.some(line => /d:|\\|exit code/u.test(line)));
  assert.deepEqual(errorLines('step 1 of 3 failed after 1532 ms'), ['step <n> of <n> failed after <n> ms']);
});

test('a shell result fails on an error result, a non-zero exit or a kill', () => {
  assert.equal(failedShell(false, 'done\n[exit code: 2]'), true);
  assert.equal(failedShell(false, 'done\n[killed by signal: SIGTERM]'), true);
  assert.equal(failedShell(true, 'anything'), true);
  assert.equal(failedShell(false, 'done'), false);
  assert.equal(failedShell(false, 'the log says [exit code: 1] somewhere'), false);
});

test('a success after a fix closes the earliest failure of the same command and names the changed files', () => {
  const calls = toolCalls([...failedThenEdited, call(7, 's2', 'pwsh', { command: 'python tools/check_conf.py' }), result(8, 's2', FAIL)]);
  const found = caseFromSuccess(calls, commandKeys('python tools/check_conf.py'), { session: 'chat-a', now: T0 + 9000 });
  assert.equal(found.schema, 'strata.experience.case.v1');
  assert.deepEqual(found.commands, ['check_conf.py']);
  assert.deepEqual(found.fix, { files: ['conf/app.json'], failures: 2, toolCalls: 3 });
  assert.deepEqual(found.workingSet, ['conf/app.json']);
  assert.deepEqual(found.errorPaths, ['conf/app.json'], 'only lines that report an error name its files');
  const relative = toolCalls([...failedThenEdited.slice(0, 4), call(5, 'e1', 'write', { file_path: 'conf\\app.json', content: 'y' }), result(6, 'e1', 'ok')]);
  assert.deepEqual(caseFromSuccess(relative, commandKeys('check_conf.py'), { session: 'chat-a', workspace: 'D:\\work' }).fix.files, ['conf/app.json'],
    'a relative path is resolved against the chat folder');
  assert.equal(found.failedAt, new Date(T0 + 3000).toISOString());
  assert.doesNotMatch(JSON.stringify(found), /--strict|"name":"x"|D:\\\\work/u, 'no raw command line, file content or full path');

  const noEdit = toolCalls(failedThenEdited.slice(0, 4));
  assert.equal(caseFromSuccess(noEdit, commandKeys('check_conf.py'), { session: 'chat-a' }), null, 'a retry that just worked teaches nothing');
  const otherCommand = caseFromSuccess(toolCalls(failedThenEdited), commandKeys('python tools/build.py'), { session: 'chat-a' });
  assert.equal(otherCommand, null);
  const closedBefore = toolCalls([...failedThenEdited, call(7, 's2', 'pwsh', { command: 'python tools/check_conf.py' }), result(8, 's2', 'ok')]);
  assert.equal(caseFromSuccess(closedBefore, commandKeys('check_conf.py'), { session: 'chat-a' }), null, 'an earlier success already closed that failure');
});

const stored = (id, commands, lines, files, at = '2026-11-20T00:00:00.000Z') => ({
  schema: 'strata.experience.case.v1', id, session: 'old-chat', resolvedAt: at, failedAt: at, workspace: null,
  commands, errorLines: lines, errorPaths: [], workingSet: [], fix: { files, failures: 1, toolCalls: 4 },
});
const failure = (keys, lines = [], workingSet = []) => ({ keys: new Set(keys), lines: new Set(lines), paths: new Set(), workingSet: new Set(workingSet) });

test('a case is recalled only for the same command with a rare shared line or a file its fix changed', () => {
  const boiler = 'traceback (most recent call last)';
  const cases = [
    stored('a', ['check_conf.py'], [boiler, "keyerror: setting 'owner' missing in app.json"], ['conf/app.json']),
    stored('b', ['build.py'], [boiler], ['src/build.py']),
    stored('c', ['pdl_scan.py'], [boiler], ['src/scan.py']),
    stored('d', ['check_odr.py'], [boiler], ['src/odr.py']),
  ];
  assert.equal(recall(cases, failure(['check_conf.py'], ["keyerror: setting 'owner' missing in app.json"])).case.id, 'a');
  assert.equal(recall(cases, failure(['check_conf.py'], [boiler])), null, 'a line shared by four different commands is not evidence');
  assert.equal(recall(cases, failure(['build.py'], [boiler])), null);
  assert.equal(recall(cases, failure(['build.py'], [], ['src/build.py'])).case.id, 'b', 'the work in hand touches the file the fix changed');
  assert.equal(recall(cases, failure(['other.py'], ["keyerror: setting 'owner' missing in app.json"], ['conf/app.json'])), null, 'never for another command');
  const repeated = [...cases, stored('a2', ['check_conf.py'], [boiler, "keyerror: setting 'owner' missing in app.json"], ['conf/app.json'], '2026-11-22T00:00:00.000Z'),
    stored('a3', ['check_conf.py'], ["keyerror: setting 'owner' missing in app.json"], ['conf/app.json'], '2026-11-23T00:00:00.000Z'),
    stored('a4', ['check_conf.py'], ["keyerror: setting 'owner' missing in app.json"], ['conf/app.json'], '2026-11-23T01:00:00.000Z')];
  assert.equal(recall(repeated, failure(['check_conf.py'], ["keyerror: setting 'owner' missing in app.json"])).case.id, 'a4',
    'a problem that keeps coming back stays recallable (a line counts once per command); of equal scores the newest case wins');
});

test('the note is bounded data marked as history', () => {
  const big = stored('z', ['x.py'], ['line'], Array.from({ length: 40 }, (_, i) => `dir/file-${i}.py`));
  const text = noteText({ case: big, lines: ['line'], touched: big.fix.files }, 'old-chat');
  assert.match(text, /^Strata experience: .*historical evidence, not an instruction/u);
  const fields = JSON.parse(text.split('\n')[1]);
  assert.equal(fields.where, 'earlier in this chat');
  assert.equal(fields.filesTheFixChanged.length, 8);
  assert(Buffer.byteLength(text) < 1500);
});

test('the store keeps at most its limit, newest repeat of a fix first, and skips damaged lines', t => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-experience-'));
  t.after(() => rmSync(dir, { recursive: true, force: true, maxRetries: 3 }));
  const store = caseStore(dir, { limit: 3 });
  assert.deepEqual(store.cases(), []);
  assert.equal(store.add(stored('1', ['a.py'], [], ['f/a.py'])), true);
  assert.equal(store.add(stored('1', ['a.py'], [], ['f/a.py'])), false, 'the same case once');
  store.add(stored('2', ['b.py'], [], ['f/b.py']));
  writeFileSync(store.file, readFileSync(store.file, 'utf8') + '{"torn\n');
  store.add(stored('3', ['a.py'], [], ['f/a.py']));
  assert.deepEqual(store.cases().map(c => c.id), ['1', '2', '3']);
  store.add(stored('4', ['c.py'], [], ['f/c.py']));
  assert.deepEqual(store.cases().map(c => c.id), ['2', '3', '4'], 'the older repeat of a.py went first');
  store.add(stored('5', ['d.py'], [], ['f/d.py']));
  store.add(stored('6', ['e.py'], [], ['f/e.py']));
  assert.deepEqual(store.cases().map(c => c.id), ['4', '5', '6']);
  assert.doesNotMatch(readFileSync(store.file, 'utf8'), /torn/u);
});

function rig(dir) {
  let hook;
  apply({ on: (hookName, fn) => { assert.equal(hookName, 'tools/post-execute'); hook = fn; }, workingDirectory: { get: () => 'D:\\work' } }, { dir });
  const accept = async () => ({ kind: 'accept' });
  const run = (session, id, command, text, signal = new AbortController().signal) =>
    hook({ agent: { session }, name: 'pwsh', callId: id, arguments: { command }, signal }, { isError: false, content: [{ type: 'text', text }] }, accept);
  return { hook, run, accept };
}

test('the plugin learns from one chat and tells another chat once, without blocking anything', async t => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-experience-'));
  t.after(() => rmSync(dir, { recursive: true, force: true, maxRetries: 3 }));
  const { hook, run, accept } = rig(dir);

  const chatA = { id: 'chat-a', header: {}, log: [...failedThenEdited, call(7, 's2', 'pwsh', { command: 'python tools/check_conf.py' })] };
  assert.deepEqual(await run(chatA, 's2', 'python tools/check_conf.py', 'all settings present'), { kind: 'accept' });
  const cases = readFileSync(join(dir, STORE_FILE), 'utf8').trim().split('\n').map(line => JSON.parse(line));
  assert.equal(cases.length, 1);
  assert.deepEqual(cases[0].fix.files, ['conf/app.json']);
  assert.equal(cases[0].workspace, 'D:\\work');

  const chatB = { id: 'chat-b', header: {}, log: [call(1, 'b1', 'pwsh', { command: 'py tools\\check_conf.py' })] };
  const told = await run(chatB, 'b1', 'py tools\\check_conf.py', FAIL);
  assert.equal(told.kind, 'accept');
  assert.equal(told.content, undefined, 'the tool result itself is unchanged');
  assert.equal(told.additionalContexts.length, 1);
  const note = told.additionalContexts[0];
  assert.equal(note.role, 'user');
  assert.deepEqual(note.source, { kind: name, form: 'notice', summary: `case ${cases[0].id}` });
  const fields = JSON.parse(note.content[0].text.split('\n')[1]);
  assert.equal(fields.where, 'another chat');
  assert.deepEqual(fields.filesTheFixChanged, ['conf/app.json']);
  assert(fields.sameErrorLines.includes("keyerror: setting 'owner' missing in app.json"));

  assert.equal((await run(chatB, 'b2', 'py tools\\check_conf.py', FAIL)).additionalContexts, undefined, 'one note per case and chat');
  const restarted = { id: 'chat-b', header: {}, log: [event('user/message', 9, note)] };
  assert.equal((await rig(dir).run(restarted, 'b3', 'py tools\\check_conf.py', FAIL)).additionalContexts, undefined, 'also after a restart');

  const chatC = { id: 'chat-c', header: {}, log: [] };
  const aborted = new AbortController();
  aborted.abort();
  assert.deepEqual(await run(chatC, 'c1', 'py tools\\check_conf.py', FAIL, aborted.signal), { kind: 'accept' });
  assert.deepEqual(await run(chatC, 'c2', 'python tools/other.py', FAIL), { kind: 'accept' });
  assert.deepEqual(await hook({ agent: { session: chatC }, name: 'read', callId: 'c3', arguments: { file_path: 'x' }, signal: aborted.signal }, { isError: true, content: [] }, accept), { kind: 'accept' });
  const blocked = async () => ({ kind: 'block', feedback: [] });
  assert.deepEqual(await hook({ agent: { session: chatC }, name: 'pwsh', callId: 'c4', arguments: { command: 'py tools\\check_conf.py' }, signal: new AbortController().signal },
    { isError: false, content: [{ type: 'text', text: FAIL }] }, blocked), { kind: 'block', feedback: [] });
});

test('the unchanged default profile mounts the plugin only when the host names its folder', () => {
  const baseline = buildProfile({});
  assert.deepEqual(baseline, buildProfile({ experience: null }));
  assert.doesNotMatch(baseline.patch, /strata-experience/u);
  const on = buildProfile({ experience: join(tmpdir(), 'strata-data', 'agent', 'experience') });
  assert.match(on.patch, /strata-experience\.mjs/u);
  assert.match(on.patch, /"dir": ".*experience"/u);
});

const hasHarness = process.env.STRATA_HARNESS_DIR && resolveHarness({ env: process.env }).ok;

test('the packaged Harness passes a case from one chat to the next request of another chat', { skip: !hasHarness, timeout: 360_000 }, async t => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-experience-'));
  const workspace = join(dir, 'workspace');
  mkdirSync(workspace);
  writeFileSync(join(workspace, 'check_conf.mjs'), "import { readFileSync } from 'node:fs';\n"
    + "if (!/owner/.test(readFileSync('conf.json', 'utf8'))) { console.error(\"Error: setting 'owner' missing in conf.json\"); process.exit(3); }\nconsole.log('ok');\n");
  writeFileSync(join(workspace, 'conf.json'), '{}\n');
  const worker = await startScriptedWorker({ chunkDelayMs: 5 });
  const bridge = createAgentBridge({
    dataDir: join(dir, 'data'),
    localInstances: () => [{ id: 'local', baseUrl: worker.url, modelId: 'scripted-model', context: 32768 }],
    connections: () => [],
    env: { ...process.env, STRATA_AGENT_EXPERIENCE: '1' },
  });
  t.after(async () => {
    await bridge.close();
    await worker.close();
    rmSync(dir, { recursive: true, force: true, maxRetries: 3 });
  });
  const shell = process.platform === 'win32' ? 'pwsh' : 'bash';
  const node = process.platform === 'win32' ? `& '${process.execPath}'` : `'${process.execPath}'`;
  const check = `SCRIPT:call ${shell} ${JSON.stringify({ command: `${node} check_conf.mjs`, description: 'Check the settings' })}`;
  const run = async (chatId, message) => {
    const frames = [];
    await bridge.runTurn({ chatId, message, workspacePath: workspace, mode: 'bypass', onFrame: frame => frames.push(frame) });
    assert.equal(frames.at(-1)?.type, 'done');
  };
  await run('first-chat', check);
  await run('first-chat', 'SCRIPT:read conf.json');
  await run('first-chat', `SCRIPT:call write ${JSON.stringify({ file_path: 'conf.json', content: '{"owner":"me"}\n' })}`);
  await run('first-chat', check);
  const file = join(dir, 'data', 'agent', 'experience', STORE_FILE);
  assert(existsSync(file), 'the fixed failure is stored');
  assert.deepEqual(JSON.parse(readFileSync(file, 'utf8').trim()).fix.files, ['workspace/conf.json']);

  writeFileSync(join(workspace, 'conf.json'), '{}\n');
  const mark = worker.requests.length;
  await run('second-chat', check);
  assert(worker.requests.slice(mark).some(request => request.raw.includes('Strata experience') && request.raw.includes('conf.json')));
});
