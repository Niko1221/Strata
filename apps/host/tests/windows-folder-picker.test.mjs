import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { build, transform } from 'esbuild';
import { fileURLToPath } from 'node:url';
import { chooseFolder } from '../server/folder-dialog.mjs';

const env = { STRATA_SHELL_BRIDGE: 'http://127.0.0.1:12345', STRATA_SHELL_TOKEN: 'private-token' };
const noShell = () => { assert.fail('the desktop picker must never start PowerShell'); };

test('Windows desktop uses the authenticated shell bridge and preserves Unicode paths and roots', async () => {
  for (const path of ['C:\\Users\\Jörg\\模型\\', 'D:\\']) {
    const picked = await chooseFolder('Choose & import?', { platform: 'win32', env, run: noShell, fetcher: async (url, options) => {
      assert.equal(new URL(url).pathname, '/shell/folder');
      assert.equal(new URL(url).searchParams.get('title'), 'Choose & import?');
      assert.equal(options.method, 'POST');
      assert.equal(options.headers.authorization, 'Bearer private-token');
      return { ok: true, json: async () => ({ path }) };
    } });
    assert.equal(picked.path, path === 'D:\\' ? path : path.slice(0, -1));
  }
});

test('native cancellation is not a failure and never invokes a second picker', async () => {
  assert.deepEqual(await chooseFolder('Pick', { platform: 'win32', env, run: noShell,
    fetcher: async () => ({ ok: true, json: async () => ({ cancelled: true }) }) }), { cancelled: true });
});

test('an unavailable or malformed native response offers typing, never PowerShell', async () => {
  for (const fetcher of [async () => { throw new Error('offline'); }, async () => ({ ok: false, status: 503 }),
    async () => ({ ok: true, json: async () => ({}) })]) {
    const picked = await chooseFolder('Pick', { platform: 'win32', env, run: noShell, fetcher });
    assert.equal(picked.unavailable, true);
    assert.equal(picked.failed, true);
  }
});

async function folderUI(post) {
  const result = await build({ entryPoints: [fileURLToPath(new URL('../src/lib/folder-pick.ts', import.meta.url))],
    bundle: true, write: false, format: 'esm', plugins: [{ name: 'folder-test', setup(b) {
      b.onResolve({ filter: /^(react|\.\/client)$/ }, args => ({ path: args.path, namespace: 'stub' }));
      b.onLoad({ filter: /.*/, namespace: 'stub' }, args => ({ contents: args.path === 'react'
        ? 'export const useSyncExternalStore = (subscribe, get) => get();'
        : 'export class ApiError extends Error { constructor(message, status, body) { super(message); this.status = status; this.body = body; } } globalThis.__FolderApiError = ApiError; export const api = { post: (...args) => globalThis.__folderPost(...args) };' }));
    } }] });
  globalThis.__folderPost = post;
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text + `\n// ${Math.random()}`).toString('base64')}`);
}

test('manual entry is available immediately; dismissing it posts nothing', async () => {
  const ui = await folderUI(() => assert.fail('cancel must not change host state'));
  const pending = ui.postWithFolderFallback('/api/skills/add', true);
  assert.equal(typeof ui.useFolderPrompt().browse, 'function');
  ui.useFolderPrompt().resolve(null);
  assert.deepEqual(await pending, { cancelled: true });
});

test('browsing a skill only picks a path; adding waits for explicit confirmation', async () => {
  const calls = [];
  const ui = await folderUI(async (url, body) => {
    calls.push({ url, body });
    return url === '/api/workspaces/pick' ? { path: 'C:\\Skills' } : { added: 'skill-id' };
  });
  const pending = ui.postWithFolderFallback('/api/skills/add', true);
  const prompt = ui.useFolderPrompt();
  assert.equal(await prompt.browse(), 'C:\\Skills');
  assert.deepEqual(calls, [{ url: '/api/workspaces/pick', body: {} }]);
  prompt.resolve('C:\\Skills');
  assert.deepEqual(await pending, { added: 'skill-id' });
  assert.deepEqual(calls[1], { url: '/api/skills/add', body: { path: 'C:\\Skills' } });
});

test('canceling native browse leaves the manual prompt open; invalid typing can be corrected', async () => {
  let ui;
  ui = await folderUI(async (url, body) => {
    if (!body.path) return { cancelled: true };
    if (body.path === 'bad') throw new globalThis.__FolderApiError('That folder does not exist.', 400);
    return { path: body.path };
  });
  const pending = ui.postWithFolderFallback('/api/workspaces/pick', true);
  const prompt = ui.useFolderPrompt();
  assert.equal(await prompt.browse(), null);
  assert.equal(ui.useFolderPrompt(), prompt);
  prompt.resolve('bad');
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(ui.useFolderPrompt().message, 'That folder does not exist.');
  ui.useFolderPrompt().resolve('C:\\Workspace');
  assert.deepEqual(await pending, { path: 'C:\\Workspace' });
});

test('the GGUF preview and typed path are changed only after selection, not on cancel', async () => {
  const source = await readFile(new URL('../src/components/NativeLocalModel.tsx', import.meta.url), 'utf8');
  // Exercise the actual component's check function with state setters and an API stub, without rendering a window.
  const check = source.slice(source.indexOf('    async function check('), source.indexOf('    /* The first-run'));
  const { code } = await transform(`export default function(api, setPending, setError, setChecked, setTyped) { ${check}; return check; }`, { loader: 'ts', format: 'esm' });
  const { default: makeCheck } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
  const state = { pending: false, error: 'previous error', checked: { path: 'old' }, typed: 'old' };
  const setter = key => value => { state[key] = value; };
  const invoke = response => makeCheck({ post: async () => response }, ...['pending', 'error', 'checked', 'typed'].map(setter))('');
  await invoke({ cancelled: true });
  assert.deepEqual(state, { pending: false, error: 'previous error', checked: { path: 'old' }, typed: 'old' });
  await invoke({ path: 'new', preview: {} });
  assert.equal(state.typed, 'new');
  assert.equal(state.checked.path, 'new');
  assert.equal(state.error, '');
});
