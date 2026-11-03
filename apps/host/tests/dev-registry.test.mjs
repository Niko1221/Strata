// Strata Dev (docs/DEV_BUILD.md): the UI slot registry (src/dev/registry.ts) and the Dev overrides the app logic reads (src/dev/overrides.ts).
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';

const load = async file => {
  const compiled = await build({ stdin: { contents: `export * from './${file}';`, resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node' });
  return import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);
};
const R = await load('src/dev/registry');
const O = await load('src/dev/overrides');

const builtins = [
  { id: 'sidebar.skills', label: 'Skills', item: 'skills' },
  { id: 'sidebar.monitor', label: 'Monitor', item: 'monitor' },
  { id: 'sidebar.models', label: 'Models', item: 'models' },
];

test('with no overrides and no contributions a list is the app\'s own list, in its own order', () => {
  const rows = R.arrange('sidebar', builtins, [], {});
  assert.deepEqual(rows.map(r => [r.kind, r.id, r.label]), [['builtin', 'sidebar.skills', 'Skills'], ['builtin', 'sidebar.monitor', 'Monitor'], ['builtin', 'sidebar.models', 'Models']]);
});

test('an element can be hidden, renamed, moved and replaced; contributions come after the app\'s own unless they name an order', () => {
  const render = () => null;
  const rows = R.arrange('sidebar', builtins, [{ id: 'notes', extension: 'x', label: 'Notes', render }, { id: 'first', extension: 'x', order: 1, render }], {
    'sidebar.monitor': { hidden: true },
    'sidebar.models': { label: 'Modelle', order: 5 },
    'sidebar.skills': { replace: 'my-skills' },
  });
  assert.deepEqual(rows.map(r => r.id), ['sidebar.first', 'sidebar.models', 'sidebar.skills', 'sidebar.notes']);
  assert.equal(rows.find(r => r.id === 'sidebar.models').label, 'Modelle');
  assert.equal(rows.find(r => r.id === 'sidebar.skills').replace, 'my-skills');
  assert.equal(rows.find(r => r.id === 'sidebar.notes').kind, 'contribution');
});

test('the order is stable for equal order values', () => {
  const rows = R.arrange('settings.tabs', builtins, [], { 'sidebar.models': { order: 10 } });
  assert.deepEqual(rows.map(r => r.id), ['sidebar.skills', 'sidebar.models', 'sidebar.monitor']);
});

test('dev.config.json wins over an extension', () => {
  assert.deepEqual(R.mergeOverrides({ a: { hidden: true, label: 'X' } }, { a: { hidden: false } }), { a: { hidden: false, label: 'X' } });
  assert.deepEqual(R.resolveElement('a', { a: { hidden: true } }), { hidden: true });
  assert.deepEqual(R.resolveElement('b', {}), { hidden: false });
});

test('the registry keeps contributions per extension; switching one off removes them, forgetting clears them', () => {
  const reg = R.createRegistry();
  let calls = 0;
  reg.subscribe(() => calls++);
  reg.declare('info', true);
  reg.contribute('chat.toolbar', { id: 'info', extension: 'info', render: 'r' });
  reg.override('info', 'sidebar.monitor', { hidden: true });
  reg.replace('info', 'screen.models', 'Component');
  assert.equal(reg.slot('chat.toolbar').length, 1);
  assert.equal(reg.overrides()['sidebar.monitor'].hidden, true);
  assert.equal(reg.overrides()['screen.models'].replace, 'screen.models');
  assert.equal(reg.replacement('screen.models').component, 'Component');
  reg.setEnabled('info', false);
  assert.equal(reg.slot('chat.toolbar').length, 0);
  assert.equal(reg.overrides()['sidebar.monitor'], undefined);
  assert.equal(reg.replacement('screen.models'), null);
  reg.setEnabled('info', true);
  reg.setConfig({ 'sidebar.monitor': { hidden: false } });
  assert.equal(reg.overrides()['sidebar.monitor'].hidden, false, 'the config has the last word');
  reg.forget('info');
  assert.equal(reg.slot('chat.toolbar').length, 0);
  assert.deepEqual(reg.extensions().find(e => e.name === 'info').contributions, []);
  assert.ok(calls >= 8, 'every change tells the subscribers');
  assert.ok(reg.version() > 0);
});

test('an unknown slot or a contribution without an id is refused; errors are kept per extension', () => {
  const reg = R.createRegistry();
  assert.throws(() => reg.contribute('nowhere', { id: 'a', extension: 'x', render: null }), /Unknown slot "nowhere"/);
  assert.throws(() => reg.contribute('sidebar', { id: '', extension: 'x', render: null }), /needs an id/);
  reg.fail('x', 'boom', 'stack');
  assert.equal(reg.extensions().find(e => e.name === 'x').error, 'boom');
  reg.clearError('x');
  assert.equal(reg.extensions().find(e => e.name === 'x').error, null);
  assert.deepEqual([...R.SLOTS], ['sidebar', 'chat.toolbar', 'chat.message.actions', 'settings.tabs', 'monitor.widgets', 'preview.panel']);
});

test('the overrides the app logic reads change nothing until Dev sets them', () => {
  assert.equal(O.devLoopEnabled(), true);
  assert.equal(O.devLoopMaxRuns(100), 100);
  assert.equal(O.devDefaultMode('manual'), 'manual');
  assert.deepEqual(O.devModeOverride(undefined), {});
  O.setDevOverrides({ loop: { enabled: false, maxRuns: 5 }, permissions: { defaultMode: 'plan' } });
  assert.equal(O.devLoopEnabled(), false);
  assert.equal(O.devLoopMaxRuns(100), 5);
  assert.equal(O.devDefaultMode('manual'), 'plan');
  assert.deepEqual(O.devModeOverride(undefined), { mode: 'plan' });
  assert.deepEqual(O.devModeOverride('bypass'), {}, 'the chat keeps its own mode');
  O.setDevOverrides(undefined);
  assert.equal(O.devLoopEnabled(), true);
});
