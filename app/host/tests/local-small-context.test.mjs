import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { createRequire } from 'node:module';
import { build } from 'esbuild';
import { ggufFixture } from './helpers/local-gguf.mjs';
import { inspectLocalModel } from '../server/local-models.mjs';
import { CONTEXT_TOKENS, expertParams } from '../server/catalog.mjs';
import { modelReservation } from '../server/runtime-models.mjs';

test('1K, 2K and 3K GGUF inspection and reservation preserve the model maximum', t => {
  const root = mkdtempSync(join(tmpdir(), 'strata-small-context-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  for (const context of [1024, 2048, 3072]) {
    const file = join(root, `model-${context}.gguf`);
    ggufFixture(file, { context });
    const { variant } = inspectLocalModel(file);
    const label = `${context / 1024}k`;
    assert.equal(variant.contextMax, context);
    assert.equal(CONTEXT_TOKENS[label], context);
    const config = { beginner: { contextLength: label }, expert: [] };
    assert.equal(modelReservation(variant, config, { memory: { totalGb: 32 }, gpus: [] }, []).context, context);
    assert.throws(() => modelReservation(variant, { ...config, beginner: { contextLength: '4k' } }, { memory: { totalGb: 32 }, gpus: [] }, []), /supported conversation context/);
  }
});

// Render the real Tuning screen with React hooks/SSR. Only external state and
// surrounding panels are stubbed; the slider, config conversion and handlers run.
const root = resolve(import.meta.dirname, '..');
const stubs = {
  store: `export const useStore = select => select(globalThis.__smallContextState);
    export const setConfigDraft = text => { globalThis.__smallContextState.configDraft = text; };
    export const useLifecycle = () => ({allowed:true});
    export const lifecycleAvailable = () => true;
    export const saveEngine = () => {}; export const restartEngine = () => {};
    export const validateEngine = () => {}; export const pushToast = () => {};`,
  i18n: 'export const tr = value => value; export const localeTag = () => "en-US";',
  ui: `export const Btn = props => <button {...props}/>;
    export const ActionError = () => null; export const Dialog = () => null;
    export const ErrorNotice = () => null; export const Icon = () => null; export const Select = () => null;`,
  layout: 'export const Screen = ({children, actions}) => <main>{actions}{children}</main>; export const SectionHead = () => null;',
  EngineExperience: 'export const DemoTest = () => null; export const PerformancePreview = () => null;',
  AutoTune: 'export const AutoTune = () => null;',
  LifecycleNotice: 'export const LifecycleNotice = () => null;',
  FloatingMenu: 'export const FloatingMenu = () => null;',
};
const bundle = await build({
  stdin: { contents: `import React from 'react'; import {renderToStaticMarkup} from 'react-dom/server';
    import {TuningScreen} from './src/screens/Tuning';
    let tree; function Capture() { tree = TuningScreen(); return tree; }
    export const render = () => ({html:renderToStaticMarkup(<Capture/>), tree});`, resolveDir: root, loader: 'tsx' },
  bundle: true, write: false, platform: 'node', format: 'cjs', jsx: 'automatic',
  plugins: [{ name: 'isolated-panels', setup(b) {
    b.onResolve({ filter: /\/(store|i18n|ui|layout|EngineExperience|AutoTune|LifecycleNotice|FloatingMenu)$/ }, args => ({ path: args.path.split('/').pop(), namespace: 'stub' }));
    b.onLoad({ filter: /.*/, namespace: 'stub' }, args => ({ contents: stubs[args.path], loader: 'tsx', resolveDir: root }));
  } }],
});
const mod = { exports: {} };
new Function('require', 'module', 'exports', bundle.outputFiles[0].text)(createRequire(import.meta.url), mod, mod.exports);
function elements(node) {
  if (Array.isArray(node)) return node.flatMap(elements);
  if (!node?.props) return [];
  return [node, ...elements(node.props.children), ...elements(node.props.actions)];
}
function state(max, value) {
  const expert = expertParams(8).map(p => p.id === 'ctx.length' ? { ...p, value: `${value}K` } : p);
  return { connection: 'offline', configDraft: null, hardware: null,
    engine: { beginner: { contextLength: `${value}k` }, expert, predicted: { notes: [] } },
    library: { loaded: 'local', mode: 'real', variants: [{ id: 'local', contextMax: max * 1024 }] } };
}
for (const max of [1, 2, 3, 32, 128, 256, 1024]) {
  test(`Tuning renders and edits a ${max}K model without inflating its maximum`, () => {
    const value = Math.min(max, 128), min = Math.min(4, max);
    globalThis.__smallContextState = state(max, value);
    const { tree, html } = mod.exports.render();
    const slider = elements(tree).find(n => n.type === 'input' && n.props.type === 'range');
    assert.ok(slider, 'the real screen rendered its context slider');
    assert.deepEqual([slider.props.min, slider.props.max, slider.props.value], [min, max, value]);
    assert.equal(slider.props['aria-valuetext'], `${value}K tokens`);
    assert.match(html, new RegExp(`min="${min}" max="${max}" step="1" value="${value}"`));
    assert.match(html, new RegExp(`Recommended: ${Math.min(max, 128)}K`));
    assert.doesNotMatch(html, /NaN|Infinity/);
    const edited = Math.max(min, max - 1);
    slider.props.onChange({ target: { value: String(edited) } });
    const draft = JSON.parse(globalThis.__smallContextState.configDraft);
    assert.equal(draft.beginner.contextLength, `${edited}k`);
    assert.equal(draft.expert['ctx.length'], `${edited}K`);
    assert.equal(elements(mod.exports.render().tree).find(n => n.type === 'input' && n.props.type === 'range').props.value, edited);
    const reset = elements(tree).find(n => n.props.children === 'Reset to recommended');
    reset.props.onClick();
    const recommended = JSON.parse(globalThis.__smallContextState.configDraft);
    assert.equal(recommended.expert['ctx.length'], `${Math.min(max, 128)}K`);
    assert.equal(recommended.beginner.contextLength, max >= 128 ? 'recommended' : `${max}k`);
  });
}
