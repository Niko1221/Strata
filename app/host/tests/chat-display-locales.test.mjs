/** Actual React rendering and catalogs for the reported Japanese chat display. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {build} from 'esbuild';

const compiled = await build({
  stdin: {contents: `
    import React from 'react';
    import {renderToStaticMarkup} from 'react-dom/server';
    import {ChatTelemetry} from './src/components/TelemetryPopover';
    import {ErrorNotice} from './src/components/ui';
    import {setTranslationLanguage, tr, LANGUAGES, loadCatalog} from './src/lib/i18n';
    export {tr, LANGUAGES, setTranslationLanguage, loadCatalog};
    export const telemetry = (language, message) => {
      setTranslationLanguage(language);
      return renderToStaticMarkup(React.createElement(ChatTelemetry, {session: {
        id: 'synthetic-chat', statsAvailable: false, messages: [{
          id: 'synthetic-reply', role: 'assistant', content: 'Synthetic answer',
          at: '2026-11-07T00:00:00Z', ...message
        }]
      }}));
    };
    export const errorNotice = (language, message) => {
      setTranslationLanguage(language);
      return renderToStaticMarkup(React.createElement(ErrorNotice, {message}));
    };
  `, resolveDir: process.cwd(), loader: 'tsx'},
  bundle: true, write: false, format: 'cjs', platform: 'node', jsx: 'automatic',
  external: ['react', 'react-dom', 'react-dom/server'], logLevel: 'silent',
  plugins: [{name: 'isolated-readonly-state', setup(b) {
    b.onResolve({filter: /\/lib\/(store|native)$/}, args => ({path: args.path.endsWith('/store') ? 'store' : 'native', namespace: 'state'}));
    b.onLoad({filter: /.*/, namespace: 'state'}, args => ({loader: 'js', contents: args.path === 'store'
      ? `export const useStore = fn => fn({monitor: null, engine: {beginner: {contextLength: '4k'}}});
         export const dismissActionError=()=>{},dismissToast=()=>{},holdToast=()=>{},releaseToast=()=>{};`
      : `export const useNative = () => ({instances: [], defaultId: null});`}));
  }}],
});
const module = {exports: {}};
new Function('require', 'module', 'exports', compiled.outputFiles[0].text)(createRequire(import.meta.url), module, module.exports);
const {telemetry, errorNotice, tr, LANGUAGES, setTranslationLanguage, loadCatalog} = module.exports;
await Promise.all(LANGUAGES.map(({value}) => loadCatalog(value)));
test('document language follows the selected UI locale without changing output code points', async () => {
  const saved=globalThis.document;
  globalThis.document={documentElement:{lang:'en'}};
  try {
    await setTranslationLanguage('ja'); assert.equal(document.documentElement.lang,'ja');
    const literal='日本語 國 臺灣 𠮷'; assert.equal(tr(literal),literal);
    await setTranslationLanguage('zh'); assert.equal(document.documentElement.lang,'zh-CN');
    await setTranslationLanguage('en'); assert.equal(document.documentElement.lang,'en');
  } finally { if(saved===undefined)delete globalThis.document;else globalThis.document=saved; }
});
const text = html => html.replace(/<[^>]*>/g, '').replace(/&#x27;/g, "'").replace(/&quot;/g, '"').replace(/&amp;/g, '&');

for (const {value: language} of LANGUAGES) {
  test(`${language}: live and finished speed render numbers or unknown, never template slots`, () => {
    for (const streaming of [true, false]) for (const speed of [undefined, 12.4]) {
      const html = telemetry(language, {streaming, tps: speed});
      const gauge = /<button[^>]*class="telemetry__chip"[^>]*>[\s\S]*?<\/button>/.exec(html)?.[0];
      assert.ok(gauge, 'real telemetry chip rendered');
      const shown = text(gauge);
      assert.doesNotMatch(shown, /\{\d+\}|\u0001/, 'no raw substitution/sentinel text');
      assert.ok(shown.includes(speed === undefined ? '—' : '12'), 'reported/unknown value is visible');
      setTranslationLanguage(language);
      if (streaming) assert.ok(!shown.includes(tr('· last reply')), 'live response is not labelled as an older reply');
      else assert.ok(shown.includes(tr('· last reply')), 'last reply suffix uses the active language');
      if (language === 'ja') assert.doesNotMatch(shown, /last reply/);
    }
  });
}

for (const message of [
  'The reply was cut off. The partial answer is kept.',
  'Response interrupted. Ask again to retry; the partial answer is kept.',
]) {
  test(`saved recovery notice translates on render: ${message}`, () => {
    for (const {value: language} of LANGUAGES) {
      const shown = text(errorNotice(language, message));
      if (language === 'en') assert.ok(shown.includes(message));
      else assert.ok(!shown.includes(message), `${language}: no raw English recovery sentence`);
      assert.doesNotMatch(shown, /\{\d+\}/);
    }
    assert.match(text(errorNotice('ja', message)), /回答|応答/);
  });
}
