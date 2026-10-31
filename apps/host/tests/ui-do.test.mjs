// The 0.2.9.1 interface changes (branch f-0291-ui): one section per issue. Source-level checks for what only a browser can show are
// backed by scripts/ui-do-drive.mjs, which drives the built app.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { build, transform } from 'esbuild';

const src = name => readFileSync(new URL(`../src/${name}`, import.meta.url), 'utf8');
const locale = name => JSON.parse(readFileSync(new URL(`../src/locales/${name}.json`, import.meta.url), 'utf8'));
const load = async name => {
  const { code } = await transform(src(name), { loader: 'ts', format: 'esm' });
  return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
};
const withI18n = async entry => {
  const built = await build({ stdin: { contents: `export * from './src/lib/i18n'; export * from './src/lib/${entry}';`, resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
  return import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
};

const languages = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr'];

test('the ui catalogs have the same keys and placeholders in all nine languages, none still English, none in another catalog', () => {
  const en = locale('ui-en');
  const slots = text => [...text.matchAll(/\{\d+\}/g)].map(m => m[0]).sort();
  const others = new Set();
  for (const name of ['en', 'tune-en', 'tools-en', 'fix-en', 'gh-en']) for (const v of Object.values(locale(name))) others.add(v);
  for (const l of languages) {
    const catalog = locale(`ui-${l}`);
    assert.deepEqual(Object.keys(catalog), Object.keys(en), l);
    for (const [id, source] of Object.entries(en)) {
      assert.ok(catalog[id].trim(), `${l}:${id} is empty`);
      assert.deepEqual(slots(catalog[id]), slots(source), `${l}:${id}`);
      if (l !== 'en') assert.notEqual(catalog[id], source, `${l}:${id} is still English`);
    }
  }
  for (const [id, text] of Object.entries(en)) assert.ok(!others.has(text), `already in another catalog: ${id} ${text}`);
});

// ---------------------------------------------------------------------------------------------------------------- #4725
test('#4725 the pending approvals of a chat are listed oldest first, and nothing else is', async () => {
  const { pendingApprovals } = await load('lib/approvals.ts');
  const tool = (id, state) => ({ id, name: 'write', state });
  const session = { messages: [
    { id: 'm1', tools: [tool('a', 'done'), tool('b', 'proposed')] },
    { id: 'm2', tools: [tool('c', 'denied'), tool('d', 'running')] },
    { id: 'm3' },
    { id: 'm4', tools: [tool('e', 'proposed')] },
  ] };
  assert.deepEqual(pendingApprovals(session).map(x => [x.messageId, x.tool.id]), [['m1', 'b'], ['m4', 'e']]);
  assert.deepEqual(pendingApprovals(null), []);
  assert.deepEqual(pendingApprovals({ messages: [] }), []);
});

test('#4725 the Approve and Deny buttons live in the dock above the message box, and the reply only says that it waits', () => {
  const chat = src('screens/Chat.tsx');
  assert.match(chat, /<ApprovalDock sessionId=\{session\?\.id\}\/>/);
  const toolCard = chat.slice(chat.indexOf('function ToolCard'));
  assert.doesNotMatch(toolCard, /\{tr\("(Approve|Deny)"\)\}|resolveTool/,'the card in the reply no longer carries the buttons');
  assert.match(toolCard, /Waiting for your answer below/);
  const dock = src('components/ApprovalCard.tsx');
  assert.match(dock, /resolveTool\(messageId, t\.id, true\)/);
  assert.match(dock, /resolveTool\(messageId, t\.id, false\)/);
  assert.equal((dock.match(/tr\("Approve"\)/g) ?? []).length, 2, 'one Approve and Deny pair per card style, nowhere else');
});

test('#4725 the dock and the waiting line are translated', async () => {
  const { tr, setTranslationLanguage } = await withI18n('approvals.ts');
  for (const l of languages.slice(1)) {
    setTranslationLanguage(l);
    for (const text of ['Waiting for your answer', 'Waiting for your answer below']) assert.notEqual(tr(text), text, `${l}: ${text}`);
  }
  setTranslationLanguage('en');
});

// ---------------------------------------------------------------------------------------------------------------------- #4748
import { readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
const walk = dir => readdirSync(dir).flatMap(name => { const p = join(dir, name); return statSync(p).isDirectory() ? (name === 'locales' ? [] : walk(p)) : /\.tsx?$/.test(name) ? [p] : []; });

test('#4748 every Copy control goes through one shared behaviour: no bare clipboard call, no copy icon button of its own', () => {
  const files = walk(new URL('../src', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'));
  const bare = [], own = [];
  for (const file of files) {
    if (/lib[\/]clipboard\.ts$/.test(file)) continue;
    const text = readFileSync(file, 'utf8');
    if (/navigator\.clipboard\.writeText/.test(text)) bare.push(file);
    if (/<IconBtn name="copy"/.test(text)) own.push(file);
  }
  assert.deepEqual(bare, [], 'a bare clipboard call reports nothing when the clipboard is refused');
  assert.deepEqual(own, [], 'a copy icon button that is not CopyBtn shows nothing on itself');
});

test('#4748 the shared copy shows its state on the control for about two seconds, then goes back; a menu item says Copied and then closes', () => {
  const hook = src('lib/use-copy.ts');
  assert.match(hook, /COPIED_MS = 1800/);
  assert.match(hook, /COPIED_MENU_MS = 900/);
  assert.match(hook, /if \(ok\) \{\s*setCopied\(true\);/, 'only a copy that worked is shown');
  const btn = src('components/CopyBtn.tsx');
  assert.match(btn, /data-copied=\{copied \|\| undefined\}/);
  assert.match(btn, /name=\{copied \? 'check' : 'copy'\}/);
  assert.match(btn, /role="status">\{copied \? tr\('Copied'\) : ''\}/, 'a screen reader hears Copied');
  assert.match(btn, /e\.stopPropagation\(\)/, 'the menu does not close before the person sees it');
  const css = src('styles/app.css');
  assert.match(css, /\.st-msg__actions:has\(\[data-copied\]\)\s*\{\s*opacity:\s*1/, 'the hover-only action row stays while it says Copied');
});

test('#4748 Copied is translated in every language', async () => {
  const { tr, setTranslationLanguage } = await withI18n('approvals.ts');
  for (const l of languages.slice(1)) { setTranslationLanguage(l); assert.notEqual(tr('Copied'), 'Copied', l); }
  setTranslationLanguage('en');
});
