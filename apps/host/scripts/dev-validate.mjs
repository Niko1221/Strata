#!/usr/bin/env node
/**
 * Strata Dev (docs/DEV_BUILD.md): checks everything in dev/ and dev.config.json before a build, for dev-check.ps1 (and by hand:
 * `node apps/host/scripts/dev-validate.mjs [--json]`). Uses the same checks the host uses at runtime, so a pass here is a load there.
 *
 *   dev.config.json        valid JSON, known sections, the warnings the host would show
 *   dev/extensions/ui      every extension has an entry file that parses and has a default export
 *   dev/extensions/backend every JS hook loads (dev-hooks.mjs) with its events and routes; every .py hook names known events
 *   dev/skills             the skill validator (dev-skills.mjs): frontmatter, kebab-case name, description, 500 lines; tools.json
 *   dev/models             the overlay checks (dev-catalog.mjs)
 *
 * Exit code 1 when anything is an error; warnings only are 0.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { transform } from 'esbuild';
import { devExtensionsDir, devModelsDir, devSettings, devSkillsDir, loadDevConfig } from '../server/dev-mode.mjs';
import { createDevHooks, pythonEvents } from '../server/dev-hooks.mjs';
import { readSkillRoot } from '../server/dev-skills.mjs';
import { readOverlays } from '../server/dev-catalog.mjs';
import { llamaSettings, settingsProblem } from '../server/dev-llama.mjs';

const REPO = resolve(fileURLToPath(new URL('.', import.meta.url)), '..', '..', '..');
const env = { ...process.env, STRATA_DEV_CONFIG: process.env.STRATA_DEV_CONFIG || join(REPO, 'dev.config.json') };
const rows = [];
const add = (area, item, level, text) => rows.push({ area, item, level, text });

/* dev.config.json */
const loaded = loadDevConfig({ file: env.STRATA_DEV_CONFIG });
if (!existsSync(env.STRATA_DEV_CONFIG)) add('config', 'dev.config.json', 'warn', `not found at ${env.STRATA_DEV_CONFIG}; Dev starts with the empty config`);
for (const w of loaded.warnings) add('config', 'dev.config.json', 'error', w);
const settings = devSettings(loaded.config);
const KNOWN = new Set(['$comment', 'version', 'badge', 'logging', 'features', 'overrides', 'tokens', 'ui', 'hooks', 'llamacpp']);
for (const key of Object.keys(loaded.config)) if (!KNOWN.has(key)) add('config', 'dev.config.json', 'warn', `"${key}" is not a key Strata Dev reads`);
if (loaded.config.llamacpp) {
  const problem = settingsProblem(llamaSettings(loaded.config.llamacpp));
  add('config', 'llamacpp', problem ? 'warn' : 'ok', problem ?? `llama-server found (${llamaSettings(loaded.config.llamacpp).server})`);
}
if (!loaded.warnings.length) add('config', 'dev.config.json', 'ok', `${Object.keys(settings.tokens).length} token(s), ${Object.keys(settings.ui.elements).length} element override(s)`);

/* UI extensions */
const uiDir = join(devExtensionsDir(env), 'ui');
for (const entry of existsSync(uiDir) ? readdirSync(uiDir).sort() : []) {
  if (entry.startsWith('_') || entry.startsWith('.')) continue;
  const path = join(uiDir, entry);
  const file = statSync(path).isDirectory() ? ['index.tsx', 'index.ts', 'index.jsx', 'index.js'].map(n => join(path, n)).find(existsSync) : /\.(tsx|ts|jsx|js)$/.test(entry) && !entry.endsWith('.d.ts') ? path : null;
  if (!file) { add('ui', entry, statSync(path).isDirectory() ? 'error' : 'warn', statSync(path).isDirectory() ? 'no index.tsx/ts/jsx/js in the folder' : 'not an extension file (ignored)'); continue; }
  const source = readFileSync(file, 'utf8');
  try {
    await transform(source, { loader: file.endsWith('x') ? (file.endsWith('.tsx') ? 'tsx' : 'jsx') : file.endsWith('.ts') ? 'ts' : 'js', jsx: 'automatic', format: 'esm' });
    if (!/export\s+default\b/.test(source)) add('ui', entry, 'error', 'no default export (export default function setup(strata) { ... })');
    else add('ui', entry, 'ok', `${file.slice(uiDir.length + 1)} parses`);
  } catch (e) { add('ui', entry, 'error', `does not parse: ${(e.errors?.[0]?.text ?? e.message).split('\n')[0]}`); }
}

/* backend hooks */
const backendDir = join(devExtensionsDir(env), 'backend');
const hooks = createDevHooks({ dir: backendDir, settings: settings.hooks, onLog: () => {} });
for (const h of await hooks.load()) {
  if (h.error) add('backend', h.file, 'error', h.error);
  else if (h.kind === 'py' && !h.events.length) add('backend', h.file, 'warn', 'names no known event in its "# strata-hooks:" line');
  else if (!h.events.length && !h.routes.length) add('backend', h.file, 'warn', 'exports no hook and no routes');
  else add('backend', h.file, 'ok', [h.events.join(', '), h.routes.join(', ')].filter(Boolean).join(' · '));
}
for (const entry of existsSync(backendDir) ? readdirSync(backendDir).filter(n => n.startsWith('_') && n.endsWith('.py')) : []) {
  if (!pythonEvents(readFileSync(join(backendDir, entry), 'utf8')).length) add('backend', entry, 'warn', 'example (not loaded) names no known event');
}

/* skills */
for (const s of readSkillRoot(devSkillsDir(env), { source: 'dev', prefix: 'dev-' })) {
  const v = s.validation;
  for (const e of v.errors) add('skills', s.folder, 'error', e);
  for (const w of v.warnings) add('skills', s.folder, 'warn', w);
  let declared = 0;
  try { declared = (JSON.parse(readFileSync(join(s.dir, 'tools.json'), 'utf8')).tools ?? []).length; } catch (e) { if (existsSync(join(s.dir, 'tools.json'))) add('skills', s.folder, 'error', `tools.json is not valid JSON: ${e.message}`); }
  if (declared > s.tools.length) add('skills', s.folder, 'warn', `${declared - s.tools.length} tool(s) in tools.json left out (name a-z0-9_-, a one-line command)`);
  if (!v.errors.length) add('skills', s.folder, 'ok', `${v.lines} lines, ${s.files.length} file(s)${s.tools.length ? `, tools: ${s.tools.map(t => t.name).join(', ')}` : ''}`);
}

/* models */
for (const o of readOverlays(devModelsDir(env))) {
  if (o.problems.length) for (const p of o.problems) add('models', `${o.id}.json`, 'error', p);
  else add('models', `${o.id}.json`, 'ok', `applies to ${o.data.model}${o.data.runner ? ` · runner ${o.data.runner}` : ''}`);
}

const errors = rows.filter(r => r.level === 'error').length, warnings = rows.filter(r => r.level === 'warn').length;
if (process.argv.includes('--json')) {
  console.log(JSON.stringify({ errors, warnings, rows }, null, 1));
} else {
  const mark = { ok: 'OK  ', warn: 'WARN', error: 'FEHL' };
  for (const r of rows) console.log(`${mark[r.level]}  ${r.area.padEnd(8)} ${r.item}: ${r.text}`);
  console.log(`dev-validate: ${errors} Fehler, ${warnings} Warnung(en), ${rows.filter(r => r.level === 'ok').length} OK`);
}
process.exitCode = errors ? 1 : 0;
