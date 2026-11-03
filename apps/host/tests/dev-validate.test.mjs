// Strata Dev (docs/DEV_BUILD.md): the validator dev-check.ps1 runs first (scripts/dev-validate.mjs), on a scratch dev/ folder.
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const SCRIPT = fileURLToPath(new URL('../scripts/dev-validate.mjs', import.meta.url));
const write = (path, text) => { mkdirSync(join(path, '..'), { recursive: true }); writeFileSync(path, text); };
const run = root => {
  const r = spawnSync(process.execPath, [SCRIPT, '--json'], { env: { ...process.env, STRATA_DEV_CONFIG: join(root, 'dev.config.json'), STRATA_DEV_EXTENSIONS: '', STRATA_DEV_SKILLS: '', STRATA_DEV_MODELS: '' }, encoding: 'utf8' });
  return { code: r.status, report: JSON.parse(r.stdout) };
};

test('a clean dev folder passes; every kind of broken entry is named and fails the check', () => {
  const root = mkdtempSync(join(tmpdir(), 'strata-dev-validate-'));
  try {
    write(join(root, 'dev.config.json'), JSON.stringify({ version: 1, badge: { label: 'DEV' } }));
    write(join(root, 'dev', 'extensions', 'ui', 'ok', 'index.tsx'), 'export default function setup(strata) { strata.log(<b>hi</b>); }');
    write(join(root, 'dev', 'extensions', 'backend', 'ok.mjs'), 'export function before_request(b) { return b; }');
    write(join(root, 'dev', 'skills', 'good-skill', 'SKILL.md'), '---\nname: good-skill\ndescription: Does a thing when asked.\n---\n\nDo it.\n');
    write(join(root, 'dev', 'models', 'q.json'), JSON.stringify({ model: 'qwen*' }));
    const clean = run(root);
    assert.equal(clean.code, 0, JSON.stringify(clean.report.rows));
    assert.equal(clean.report.errors, 0);

    write(join(root, 'dev', 'extensions', 'ui', 'nodefault', 'index.ts'), 'export const x = 1;');
    write(join(root, 'dev', 'extensions', 'ui', 'broken', 'index.tsx'), 'export default function ( {');
    write(join(root, 'dev', 'extensions', 'ui', 'empty', 'readme.txt'), 'no entry');
    write(join(root, 'dev', 'extensions', 'backend', 'bad.mjs'), 'export function before_request( {');
    write(join(root, 'dev', 'skills', 'Bad_Skill', 'SKILL.md'), '---\nname: Bad_Skill\ndescription:\n---\n\nx\n');
    write(join(root, 'dev', 'skills', 'good-skill', 'tools.json'), '{ not json');
    write(join(root, 'dev', 'models', 'm.json'), JSON.stringify({ context: 5 }));
    write(join(root, 'dev.config.json'), JSON.stringify({ version: 1, overrides: [], unknownKey: 1 }));
    const broken = run(root);
    assert.equal(broken.code, 1);
    const errorsFor = item => broken.report.rows.filter(r => r.item === item && r.level === 'error').map(r => r.text).join(' | ');
    assert.match(errorsFor('nodefault'), /no default export/);
    assert.match(errorsFor('broken'), /does not parse/);
    assert.match(errorsFor('empty'), /no index/);
    assert.match(errorsFor('bad.mjs'), /^load:/);
    assert.match(errorsFor('Bad_Skill'), /kebab-case/);
    assert.match(errorsFor('good-skill'), /tools\.json is not valid JSON/);
    assert.match(errorsFor('m.json'), /"model"/);
    assert.match(errorsFor('dev.config.json'), /"overrides" must be an object/);
    assert.ok(broken.report.rows.some(r => r.level === 'warn' && /unknownKey/.test(r.text)));
  } finally { rmSync(root, { recursive: true, force: true }); }
});
