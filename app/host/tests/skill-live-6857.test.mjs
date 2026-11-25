/**
 * #6857: an update deleted the skills a person had in the agent's skills folder (skills-live). `materializeSkills` removed every folder that was not in
 * the current list. Now only folders Strata wrote are ever taken away, and those are moved aside, not deleted.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { SKILLS_REMOVED, materializeSkills, skillFileText } from '../server/skill-files.mjs';

const scratch = () => mkdtempSync(join(tmpdir(), 'strata-skill-live-'));
const cleanup = dir => { try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } };
const put = (file, text) => { mkdirSync(join(file, '..'), { recursive: true }); writeFileSync(file, text); };
const skill = (id, body = 'Do the thing.') => ({ id, name: id, description: 'd', body, enabled: true });
/** A folder the way 0.2.20 left it: the SKILL.md `skillFileText` writes, and, for a skill with a folder of its own, the copy note. */
const oldLayout = (live, name, { note = false } = {}) => {
  put(join(live, name, 'SKILL.md'), skillFileText({ name, description: `${name}. d`, body: 'old body' }));
  if (note) writeFileSync(join(live, name, '.strata-copied.json'), '[]');
};

test('a folder the person put in the skills folder survives, with or without anything else in it (#6857)', () => {
  const dir = scratch(), live = join(dir, 'agent', 'skills-live');
  try {
    // a hand-written skill with no Strata marker, a bare folder, and one that merely looks like Strata's quoted frontmatter but has other fields
    put(join(live, 'behavior-guidelines', 'SKILL.md'), '---\nname: behavior-guidelines\ndescription: Be brief\n---\nShort answers.');
    put(join(live, 'concise-answers', 'SKILL.md'), '---\nname: concise-answers\ndescription: "Be brief"\nallowed-tools: read\n---\nShort.');
    put(join(live, 'concise-answers', 'helper.ps1'), 'Write-Host hi');
    mkdirSync(join(live, 'empty-folder'), { recursive: true });
    for (let round = 0; round < 3; round++) materializeSkills(live, [skill('commit-messages')]);   // an update restarts the host: several syncs
    assert.equal(readFileSync(join(live, 'behavior-guidelines', 'SKILL.md'), 'utf8').includes('Short answers.'), true);
    assert.equal(readFileSync(join(live, 'concise-answers', 'helper.ps1'), 'utf8'), 'Write-Host hi');
    assert.equal(existsSync(join(live, 'empty-folder')), true);
    assert.equal(existsSync(join(live, 'commit-messages', 'SKILL.md')), true);
  } finally { cleanup(dir); }
});

test('an upgrade from the 0.2.20 layout keeps the person\'s folders and still retires Strata\'s own (#6857)', () => {
  const dir = scratch(), live = join(dir, 'agent', 'skills-live');
  try {
    // 0.2.20 left: two bundled skills (no copy note), one skill of a workspace folder (copy note), and no .strata-skills.json
    oldLayout(live, 'code-review');
    oldLayout(live, 'hardware-report');
    oldLayout(live, 'gone-skill', { note: true });
    put(join(live, 'user-profile', 'SKILL.md'), '---\nname: user-profile\ndescription: About me\n---\nI am Ann.');
    const list = [skill('code-review', 'new body'), skill('hardware-report', 'new body')];
    materializeSkills(live, list);
    assert.match(readFileSync(join(live, 'code-review', 'SKILL.md'), 'utf8'), /new body/, 'a bundled skill from the old layout is updated in place, not duplicated');
    assert.equal(existsSync(join(live, 'code-review-2')), false);
    assert.equal(existsSync(join(live, 'user-profile', 'SKILL.md')), true, 'the person\'s skill stays');
    assert.equal(existsSync(join(live, 'gone-skill')), false, 'a skill of Strata\'s that is no longer in the list is retired');
    const kept = readdirSync(join(dir, 'agent', SKILLS_REMOVED));
    assert.equal(kept.length, 1);
    assert.match(readFileSync(join(dir, 'agent', SKILLS_REMOVED, kept[0], 'SKILL.md'), 'utf8'), /old body/, 'and put aside, not deleted');
    assert.deepEqual(JSON.parse(readFileSync(join(live, '.strata-skills.json'), 'utf8')).folders, ['code-review', 'hardware-report']);
  } finally { cleanup(dir); }
});

test('after the first run only the folders named in the note are Strata\'s: a skill taken out of the app goes, the person\'s folder next to it stays (#6857)', () => {
  const dir = scratch(), live = join(dir, 'agent', 'skills-live');
  try {
    materializeSkills(live, [skill('alpha'), skill('beta')]);
    put(join(live, 'mine', 'SKILL.md'), '---\nname: mine\ndescription: m\n---\nmine');
    materializeSkills(live, [skill('alpha')]);
    assert.equal(existsSync(join(live, 'beta')), false);
    assert.equal(existsSync(join(live, 'alpha', 'SKILL.md')), true);
    assert.equal(existsSync(join(live, 'mine', 'SKILL.md')), true);
    assert.equal(readdirSync(join(dir, 'agent', SKILLS_REMOVED)).length, 1);
    // a note that cannot be read does not make every folder Strata's
    writeFileSync(join(live, '.strata-skills.json'), '{not json');
    materializeSkills(live, [skill('alpha')]);
    assert.equal(existsSync(join(live, 'mine', 'SKILL.md')), true);
  } finally { cleanup(dir); }
});

test('a skill whose name a person\'s folder already has is written under another name, and the person\'s file is not touched (#6857)', () => {
  const dir = scratch(), live = join(dir, 'agent', 'skills-live');
  try {
    materializeSkills(live, [skill('seed')]);                    // the note exists, so the old-layout leniency is off
    put(join(live, 'notes', 'SKILL.md'), '---\nname: notes\ndescription: mine\n---\nmy own notes');
    materializeSkills(live, [skill('seed'), skill('notes', 'the app\'s notes')]);
    assert.match(readFileSync(join(live, 'notes', 'SKILL.md'), 'utf8'), /my own notes/);
    assert.match(readFileSync(join(live, 'notes-2', 'SKILL.md'), 'utf8'), /the app's notes/);
    materializeSkills(live, [skill('seed'), skill('notes', 'the app\'s notes')]);   // stable on the next sync
    assert.equal(existsSync(join(live, 'notes-3')), false);
  } finally { cleanup(dir); }
});

test('a skill the app lists and that has a folder of its own keeps its copy note behaviour (#6245 still holds)', () => {
  const dir = scratch(), live = join(dir, 'agent', 'skills-live');
  try {
    const src = join(dir, 'ws', 'skills', 'demo');
    put(join(src, 'SKILL.md'), '---\nname: demo\ndescription: d\n---\nuse assets/a.txt');
    put(join(src, 'assets', 'a.txt'), 'a');
    materializeSkills(live, [skill('ws-demo', 'use assets/a.txt')], new Map([['ws-demo', src]]));
    assert.equal(readFileSync(join(live, 'demo', 'assets', 'a.txt'), 'utf8'), 'a');
    assert.equal(existsSync(join(live, 'demo', '.strata-copied.json')), true);
    materializeSkills(live, [], new Map());
    assert.equal(existsSync(join(live, 'demo')), false, 'the skill was taken out of the app');
    assert.equal(readFileSync(join(dir, 'agent', SKILLS_REMOVED, readdirSync(join(dir, 'agent', SKILLS_REMOVED))[0], 'assets', 'a.txt'), 'utf8'), 'a');
  } finally { cleanup(dir); }
});
