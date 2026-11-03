// Strata Dev (docs/DEV_BUILD.md): skills as whole folders (server/dev-skills.mjs), the .zip helper (dev-zip.mjs), the description-based
// selection (skill-selection.mjs), skill tools in the tool file (agent/tools-settings.mjs), the Dev models routes' checks (dev-catalog.mjs)
// and the DeepSeek fixes on the native path (dev-native.mjs).
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createSkillVersions, exportSkillZip, importSkill, materializeSkillFolders, parseFrontmatter, readSkillRoot, skillTextKeeping, skillToolCommands, validateSkill, writeSkillEdit } from '../server/dev-skills.mjs';
import { crc32, readZip, safeEntryName, writeZip } from '../server/dev-zip.mjs';
import { descriptionMatch, selectSkills, selectSkillsDev } from '../server/skill-selection.mjs';
import { materializeSkills } from '../server/skill-files.mjs';
import { createAgentTools } from '../server/agent/tools-settings.mjs';
import { memoryWarning, overlayProblems, readOverlays } from '../server/dev-catalog.mjs';
import { devExpertProfileArgs, devLoadAfterModel } from '../server/dev-native.mjs';

const tmp = () => mkdtempSync(join(tmpdir(), 'strata-dev-skills-'));
const write = (path, text) => { mkdirSync(join(path, '..'), { recursive: true }); writeFileSync(path, text); };
const SKILL = `---\nname: cloud-deploy\ndescription: Deploys a container to AWS, Google Cloud or Azure.\nmetadata:\n  variants: [aws, gcp]\nlicense: MIT\n---\n\nRead references/aws.md for AWS.\n`;

function skillTree(root) {
  write(join(root, 'cloud-deploy', 'SKILL.md'), SKILL);
  write(join(root, 'cloud-deploy', 'references', 'aws.md'), '# AWS');
  write(join(root, 'cloud-deploy', 'scripts', 'x.py'), 'print(1)');
  write(join(root, 'cloud-deploy', 'tools.json'), JSON.stringify({ tools: [{ name: 'deploy-check', description: 'checks', command: 'python scripts/x.py' }, { name: 'bad name!', command: 'x' }, { name: 'multi', command: 'a\nb' }] }));
  write(join(root, 'Bad_Skill', 'SKILL.md'), `---\nname: Bad_Skill\ndescription: \n---\n\n${'line\n'.repeat(520)}`);
  write(join(root, 'no-frontmatter', 'SKILL.md'), 'Just text.');
  write(join(root, '_hidden', 'SKILL.md'), SKILL);
}

test('frontmatter: unknown fields are kept as written; name and description are set', () => {
  const parsed = parseFrontmatter(SKILL);
  assert.equal(parsed.value('name'), 'cloud-deploy');
  assert.equal(parsed.value('license'), 'MIT');
  const text = skillTextKeeping(parsed, { name: 'cloud-deploy-2', description: 'New', body: 'Body', hidden: true });
  assert.match(text, /^---\nname: cloud-deploy-2\ndescription: "New"\nmetadata:\n {2}variants: \[aws, gcp\]\nlicense: MIT\ndisable-model-invocation: true\n---\n\nBody\n$/);
  assert.equal(parseFrontmatter('---\ndescription: >\n  folded\n  text\n---\nx').value('description'), 'folded text');
  assert.equal(parseFrontmatter('no frontmatter').ok, false);
});

test('the validator: frontmatter, kebab-case name, a description, the 500-line warning', () => {
  assert.deepEqual(validateSkill(SKILL, 'cloud-deploy').errors, []);
  const bad = validateSkill(`---\nname: Bad_Skill\ndescription: \n---\n\n${'line\n'.repeat(520)}`, 'Bad_Skill');
  assert.ok(bad.errors.some(e => /kebab-case/.test(e)));
  assert.ok(bad.errors.some(e => /description/.test(e)));
  assert.ok(bad.warnings.some(w => /under 500/.test(w)));
  assert.match(validateSkill('Just text.').errors[0], /no frontmatter/);
  assert.match(validateSkill(SKILL, 'other').warnings[0], /differs from the folder/);
});

test('dev/skills is read as folders; the agent folder gets the whole folder, unknown fields kept, unchanged folders not rewritten', () => {
  const root = tmp(), live = tmp();
  try {
    skillTree(root);
    const skills = readSkillRoot(root, { source: 'dev', prefix: 'dev-' });
    assert.deepEqual(skills.map(s => s.id), ['dev-Bad_Skill', 'dev-cloud-deploy', 'dev-no-frontmatter']);
    const cloud = skills.find(s => s.id === 'dev-cloud-deploy');
    assert.deepEqual(cloud.files, ['references/aws.md', 'scripts/x.py', 'SKILL.md', 'tools.json']);
    assert.deepEqual(cloud.tools.map(t => t.name), ['deploy-check'], 'tools with a bad name or a multi-line command are left out');
    const names = materializeSkillFolders(live, [cloud, { id: 'commit-messages', name: 'Commit messages', description: 'd', body: 'b', enabled: false }]);
    assert.deepEqual(names, ['cloud-deploy', 'commit-messages']);
    assert.ok(existsSync(join(live, 'cloud-deploy', 'references', 'aws.md')));
    assert.ok(existsSync(join(live, 'cloud-deploy', 'scripts', 'x.py')));
    const text = readFileSync(join(live, 'cloud-deploy', 'SKILL.md'), 'utf8');
    assert.match(text, /license: MIT/);
    assert.match(text, /description: "Cloud deploy\. Deploys a container/);
    assert.match(readFileSync(join(live, 'commit-messages', 'SKILL.md'), 'utf8'), /disable-model-invocation: true/);
    // Unchanged: not rewritten (the Harness reloads on every write). Changed: copied again; removed: gone.
    writeFileSync(join(live, 'cloud-deploy', 'marker'), 'x');
    materializeSkillFolders(live, [cloud]);
    assert.ok(existsSync(join(live, 'cloud-deploy', 'marker')), 'an unchanged folder is left alone');
    assert.ok(!existsSync(join(live, 'commit-messages')));
    write(join(root, 'cloud-deploy', 'references', 'gcp.md'), '# GCP');
    materializeSkillFolders(live, readSkillRoot(root, { source: 'dev', prefix: 'dev-' }).filter(s => s.id === 'dev-cloud-deploy'));
    assert.ok(existsSync(join(live, 'cloud-deploy', 'references', 'gcp.md')));
    assert.ok(!existsSync(join(live, 'cloud-deploy', 'marker')));
  } finally { rmSync(root, { recursive: true, force: true }); rmSync(live, { recursive: true, force: true }); }
});

test('Live keeps its text-only copy (skill-files.mjs is unchanged): no resources next to SKILL.md', () => {
  const live = tmp();
  try {
    materializeSkills(live, [{ id: 'ws-x', name: 'X', description: 'd', body: 'b' }]);
    assert.deepEqual(readSkillRoot(live, { source: 'x', prefix: '' })[0].files, ['SKILL.md']);
  } finally { rmSync(live, { recursive: true, force: true }); }
});

test('versions: a change is a new version, the same state is not, a version restores files and the edit', () => {
  const root = tmp(), store = tmp();
  try {
    skillTree(root);
    const skill = readSkillRoot(root, { source: 'dev', prefix: 'dev-' }).find(s => s.id === 'dev-cloud-deploy');
    const v = createSkillVersions(store);
    assert.equal(v.record(skill), 1);
    assert.equal(v.record(skill), null, 'nothing changed');
    writeFileSync(join(root, 'cloud-deploy', 'references', 'aws.md'), '# AWS changed');
    write(join(root, 'cloud-deploy', 'references', 'new.md'), 'new');
    assert.equal(v.record(skill, { edit: { triggers: ['x'] } }), 2);
    assert.deepEqual(v.list(skill.id).map(x => x.n), [2, 1]);
    assert.equal(v.restore(skill, 1), null);
    assert.equal(readFileSync(join(root, 'cloud-deploy', 'references', 'aws.md'), 'utf8'), '# AWS');
    assert.ok(!existsSync(join(root, 'cloud-deploy', 'references', 'new.md')), 'a file the version did not have is removed');
    assert.deepEqual(v.restore(skill, 2), { triggers: ['x'] });
    // An edit made in the app goes into the Dev skill's own SKILL.md, frontmatter kept.
    writeSkillEdit(skill, { description: 'Changed description', body: 'New body' });
    const text = readFileSync(join(root, 'cloud-deploy', 'SKILL.md'), 'utf8');
    assert.match(text, /name: cloud-deploy\ndescription: "Changed description"\nmetadata:/);
    assert.match(text, /license: MIT/);
    assert.match(text, /\n\nNew body\n$/);
  } finally { rmSync(root, { recursive: true, force: true }); rmSync(store, { recursive: true, force: true }); }
});

test('zip: written and read back; a path out of the folder is refused', () => {
  const zip = writeZip([{ name: 'a/SKILL.md', data: Buffer.from('hello') }, { name: 'a\\b\\c.txt', data: Buffer.from('x'.repeat(5000)) }]);
  assert.deepEqual(readZip(zip).map(f => [f.name, f.data.length]), [['a/SKILL.md', 5], ['a/b/c.txt', 5000]]);
  assert.equal(crc32(Buffer.from('123456789')), 0xcbf43926);
  for (const bad of ['../x', '/etc/x', 'C:/x', 'a/../../x']) assert.equal(safeEntryName(bad), null, bad);
  assert.throws(() => readZip(writeZip([{ name: '../evil', data: Buffer.from('x') }])), /outside its folder/);
  assert.throws(() => readZip(Buffer.from('not a zip at all, definitely not')), /not a \.zip/);
});

test('import and export: a folder or a .zip into dev/skills, an existing one only with replace', () => {
  const root = tmp(), target = tmp(), out = tmp();
  try {
    skillTree(root);
    const r = importSkill(join(root, 'cloud-deploy'), target);
    assert.equal(r.folder, 'cloud-deploy');
    assert.deepEqual(r.validation.errors, []);
    assert.throws(() => importSkill(join(root, 'cloud-deploy'), target), /exists already/);
    const skill = readSkillRoot(target, { source: 'dev', prefix: 'dev-' })[0];
    const zipFile = join(out, 'cloud.zip');
    writeFileSync(zipFile, exportSkillZip(skill));
    assert.ok(readZip(readFileSync(zipFile)).every(f => f.name.startsWith('cloud-deploy/')));
    rmSync(join(target, 'cloud-deploy'), { recursive: true });
    const again = importSkill(zipFile, target);
    assert.equal(again.folder, 'cloud-deploy');
    assert.equal(readFileSync(join(target, 'cloud-deploy', 'references', 'aws.md'), 'utf8'), '# AWS');
    // A skill without a folder exports a SKILL.md made from its fields.
    const built = readZip(exportSkillZip({ id: 'commit-messages', name: 'Commit messages', description: 'Writes commits', body: 'Do it' }));
    assert.equal(built[0].name, 'commit-messages/SKILL.md');
    assert.match(built[0].data.toString(), /name: commit-messages\ndescription: "Writes commits"/);
    assert.throws(() => importSkill(join(root, 'nowhere'), target), /Nothing is at/);
    writeFileSync(join(out, 'x.txt'), 'x');
    assert.throws(() => importSkill(join(out, 'x.txt'), target), /skill folder or a \.zip/);
  } finally { for (const d of [root, target, out]) rmSync(d, { recursive: true, force: true }); }
});

test('selection: Live keeps triggers only; Dev adds the description and the chat\'s own choice', () => {
  const skills = [
    { id: 'a', name: 'Cloud deploy', description: 'Deploys containers to AWS, Google Cloud or Azure with a checklist.', body: 'x', enabled: true, triggers: ['deploy it'] },
    { id: 'b', name: 'Meeting notes', description: 'Turns meeting notes into a protocol with decisions and action items.', body: 'y', enabled: true, triggers: ['minutes'] },
    { id: 'c', name: 'Off', description: 'meeting protocol decisions', body: 'z', enabled: false, triggers: [] },
  ];
  const msg = 'Please make a protocol of these meeting notes with all decisions';
  assert.deepEqual(selectSkills(skills, msg, null).map(s => s.id), [], 'Live: no trigger phrase, no skill');
  assert.deepEqual(selectSkillsDev(skills, msg, null).map(s => s.id), ['b']);
  assert.ok(descriptionMatch(skills[1], msg) >= 2);
  assert.deepEqual(selectSkillsDev(skills, 'deploy it please', null).map(s => s.id), ['a'], 'triggers still work');
  assert.deepEqual(selectSkillsDev(skills, msg, null, { allowed: ['a'] }).map(s => s.id), [], 'only the chat\'s skills');
});

test('skill tools: written as command tools beside the person\'s own, only from skills that are on', () => {
  const data = tmp(), root = tmp();
  try {
    skillTree(root);
    const skills = readSkillRoot(root, { source: 'dev', prefix: 'dev-' });
    const commands = skillToolCommands(skills);
    assert.deepEqual(Object.keys(commands), ['skill-cloud-deploy-deploy-check']);
    assert.equal(commands['skill-cloud-deploy-deploy-check'].cwd, join(root, 'cloud-deploy'));
    assert.deepEqual(skillToolCommands(skills.map(s => ({ ...s, enabled: false }))), {});
    let saved = {};
    const tools = createAgentTools({ read: () => saved, write: v => { saved = v; }, dataDir: data, extraCommands: () => commands });
    tools.ensureFile();
    const file = JSON.parse(readFileSync(tools.filePath, 'utf8'));
    assert.deepEqual(Object.keys(file.commands), ['skill-cloud-deploy-deploy-check']);
    assert.equal(tools.profileView().hasCommandTools, true);
    const live = createAgentTools({ read: () => ({}), write: () => {}, dataDir: data });
    live.ensureFile();
    assert.deepEqual(JSON.parse(readFileSync(live.filePath, 'utf8')).commands, {}, 'without extraCommands (Live) nothing is added');
    assert.equal(live.profileView().hasCommandTools, false);
  } finally { rmSync(data, { recursive: true, force: true }); rmSync(root, { recursive: true, force: true }); }
});

test('Dev models: overlay checks, the RAM/VRAM warning (never a refusal), the listing', () => {
  assert.deepEqual(overlayProblems({ model: 'qwen*', context: 65536, args: { '--kv': 'q8_0' } }), []);
  assert.ok(overlayProblems({}).some(p => /"model"/.test(p)));
  assert.ok(overlayProblems({ model: 'x', context: 10 }).some(p => /context/.test(p)));
  assert.ok(overlayProblems({ model: 'x', source: { url: 'y' } }).some(p => /source/.test(p)));
  assert.equal(memoryWarning(8, { vramGb: 12, ramGb: 64 }), null);
  assert.equal(memoryWarning(30, { vramGb: 12, ramGb: 64 }).level, 'warn');
  assert.equal(memoryWarning(300, { vramGb: 12, ramGb: 64 }).level, 'danger');
  assert.equal(memoryWarning(30, null), null);
  const dir = tmp();
  try {
    writeFileSync(join(dir, 'a.json'), JSON.stringify({ model: 'qwen*' }));
    writeFileSync(join(dir, 'b.json'), '{ broken');
    writeFileSync(join(dir, '_c.json'), '{}');
    const list = readOverlays(dir);
    assert.deepEqual(list.map(o => [o.id, o.problems.length > 0]), [['a', false], ['b', true]]);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('DeepSeek in Dev: the shared pack\'s expert profile is copied (or read only), and loadAfter takes the installed id', () => {
  const home = tmp(), pack = tmp();
  try {
    const profile = join(pack, 'data', 'expert-profile.bin');
    write(profile, 'profile');
    const args = ['--pack', 'x', '--expert-profile', profile, '--gpus', '0'];
    const out = devExpertProfileArgs(args, { devHome: home });
    assert.notEqual(out.args[3], profile);
    assert.ok(out.args[3].startsWith(home));
    assert.equal(readFileSync(out.args[3], 'utf8'), 'profile');
    assert.match(out.note, /copy/);
    assert.deepEqual(args[3], profile, 'the input list is not changed');
    const ro = devExpertProfileArgs(['--expert-profile', join(pack, 'missing.bin')], { devHome: home });
    assert.deepEqual(ro.args.slice(-1), ['--profile-readonly']);
    assert.deepEqual(devExpertProfileArgs(['--expert-profile', out.args[3]], { devHome: home }).args, ['--expert-profile', out.args[3]], 'already in Dev: unchanged');
    assert.deepEqual(devExpertProfileArgs(args, { devHome: null }).args, args);
  } finally { rmSync(home, { recursive: true, force: true }); rmSync(pack, { recursive: true, force: true }); }
  const installed = [{ model: 'q2_0' }, { model: 'deepseek-v4.1-flash-ist-das-2.00bpw' }];
  assert.equal(devLoadAfterModel('deepseek41', 'IST-DAS-2.00bpw', installed), 'deepseek-v4.1-flash-ist-das-2.00bpw');
  assert.equal(devLoadAfterModel('qwen', 'Q2_0', installed), 'q2_0');
  assert.equal(devLoadAfterModel('deepseek41', 'other', installed), 'deepseek41-other', 'nothing fits: the old guess');
});
