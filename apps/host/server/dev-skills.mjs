/**
 * Strata Dev (docs/DEV_BUILD.md): skills as whole folders, in the Claude skill format (skill-struktur.md). Only index.mjs uses this, and
 * only when STRATA_DEV=1; Live keeps skill-files.mjs's text-only copy.
 *
 *   - a third source beside <data>/skills and <workspace>/skills: dev/skills/<name>/ (ids `dev-<name>`);
 *   - the agent's folder (<data>/agent/skills-live) gets the whole skill folder (scripts/, references/, assets/, any other), so
 *     "references/aws.md" or "scripts/x.py" resolve against the "Base directory" the Harness names when it loads the skill;
 *     the SKILL.md written there keeps every frontmatter field it had (only name, description and disable-model-invocation are set);
 *   - a validator (frontmatter, kebab-case name, a description, under 500 lines);
 *   - versions: every change is kept, and a version can be restored;
 *   - tools a skill brings (tools.json), served as command tools through the app's own tool server (agent/tools-settings.mjs);
 *   - import and export as a folder or a .zip.
 * Scripts in a skill are never run by the app itself: the agent runs them with its shell, under the chat's Manual/Plan/Bypass mode,
 * or a tool from tools.json runs them, which Manual asks for like every command tool.
 */
import { createHash } from 'node:crypto';
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { basename, dirname, extname, join, relative, resolve, sep } from 'node:path';
import { skillSlug } from './skill-files.mjs';
import { readZip, safeEntryName, writeZip } from './dev-zip.mjs';

export const MAX_LINES = 500;
const KEBAB = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const SKIP_DIRS = new Set(['.git', 'node_modules', '__pycache__', '.venv']);
const SIG_FILE = '.strata-skill-sig';
const MAX_SNAPSHOT_BYTES = 5 * 1024 * 1024;
const MAX_VERSIONS = 30;

/* --------------------------------------------------------------------------------------------- frontmatter --- */

/**
 * The frontmatter as ordered raw fields (`key` -> the lines as written, nested lines included), so unknown fields go back
 * unchanged; `value(key)` gives a plain scalar. No frontmatter: `ok` is false and the whole text is the body.
 */
export function parseFrontmatter(text) {
  const raw = String(text ?? '').replace(/^﻿/, '').replace(/\r\n/g, '\n');
  const m = /^---\n([\s\S]*?)\n---[ \t]*(?:\n|$)/.exec(raw);
  if (!m) return { ok: false, fields: [], body: raw.trim(), value: () => null };
  const fields = [];
  for (const line of m[1].split('\n')) {
    const key = /^([A-Za-z0-9_-]+):(.*)$/.exec(line);
    if (key) fields.push({ key: key[1], lines: [line] });
    else if (fields.length) fields[fields.length - 1].lines.push(line);
  }
  const value = name => {
    const f = fields.find(x => x.key === name);
    if (!f) return null;
    const first = f.lines[0].slice(name.length + 1).trim();
    const folded = /^[>|][+-]?$/.test(first) ? f.lines.slice(1).map(l => l.trim()).join(first.startsWith('>') ? ' ' : '\n').trim() : first;
    if (/^"/.test(folded)) { try { return String(JSON.parse(folded)); } catch { /* not JSON-quoted */ } }
    if (/^'.*'$/.test(folded)) return folded.slice(1, -1).replace(/''/g, "'");
    return folded;
  };
  return { ok: true, fields, body: raw.slice(m[0].length).trim(), value };
}

const yamlText = value => JSON.stringify(String(value ?? '').replace(/\s+/g, ' ').trim());

/** SKILL.md text: name and description set (and disable-model-invocation when off), every other field as it was. */
export function skillTextKeeping(parsed, { name, description, body, hidden = false }) {
  const own = new Set(['name', 'description', 'disable-model-invocation']);
  const kept = parsed.fields.filter(f => !own.has(f.key)).flatMap(f => f.lines);
  return `---\nname: ${name}\ndescription: ${yamlText(description)}\n${kept.length ? `${kept.join('\n')}\n` : ''}${hidden ? 'disable-model-invocation: true\n' : ''}---\n\n${String(body ?? '').trim()}\n`;
}

/* ----------------------------------------------------------------------------------------------- validator --- */

/** `{ errors, warnings, lines }`: what keeps the Harness from loading the skill, and what is only advice. */
export function validateSkill(text, folder = null) {
  const parsed = parseFrontmatter(text);
  const errors = [], warnings = [];
  const lines = String(text ?? '').replace(/\r\n/g, '\n').split('\n').length;
  if (!parsed.ok) errors.push('SKILL.md has no frontmatter (--- name, description --- at the top).');
  const name = parsed.value('name'), description = parsed.value('description');
  if (parsed.ok && !name) errors.push('The frontmatter has no "name".');
  else if (name && !KEBAB.test(name)) errors.push(`"name" must be kebab-case (lower-case letters, digits and dashes): "${name}".`);
  if (parsed.ok && !String(description ?? '').trim()) errors.push('"description" is empty: it is what the assistant decides by.');
  if (lines > MAX_LINES) warnings.push(`SKILL.md has ${lines} lines; keep it under ${MAX_LINES} and move details to references/.`);
  if (name && folder && name !== folder) warnings.push(`"name" (${name}) differs from the folder name (${folder}).`);
  if (!parsed.body) warnings.push('SKILL.md has no instructions below the frontmatter.');
  return { errors, warnings, lines };
}

/* ----------------------------------------------------------------------------------------------- folders --- */

/** Every file below `dir` (relative paths with "/"), the usual clutter folders left out. */
export function listFiles(dir) {
  const out = [];
  const walk = (at, rel) => {
    for (const entry of readdirSync(at, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
      if (entry.name === SIG_FILE) continue;
      const full = join(at, entry.name), r = rel ? `${rel}/${entry.name}` : entry.name;
      if (entry.isDirectory()) { if (!SKIP_DIRS.has(entry.name)) walk(full, r); }
      else if (entry.isFile()) { const s = statSync(full); out.push({ rel: r, size: s.size, mtimeMs: s.mtimeMs }); }
    }
  };
  if (existsSync(dir)) walk(dir, '');
  return out;
}

/** tools.json of a skill: `{ "tools": [{ "name", "description", "command", "shell"?, "timeoutSec"? }] }`. Wrong entries are left out. */
export function readSkillTools(dir) {
  let data;
  try { data = JSON.parse(readFileSync(join(dir, 'tools.json'), 'utf8')); } catch { return []; }
  const list = Array.isArray(data?.tools) ? data.tools : [];
  return list.filter(t => t && typeof t.name === 'string' && /^[a-z0-9][a-z0-9_-]{0,40}$/i.test(t.name) && typeof t.command === 'string' && t.command.trim() && !/[\r\n]/.test(t.command))
    .map(t => ({ name: t.name, description: String(t.description ?? '').slice(0, 500), command: t.command.trim(), shell: ['auto', 'cmd', 'powershell', 'bash', 'sh'].includes(t.shell) ? t.shell : 'auto', timeoutSec: Number.isInteger(t.timeoutSec) ? t.timeoutSec : 120 }));
}

/** One skill folder as a skill entry (the shape index.mjs's skills() makes, with the Dev fields: dir, folder, validation, files, tools). */
export function readSkillFolder(dir, { source, prefix }) {
  const folder = basename(dir);
  let text = '';
  try { text = readFileSync(join(dir, 'SKILL.md'), 'utf8'); } catch { /* listed, with the validator saying why it is empty */ }
  const parsed = parseFrontmatter(text);
  const files = listFiles(dir);
  return {
    id: `${prefix}${folder}`,
    name: parsed.value('title') ?? folder.replace(/-/g, ' ').replace(/^./, c => c.toUpperCase()),
    description: parsed.value('description') ?? `Skill loaded from ${folder}/SKILL.md`,
    source,
    enabled: true,
    triggers: [folder.replace(/-/g, ' ')],
    body: parsed.body,
    sizeKb: Number(Math.max(0.1, files.reduce((s, f) => s + f.size, 0) / 1024).toFixed(1)),
    dir,
    folder,
    validation: text ? validateSkill(text, folder) : { errors: ['The folder has no SKILL.md.'], warnings: [], lines: 0 },
    files: files.map(f => f.rel),
    tools: readSkillTools(dir),
    compatibility: parsed.value('compatibility'),
  };
}

/** The skills in dev/skills (and the Dev fields for created and workspace skills, which are folders too). */
export function readSkillRoot(root, { source, prefix }) {
  if (!existsSync(root)) return [];
  const out = [];
  for (const name of readdirSync(root).sort()) {
    if (name.startsWith('_') || name.startsWith('.')) continue;
    const dir = join(root, name);
    try { if (!statSync(dir).isDirectory()) continue; } catch { continue; }
    out.push(readSkillFolder(dir, { source, prefix }));
  }
  return out;
}

/* ----------------------------------------------------------------------------------------- the agent's folder --- */

function treeSignature(dir, text) {
  const h = createHash('sha256').update(text);
  for (const f of listFiles(dir)) h.update(`${f.rel}\0${f.size}\0${f.mtimeMs}\n`);
  return h.digest('hex').slice(0, 20);
}

/**
 * Make `root` hold exactly the skills the agent may use. A skill with a folder (`dir`) is copied whole, then its SKILL.md is
 * written with the unique name and its current description and instructions (an edit made in the app included), every other
 * frontmatter field kept. A skill without a folder (the built-in ones) is a SKILL.md alone, as skill-files.mjs writes it.
 * A folder is copied again only when it or its SKILL.md changed (the Harness treats every write as a change to its list).
 */
export function materializeSkillFolders(root, skills) {
  mkdirSync(root, { recursive: true });
  const keep = new Set();
  for (const s of skills) {
    const slug = skillSlug(s.id.replace(/^(?:ws|my|dev)-/, '') || s.name);
    if (!slug || !s.body?.trim()) continue;
    let unique = slug, n = 2;
    while (keep.has(unique)) unique = `${slug}-${n++}`;
    keep.add(unique);
    const target = join(root, unique);
    let parsed = { fields: [] };
    if (s.dir) { try { parsed = parseFrontmatter(readFileSync(join(s.dir, 'SKILL.md'), 'utf8')); } catch { /* written from the entry */ } }
    const text = skillTextKeeping(parsed, { name: unique, description: `${s.name}. ${s.description ?? ''}`.trim(), body: s.body, hidden: s.enabled === false });
    const sig = s.dir ? treeSignature(s.dir, text) : createHash('sha256').update(text).digest('hex').slice(0, 20);
    let same = false;
    try { same = readFileSync(join(target, SIG_FILE), 'utf8') === sig; } catch { /* new */ }
    if (same) continue;
    rmSync(target, { recursive: true, force: true });
    mkdirSync(target, { recursive: true });
    if (s.dir && existsSync(s.dir)) cpSync(s.dir, target, { recursive: true, filter: src => !SKIP_DIRS.has(basename(src)) });
    writeFileSync(join(target, 'SKILL.md'), text);
    writeFileSync(join(target, SIG_FILE), sig);
  }
  for (const entry of readdirSync(root, { withFileTypes: true })) if (entry.isDirectory() && !keep.has(entry.name)) rmSync(join(root, entry.name), { recursive: true, force: true });
  return [...keep];
}

/** The command tools the skills bring, for agent/tools-settings.mjs (`extraCommands`): `skill-<skill>-<tool>`, run in the skill's folder. */
export function skillToolCommands(skills) {
  const out = {};
  for (const s of skills) {
    if (s.enabled === false || !s.dir) continue;
    for (const t of s.tools ?? []) {
      const key = `skill-${skillSlug(s.folder ?? s.id)}-${t.name}`.slice(0, 60);
      out[key] = { description: `${t.description || t.name} (from the skill "${s.name}")`, command: t.command, cwd: s.dir, shell: t.shell, timeoutSec: t.timeoutSec };
    }
  }
  return out;
}

/* ------------------------------------------------------------------------------------------------- versions --- */

function snapshotFiles(dir) {
  const files = [];
  let total = 0, truncated = false;
  for (const f of listFiles(dir)) {
    if (total + f.size > MAX_SNAPSHOT_BYTES) { truncated = true; continue; }
    total += f.size;
    files.push({ rel: f.rel, b64: readFileSync(join(dir, ...f.rel.split('/'))).toString('base64') });
  }
  return { files, truncated };
}

/**
 * Versions of every skill under `root/<id>/<n>.json`: the files of its folder (up to 5 MB) and the app's edit of it.
 * A version is only written when something differs from the newest one.
 */
export function createSkillVersions(root) {
  const folderOf = id => join(root, skillSlug(id) || 'skill');
  const numbers = id => { try { return readdirSync(folderOf(id)).map(n => /^(\d+)\.json$/.exec(n)?.[1]).filter(Boolean).map(Number).sort((a, b) => a - b); } catch { return []; } };
  const read = (id, n) => JSON.parse(readFileSync(join(folderOf(id), `${String(n).padStart(4, '0')}.json`), 'utf8'));
  return {
    list(id) {
      return numbers(id).map(n => { const v = read(id, n); return { n: v.n, at: v.at, reason: v.reason, files: v.files.length, truncated: v.truncated, edited: !!v.edit && Object.keys(v.edit).length > 0 }; }).reverse();
    },
    get: read,
    /** A new version of `skill` (its folder, `edit`) unless the newest one is the same. Returns its number, or null. */
    record(skill, { edit = null, reason = 'change' } = {}) {
      const { files, truncated } = skill.dir && existsSync(skill.dir) ? snapshotFiles(skill.dir) : { files: [], truncated: false };
      const hash = createHash('sha256').update(JSON.stringify([files, edit ?? null])).digest('hex').slice(0, 20);
      const have = numbers(skill.id);
      const last = have.length ? read(skill.id, have.at(-1)) : null;
      if (last?.hash === hash) return null;
      const n = (have.at(-1) ?? 0) + 1;
      mkdirSync(folderOf(skill.id), { recursive: true });
      writeFileSync(join(folderOf(skill.id), `${String(n).padStart(4, '0')}.json`), JSON.stringify({ n, at: new Date().toISOString(), reason, hash, edit, files, truncated }));
      for (const old of have.slice(0, Math.max(0, have.length + 1 - MAX_VERSIONS))) rmSync(join(folderOf(skill.id), `${String(old).padStart(4, '0')}.json`), { force: true });
      return n;
    },
    /** Puts version `n` back: its files into the skill's folder (files it did not have are removed, unless it was cut at 5 MB). Returns its edit. */
    restore(skill, n) {
      const v = read(skill.id, n);
      if (skill.dir && v.files.length) {
        const keep = new Set(v.files.map(f => f.rel));
        if (!v.truncated) for (const f of listFiles(skill.dir)) if (!keep.has(f.rel)) rmSync(join(skill.dir, ...f.rel.split('/')), { force: true });
        for (const f of v.files) {
          const path = join(skill.dir, ...f.rel.split('/'));
          mkdirSync(dirname(path), { recursive: true });
          writeFileSync(path, Buffer.from(f.b64, 'base64'));
        }
      }
      return v.edit ?? null;
    },
  };
}

/* -------------------------------------------------------------------------------------------- import / export --- */

/** The files of a skill to export: its folder, or a SKILL.md made from the entry for a skill without one. */
export function exportSkillZip(skill) {
  const folder = skill.folder ?? skillSlug(skill.id.replace(/^(?:ws|my|dev)-/, '')) ?? 'skill';
  if (skill.dir && existsSync(skill.dir)) {
    return writeZip(listFiles(skill.dir).map(f => ({ name: `${folder}/${f.rel}`, data: readFileSync(join(skill.dir, ...f.rel.split('/'))) })));
  }
  const text = skillTextKeeping({ fields: [] }, { name: skillSlug(folder), description: skill.description ?? skill.name, body: skill.body ?? '' });
  return writeZip([{ name: `${folder}/SKILL.md`, data: Buffer.from(text) }]);
}

/**
 * Imports a skill folder or a .zip of one into `root` (dev/skills). The name comes from the folder (a .zip's single top folder, or
 * its file name). Refuses a name that exists unless `replace` is set. Returns `{ folder, dir, validation }`.
 */
export function importSkill(source, root, { replace = false } = {}) {
  const from = resolve(String(source ?? ''));
  if (!existsSync(from)) throw new Error(`Nothing is at ${from}.`);
  let files, name;
  if (statSync(from).isDirectory()) {
    if (!existsSync(join(from, 'SKILL.md'))) throw new Error('That folder has no SKILL.md, so it is not a skill.');
    name = basename(from);
    files = listFiles(from).map(f => ({ name: f.rel, data: readFileSync(join(from, ...f.rel.split('/'))) }));
  } else if (extname(from).toLowerCase() === '.zip' || extname(from).toLowerCase() === '.skill') {
    const all = readZip(readFileSync(from));
    const tops = new Set(all.map(f => f.name.split('/')[0]));
    const nested = tops.size === 1 && all.every(f => f.name.includes('/'));
    name = nested ? [...tops][0] : basename(from, extname(from));
    files = all.map(f => ({ name: nested ? f.name.slice(name.length + 1) : f.name, data: f.data }));
    if (!files.some(f => f.name === 'SKILL.md')) throw new Error('The archive has no SKILL.md at its top (or in its one top folder).');
  } else {
    throw new Error('Choose a skill folder or a .zip file.');
  }
  const folder = skillSlug(name);
  if (!folder) throw new Error('The skill needs a name with letters or digits.');
  const dir = join(root, folder);
  if (existsSync(dir) && !replace) throw new Error(`dev/skills/${folder} exists already. Remove or rename it, or import with "replace".`);
  rmSync(dir, { recursive: true, force: true });
  for (const f of files) {
    const rel = safeEntryName(f.name);
    if (!rel) continue;
    const path = join(dir, ...rel.split('/'));
    if (!resolve(path).startsWith(resolve(dir) + sep)) continue;
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, f.data);
  }
  return { folder, dir, validation: validateSkill(readFileSync(join(dir, 'SKILL.md'), 'utf8'), folder) };
}

/** Writes an edit made in the app into a Dev skill's own SKILL.md (frontmatter kept), so the folder is the truth. */
export function writeSkillEdit(skill, edit) {
  const file = join(skill.dir, 'SKILL.md');
  const parsed = parseFrontmatter(existsSync(file) ? readFileSync(file, 'utf8') : '');
  const name = parsed.value('name') ?? skill.folder;
  const fields = parsed.fields.filter(f => f.key !== 'title' || edit.name === undefined);
  const titled = edit.name !== undefined && edit.name !== skill.folder ? [...fields, { key: 'title', lines: [`title: ${yamlText(edit.name)}`] }] : fields;
  writeFileSync(file, skillTextKeeping({ fields: titled }, { name, description: edit.description ?? parsed.value('description') ?? '', body: edit.body ?? parsed.body }));
}

export const relativeTo = (root, path) => relative(root, path).split(sep).join('/');
