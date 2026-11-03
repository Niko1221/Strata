/**
 * Strata Dev (docs/DEV_BUILD.md): the host routes for Dev skills and Dev models. index.mjs answers them only when STRATA_DEV=1.
 *
 *   GET    /api/dev/skills                         every skill with the Dev fields (source, folder, validator, files, tools, versions)
 *   GET    /api/dev/skills/<id>/versions           its versions, newest first
 *   POST   /api/dev/skills/<id>/rollback {n}       version n back (folder files and the app's edit), recorded as a new version
 *   GET    /api/dev/skills/<id>/export             the skill as a .zip
 *   POST   /api/dev/skills/import {path, replace}  a skill folder or .zip into dev/skills
 *
 *   GET    /api/dev/models                         dev/models overlays, profiles, the profile chosen per overlay, the installed models
 *   POST   /api/dev/models {id, data}              writes dev/models/<id>.json (new, duplicate or edit)
 *   DELETE /api/dev/models/<id>                    removes it
 *   POST   /api/dev/models/<id>/profile {profile}  the profile Control uses for it (null: the file's own default)
 *   POST   /api/dev/models/inspect {path}          a local .gguf: its size, and whether the Strata engine knows it (Control's check)
 */
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { basename, join } from 'node:path';
import { exportSkillZip, importSkill } from './dev-skills.mjs';

/* --------------------------------------------------------------------------------------------------- skills --- */

export async function devSkillsApi(req, res, path, method, url, deps) {
  if (!path.startsWith('/api/dev/skills')) return undefined;
  const { json, error, readBody, skills, versions, rescan, sync, edits, devDir } = deps;
  const find = id => skills().find(s => s.id === id);
  if (path === '/api/dev/skills' && method === 'GET') {
    return json(res, 200, { dir: devDir, skills: skills().map(s => ({ id: s.id, name: s.name, source: s.source, folder: s.folder ?? null, dir: s.dir ?? null, validation: s.validation ?? null, files: s.files ?? [], tools: (s.tools ?? []).map(t => t.name), compatibility: s.compatibility ?? null, versions: versions.list(s.id).length })) });
  }
  if (path === '/api/dev/skills/import' && method === 'POST') {
    const b = await readBody(req);
    try {
      const result = importSkill(b.path, devDir, { replace: b.replace === true });
      rescan(); sync();
      return json(res, 200, { id: `dev-${result.folder}`, validation: result.validation });
    } catch (e) { return error(res, 400, e.message); }
  }
  const m = /^\/api\/dev\/skills\/([^/]+)\/(versions|rollback|export)$/.exec(path);
  if (!m) return error(res, 404, `No route for ${method} ${path}`);
  const skill = find(decodeURIComponent(m[1]));
  if (!skill) return error(res, 404, 'Skill not found.');
  if (m[2] === 'versions' && method === 'GET') return json(res, 200, { versions: versions.list(skill.id) });
  if (m[2] === 'export' && method === 'GET') {
    const zip = exportSkillZip(skill);
    res.writeHead(200, { 'content-type': 'application/zip', 'content-disposition': `attachment; filename="${(skill.folder ?? skill.id).replace(/[^A-Za-z0-9_.-]/g, '_')}.zip"`, 'content-length': zip.length });
    res.end(zip);
    return true;
  }
  if (m[2] === 'rollback' && method === 'POST') {
    const b = await readBody(req);
    if (!Number.isInteger(b.n)) return error(res, 400, 'n (the version number) is required.');
    try {
      const edit = versions.restore(skill, b.n);
      edits.set(skill.id, edit);
      rescan(); sync();
      return json(res, 200, { restored: b.n, versions: versions.list(skill.id) });
    } catch (e) { return error(res, 400, `Version ${b.n} could not be restored: ${e.message}`); }
  }
  return error(res, 404, `No route for ${method} ${path}`);
}

/* --------------------------------------------------------------------------------------------------- models --- */

export const BUILTIN_PROFILES = Object.freeze([
  { id: 'schnell', label: 'Schnell' },
  { id: 'coder', label: 'Coder' },
  { id: 'max-qualitaet', label: 'Max Qualität' },
]);
const OVERLAY_ID = /^[a-z0-9][a-z0-9_.-]{0,63}$/;

/** dev/models/*.json (files starting with "_" left out), each with its id and the problems that keep Control from using it. */
export function readOverlays(dir) {
  if (!existsSync(dir)) return [];
  return readdirSync(dir).filter(n => n.endsWith('.json') && !n.startsWith('_')).sort().map(name => {
    const id = name.slice(0, -5);
    try {
      const data = JSON.parse(readFileSync(join(dir, name), 'utf8'));
      const problems = overlayProblems(data);
      return { id, data, problems };
    } catch (e) { return { id, data: null, problems: [`not valid JSON: ${e.message}`] }; }
  });
}

/** What is wrong with an overlay (empty: fine). The same rule Control's dev_models.py applies, plus the field types. */
export function overlayProblems(data) {
  const out = [];
  if (!data || typeof data !== 'object' || Array.isArray(data)) return ['must be a JSON object'];
  if (typeof data.model !== 'string' || !data.model.trim()) out.push('"model" (which run configs: their model_name, a glob like qwen3.8-*) is required');
  if (data.context !== undefined && !(Number.isInteger(data.context) && data.context >= 1024)) out.push('"context" must be a whole number of tokens (at least 1024)');
  if (data.args !== undefined && (typeof data.args !== 'object' || Array.isArray(data.args))) out.push('"args" must be an object: { "--flag": value | null }');
  if (data.env !== undefined && (typeof data.env !== 'object' || Array.isArray(data.env))) out.push('"env" must be an object');
  if (data.sampling !== undefined && (typeof data.sampling !== 'object' || Array.isArray(data.sampling))) out.push('"sampling" must be an object');
  if (data.source !== undefined && !(data.source && (typeof data.source.hf === 'string' || typeof data.source.gguf === 'string' || typeof data.source.server === 'string'))) out.push('"source" is { "hf": "org/repo" } or { "gguf": "C:/path/file.gguf" } or { "server": "http://..." }');
  return out;
}

/**
 * RAM/VRAM warning for a model of `sizeGb` (a warning, never a refusal): it fits the graphics memory, needs RAM beside it, or does
 * not fit both and will read from the disk. `hardware`: { vramGb, ramGb }.
 */
export function memoryWarning(sizeGb, hardware) {
  if (!Number.isFinite(sizeGb) || sizeGb <= 0 || !hardware) return null;
  const vram = hardware.vramGb ?? 0, ram = hardware.ramGb ?? 0;
  if (sizeGb <= vram * 0.9) return null;
  if (sizeGb <= vram + ram * 0.75) return { level: 'warn', text: `About ${sizeGb.toFixed(1)} GB: more than the ${vram.toFixed(0)} GB of graphics memory; the rest runs from RAM (${ram.toFixed(0)} GB) and is slower. It can still be added.` };
  return { level: 'danger', text: `About ${sizeGb.toFixed(1)} GB: more than graphics memory and RAM together (${(vram + ram).toFixed(0)} GB); it would read from the disk and be very slow. It can still be added.` };
}

export async function devModelsApi(req, res, path, method, url, deps) {
  if (!path.startsWith('/api/dev/models')) return undefined;
  const { json, error, readBody, dir, choicesFile, installed, inspectLocal, hardware } = deps;
  const choices = () => { try { return JSON.parse(readFileSync(choicesFile, 'utf8')); } catch { return {}; } };
  const state = () => ({ dir, overlays: readOverlays(dir), profiles: BUILTIN_PROFILES, choices: choices(), installed: installed(), hardware: hardware() });
  if (path === '/api/dev/models' && method === 'GET') return json(res, 200, state());
  if (path === '/api/dev/models' && method === 'POST') {
    const b = await readBody(req);
    const id = String(b.id ?? '').trim().toLowerCase();
    if (!OVERLAY_ID.test(id)) return error(res, 400, 'The id is a file name: lower-case letters, digits, dots, dashes (for example qwen-schnell).');
    const problems = overlayProblems(b.data);
    if (problems.length) return error(res, 400, problems.join('; '));
    if (b.create === true && existsSync(join(dir, `${id}.json`))) return error(res, 409, `dev/models/${id}.json exists already. Choose another id.`);
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, `${id}.json`), `${JSON.stringify(b.data, null, 2)}\n`);
    return json(res, 200, state());
  }
  if (path === '/api/dev/models/inspect' && method === 'POST') {
    const b = await readBody(req);
    const file = String(b.path ?? '').trim();
    if (!file || !existsSync(file)) return error(res, 400, 'No file or folder at that path.');
    const files = statSync(file).isDirectory() ? readdirSync(file).filter(n => n.toLowerCase().endsWith('.gguf')).map(n => join(file, n)) : [file];
    const bytes = files.reduce((s, f) => s + statSync(f).size, 0);
    let strata = null;
    try { strata = await inspectLocal(file); } catch { /* Control could not check it: the app says "unknown" */ }
    const sizeGb = bytes / 1024 ** 3;
    return json(res, 200, { path: file, files: files.map(f => basename(f)), sizeGb, engine: strata?.selection ? 'strata' : strata ? 'llama.cpp' : 'unknown', selection: strata?.selection ?? null, warning: memoryWarning(sizeGb, hardware()) });
  }
  const one = /^\/api\/dev\/models\/([a-z0-9][a-z0-9_.-]{0,63})(\/profile)?$/.exec(path);
  if (one && !one[2] && method === 'DELETE') { rmSync(join(dir, `${one[1]}.json`), { force: true }); return json(res, 200, state()); }
  if (one && one[2] && method === 'POST') {
    const b = await readBody(req);
    const next = { ...choices() };
    if (b.profile === null || b.profile === '') delete next[one[1]];
    else if (typeof b.profile === 'string' && /^[a-z0-9][a-z0-9_.-]{0,40}$/i.test(b.profile)) next[one[1]] = b.profile;
    else return error(res, 400, 'profile is a profile name or null.');
    mkdirSync(join(choicesFile, '..'), { recursive: true });
    writeFileSync(choicesFile, `${JSON.stringify(next, null, 2)}\n`);
    return json(res, 200, state());
  }
  return error(res, 404, `No route for ${method} ${path}`);
}
