/** Immutable copies of the App's own frontend. Does not serve, install or activate it. */
import {constants} from 'node:fs';
import {lstat, mkdir, open, readdir, writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {dirname, isAbsolute, join, relative, resolve, sep} from 'node:path';
import {inspectApp} from './contract.mjs';

const MAX_FILE = 64 * 1024 * 1024;
const MAX_TOTAL = 256 * 1024 * 1024;
const MAX_FILES = 2000;
const HASH = /^[a-f0-9]{64}$/;
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const canonical = value => JSON.stringify(value, (_, item) =>
  item && typeof item === 'object' && !Array.isArray(item)
    ? Object.fromEntries(Object.keys(item).sort().map(key => [key, item[key]])) : item);

export async function noLinks(path) {
  let current = resolve(path);
  while (true) {
    if ((await lstat(current)).isSymbolicLink()) throw Error('linked_asset_path');
    const parent = dirname(current);
    if (parent === current) break;
    current = parent;
  }
}

export async function regularBytes(path, limit = MAX_FILE) {
  await noLinks(path);
  const file = await open(path, constants.O_RDONLY | (constants.O_NOFOLLOW || 0));
  try {
    const before = await file.stat();
    if (!before.isFile() || before.size > limit) throw Error('invalid_asset_size');
    const data = Buffer.alloc(before.size + 1);
    let size = 0;
    while (size < data.length) {
      const read = await file.read(data, size, data.length - size, size);
      if (!read.bytesRead) break;
      size += read.bytesRead;
    }
    const after = await file.stat();
    if (size !== before.size || after.size !== before.size || after.mtimeMs !== before.mtimeMs)
      throw Error('asset_changed_during_read');
    await noLinks(path);
    return data.subarray(0, size);
  } finally { await file.close(); }
}

function assetName(name) {
  if (typeof name !== 'string' || !name || /[\\%?#\x00-\x1f]/.test(name)
      || name.split('/').some(part => !part || part.startsWith('.')) || name.includes(':'))
    throw Error('invalid_asset_name');
  return name;
}

export async function inventory(root) {
  await noLinks(root);
  if (!(await lstat(root)).isDirectory()) throw Error('asset_directory_required');
  const files = [];
  let total = 0, directories = 0;
  async function visit(folder, prefix = '') {
    if (++directories > MAX_FILES) throw Error('excessive_asset_directories');
    for (const entry of (await readdir(folder, {withFileTypes:true})).sort((a,b) => a.name < b.name ? -1 : 1)) {
      const name = assetName(prefix + entry.name), path = join(folder, entry.name);
      if (entry.isSymbolicLink()) throw Error('linked_asset_path');
      if (entry.isDirectory()) await visit(path, name + '/');
      else if (entry.isFile()) {
        if (files.length >= MAX_FILES) throw Error('excessive_asset_files');
        const data = await regularBytes(path);
        total += data.length;
        if (total > MAX_TOTAL) throw Error('excessive_asset_bytes');
        files.push({path:name, size:data.length, sha256:hash(data)});
      } else throw Error('non_regular_asset');
    }
  }
  await visit(root);
  files.sort((a,b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0);
  if (!files.some(file => file.path === 'index.html')) throw Error('app_entry_missing');
  return files;
}

function identity(current) {
  if (!current || current.state !== 'compatible' || !HASH.test(current.app?.fingerprint || '')
      || typeof current.app.version !== 'string') throw Error('app_contract_needs_validation');
  return current.app;
}

export async function verifyCandidate(root, {manifestSha256, current} = {}) {
  if (!isAbsolute(root) || !HASH.test(manifestSha256 || '')) throw Error('pinned_manifest_required');
  await noLinks(root);
  const names = (await readdir(root)).sort();
  if (canonical(names) !== canonical(['dist', 'manifest.json'])) throw Error('candidate_inventory_changed');
  const raw = await regularBytes(join(root, 'manifest.json'), 2 * 1024 * 1024);
  if (hash(raw) !== manifestSha256) throw Error('candidate_manifest_changed');
  const manifest = JSON.parse(raw);
  if (canonical(Object.keys(manifest).sort()) !== canonical(['app','capabilities','component','files','schema','upstreamPatches'])
      || manifest.schema !== 1 || manifest.component !== 'strata-app-surface'
      || canonical(manifest.upstreamPatches) !== '[]' || !Array.isArray(manifest.files)
      || canonical(identity(manifest.capabilities)) !== canonical(manifest.app)) throw Error('invalid_candidate_manifest');
  for (const file of manifest.files) {
    assetName(file.path);
    if (canonical(Object.keys(file).sort()) !== canonical(['path','sha256','size'])
        || !Number.isSafeInteger(file.size) || file.size < 0 || file.size > MAX_FILE || !HASH.test(file.sha256 || ''))
      throw Error('invalid_candidate_inventory');
  }
  if (canonical(await inventory(join(root, 'dist'))) !== canonical(manifest.files)) throw Error('candidate_assets_changed');
  if (canonical(identity(current)) !== canonical(manifest.app)) throw Error('app_source_changed');
  return {schema:1, state:'verified', app:manifest.app, manifestSha256,
    fileCount:manifest.files.length, verification:'assets-and-source-identity', activated:false};
}

export async function exportCandidate(inspected, destination, snapshot, {reinspect = inspectApp} = {}) {
  const app = identity(snapshot), host = resolve(inspected.hostDir);
  if (app.fingerprint !== inspected.fingerprint || app.version !== inspected.appVersion
      || app.controlVersion !== inspected.controlVersion) throw Error('app_source_changed');
  if (!isAbsolute(destination)) throw Error('absolute_candidate_directory_required');
  const output = resolve(destination), within = relative(host, output);
  if (!within || (!within.startsWith('..' + sep) && within !== '..' && !isAbsolute(within)))
    throw Error('candidate_must_be_outside_app');
  await noLinks(dirname(output));
  const files = await inventory(join(host, 'dist'));
  // Exclusive creation: interrupted candidates remain evidence, never an overwrite target.
  await mkdir(output);
  await mkdir(join(output, 'dist'));
  for (const file of files) {
    const data = await regularBytes(join(host, 'dist', file.path));
    if (data.length !== file.size || hash(data) !== file.sha256) throw Error('app_source_changed');
    const target = join(output, 'dist', file.path);
    await mkdir(dirname(target), {recursive:true});
    await writeFile(target, data, {flag:'wx'});
  }
  const fresh = await reinspect(host);
  if (fresh.fingerprint !== inspected.fingerprint || fresh.appVersion !== inspected.appVersion
      || fresh.controlVersion !== inspected.controlVersion) throw Error('app_source_changed');
  const manifest = {schema:1, component:'strata-app-surface', app, capabilities:snapshot, files, upstreamPatches:[]};
  const raw = Buffer.from(canonical(manifest) + '\n');
  await writeFile(join(output, 'manifest.json'), raw, {flag:'wx'});
  return verifyCandidate(output, {manifestSha256:hash(raw), current:snapshot});
}
