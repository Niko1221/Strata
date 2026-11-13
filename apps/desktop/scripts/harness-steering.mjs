#!/usr/bin/env node
/** Apply the reviewed Harness SOURCE patch, then attest its normal build outputs. */
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const directory = fileURLToPath(new URL('../../../packaging/harness/', import.meta.url));
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const definition = () => JSON.parse(readFileSync(join(directory, 'soft-cut.json'), 'utf8'));
const patchFile = join(directory, 'soft-cut.patch');

function checkSources(root, def, side) {
  for (const file of def.sources) {
    const path = join(root, file.path);
    if (file[side] === null ? existsSync(path) : !existsSync(path) || digest(readFileSync(path)) !== file[side]) throw new Error(`Unexpected Harness source: ${file.path} (${side}).`);
  }
}

export function applyHarnessSteeringSource(root) {
  const def = definition();
  if (digest(readFileSync(patchFile)) !== def.patchSha256) throw new Error('The Harness patch hash does not match its manifest.');
  checkSources(root, def, 'before');
  for (const args of [['apply', '--check', patchFile], ['apply', patchFile]]) {
    const result = spawnSync('git', args, { cwd: root, encoding: 'utf8' });
    if (result.status !== 0) throw new Error(result.stderr || 'Could not apply the Harness source patch.');
  }
  checkSources(root, def, 'after');
}

export function recordHarnessSteeringBuild(root) {
  const def = definition();
  checkSources(root, def, 'after');
  const artifacts = def.artifacts.map(file => ({ ...file, sha256: digest(readFileSync(join(root, file.source))) }));
  const validator = readFileSync(join(root, 'packages/api/session-controller/lib/typert.host.js'), 'utf8');
  if (!validator.includes('soft-cut') || !validator.includes('expectedTurn')) throw new Error('Build the updated Harness protocol validators before recording provenance.');
  const record = { id: def.id, commit: def.commit, patchSha256: def.patchSha256, artifacts };
  writeFileSync(join(root, 'BUNDLE.json'), JSON.stringify({ component: 'deepseek-harness-source', commit: def.commit, patches: [record] }, null, 2));
  return record;
}

/** Validate source and staged outputs; never edit compiled Harness modules. */
export function stagedHarnessSteeringPatch(source, staged) {
  if (!existsSync(join(source, 'BUNDLE.json'))) return null;
  const def = definition();
  const record = JSON.parse(readFileSync(join(source, 'BUNDLE.json'), 'utf8')).patches?.find(p => p.id === def.id);
  if (!record) return null;
  if (record.commit !== def.commit || record.patchSha256 !== def.patchSha256) throw new Error('The Harness steering provenance does not match the reviewed source patch.');
  checkSources(source, def, 'after');
  if (record.artifacts.length !== def.artifacts.length) throw new Error('Incomplete Harness steering build provenance.');
  for (const file of def.artifacts) {
    const built = record.artifacts.find(p => p.source === file.source && p.staged === file.staged);
    if (!built || digest(readFileSync(join(source, file.source))) !== built.sha256 || digest(readFileSync(join(staged, file.staged))) !== built.sha256) throw new Error(`Harness steering build mismatch: ${file.source}`);
  }
  return record;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const source = process.argv.indexOf('--src');
  if (source < 0 || !process.argv[source + 1]) throw new Error('Pass --src <pinned Harness source checkout>.');
  const root = resolve(process.argv[source + 1]);
  if (process.argv.includes('--apply')) applyHarnessSteeringSource(root);
  else if (process.argv.includes('--record')) recordHarnessSteeringBuild(root);
  else throw new Error('Choose --apply before building or --record after building.');
}
