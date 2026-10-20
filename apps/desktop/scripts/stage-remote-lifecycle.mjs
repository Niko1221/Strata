#!/usr/bin/env node
/** Package only tracked application source and the two fixed lifecycle helpers. */
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { existsSync, lstatSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';

export function stageRemoteLifecycle({ root = fileURLToPath(new URL('../../..', import.meta.url)), output } = {}) {
  output ??= resolve(root, 'apps/host/python/remote-lifecycle-source.json.gz');
  const roots = ['serve', 'strata_control', 'strata_cli', 'tools', 'data', 'setup.py', 'strata.py', 'requirements.txt', 'requirements-build.txt', 'requirements-monitor.txt'];
  const tracked = execFileSync('git', ['ls-files', '-z', '--', ...roots], { cwd: root, maxBuffer: 2 * 1024 * 1024 }).toString().split('\0').filter(Boolean);
  if (!tracked.includes('serve/server.py') || !tracked.includes('strata_control/runtime.py')) throw new Error('Build lifecycle resources from a Strata source checkout.');
  const paths = tracked.map(path => ({ path, source: resolve(root, path) }));
  if (existsSync(resolve(root, 'VERSION'))) paths.push({ path: 'VERSION', source: resolve(root, 'VERSION') });
  for (const name of ['remote_model_manager.py', 'remote_model_lifecycle.py']) paths.push({ path: name, source: resolve(root, 'apps/host/python', name) });
  let total = 0;
  const files = paths.sort((a, b) => a.path.localeCompare(b.path, 'en')).map(({ path, source }) => {
    const stat = lstatSync(source);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.size > 8 * 1024 * 1024 || /(?:^|\/)(?:__pycache__|\.\.?)(?:\/|$)/.test(path)) throw new Error('Lifecycle source contains an unsupported file.');
    const bytes = readFileSync(source); total += bytes.length;
    if (total > 24 * 1024 * 1024) throw new Error('Lifecycle source exceeds the bundle limit.');
    return { path, sha256: createHash('sha256').update(bytes).digest('hex'), data: bytes.toString('base64') };
  });
  const archive = gzipSync(JSON.stringify({ schema: 1, files }), { level: 9 });
  if (archive.length > 8 * 1024 * 1024) throw new Error('Lifecycle source archive exceeds the bundle limit.');
  mkdirSync(dirname(output), { recursive: true }); writeFileSync(output, archive);
  return { files: files.length, bytes: archive.length, sha256: createHash('sha256').update(archive).digest('hex') };
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) console.log(JSON.stringify(stageRemoteLifecycle()));
