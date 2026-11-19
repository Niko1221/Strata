/** Synthetic adapter package for lifecycle tests; never installed model weights. */
import {mkdirSync, writeFileSync, readFileSync, copyFileSync, chmodSync} from 'node:fs';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {LOCAL_BACKENDS} from '../../server/local-backends.mjs';

export function verifiedLocalRuntime(t, root, {realChild = false} = {}) {
  const components = join(root, 'components');
  const folder = join(components, 'local-backends', 'prism');
  mkdirSync(folder, {recursive: true});
  const name = process.platform === 'win32' ? 'llama-server.exe' : 'llama-server';
  const binary = join(folder, name);
  if (realChild) { copyFileSync(process.execPath, binary); if (process.platform !== 'win32') chmodSync(binary, 0o700); }
  else writeFileSync(binary, 'synthetic runtime; spawn is mocked by the test');
  const definition = LOCAL_BACKENDS.PrismML;
  writeFileSync(join(folder, 'runtime.json'), JSON.stringify({
    schema: 1, backend: definition.id, version: '0.4.0',
    source: {repository: definition.repository, commit: definition.versions['0.4.0'].commit, dirty: false, patches: []},
    cacheTypes: ['f16', 'q8_0'],
    files: [{path: name, sha256: createHash('sha256').update(readFileSync(binary)).digest('hex')}],
  }));
  const previous = process.env.STRATA_COMPONENTS_DIR;
  process.env.STRATA_COMPONENTS_DIR = components;
  t.after(() => { if (previous === undefined) delete process.env.STRATA_COMPONENTS_DIR; else process.env.STRATA_COMPONENTS_DIR = previous; });
  return [{id: 'runtime.backend', value: 'PrismML'}];
}
