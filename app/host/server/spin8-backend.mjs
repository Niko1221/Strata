/** Honest capability boundary for the separately built Bonsai runtime.
 * A Bonsai quantization backend is not an implementation of Spin(8) attention.
 * Keep this guard closed until a graph hook, not merely CLI flags, exists.
 */
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { dirname, isAbsolute, relative, resolve } from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { explicitKvEnvironment } from './kv-settings.mjs';

const exec = promisify(execFile);
export const SPIN8_MODES = Object.freeze(['off', 'vector', 'spinor-plus', 'spinor-minus', 'consensus']);
export const BONSAI_SOURCES = Object.freeze({
  prism: { repository: 'https://github.com/PrismML-Eng/llama.cpp.git', commit: '73bfe6cc485554822f80c473a3c0c4f25e80f41b' },
  zapabob: { repository: 'https://github.com/zapabob/llama.cpp.git', commit: '2c8e9655cdc6ea31051d4437a71492e717d046a4' },
});
export const BONSAI_SM120_PATCH = Object.freeze({
  id: 'prism-standard-sm120-v1', sha256: 'a291a22cd8915edee441e39b08b1c948e5ecc18b41060529c120701bfcf26ddd',
});

export class BackendCapabilityError extends Error {
  constructor(code, message) {
    super(message);
    this.name = 'BackendCapabilityError';
    this.code = code;
  }
}

export function spin8Request(config) {
  const request = config?.spin8;
  if (request === undefined) return { mode: 'off', profilePath: null };
  if (!request || typeof request !== 'object' || Array.isArray(request)) {
    throw new BackendCapabilityError('SPIN8_INVALID_CONFIG', 'Spin8 configuration must be an object.');
  }
  const mode = request.mode ?? 'off';
  if (!SPIN8_MODES.includes(mode)) {
    throw new BackendCapabilityError('SPIN8_INVALID_MODE', `Unknown Spin8 mode: ${String(mode)}`);
  }
  const profilePath = request.profilePath ?? null;
  if (profilePath !== null && (typeof profilePath !== 'string' || !profilePath.trim() || profilePath.includes('\0'))) {
    throw new BackendCapabilityError('SPIN8_INVALID_PROFILE', 'Spin8 profilePath must be a nonempty path.');
  }
  return { mode, profilePath };
}

/** Off is byte-identical to the existing launch command. No fake enable flags. */
export function buildSpin8Args(config) {
  const { mode, profilePath } = spin8Request(config);
  if (mode === 'off' && profilePath === null) return [];
  throw new BackendCapabilityError('SPIN8_BACKEND_UNAVAILABLE',
    'The pinned Bonsai runtime has no Spin8 attention graph hook. Spin8 modes and profiling are unavailable.');
}

async function sha256(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

/** Probe a locally selected binary only; never downloads, starts or stops a server.
 * A manifest is local build provenance, not a vendor signature or a model-load test.
 */
export async function probeBackendCapabilities(binaryPath, { manifestPath = null, timeoutMs = 5000 } = {}) {
  const result = {
    available: false, code: 'BACKEND_UNAVAILABLE', binaryPath: binaryPath ?? null,
    bonsai: { available: false, code: 'BONSAI_UNVERIFIED', formats: [], source: null },
    spin8: { available: false, code: 'SPIN8_BACKEND_UNAVAILABLE', modes: ['off'], profiling: false,
      reason: 'No Spin8 attention graph hook is implemented in the pinned Bonsai runtime.' },
  };
  if (typeof binaryPath !== 'string' || !binaryPath || binaryPath.includes('\0')) return result;
  try {
    result.sha256 = await sha256(binaryPath);
    const { stdout, stderr } = await exec(binaryPath, ['--version'], {
      timeout: timeoutMs, maxBuffer: 1024 * 1024, windowsHide: true, env: explicitKvEnvironment(),
    });
    result.version = `${stdout}\n${stderr}`.trim();
    result.available = true;
    result.code = 'BACKEND_EXECUTABLE';
  } catch (error) {
    result.reason = error.message;
    return result;
  }
  if (!manifestPath) return result;
  try {
    const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
    const pinned = BONSAI_SOURCES[manifest.backend];
    const patches = manifest.source?.patches ?? [];
    const standardSm120 = manifest.backend === 'prism' && manifest.source?.dirty === true &&
      patches.length === 1 && patches[0].id === BONSAI_SM120_PATCH.id && patches[0].sha256 === BONSAI_SM120_PATCH.sha256 &&
      manifest.build?.device === 'cuda' && manifest.build?.cudaArchitecture === '120' &&
      manifest.files?.some(file => file.path === 'source-patch.diff' && file.sha256 === BONSAI_SM120_PATCH.sha256);
    if (manifest.schemaVersion !== 1 || !pinned || manifest.source?.commit !== pinned.commit ||
        manifest.source?.repository !== pinned.repository ||
        !(standardSm120 || (manifest.source?.dirty === false && patches.length === 0)) ||
        manifest.build?.status !== 'success' || manifest.binary?.sha256 !== result.sha256 ||
        !Array.isArray(manifest.files) || !manifest.files.length) {
      throw new Error('Manifest does not identify a pinned successful build with the exact approved source patch and matching binary.');
    }
    const root = dirname(resolve(manifestPath));
    let hasBinary = false;
    for (const file of manifest.files) {
      if (!file || typeof file.path !== 'string' || isAbsolute(file.path) || !/^[a-f0-9]{64}$/i.test(file.sha256 ?? '')) {
        throw new Error('Invalid manifest file entry.');
      }
      const path = resolve(root, file.path);
      const rel = relative(root, path);
      if (rel === '..' || rel.startsWith('../') || rel.startsWith('..\\') || isAbsolute(rel)) {
        throw new Error('Manifest file path leaves the build directory.');
      }
      if (await sha256(path) !== file.sha256.toLowerCase()) throw new Error(`Build artifact digest mismatch: ${file.path}`);
      if (path === resolve(binaryPath)) hasBinary = true;
    }
    if (!hasBinary) throw new Error('Manifest does not include the selected binary.');
    const versionCommit = /\bcommit ([a-f0-9]{7,40})(?:-dirty)?\b/i.exec(result.version)?.[1]?.toLowerCase();
    if (!versionCommit || !pinned.commit.startsWith(versionCommit)) {
      throw new Error('Executable version does not report the pinned source commit.');
    }
    // These are source-level format support claims from this exact pinned fork.
    // Only a real model load can establish compatibility with a particular GGUF.
    result.bonsai = { available: true, code: 'BONSAI_PINNED_BUILD',
      formats: manifest.backend === 'prism' ? ['PQ2_0', 'PTQ1_0', 'Q2_0'] : ['Q2_0'], source: { ...pinned, patches },
      evidence: 'Local pinned source build manifest, recorded source patch and artifact digests; model load not tested.' };
  } catch (error) {
    result.bonsai.code = 'BONSAI_MANIFEST_INVALID';
    result.bonsai.reason = error.message;
  }
  return result;
}
