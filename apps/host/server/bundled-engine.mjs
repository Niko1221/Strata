/**
 * The engine that ships inside the installer (<install>/engine-bundle/<package folder>, the same layout Strata Control's
 * packaging/package_engine.py writes: engine-component.json, schema 2, SHA-256 and size for every file).
 *
 * On start, when this PC has no engine package for the bundle's platform and backend, or only an older one, the bundled
 * package is checked against its own manifest and copied into the normal components folder (<components>/engine/<folder>),
 * where Control and the "engine comes with the model" flow find it like any other installed engine. Nothing is downloaded.
 * An installed engine of the same version or newer is never touched; a damaged copy of the very same package is repaired.
 * The copy is made under a hidden temporary name and renamed into place, so an interrupted run leaves nothing half-installed
 * (the next start removes the leftover and tries again).
 */
import { createReadStream, existsSync, readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { copyFile, mkdir, readdir, readFile, rename, rm, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, join, resolve, sep } from 'node:path';

const MANIFEST = 'engine-component.json';

const versionKey = value => String(value ?? '').split('.').map(part => Number.parseInt(part, 10) || 0);
export function compareVersions(a, b) {
  const x = versionKey(a), y = versionKey(b);
  for (let i = 0; i < Math.max(x.length, y.length); i++) if ((x[i] ?? 0) !== (y[i] ?? 0)) return (x[i] ?? 0) > (y[i] ?? 0) ? 1 : -1;
  return 0;
}

async function readManifest(folder) {
  try {
    const data = JSON.parse((await readFile(join(folder, MANIFEST), 'utf8')).replace(/^﻿/, ''));
    if (!data || data.id !== 'strata-engine' || typeof data.version !== 'string' || typeof data.platform !== 'string' || typeof data.backend !== 'string') return null;
    if (!Array.isArray(data.files) || !data.files.length || typeof data.entrypoint !== 'string') return null;
    return data;
  } catch { return null; }
}

function plainRelative(path) {
  if (typeof path !== 'string' || !path || path.startsWith('/') || path.startsWith('\\') || /^[a-z]:/i.test(path)) return null;
  const parts = path.split(/[\\/]/);
  return parts.some(p => !p || p === '..' || p === '.') ? null : parts.join('/');
}

function sha256(path) {
  return new Promise((resolveHash, reject) => {
    const hash = createHash('sha256');
    createReadStream(path).on('data', d => hash.update(d)).on('error', reject).on('end', () => resolveHash(hash.digest('hex')));
  });
}

/**
 * The NVIDIA driver version of this PC (the oldest when several cards disagree), or null when there is none or it cannot be read.
 * Linux: the kernel module's own file; anywhere: nvidia-smi. Never throws.
 */
export function nvidiaDriverVersion() {
  try {
    const text = readFileSync('/proc/driver/nvidia/version', 'utf8');
    const found = /Kernel Module\s+(\d+(?:\.\d+)+)/.exec(text);
    if (found) return found[1];
  } catch { /* not Linux, or no NVIDIA driver */ }
  try {
    const out = execFileSync('nvidia-smi', ['--query-gpu=driver_version', '--format=csv,noheader'], { encoding: 'utf8', timeout: 8000, windowsHide: true, stdio: ['ignore', 'pipe', 'ignore'] });
    const versions = out.split(/\r?\n/).map(line => line.trim()).filter(line => /^\d+(\.\d+)+$/.test(line));
    if (versions.length) return versions.sort(compareVersions)[0];
  } catch { /* no nvidia-smi */ }
  return null;
}

/** One engine can ship as several builds of the same backend (CUDA 13 and CUDA 12 differ in the oldest driver they run on). */
const variantOf = manifest => `${manifest.platform}/${manifest.backend}/${manifest.min_driver?.nvidia ?? ''}`;

/** Does every file the manifest lists exist with its size (and, when `deep`, its SHA-256)? */
export async function packageIsWhole(folder, manifest, { deep = false } = {}) {
  for (const item of manifest.files) {
    const rel = plainRelative(item?.path);
    if (!rel || !/^[0-9a-f]{64}$/i.test(String(item.sha256 ?? '')) || !Number.isInteger(item.bytes)) return false;
    const path = join(folder, ...rel.split('/'));
    let info;
    try { info = await stat(path); } catch { return false; }
    if (!info.isFile() || info.size !== item.bytes) return false;
    if (deep && (await sha256(path)).toLowerCase() !== String(item.sha256).toLowerCase()) return false;
  }
  return true;
}

/**
 * Install the bundled engine when it is needed.
 * Returns { action: 'installed' | 'repaired' | 'kept' | 'none' | 'refused', folder?, version?, reason? }; never throws.
 */
export async function installBundledEngine({ bundleDir, components, platform = null, log = () => {}, driver = undefined } = {}) {
  try {
    if (!bundleDir || !components || !existsSync(bundleDir)) return { action: 'none' };
    let packages = [];
    for (const entry of await readdir(bundleDir, { withFileTypes: true })) {
      if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
      const manifest = await readManifest(join(bundleDir, entry.name));
      if (manifest) packages.push({ name: entry.name, manifest });
    }
    if (platform) packages = packages.filter(p => p.manifest.platform === platform);
    if (!packages.length) return { action: 'none' };
    // The CUDA builds the NVIDIA driver here can run: all of them when the driver is unknown or fits several; when none fits, the one with the lowest
    // requirement, so that Control can name the driver to install (copying the others would only waste disk).
    const driverHere = driver === undefined ? nvidiaDriverVersion() : driver;
    const need = item => item.manifest.min_driver?.nvidia ?? null;
    if (driverHere) {
      const cuda = packages.filter(p => need(p));
      const fits = cuda.filter(p => compareVersions(driverHere, need(p)) >= 0);
      if (cuda.length && !fits.length) {
        const lowest = [...cuda].sort((a, b) => compareVersions(need(a), need(b)))[0];
        packages = packages.filter(p => !need(p) || p === lowest);
      } else {
        packages = packages.filter(p => !need(p) || fits.includes(p));
      }
    }
    // the newest bundled package of each platform/backend/driver requirement
    packages.sort((a, b) => compareVersions(b.manifest.version, a.manifest.version));
    const engineRoot = join(components, 'engine');
    let last = { action: 'none' };
    const done = new Set();
    for (const { name, manifest } of packages) {
      const key = variantOf(manifest);
      if (done.has(key)) continue;
      done.add(key);
      if (!/^[\w.+-]+$/.test(name)) { last = { action: 'refused', reason: 'the package folder name is not plain' }; continue; }
      // what is installed for the same platform and backend
      let newestInstalled = null, sameFolder = null;
      try {
        for (const entry of await readdir(engineRoot, { withFileTypes: true })) {
          if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
          const installed = await readManifest(join(engineRoot, entry.name));
          if (!installed || variantOf(installed) !== variantOf(manifest)) continue;
          if (entry.name === name) sameFolder = installed;
          if (!newestInstalled || compareVersions(installed.version, newestInstalled.version) > 0) newestInstalled = installed;
        }
      } catch { /* no engine folder yet */ }
      let repair = false;
      if (newestInstalled && compareVersions(newestInstalled.version, manifest.version) >= 0) {
        // an engine of the same version or newer is already there; only a damaged copy of this very package is put right
        if (sameFolder && !(await packageIsWhole(join(engineRoot, name), sameFolder))) repair = true;
        else { last = { action: 'kept', version: newestInstalled.version }; continue; }
      }
      const source = join(bundleDir, name);
      if (!(await packageIsWhole(source, manifest, { deep: true }))) {
        last = { action: 'refused', reason: 'the engine inside the installer does not match its manifest' };
        log(`bundled engine ${name}: ${last.reason}`);
        continue;
      }
      await mkdir(engineRoot, { recursive: true });
      const temp = join(engineRoot, `.bundle-${name}.tmp`);
      await rm(temp, { recursive: true, force: true });
      try {
        await mkdir(temp, { recursive: true });
        for (const item of [...manifest.files.map(f => f.path), MANIFEST]) {
          const rel = plainRelative(item).split('/');
          const target = join(temp, ...rel);
          if (!resolve(target).startsWith(resolve(temp) + sep)) throw new Error('a path leaves the package');
          await mkdir(dirname(target), { recursive: true });
          await copyFile(join(source, ...rel), target);
        }
        if (!(await packageIsWhole(temp, manifest, { deep: true }))) throw new Error('the copy does not match the manifest');
        const finalFolder = join(engineRoot, name);
        if (repair) await rm(finalFolder, { recursive: true, force: true });
        await rename(temp, finalFolder);
      } catch (error) {
        await rm(temp, { recursive: true, force: true }).catch(() => {});
        last = { action: 'refused', reason: error?.message ?? String(error) };
        log(`bundled engine ${name}: ${last.reason}`);
        continue;
      }
      last = { action: repair ? 'repaired' : 'installed', folder: join(engineRoot, name), version: manifest.version };
      log(`bundled engine ${name}: ${last.action}`);
    }
    return last;
  } catch (error) {
    return { action: 'refused', reason: error?.message ?? String(error) };
  }
}
