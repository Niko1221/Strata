/**
 * Strata Dev: the developer variant (docs/DEV_BUILD.md). On when STRATA_DEV=1, which the Dev app (apps/desktop, `--features dev-build`)
 * and dev-build.ps1 set; off everywhere else, and then nothing here changes what the host does.
 *
 * In Dev the host reads dev.config.json (feature switches and overrides for the later phases). STRATA_DEV_CONFIG names the file; else it is
 * the one at the root of the checkout. A missing file is the empty config; a broken one is a warning (in the log and in the page), never a
 * stop: the host starts with the empty config.
 */
import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = resolve(fileURLToPath(new URL('.', import.meta.url)), '..', '..', '..');

export const DEV_PORTS = Object.freeze({ window: 47832, host: 8792, engine: 8081 });

export function isDevMode(env = process.env) {
  return env.STRATA_DEV === '1';
}

/** The host's own port when nothing names one: 8791, or 8792 in Dev. */
export function defaultHostPort(env = process.env) {
  return isDevMode(env) ? DEV_PORTS.host : 8791;
}

export function devConfigFile(env = process.env) {
  return env.STRATA_DEV_CONFIG ? resolve(env.STRATA_DEV_CONFIG) : join(REPO_ROOT, 'dev.config.json');
}

const EMPTY = Object.freeze({ version: 1, badge: { label: 'DEV' }, features: {}, overrides: {} });

/** `{ file, config, warnings }`. Never throws: a missing file is the empty config, a broken one a warning plus the empty config. */
export function loadDevConfig({ file = devConfigFile(), read = readFileSync } = {}) {
  let text;
  try {
    text = read(file, 'utf8');
  } catch (error) {
    if (error?.code === 'ENOENT') return { file, config: { ...EMPTY }, warnings: [] };
    return { file, config: { ...EMPTY }, warnings: [`dev.config.json could not be read (${error?.message ?? error}). Dev starts without it.`] };
  }
  let raw;
  try {
    raw = JSON.parse(String(text).replace(/^﻿/, ''));
  } catch (error) {
    return { file, config: { ...EMPTY }, warnings: [`dev.config.json is not valid JSON (${error.message}). Dev starts without it.`] };
  }
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return { file, config: { ...EMPTY }, warnings: ['dev.config.json must hold one JSON object. Dev starts without it.'] };
  }
  const warnings = [];
  const section = name => {
    if (raw[name] === undefined) return {};
    if (raw[name] && typeof raw[name] === 'object' && !Array.isArray(raw[name])) return raw[name];
    warnings.push(`dev.config.json: "${name}" must be an object; it is ignored.`);
    return {};
  };
  const badge = section('badge');
  const label = typeof badge.label === 'string' && badge.label.trim() ? badge.label.trim().slice(0, 24) : EMPTY.badge.label;
  return { file, config: { ...raw, badge: { ...badge, label }, features: section('features'), overrides: section('overrides') }, warnings };
}

/** What the page learns about Dev (GET /api/app/mode carries it as `dev` in Dev only). */
export function devSummary(loaded) {
  return { label: loaded.config.badge.label, configFile: loaded.file, warnings: loaded.warnings };
}
