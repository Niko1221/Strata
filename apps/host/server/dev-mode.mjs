/**
 * Strata Dev: the developer variant (docs/DEV_BUILD.md). On when STRATA_DEV=1, which the Dev app (apps/desktop, `--features dev-build`)
 * and dev-build.ps1 set; off everywhere else, and then nothing here changes what the host does.
 *
 * In Dev the host reads dev.config.json (feature switches and overrides for the later phases). STRATA_DEV_CONFIG names the file; else it is
 * the one at the root of the checkout. A missing file is the empty config; a broken one is a warning (in the log and in the page), never a
 * stop: the host starts with the empty config.
 */
import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
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

/**
 * dev/extensions (ui/ and backend/): STRATA_DEV_EXTENSIONS names it; else it sits beside dev.config.json, so the installed Dev app
 * (whose STRATA_DEV_CONFIG points into the checkout) finds the checkout's extensions.
 */
export function devExtensionsDir(env = process.env) {
  return env.STRATA_DEV_EXTENSIONS ? resolve(env.STRATA_DEV_EXTENSIONS) : join(dirname(devConfigFile(env)), 'dev', 'extensions');
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
  warnings.push(...contextGuardOf(raw.contextGuard).warnings);
  const label = typeof badge.label === 'string' && badge.label.trim() ? badge.label.trim().slice(0, 24) : EMPTY.badge.label;
  return { file, config: { ...raw, badge: { ...badge, label }, features: section('features'), overrides: section('overrides') }, warnings };
}

/** What the page learns about Dev (GET /api/app/mode carries it as `dev` in Dev only). */
export function devSummary(loaded) {
  const settings = devSettings(loaded.config);
  return { label: loaded.config.badge.label, configFile: loaded.file, warnings: loaded.warnings, verbose: settings.verbose, chatOverflow: settings.chatOverflow, tokens: settings.tokens, ui: settings.ui, overrides: { loop: settings.overrides.loop, folder: settings.overrides.folder, permissions: settings.overrides.permissions } };
}

/* ---------------------------------------------------------------- Phase 2: what dev.config.json may set --- */

const MODES = ['manual', 'plan', 'bypass'];
const isObject = value => !!value && typeof value === 'object' && !Array.isArray(value);
const objectOr = (value, fallback = {}) => (isObject(value) ? value : fallback);
const numberIn = (value, min, max, fallback) => (Number.isFinite(value) && value >= min && value <= max ? value : fallback);
/** A design token is a custom property (`--st-...`) with a plain value: nothing that could end the declaration or open a block. */
const TOKEN_NAME = /^--[A-Za-z0-9-]{1,64}$/;
const TOKEN_VALUE = /^[^;{}<>]{1,200}$/;

/**
 * The Phase 2 settings in plain form, with every unknown or wrong value left out (never a throw):
 *   logging.verbose                       more host logging (every /api request, every hook run)
 *   tokens  { "--st-accent": "#e0218a" }   design tokens (src/styles/kit/tokens.css) the page overrides
 *   ui.elements { "<id>": { hidden, label, order, replace } }   existing UI elements by their fixed id
 *   ui.extensions { "<name>": { enabled } }                    UI extensions in dev/extensions/ui
 *   hooks { timeoutMs, tokenBudgetMs, disabled: [file] }       backend hooks in dev/extensions/backend
 *   overrides.webSearch { enabled, provider, withoutAsking }
 *   overrides.loop { enabled, maxRuns }
 *   overrides.folder { enabled, defaultPath }
 *   overrides.permissions { defaultMode, allowedModes }
 */
export function devSettings(config = {}) {
  const logging = objectOr(config.logging);
  const tokens = Object.fromEntries(Object.entries(objectOr(config.tokens)).filter(([name, value]) => TOKEN_NAME.test(name) && typeof value === 'string' && TOKEN_VALUE.test(value.trim())).map(([name, value]) => [name, value.trim()]));
  const ui = objectOr(config.ui);
  const elements = {};
  for (const [id, raw] of Object.entries(objectOr(ui.elements))) {
    if (!isObject(raw)) continue;
    const entry = {};
    if (typeof raw.hidden === 'boolean') entry.hidden = raw.hidden;
    if (typeof raw.label === 'string' && raw.label.trim()) entry.label = raw.label.trim().slice(0, 80);
    if (Number.isFinite(raw.order)) entry.order = raw.order;
    if (typeof raw.replace === 'string' && raw.replace.trim()) entry.replace = raw.replace.trim();
    elements[id] = entry;
  }
  const extensions = {};
  for (const [name, raw] of Object.entries(objectOr(ui.extensions))) if (isObject(raw) && typeof raw.enabled === 'boolean') extensions[name] = { enabled: raw.enabled };
  const hooks = objectOr(config.hooks);
  const overrides = objectOr(config.overrides);
  const web = objectOr(overrides.webSearch), loop = objectOr(overrides.loop), folder = objectOr(overrides.folder), permissions = objectOr(overrides.permissions);
  const allowedModes = Array.isArray(permissions.allowedModes) ? MODES.filter(m => permissions.allowedModes.includes(m)) : null;
  return {
    verbose: logging.verbose === true,
    chatOverflow: chatOverflowSettings(config.chatOverflowGuard),
    contextGuard: contextGuardOf(config.contextGuard).settings,
    tokens,
    ui: { elements, extensions },
    hooks: {
      timeoutMs: numberIn(hooks.timeoutMs, 50, 60_000, 2000),
      tokenBudgetMs: numberIn(hooks.tokenBudgetMs, 1, 1000, 50),
      disabled: Array.isArray(hooks.disabled) ? hooks.disabled.filter(n => typeof n === 'string') : [],
    },
    overrides: {
      webSearch: {
        ...(typeof web.enabled === 'boolean' ? { enabled: web.enabled } : {}),
        ...(typeof web.provider === 'string' && ['duckduckgo', 'exa', 'perplexity', 'deepseek'].includes(web.provider) ? { provider: web.provider } : {}),
        ...(typeof web.withoutAsking === 'boolean' ? { withoutAsking: web.withoutAsking } : {}),
      },
      loop: {
        ...(typeof loop.enabled === 'boolean' ? { enabled: loop.enabled } : {}),
        ...(Number.isInteger(loop.maxRuns) && loop.maxRuns >= 1 && loop.maxRuns <= 10_000 ? { maxRuns: loop.maxRuns } : {}),
      },
      folder: {
        ...(typeof folder.enabled === 'boolean' ? { enabled: folder.enabled } : {}),
        ...(typeof folder.defaultPath === 'string' && folder.defaultPath.trim() ? { defaultPath: folder.defaultPath.trim() } : {}),
      },
      permissions: {
        ...(MODES.includes(permissions.defaultMode) && (!allowedModes || allowedModes.includes(permissions.defaultMode)) ? { defaultMode: permissions.defaultMode } : {}),
        ...(allowedModes && allowedModes.length ? { allowedModes } : {}),
      },
    },
  };
}

/**
 * "chatOverflowGuard" (server/chat-overflow.mjs): true/false, or { enabled, maxTotalChars }. On unless switched off. maxTotalChars
 * (10,000 to 3,500,000) makes the saved chat list smaller, only to try the guard with a few chats.
 */
function chatOverflowSettings(raw) {
  const entry = isObject(raw) ? raw : { enabled: raw };
  const max = entry.maxTotalChars;
  return { enabled: entry.enabled !== false, ...(Number.isInteger(max) && max >= 10_000 && max <= 3_500_000 ? { maxTotalChars: max } : {}) };
}

/**
 * "contextGuard" (agent/agent-bridge.mjs, agent/profile.mjs, agent/context-fit.mjs): summarise a local chat early so it never fills up.
 * { enabled: true, midTaskAt, compactAfterTurnAt, compactBeforeSendAt }, each a share of the window from 0.3 to 0.9. Off unless enabled is
 * true; a missing or wrong value switches the whole guard off and gives a warning (never a throw). Returns { settings: null | {...}, warnings }.
 */
export function contextGuardOf(raw) {
  if (raw === undefined || raw === null || raw === false) return { settings: null, warnings: [] };
  if (!isObject(raw)) return { settings: null, warnings: ['dev.config.json: "contextGuard" must be an object; the context guard is off.'] };
  if (raw.enabled !== true) return { settings: null, warnings: raw.enabled === false || raw.enabled === undefined ? [] : ['dev.config.json: "contextGuard.enabled" must be true or false; the context guard is off.'] };
  const settings = {}, bad = [];
  for (const key of ['midTaskAt', 'compactAfterTurnAt', 'compactBeforeSendAt']) {
    const v = raw[key];
    if (typeof v === 'number' && Number.isFinite(v) && v >= 0.3 && v <= 0.9) settings[key] = v; else bad.push(key);
  }
  if (bad.length) return { settings: null, warnings: [`dev.config.json: "contextGuard.${bad.join('", "contextGuard.')}" must be a number from 0.3 to 0.9; the context guard is off.`] };
  return { settings: { enabled: true, ...settings }, warnings: [] };
}

/** The Dev settings the host runs with now, or null outside Dev. index.mjs sets it at start and on Reload; the agent profile reads it. */
let active = null;
export function setActiveDevSettings(settings) { active = settings ?? null; }
export function activeDevSettings() { return active; }

/** The chat mode a turn runs in: unchanged outside Dev; in Dev a mode dev.config.json does not allow becomes its default (or the first allowed one). */
export function devChatMode(mode, settings = active) {
  const p = settings?.overrides?.permissions;
  if (!p) return mode;
  const allowed = p.allowedModes ?? MODES;
  const fallback = p.defaultMode ?? allowed[0];
  if (mode == null) return p.defaultMode ?? mode;
  return allowed.includes(mode) ? mode : fallback;
}

/**
 * The agent's tool settings (agent/tools-settings.mjs) with dev.config.json's web search override on top: `enabled` turns the
 * search group on or off, `provider` picks the service (a keyed one only works with its key saved in Settings -> Tools),
 * `withoutAsking` skips Manual's web question. Returns a new object; the saved settings are never changed.
 */
export function devToolsSettings(tools, settings = active) {
  const web = settings?.overrides?.webSearch;
  if (!web || !Object.keys(web).length) return tools;
  const off = new Set(tools.off ?? []);
  if (web.enabled === false) off.add('search');
  if (web.enabled === true) off.delete('search');
  const provider = web.provider === 'duckduckgo' ? 'builtin' : web.provider;
  return {
    ...tools,
    off,
    ...(provider ? { search: { ...(tools.search ?? {}), provider } } : {}),
    ...(typeof web.withoutAsking === 'boolean' ? { webWithoutAsking: web.withoutAsking } : {}),
  };
}

/** dev/skills and dev/models (Phase 3): STRATA_DEV_SKILLS / STRATA_DEV_MODELS name them; else beside dev.config.json (as Control's dev_models.py). */
export function devSkillsDir(env = process.env) {
  return env.STRATA_DEV_SKILLS ? resolve(env.STRATA_DEV_SKILLS) : join(dirname(devConfigFile(env)), 'dev', 'skills');
}
export function devModelsDir(env = process.env) {
  return env.STRATA_DEV_MODELS ? resolve(env.STRATA_DEV_MODELS) : join(dirname(devConfigFile(env)), 'dev', 'models');
}
