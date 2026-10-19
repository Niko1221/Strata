import { migrateSetup } from './setup.mjs';
/** Tiny persisted state for the host: one JSON file under STRATA_HOME. */
import { accessSync, constants, mkdirSync, readFileSync, writeFileSync, existsSync, realpathSync, readdirSync, statSync, statfsSync, symlinkSync } from 'node:fs';
import { dirname, isAbsolute, join, parse, relative, resolve } from 'node:path';
import { homedir } from 'node:os';
import { defaultModelDirectory } from './model-directory-default.mjs';

/**
 * Prefer ~/.strata, the way the packaged app stores its data. Sandboxes and
 * locked-down accounts cannot write to the home directory, so fall back to a
 * folder next to the checkout rather than failing to boot.
 */
function resolveHome() {
  const candidates = [process.env.STRATA_HOME, join(homedir(), '.strata'), join(process.cwd(), '.strata')].filter(
    Boolean,
  );
  for (const dir of candidates) {
    try {
      mkdirSync(join(dir, 'models'), { recursive: true });
      mkdirSync(join(dir, 'data'), { recursive: true });
      return dir;
    } catch {
      /* try the next candidate */
    }
  }
  throw new Error('No writable data directory found — set STRATA_HOME to a writable path');
}

export const HOME = resolveHome();
export const DATA_DIR = join(HOME, 'data');
const STATE_FILE = join(DATA_DIR, 'state.json');

const DEFAULTS = {
  version: 1,
  setup: { complete: false, flowVersion: 2, step: 1, history: [] },
  installed: {}, // id -> { installedAt, bytes }
  current: null,
  loaded: null,
  beginner: {
    priority: 'balanced',
    contextLength: 'recommended',
    readImages: false,
    gpuIdle: 'never',
  },
  expert: {}, // id -> value
  serving: { otherDevices: false, apiKey: null, pauseServing: false },
};

function read() {
  try {
    if (!existsSync(STATE_FILE)) return structuredClone(DEFAULTS);
    const parsed = JSON.parse(readFileSync(STATE_FILE, 'utf8'));
    return { ...structuredClone(DEFAULTS), ...parsed, setup: migrateSetup(parsed.setup) };
  } catch {
    return structuredClone(DEFAULTS);
  }
}

let state = read();
const DEFAULT_MODELS_DIRECTORY=defaultModelDirectory();
function initialModelsDirectory() {
 if(state.modelsDir)return state.modelsDir;
 const legacy=realpathSync(join(HOME,'models'));
 // Keep an existing library in place. A new install uses Documents/Strata/Models.
 function hasWeights(folder,depth=0){try{return readdirSync(folder,{withFileTypes:true}).some(entry=>entry.isFile()&&/\.gguf(?:\.part|\.partial)?$/i.test(entry.name)||entry.isDirectory()&&depth<4&&hasWeights(join(folder,entry.name),depth+1));}catch{return false;}}
 if(!process.env.STRATA_MODELS_DIR&&hasWeights(legacy))return legacy;
 const directory=DEFAULT_MODELS_DIRECTORY;
 mkdirSync(directory,{recursive:true});
 return realpathSync(directory);
}
export let MODELS_DIR = initialModelsDirectory();

/** Native setup writes weights to data_dir/models. Keep its data root one level up. */
export function nativeDataDirectory(input) {
 if(!input||process.env.STRATA_MODELS_DIR||!existsSync(DEFAULT_MODELS_DIRECTORY))return input;
 const root=realpathSync(DEFAULT_MODELS_DIRECTORY);
 if(resolve(input)!==resolve(DEFAULT_MODELS_DIRECTORY)&&realpathSync(input)!==root)return input;
 const data=dirname(root),nativeModels=join(data,'models');
 // On Windows/macOS case-insensitive volumes, Models already resolves as models.
 // On Linux, a relative alias keeps the native layout inside the requested folder.
 if(!existsSync(nativeModels))symlinkSync(process.platform==='win32'?root:'Models',nativeModels,process.platform==='win32'?'junction':'dir');
 const a=statSync(nativeModels),b=statSync(root);
 if(realpathSync(nativeModels)!==root&&!(a.ino&&a.ino===b.ino&&a.dev===b.dev))throw new Error('The default model folder conflicts with another library. Choose a different folder.');
 return data;
}

/** Existing folders only: a mistyped or disconnected drive must never become a new empty library. */
export function validateModelsDirectory(input) {
  if (typeof input !== 'string' || !input.trim() || input.includes('\0')) throw new Error('Enter a model directory.');
  const path = input.trim().replace(/^~(?=$|[\\/])/, homedir());
  if (!isAbsolute(path)) throw new Error('Enter the full path to an existing folder.');
  let directory;
  try {
    directory = realpathSync(path);
    if (!statSync(directory).isDirectory()) throw new Error();
    accessSync(directory, constants.R_OK | constants.W_OK | constants.X_OK);
  } catch { throw new Error('This folder is unavailable or not writable. Check the path and folder permissions.'); }
  const data = realpathSync(DATA_DIR);
  const contains = (parent, child) => { const r = relative(parent, child); return !r || !r.startsWith('..') && !isAbsolute(r); };
  if (directory === parse(directory).root || contains(directory, data) || contains(data, directory)) throw new Error('Choose a separate folder for models, outside Strata’s app data.');
  return resolve(directory);
}

export function changeModelsDirectory(directory) {
  const next = validateModelsDirectory(directory);
  if (next === MODELS_DIR) return false;
  const installedByDirectory = { ...state.installedByDirectory, [MODELS_DIR]: state.installed ?? {} };
  const updated = { ...state, modelsDir: next, installedByDirectory, installed: installedByDirectory[next] ?? {}, current: null, loaded: null };
  // Commit the preference before exposing it; a persistence error leaves the old library active.
  writeFileSync(STATE_FILE, JSON.stringify(updated, null, 2));
  state = updated; MODELS_DIR = next;
  return true;
}

export function get() {
  return state;
}

export function save() {
  try {
    writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
  } catch (err) {
    console.error('[strata] could not persist state:', err.message);
  }
}

export function update(patch) {
  state = { ...state, ...patch };
  save();
  return state;
}

export function diskFreeGb(path = HOME) {
  try {
    const s = statfsSync(path);
    return (s.bavail * s.bsize) / 1024 ** 3;
  } catch {
    return 0;
  }
}

export function diskTotalGb(path = HOME) {
  try {
    const s = statfsSync(path);
    return (s.blocks * s.bsize) / 1024 ** 3;
  } catch {
    return 0;
  }
}
