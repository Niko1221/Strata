/**
 * Strata Dev (docs/DEV_BUILD.md): two fixes on the native model path, used by native-control.mjs only when STRATA_DEV=1.
 *
 * 1. The DeepSeek engine (strata-dsv41) updates its expert profile (`--expert-profile .../data/expert-profile.bin`) while it runs. A pack
 *    in a shared models folder (Documents\Strata, which Live uses too) must not be written by Dev: Dev gets its own copy of the file in its
 *    data folder and points the engine there. If the copy cannot be made, the engine reads the shared file only (`--profile-readonly`).
 * 2. "Use a local GGUF" loads the model after preparing it. The id it guessed was `<family>-<model>` (deepseek41-ist-das-2.00bpw), but the
 *    run config setup writes is named with the family's tag (deepseek-v4.1-flash-ist-das-2.00bpw): Dev loads the installed id that fits.
 */
import { copyFileSync, existsSync, mkdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { basename, join, resolve, sep } from 'node:path';

const inside = (path, dir) => { const p = resolve(path).toLowerCase(), d = resolve(dir).toLowerCase(); return p === d || p.startsWith(d + sep.toLowerCase()) || p.startsWith(`${d}/`); };

/** `{ args, note }`: the DeepSeek engine arguments with the expert profile moved into `devHome` (or read-only). Without one: unchanged. */
export function devExpertProfileArgs(args, { devHome, exists = existsSync, copy = copyFileSync, mkdir = p => mkdirSync(p, { recursive: true }) } = {}) {
  const list = Array.isArray(args) ? [...args] : [];
  const at = list.indexOf('--expert-profile');
  if (at < 0 || at + 1 >= list.length) return { args: list, note: null };
  const source = String(list[at + 1]);
  if (!devHome || source === 'none' || inside(source, devHome)) return { args: list, note: null };
  const target = join(devHome, 'expert-profiles', createHash('sha256').update(resolve(source).toLowerCase()).digest('hex').slice(0, 12), basename(source));
  try {
    if (!exists(target)) {
      if (!exists(source)) throw new Error('the profile file is not there');
      mkdir(join(target, '..'));
      copy(source, target);
    }
    list[at + 1] = target;
    return { args: list, note: `Strata Dev: the expert profile is a copy in ${target}, so the shared pack stays as it is.` };
  } catch (e) {
    if (!list.includes('--profile-readonly')) list.push('--profile-readonly');
    return { args: list, note: `Strata Dev: the expert profile is read only (${e.message}); the shared pack is not written.` };
  }
}

/** The model id to load after an import: the installed one that is this family's `model`, else the old guess. */
export function devLoadAfterModel(family, model, installed = []) {
  const m = String(model).toLowerCase();
  const guess = (family ?? 'qwen') === 'qwen' ? m : `${family}-${m}`;
  const ids = installed.map(i => String(i?.model ?? i)).filter(Boolean);
  if (ids.includes(guess)) return guess;
  const word = String(family ?? '').replace(/\d+$/, '').toLowerCase();
  return ids.find(id => id.endsWith(`-${m}`) && (!word || id.includes(word))) ?? ids.find(id => id.endsWith(`-${m}`)) ?? guess;
}
