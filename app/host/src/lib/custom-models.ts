/**
 * Custom builds: the person's own compatible fine-tunes, merges and quantizations of a model Strata's engines already run, kept in place
 * (.strata or .gguf files on their own disk). Control writes one run config per build (`strata-<family>-custom-<name>.json`, with `custom: true`);
 * this file turns those installed rows into versions of their parent model, so they are listed beside the official builds and are picked the same way.
 * Pure logic only (no screen code): the cleaning of the name, the choice of group and the size figures.
 */
import type { NativeSize } from './native';

/** The installed-model row fields Control adds for a custom build. */
export interface CustomRow {
  model: string;
  custom?: boolean;
  display_name?: string;
  model_name?: string;
  base_family?: string;
  base_model?: string;
  custom_bytes?: number;
  diskBytes?: number;
  ready?: boolean;
  [key: string]: unknown;
}

/** The display groups the Models screen shows (native-catalog.ts), by the id of the group. */
export type CustomGroup = 'qwen3.8' | 'qwen3.6' | 'ornith' | 'hy4-flash' | 'inkling-2-mini';

/** Which display group a setup family belongs to; undefined for a family Strata does not run as a custom build. */
export function customGroupOf(baseFamily: string | undefined | null): CustomGroup | undefined {
  const f = String(baseFamily ?? '').toLowerCase();
  if (!f) return undefined;
  if (f === 'hy4-flash') return 'hy4-flash';
  if (f === 'inkling-2-mini') return 'inkling-2-mini';
  if (/ornith/.test(f)) return 'ornith';
  if (/qwen3[._-]?6|35b[-_]?a3b/.test(f)) return 'qwen3.6';
  if (['qwen', 'swift', 'coder', 'unsloth', 'gyro', 'nvfp4', 'qwen3.8'].includes(f)) return 'qwen3.8';
  return undefined;
}

/** The file types each display group's engine reads as a custom build. HY-4 Flash has no GGUF loader. */
export const CUSTOM_FILE_TYPES: Record<CustomGroup, readonly ('gguf' | 'strata')[]> = {
  'qwen3.8': ['gguf'], 'qwen3.6': ['gguf'], ornith: ['gguf'], 'hy4-flash': ['strata'], 'inkling-2-mini': ['strata', 'gguf'],
};

/** A name for the person to read: printable characters only, no path separators, one line, at most `max` characters. */
export function cleanDisplayName(value: unknown, max = 80): string {
  return String(value ?? '').replace(/[\u0000-\u001f\u007f\\/:*?"<>|]/g, ' ').replace(/\s+/g, ' ').trim().slice(0, max).trim();
}

/** The name the Display name box starts with: what the checker suggested, else the file name without its folder, extension and shard numbers. */
export function suggestedDisplayName(suggested: unknown, path: string): string {
  const fromCheck = cleanDisplayName(suggested);
  if (fromCheck) return fromCheck;
  const file = String(path ?? '').split(/[\\/]/).filter(Boolean).pop() ?? '';
  return cleanDisplayName(file.replace(/-\d{5}-of-\d{5}(?=\.gguf$)/i, '').replace(/\.(gguf|strata)$/i, ''));
}

/** The file to hand to the installer for a custom build: the picked file itself (a folder can hold the official files too), else the checked folder. */
export function customTarget(picked: string, modelDir: string | undefined, isCustom: boolean): string {
  return isCustom && /\.(gguf|strata)$/i.test(picked) ? picked : (modelDir ?? picked);
}

const finite = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n) && n >= 0;

/** The size (in bytes) of a custom build: the figure Control recorded, else what the disk row reports. */
export const customBytesOf = (row: CustomRow): number => finite(row.custom_bytes) ? row.custom_bytes : finite(row.diskBytes) ? row.diskBytes : 0;

/**
 * A custom row as a version of its parent model. The closest official build (`base_model`) gives the profile the fit figures start from; the
 * memory figures move by the difference between the real file size and that build's size, so the "needs" line follows the actual file.
 * Its id is the run config id, so Load, Context, Vision, Drafting, GPU roles and Remove all act on this build and never on the official one.
 */
export function customVersion<T extends NativeSize & { sourceFamily?: string }>(row: CustomRow, base: T | undefined, group: CustomGroup): T & { custom: true; displayName: string; customBytes: number; sourceFamily: string; edition: string; images: boolean } {
  const name = cleanDisplayName(row.display_name) || cleanDisplayName(row.model_name) || row.model;
  const bytes = customBytesOf(row), gb = bytes / 1024 ** 3;
  const baseGb = base && finite(base.download_gb) ? base.download_gb : 0;
  const delta = base && gb > 0 ? gb - baseGb : 0;
  const shift = (n: unknown) => finite(n) ? Math.max(0, +(n + delta).toFixed(3)) : 0;
  const profile = (base ?? {}) as Partial<T>;
  return {
    about: '', on_this_pc: null, ...profile,
    id: row.model, model: String(row.base_model ?? profile.model ?? ''),
    download_gb: gb, ram_needed_gb: base ? shift(profile.ram_needed_gb) : gb, experts_gb: base ? shift(profile.experts_gb) : gb,
    ...(finite(profile.fit_gb) ? { fit_gb: shift(profile.fit_gb) } : {}),
    ...(profile.inkling ? { inkling: { ...profile.inkling, recommended: false } } : {}),
    ...(profile.hy4 ? { hy4: { ...profile.hy4, recommended: false } } : {}),
    installed: true, offered: true, needsNewerEngine: false, update_available: false,
    custom: true, displayName: name, customBytes: bytes,
    sourceFamily: String(row.base_family ?? (base as { sourceFamily?: string } | undefined)?.sourceFamily ?? ''), edition: 'Custom',
    images: !!(base as { images?: boolean } | undefined)?.images || group === 'hy4-flash' || group === 'inkling-2-mini',
  } as unknown as T & { custom: true; displayName: string; customBytes: number; sourceFamily: string; edition: string; images: boolean };
}

/**
 * The display groups with every installed custom build added to its parent model, after the official builds. Groups are never changed in place.
 * A custom build whose family is not one Strata runs is left out; a group that has no entry yet (Ornith with nothing offered) is made.
 */
export function withCustomBuilds<S extends NativeSize & { sourceFamily?: string }, G extends { family: string; title: string; sizes: S[]; images?: boolean; about?: string; experimental?: boolean }>(groups: G[], rows: readonly CustomRow[] | null | undefined): G[] {
  const custom = (rows ?? []).filter(r => r?.custom === true && typeof r.model === 'string' && r.model);
  if (!custom.length) return groups;
  const out = groups.map(g => ({ ...g, sizes: [...g.sizes] }));
  const TITLES: Record<CustomGroup, string> = { 'qwen3.8': 'Qwen3.8-Flash-Next', 'qwen3.6': 'Qwen3.6-35B-A3B', ornith: 'Ornith-1.5-35B-A3B', 'hy4-flash': 'HY-4 Flash', 'inkling-2-mini': 'Inkling-2-Mini' };
  for (const row of custom) {
    const group = customGroupOf(row.base_family);
    if (!group || out.some(g => g.sizes.some(s => s.id === row.model))) continue;
    let target = out.find(g => g.family === group);
    if (!target) {
      target = { family: group, title: TITLES[group], about: '', experimental: false, images: false, sizes: [] } as unknown as G;
      out.push(target);
    }
    const base = target.sizes.find(s => !(s as { custom?: boolean }).custom && s.model === row.base_model && (!row.base_family || s.sourceFamily === row.base_family)) ?? target.sizes.find(s => !(s as { custom?: boolean }).custom && s.model === row.base_model);
    target.sizes.push(customVersion(row, base, group) as unknown as S);
  }
  return out;
}

/** The words of the small badge a custom build carries in the version lists. */
export const CUSTOM_BADGE = 'Custom';
