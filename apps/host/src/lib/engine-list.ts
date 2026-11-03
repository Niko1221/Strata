import type { NativeState } from './native';
import { nativeCatalogModels } from './native-catalog';

/** The parts of Settings > Engine that need no React: the list's shape, sizes, makes, version order, and which models need a newer engine. */
/** One engine package installed on this computer, as Control lists it (`strata components engines`, strata_control/engine_library.py). */
export interface InstalledEngine {
  family: 'qwen' | 'deepseek';
  folder: string;
  version: string;
  backend: string;
  platform: string;
  variant: string;
  bytes: number;
  /** In use: the one a model starts on. */
  active: boolean;
  /** The newest one that can run on this computer. */
  newest: boolean;
  fits: boolean;
  /** Why this computer cannot run it: a code the page words, and its values. */
  why_code?: string | null;
  why_args?: string[];
  can_use: boolean;
  can_delete: boolean;
  blocked_code?: 'active' | 'running' | null;
  build_id?: string;
  path?: string;
  root?: string;
}
export interface EngineGroup { family: 'qwen' | 'deepseek'; title: string; engines: InstalledEngine[] }
export interface EngineList { available: boolean; families: EngineGroup[]; total_bytes: number }

/** The make of graphics card an engine is built for, in the words a person knows. */
export const engineMake = (backend: string): string => ({ cuda: 'NVIDIA', hip: 'AMD', sycl: 'Intel', metal: 'Apple', cpu: 'CPU' } as Record<string, string>)[backend] ?? backend;

/** "1.0 GB", or "640 MB" below a gigabyte: decimal units, like the Terminal and the release page. */
export function engineSize(bytes: number, locale?: string): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toLocaleString(locale, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} GB`;
  return `${Math.max(1, Math.round(bytes / 1e6)).toLocaleString(locale)} MB`;
}

const parts = (v: string) => v.split(/[.\-+]/).map(p => (/^\d+$/.test(p) ? Number(p) : 0));
/** Is version a older than version b? ("0.1.9" < "0.1.10", "0.1.49" < "0.1.49.1"). */
export function versionLess(a: string, b: string): boolean {
  const x = parts(a), y = parts(b);
  for (let i = 0; i < Math.max(x.length, y.length); i++) if ((x[i] ?? 0) !== (y[i] ?? 0)) return (x[i] ?? 0) < (y[i] ?? 0);
  return false;
}

/**
 * The installed models that need a newer engine than `version` (the catalog's `engine_min`). A switch to an older engine warns about these and goes on:
 * Strata recommends, it does not force. Only the Qwen family has catalog minimums.
 */
export function modelsNeedingNewer(native: NativeState | null | undefined, engine: InstalledEngine): string[] {
  if (!native?.installed?.models.length || !native.catalog || engine.family !== 'qwen') return [];
  const sizes = nativeCatalogModels(native.catalog.families).flatMap(f => f.sizes) as unknown as { id: string; model: string; engine_min?: string | null }[];
  const names: string[] = [];
  for (const installed of native.installed.models) {
    const size = sizes.find(s => s.id === installed.model);
    if (size?.engine_min && versionLess(engine.version, size.engine_min)) names.push(installed.model_name || size.model || installed.model);
  }
  return names;
}

