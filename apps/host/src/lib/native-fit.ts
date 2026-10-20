/**
 * What a model version asks of this computer, from the catalog's own figures and the hardware report.
 *
 * The numbers are RAM + VRAM together WITHOUT the n-gram table (it stays on the SSD and is read when needed). The experts are held
 * where there is room: first in the graphics memory beyond about 5 GB (the dense weights, buffers and a 32K cache), then in RAM beyond
 * about 10 GB left for the system; whatever is left is read from the SSD. The same rules setup uses to choose its low-RAM mode
 * (setup.py: low_ram_gpu_gb, LOW_RAM_HEADROOM_GB), so the page and the engine agree.
 */
export interface FitInput { ram_needed_gb?: number; experts_gb?: number | null; memory_gb?: number; mem_total?: boolean }
export interface FitHardware { ram_gb?: number | null; vram_gb?: number | null }
export interface Fit {
  /** RAM + VRAM the version needs (the n-gram table not counted). */
  memoryGb: number;
  /** About what it takes of the graphics memory / of RAM on this PC. */
  vramUseGb: number;
  ramUseGb: number;
  /** About how much is read from the SSD while answering; 0 when it all fits in memory. */
  spillGb: number;
  known: boolean;
}

const DENSE_GB = 5, SYSTEM_ROOM_GB = 10, DENSE_EXTRA_GB = 1;
const round1 = (n: number) => Math.round(n * 10) / 10;

export function memoryNeeded(v: FitInput): number {
  if (typeof v.memory_gb === 'number' && Number.isFinite(v.memory_gb)) return v.memory_gb;
  if (v.mem_total && typeof v.ram_needed_gb === 'number') return v.ram_needed_gb;
  return round1((v.experts_gb ?? v.ram_needed_gb ?? 0) + DENSE_EXTRA_GB);
}

export function fitOn(v: FitInput, hw: FitHardware | null | undefined): Fit {
  const memoryGb = memoryNeeded(v);
  const ram = Number(hw?.ram_gb), vram = Number(hw?.vram_gb) || 0;
  if (!Number.isFinite(ram) || ram <= 0) return { memoryGb, vramUseGb: 0, ramUseGb: 0, spillGb: 0, known: false };
  const experts = Math.max(0, memoryNeeded(v) - DENSE_EXTRA_GB);
  const onGpu = Math.max(0, Math.min(experts, vram - DENSE_GB));
  const ramRoom = Math.max(0, ram - SYSTEM_ROOM_GB);
  const spillGb = Math.max(0, experts - onGpu - ramRoom);
  const inRam = Math.max(0, experts - onGpu - spillGb);
  const vramUseGb = vram > 0 ? Math.min(vram, DENSE_GB + onGpu) : 0;
  return { memoryGb, vramUseGb: round1(vramUseGb), ramUseGb: round1(inRam + DENSE_EXTRA_GB * (vram > 0 ? 0 : 1)), spillGb: round1(spillGb), known: true };
}

/** The family groups of the version list, in the order shown. `family` is the engine's family id. */
export const VERSION_GROUPS: { id: string; title: string; families: string[] }[] = [
  { id: 'swift', title: 'Swift models', families: ['swift'] },
  { id: 'gsq', title: 'GSQ-RCO models', families: ['qwen', 'coder'] },
  { id: 'nvfp4', title: 'NVFP4 models', families: ['nvfp4'] },
  { id: 'gyro', title: 'Gyro models', families: ['gyro'] },
  { id: 'unsloth', title: 'Unsloth models', families: ['unsloth'] },
];

/** The fixed "Recommended" picks, in order: engine family + model name + the short note shown beside the name. */
export const RECOMMENDED: { family: string; model: string; note: string }[] = [
  { family: 'gyro', model: 'Gyro-S', note: 'the smallest' },
  { family: 'swift', model: 'IQ2_XS', note: 'good balance' },
  { family: 'qwen', model: 'IQ3_S', note: 'best quality for its size' },
  { family: 'unsloth', model: 'UD-Q4_K_XL', note: 'more quality' },
];

/** Is a group of the version list open? What the person chose wins; until they touch it, only the group holding the selected version is open
 *  (the running version's group used to stay open whatever was clicked). */
export const groupOpen = (opened: Record<string, boolean>, id: string, containsSelected: boolean): boolean => opened[id] ?? containsSelected;

/** The recommended four, found in a version list (a version the catalog does not offer is skipped). */
export function recommendedVersions<V extends { sourceFamily: string; model: string }>(versions: V[]): { r: typeof RECOMMENDED[number]; v: V }[] {
  return RECOMMENDED.map(r => ({ r, v: versions.find(v => v.sourceFamily === r.family && v.model === r.model) })).filter((x): x is { r: typeof RECOMMENDED[number]; v: V } => !!x.v);
}
