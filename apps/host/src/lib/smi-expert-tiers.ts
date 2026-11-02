/** SMI2000: where the model's experts are - VRAM (the engine's GPU cache), RAM (its resident copy: Qwen arena_mib,
 *  DeepSeek ram_experts) and the NVMe disk (the rest, read on demand). */
import { hasReading } from './monitor-hardware';
import type { MonitorState } from './types';

const clamp = (value: number, max: number) => Math.max(0, Math.min(max, value));
export type ExpertTier = { count: number | null; gb: number | null; pct: number; approx: boolean };
export type ExpertTiers = { vram: ExpertTier; ram: ExpertTier; disk: ExpertTier; totalCount: number; totalGb: number | null };

export function expertTiers(cache: MonitorState['expertCache']): ExpertTiers | null {
    if (!['engine', 'demo'].includes(cache.source ?? '') || !hasReading(cache.used) || !hasReading(cache.total) || cache.total <= 0)
        return null;
    const totalCount = cache.total, totalGb = hasReading(cache.totalGb) && cache.totalGb > 0 ? cache.totalGb : null;
    const perGb = totalGb ? totalGb / totalCount : null;
    const vramCount = clamp(cache.used, totalCount);
    const vramGb = hasReading(cache.memoryGb) ? cache.memoryGb : perGb ? vramCount * perGb : null;
    let ramCount: number, ramGb: number | null, ramApprox = false;
    if (hasReading(cache.ramUsed)) {
        ramCount = clamp(cache.ramUsed, totalCount - vramCount);
        ramGb = perGb ? ramCount * perGb : null;
    }
    else if (hasReading(cache.ramGb) && totalGb && perGb && hasReading(vramGb)) {
        // a full RAM copy (no --resident-budget-gib) also holds the GPU's experts: they are counted once, in VRAM
        ramGb = clamp(cache.ramGb, totalGb - vramGb);
        ramCount = Math.round(ramGb / perGb);
        ramApprox = true;
    }
    else
        return null;
    const diskCount = Math.max(0, totalCount - vramCount - ramCount);
    const diskGb = totalGb && hasReading(vramGb) && hasReading(ramGb) ? Math.max(0, totalGb - vramGb - ramGb) : null;
    const share = (count: number, gb: number | null) => totalGb && hasReading(gb) ? gb / totalGb * 100 : count / totalCount * 100;
    return {
        vram: { count: vramCount, gb: vramGb, pct: share(vramCount, vramGb), approx: false },
        ram: { count: ramCount, gb: ramGb, pct: share(ramCount, ramGb), approx: ramApprox },
        disk: { count: diskCount, gb: diskGb, pct: share(diskCount, diskGb), approx: ramApprox },
        totalCount, totalGb,
    };
}

/** SMI2000: the KV cache's format (engine `kv`), its size (the server's `kv_mib_est`, an estimate) and whether it streams from RAM. */
const KV_FORMATS: Record<string, [string, string]> = {
    int8: ['8-bit keys and values.', 'The default: precise.'],
    q4_0: ['4-bit keys and values, rotated before rounding.', 'Half the memory of 8-bit; less precise on long documents.'],
    k8v4: ['8-bit keys, 4-bit values.', '23% less memory than 8-bit, so more experts fit in VRAM; prompts are 2-5% slower.'],
    fp16: ['16-bit keys and values.', 'Twice the memory of 8-bit.'],
    fp4: ['DeepSeek layout: the compressed cache in FP4, the recent window in FP8.', ''],
};
export type KvDetails = { mode: string; format: string; note: string; gb: number | null; kbPerToken: number | null; streaming: boolean };
export function kvDetails(engine: Record<string, any> | null | undefined): KvDetails | null {
    const mode = typeof engine?.kv === 'string' ? engine.kv : null;
    if (!mode)
        return null;
    const [format, note] = KV_FORMATS[mode] ?? [mode, ''];
    const est = hasReading(engine?.kv_mib_est) ? engine.kv_mib_est as number : null, context = hasReading(engine?.context) ? engine.context as number : null;
    return { mode, format, note, gb: est !== null ? est / 1024 : null, kbPerToken: est !== null && context ? Math.max(0, est - 121) * 1048576 / context / 1000 : null, streaming: hasReading(engine?.kv_resident) && engine.kv_resident > 0 };
}

/** SMI2000: what fills the GPU's memory, in GB: the engine's start figures, the KV estimate, the rest of what is used, and the free memory. */
export type VramSegment = { id: string; label: string; short: string; gb: number; pct: number; approx: boolean };
export function vramSegments(engine: Record<string, any> | null | undefined, usedGb: number | null | undefined, totalGb: number | null | undefined): VramSegment[] | null {
    if (!engine || !hasReading(usedGb) || !hasReading(totalGb) || totalGb <= 0)
        return null;
    const parts = (engine.vram_parts ?? {}) as Record<string, number>;
    const experts = engine.vram?.expert_cache_mib ?? engine.expert_cache_mib;
    let kv = hasReading(engine.kv_mib_est) ? engine.kv_mib_est as number : null;
    if (kv !== null && hasReading(engine.kv_resident) && engine.kv_resident > 0 && hasReading(engine.context) && engine.context > 0)
        kv = kv * Math.min(1, engine.kv_resident / engine.context);
    const known = [
        { id: 'weights', label: 'Model weights', short: 'Weights', mib: parts.weights, approx: false },
        { id: 'draft', label: 'Draft model (MTP)', short: 'MTP draft', mib: parts.draft, approx: false },
        { id: 'experts', label: 'Expert cache', short: 'Experts', mib: experts, approx: false },
        { id: 'kv', label: 'KV cache (context)', short: 'KV cache', mib: kv, approx: true },
        { id: 'buffers', label: 'Verify buffers', short: 'Buffers', mib: parts.buffers, approx: false },
    ].filter((p): p is { id: string; label: string; short: string; mib: number; approx: boolean } => hasReading(p.mib) && p.mib > 0);
    if (!known.some(p => p.id !== 'kv'))
        return null;
    const total = totalGb * 1024, used = Math.min(usedGb * 1024, total), sum = known.reduce((n, p) => n + p.mib, 0);
    const scale = sum > used && sum > 0 ? used / sum : 1;
    const out = [...known.map(p => ({ ...p, mib: p.mib * scale })), { id: 'other', label: 'Other (work memory, other programs)', short: 'Other', mib: Math.max(0, used - sum), approx: false }, { id: 'free', label: 'Free', short: 'Free', mib: Math.max(0, total - used), approx: false }];
    return out.map(p => ({ id: p.id, label: p.label, short: p.short, gb: p.mib / 1024, pct: p.mib / total * 100, approx: p.approx }));
}
