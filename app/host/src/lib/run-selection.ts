import { readConfig } from './config';
import { defaultProfileKey, recommendedConfig, type RunProfiles } from './run-profiles';
import { pinStrataEngineBackend } from './strata-engine-backend';
import type { EngineConfig, Hardware, ImageModelChoice, ModelVariant } from './types';

/** The viewed local model is independent of the Host's current/loaded engine. */
export const LAST_LOCAL_MODEL_KEY = 'strata.run-local-variant.v1';
export const RUN_VARIANT_KEY = 'strata.run-variant.v1';
type SelectionStorage = Pick<Storage, 'getItem' | 'setItem'>;

/** Read the durable local choice, falling back to the pre-existing session selection. */
export function readLastLocalModel(local: SelectionStorage, session: SelectionStorage): string | null {
    return local.getItem(LAST_LOCAL_MODEL_KEY) ?? session.getItem(RUN_VARIANT_KEY);
}

/** Remember a viewed identity only; this does not mutate the model registry or engine. */
export function rememberRunSelection(model: ModelVariant, local: SelectionStorage, session: SelectionStorage): void {
    session.setItem(RUN_VARIANT_KEY, model.id);
    if (model.local)
        local.setItem(LAST_LOCAL_MODEL_KEY, model.id);
}

export type LocalRunSelection =
    | { status: 'waiting' | 'none' }
    | { status: 'missing'; id: string }
    | { status: 'selected'; id: string };

/** Explicit links win. A stale saved choice never overrides the native entry screen. */
export function resolveLocalRunSelection(variants: readonly ModelVariant[] | null, explicit: string | null, last: string | null): LocalRunSelection {
    if (!variants)
        return { status: 'waiting' };
    const id = explicit ?? last;
    if (id !== null && variants.some(v => v.id === id && v.local))
        return { status: 'selected', id };
    return explicit !== null ? { status: 'missing', id: explicit } : { status: 'none' };
}

export interface RunModelDraft {
    draft: string | null;
    profile: string;
    image: ImageModelChoice;
}

/** Restore this identity's draft/default without applying settings or starting an engine. */
export function runModelSettings(model: ModelVariant, engine: EngineConfig, hardware: Hardware, profiles: RunProfiles, computer: string | null, previous?: RunModelDraft, backendHint?: string): RunModelDraft {
    if (previous) {
        let restored;
        try { restored = previous.draft ? readConfig(previous.draft) : null; }
        catch { return previous; } // Keep invalid JSON available for correction in its own model.
        const pinned = restored ? pinStrataEngineBackend(restored, model, backendHint) : null;
        return { ...previous, draft: pinned ? JSON.stringify(pinned, null, 2) : previous.draft };
    }
    const profileId = computer === null ? undefined : profiles.defaults[defaultProfileKey(computer, model.id)];
    const profile = profiles.profiles.find(p => p.id === profileId && p.variantId === model.id && p.hardware === computer);
    const config = profile ? structuredClone(profile.config) : recommendedConfig(engine, hardware);
    if (!profile && model.local) {
        const k = [1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 512, 1024].filter(k => k * 1024 <= model.contextMax).at(-1) ?? 1;
        const context = Math.min(32, k);
        config.beginner.contextLength = `${context}k`;
        config.expert['ctx.length'] = `${context}K`;
    }
    return { draft: JSON.stringify(pinStrataEngineBackend(config, model, backendHint), null, 2), profile: profile?.id ?? 'current', image: profile?.imageModel ?? 'off' };
}
