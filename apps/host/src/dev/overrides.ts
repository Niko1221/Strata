/**
 * Strata Dev (docs/DEV_BUILD.md): the dev.config.json overrides that the app's own logic reads (store.ts, loop-runner.ts). A module with no
 * imports, so those files do not pull in the Dev UI. Empty until dev.tsx activates Dev; then and only then the values below change anything.
 */
interface DevOverrides { loop?: { enabled?: boolean; maxRuns?: number }; permissions?: { defaultMode?: 'manual' | 'plan' | 'bypass' } }
let overrides: DevOverrides = {};

export function setDevOverrides(next: DevOverrides | undefined) { overrides = next ?? {}; }
/** false only when dev.config.json switches /loop off (overrides.loop.enabled). */
export const devLoopEnabled = () => overrides.loop?.enabled !== false;
/** The most answers a loop gives: dev.config.json overrides.loop.maxRuns, else the app's own cap. */
export const devLoopMaxRuns = (fallback: number) => overrides.loop?.maxRuns ?? fallback;
/** The mode a chat without its own choice starts in: dev.config.json overrides.permissions.defaultMode, else `fallback`. */
export const devDefaultMode = <T extends string | undefined>(fallback: T): T | 'manual' | 'plan' | 'bypass' => overrides.permissions?.defaultMode ?? fallback;
/** For the chat request: `{ mode }` with the Dev default when the chat has no mode of its own, else nothing (always nothing outside Dev). */
export const devModeOverride = (own: string | undefined): { mode?: 'manual' | 'plan' | 'bypass' } => (own || !overrides.permissions?.defaultMode ? {} : { mode: overrides.permissions.defaultMode });

/** Dev is on (dev.tsx calls setDevOverrides once the host said so). */
let devOn = false;
export const markDevActive = () => { devOn = true; };
export const devActive = () => devOn;
/** For the chat request: `{ skillIds }` when the chat chose its skills in Dev, else nothing (always nothing outside Dev). */
export const devSkillsOverride = (skills: string[] | null | undefined): { skillIds?: string[] } => (devOn && Array.isArray(skills) ? { skillIds: skills } : {});
