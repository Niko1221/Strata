/**
 * Strata Dev: the four chat modes. A module with no imports of the Dev UI, so store.ts and the composer can use it.
 *
 *   manual       asks before a change (the host's Manual)
 *   plan-manual  writes a plan first (a read-only turn); you approve it; the steps then run in Manual
 *   plan-auto    writes a plan first; it is approved at once and the steps run with everything allowed
 *   bypass       everything allowed, no plan ("Auto"; the host's Bypass)
 *
 * The host only knows manual, plan and bypass. plan-manual and plan-auto exist in the page: they decide how a message is sent
 * (as "/plan ..." first) and which host mode the run uses. Every function here is only called after the host said it runs as Strata Dev.
 */
export type DevMode = 'manual' | 'plan-manual' | 'plan-auto' | 'bypass';

/** Dev text is English only, like the Dev panel (no key in the language files); plain constants, not tr() calls. */
const HINT_PLAN_MANUAL = 'Write a plan first; you approve it, then each change is asked';
const HINT_PLAN_AUTO = 'Write a plan first, then run it with everything allowed';
const HINT_AUTO = 'Everything allowed, no plan';
export const DEV_MODES: { value: DevMode; label: string; hint: string }[] = [
    { value: 'manual', label: 'Manual', hint: 'Ask before making changes' },
    { value: 'plan-manual', label: 'Plan + Manual', hint: HINT_PLAN_MANUAL },
    { value: 'plan-auto', label: 'Plan + Auto', hint: HINT_PLAN_AUTO },
    { value: 'bypass', label: 'Auto', hint: HINT_AUTO },
];

/** The Dev mode a stored chat mode stands for: the host's old read-only "plan" is Plan + Manual; anything unknown is Manual. */
export function devModeOf(mode: string | null | undefined): DevMode {
    if (mode === 'plan' || mode === 'plan-manual') return 'plan-manual';
    if (mode === 'plan-auto' || mode === 'bypass') return mode;
    return 'manual';
}

/** The mode the host runs the work in. */
export const hostModeOf = (mode: string | null | undefined): 'manual' | 'bypass' => (devModeOf(mode) === 'plan-auto' || devModeOf(mode) === 'bypass' ? 'bypass' : 'manual');

export const plansFirst = (mode: string | null | undefined) => devModeOf(mode) === 'plan-manual' || devModeOf(mode) === 'plan-auto';

/**
 * Is this message to be written up as a plan first? Not when it carries out an approved plan (that is the work itself), not a command, and not
 * while a plan is under way (approved, running or paused): then "continue" or a follow-up belongs to that plan.
 */
export function wantsPlanFirst({ mode, body, carryOut = false, planState = null }: { mode: string | null | undefined; body: string; carryOut?: boolean; planState?: string | null }): boolean {
    return plansFirst(mode) && !carryOut && !!body.trim() && !body.trim().startsWith('/') && !['approved', 'running', 'paused'].includes(planState ?? '');
}

/** What goes to the host for a message typed in the box: a plain message becomes "/plan ...", a command or an empty message stays. */
export function planFirstText(text: string): string {
    const line = text.trim();
    return !line || line.startsWith('/') ? line : `/plan ${line}`;
}
