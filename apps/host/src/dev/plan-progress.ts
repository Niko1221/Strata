/**
 * Strata Dev: how far a plan has come, for the progress bar in the chat and on the task board. Only steps the assistant has finished count:
 * there is no guessed percentage. Pure, so tests/plan-progress.test.mjs can run it.
 */
import type { Plan } from '../lib/plan';

export interface PlanProgressValue { done: number; total: number; percent: number; current: string }

/** The bar for a plan that is going ahead (approved, running or paused); null for a proposal, a finished or a stopped plan. */
export function planProgress(plan: Plan | null | undefined): PlanProgressValue | null {
    if (!plan || !['approved', 'running', 'paused'].includes(plan.state) || !plan.steps.length) return null;
    const total = plan.steps.length;
    // A skipped step is finished work too: nothing is left to do for it.
    const done = plan.steps.filter(s => s.state === 'done' || s.state === 'skipped').length;
    return { done, total, percent: Math.round((100 * done) / total), current: plan.steps.find(s => s.state === 'running')?.text ?? '' };
}

/** Elapsed time for the bar: "42 s", "3:07", "1:02:09". */
export function formatElapsed(ms: number): string {
    const total = Math.max(0, Math.floor(ms / 1000));
    const h = Math.floor(total / 3600), m = Math.floor((total % 3600) / 60), sec = total % 60;
    const two = (n: number) => String(n).padStart(2, '0');
    if (h > 0) return `${h}:${two(m)}:${two(sec)}`;
    return m > 0 ? `${m}:${two(sec)}` : `${sec} s`;
}

/** Time a plan has been going ahead: from when it was first seen under way, without the time it was paused. */
export interface Clock { start: number; pausedMs: number; pausedAt: number | null }
export function tickClock(clock: Clock | undefined, paused: boolean, now: number): Clock {
    const c = clock ?? { start: now, pausedMs: 0, pausedAt: null };
    if (paused && c.pausedAt === null) return { ...c, pausedAt: now };
    if (!paused && c.pausedAt !== null) return { ...c, pausedMs: c.pausedMs + (now - c.pausedAt), pausedAt: null };
    return c;
}
export const elapsedOf = (c: Clock, now: number) => now - c.start - c.pausedMs - (c.pausedAt === null ? 0 : now - c.pausedAt);
