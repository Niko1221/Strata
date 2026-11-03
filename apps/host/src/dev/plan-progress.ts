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
