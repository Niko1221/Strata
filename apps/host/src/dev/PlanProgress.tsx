/**
 * Strata Dev: a progress bar for the plan of a chat ("3 of 5 steps"). In the chat it sits above the message box; on the task board
 * it replaces the moving bar of a running chat or loop card once that chat has a plan. Dev text is English only, like the Dev panel.
 * Outside Dev nothing is drawn and nothing is asked of the host.
 */
import { useEffect, useState } from 'react';
import { useGoal } from '../lib/goal';
import { usePlan } from '../lib/plan';
import { useDev } from './dev-state';
import { elapsedOf, formatElapsed, planProgress, remainingOf, tickClock, type Clock } from './plan-progress';

/** When each plan was first seen under way (this page only: a reload starts the count again). */
const clocks = new Map<string, Clock>();

/** The time the plan has been going, redrawn every second while it runs. null when there is no plan under way. */
function useElapsed(planId: string | undefined, going: boolean, paused: boolean): number | null {
    const [now, setNow] = useState(() => Date.now());
    useEffect(() => {
        if (!planId || !going) return;
        clocks.set(planId, tickClock(clocks.get(planId), paused, Date.now()));
        setNow(Date.now());
        if (paused) return;
        const timer = setInterval(() => setNow(Date.now()), 1000);
        return () => clearInterval(timer);
    }, [planId, going, paused]);
    const clock = planId ? clocks.get(planId) : undefined;
    return going && clock ? elapsedOf(clock, now) : null;
}

function Bar({ chatId, onPlain }: { chatId: string; onPlain: 'nothing' | 'moving' }) {
    const plan = usePlan(chatId);
    const p = planProgress(plan);
    const paused = plan?.state === 'paused';
    const elapsed = useElapsed(plan?.id, !!p, paused);
    if (!p) return onPlain === 'moving' ? <div className="bcard__bar bcard__bar--busy" aria-hidden="true"><i/></div> : null;
    return <div className="dev-progress" data-paused={paused || undefined}>
        <div className="dev-progress__text"><span>{paused ? 'Plan paused' : 'Plan'} · {p.done} of {p.total} steps</span>{p.current ? <span className="dev-progress__now" title={p.current}>{p.current}</span> : null}<output>{elapsed !== null ? `${formatElapsed(elapsed)} · ` : ''}{elapsed !== null && remainingOf(elapsed, p.done, p.total) !== null ? `about ${formatElapsed(remainingOf(elapsed, p.done, p.total) as number)} left · ` : ''}{p.percent} %</output></div>
        <div className="dev-progress__track" role="progressbar" aria-label="Plan progress" aria-valuemin={0} aria-valuemax={p.total} aria-valuenow={p.done} aria-valuetext={`${p.done} of ${p.total} steps`}><i style={{ width: `${p.percent}%` }}/></div>
    </div>;
}

/**
 * In the chat. With a goal (the plan runner) the progress is the second row of the goal bar, so there is one bar, not two:
 * `inGoalBar` is the one GoalBar draws, `alone` the one above the message box, which steps aside while a goal bar is there.
 */
export function PlanProgress({ chatId, where }: { chatId: string | null | undefined; where: 'alone' | 'inGoalBar' }) {
    const dev = useDev();
    const goal = useGoal(dev ? chatId : null);
    if (!dev || !chatId || (where === 'alone' && goal)) return null;
    return <Bar chatId={chatId} onPlain="nothing"/>;
}

/** On a running chat or loop card of the task board: the plan's bar, else the moving bar the card always had. */
export function CardProgress({ chatId }: { chatId: string }) {
    return <Bar chatId={chatId} onPlain="moving"/>;
}
