/**
 * Strata Dev: a progress bar for the plan of a chat ("3 of 5 steps"). In the chat it sits above the message box; on the task board
 * it replaces the moving bar of a running chat or loop card once that chat has a plan. Dev text is English only, like the Dev panel.
 * Outside Dev nothing is drawn and nothing is asked of the host.
 */
import { usePlan } from '../lib/plan';
import { useDev } from './dev-state';
import { planProgress } from './plan-progress';

function Bar({ chatId, onPlain }: { chatId: string; onPlain: 'nothing' | 'moving' }) {
    const plan = usePlan(chatId);
    const p = planProgress(plan);
    if (!p) return onPlain === 'moving' ? <div className="bcard__bar bcard__bar--busy" aria-hidden="true"><i/></div> : null;
    const paused = plan?.state === 'paused';
    return <div className="dev-progress" data-paused={paused || undefined}>
        <div className="dev-progress__text"><span>{paused ? 'Plan paused' : 'Plan'} · {p.done} of {p.total} steps</span>{p.current ? <span className="dev-progress__now" title={p.current}>{p.current}</span> : null}<output>{p.percent} %</output></div>
        <div className="dev-progress__track" role="progressbar" aria-label="Plan progress" aria-valuemin={0} aria-valuemax={p.total} aria-valuenow={p.done} aria-valuetext={`${p.done} of ${p.total} steps`}><i style={{ width: `${p.percent}%` }}/></div>
    </div>;
}

/** Above the message box in the chat. */
export function PlanProgress({ chatId }: { chatId: string | null | undefined }) {
    const dev = useDev();
    return dev && chatId ? <Bar chatId={chatId} onPlain="nothing"/> : null;
}

/** On a running chat or loop card of the task board: the plan's bar, else the moving bar the card always had. */
export function CardProgress({ chatId }: { chatId: string }) {
    return <Bar chatId={chatId} onPlain="moving"/>;
}
