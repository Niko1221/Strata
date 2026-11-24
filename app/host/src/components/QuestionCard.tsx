/** A question the agent asked the person, in the style of the approval card: options to pick, the person's own words, Stop / Skip / Send (#4244, #6668). */
import { useEffect, useRef, useState } from 'react';
import { tr } from '../lib/i18n';
import { Btn, Icon } from './ui';
import { answerAgentAsk, stopSession } from '../lib/store';
import { plainFailureText } from '../lib/plain-errors';
import { createImeGuard } from '../lib/ime-enter';
import { answerComplete, answeredCount, toAnswers, type AgentAsk, type AgentQuestion } from '../lib/agent-ask';

type Draft = Record<string, { selected: string[]; custom: string }>;

export function QuestionCard({ sessionId, ask }: { sessionId: string; ask: AgentAsk }) {
    const [draft, setDraft] = useState<Draft>({});
    const [busyMode, setBusyMode] = useState<'send' | 'skip' | 'stop' | null>(null);
    const [problem, setProblem] = useState('');
    const busy = busyMode !== null;
    const inputRefs = useRef<Record<string, HTMLInputElement | null>>({});
    const cardRef = useRef<HTMLElement | null>(null);
    const ime = useRef(createImeGuard());

    // Auto-focus the first question's answer field when the card appears so typing lands directly in the card (#6668).
    useEffect(() => {
        const first = ask.questions[0];
        if (!first) return;
        const frame = requestAnimationFrame(() => {
            const input = inputRefs.current[first.id];
            if (input && !input.disabled) input.focus({ preventScroll: true });
            cardRef.current?.scrollIntoView?.({ block: 'nearest' });
        });
        return () => cancelAnimationFrame(frame);
    }, [ask.callId]);

    const current = (q: AgentQuestion) => draft[q.id] ?? { selected: [], custom: '' };
    const pick = (q: AgentQuestion, label: string) => setDraft(d => {
        const cur = d[q.id] ?? { selected: [], custom: '' };
        const selected = q.multiSelect
            ? (cur.selected.includes(label) ? cur.selected.filter(x => x !== label) : [...cur.selected, label])
            : (cur.selected.length === 1 && cur.selected[0] === label ? [] : [label]);
        // Preserve any custom note alongside picked options so users can combine choices with their own words (#6668).
        return { ...d, [q.id]: { selected, custom: cur.custom } };
    });
    const type = (q: AgentQuestion, custom: string) => setDraft(d => {
        const cur = d[q.id] ?? { selected: [], custom: '' };
        return { ...d, [q.id]: { selected: cur.selected, custom } };
    });

    const complete = answerComplete(ask.questions, draft);
    const doneCount = answeredCount(ask.questions, draft);
    const multiQuestion = ask.questions.length > 1;

    async function send(skip: boolean) {
        if (busy) return;
        setBusyMode(skip ? 'skip' : 'send');
        setProblem('');
        try {
            await answerAgentAsk(sessionId, ask.callId, skip ? null : toAnswers(ask.questions, draft));
        } catch (e) {
            setProblem(plainFailureText(e instanceof Error ? e.message : String(e)).message);
            setBusyMode(null);
        }
    }

    function onStop() {
        if (busy) return;
        setBusyMode('stop');
        setProblem('');
        try {
            stopSession(sessionId);
        } catch (e) {
            setProblem(plainFailureText(e instanceof Error ? e.message : String(e)).message);
            setBusyMode(null);
        }
    }

    function onOptionKeyDown(e: React.KeyboardEvent<HTMLButtonElement>, groupEl: HTMLDivElement | null) {
        if (!groupEl) return;
        const buttons = Array.from(groupEl.querySelectorAll<HTMLButtonElement>('button.askq__opt:not(:disabled)'));
        const idx = buttons.indexOf(e.currentTarget);
        if (idx < 0 || !buttons.length) return;
        let next = -1;
        if (e.key === 'ArrowDown' || e.key === 'ArrowRight') next = (idx + 1) % buttons.length;
        else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') next = (idx - 1 + buttons.length) % buttons.length;
        else if (e.key === 'Home') next = 0;
        else if (e.key === 'End') next = buttons.length - 1;
        if (next >= 0) {
            e.preventDefault();
            buttons[next].focus();
        }
    }

    function onInputKeyDown(e: React.KeyboardEvent<HTMLInputElement>, qIndex: number) {
        if (ime.current.keyDown({ key: e.key, isComposing: e.nativeEvent.isComposing, keyCode: e.nativeEvent.keyCode }, e.timeStamp)) return;
        if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.metaKey && !e.altKey) {
            e.preventDefault();
            if (complete) {
                void send(false);
                return;
            }
            const nextUnanswered = ask.questions.find((other, i) => {
                if (i === qIndex) return false;
                const a = draft[other.id];
                return !a || (a.selected.length === 0 && a.custom.trim().length === 0);
            });
            if (nextUnanswered) inputRefs.current[nextUnanswered.id]?.focus();
        }
    }

    return <section ref={cardRef} className="askcard askq" role="region" aria-label={tr("The assistant has a question for you")} data-count={ask.questions.length}>
      {ask.agentName && <p className="xs muted askq__agent">{ask.agentName}</p>}
      <div className="askq__list">
        {ask.questions.map((q, qi) => {
          const cur = current(q);
          const hasSelection = cur.selected.length > 0;
          return <div className="askq__item" key={`${q.id}-${qi}`}>
            <div className="askcard__top">
              <span className="askcard__icon"><Icon name="chat" size={18}/></span>
              <div className="askcard__text">
                {(q.header || multiQuestion) ? <p className="xs muted askq__meta">
                  {multiQuestion ? <span className="askq__step">{qi + 1} / {ask.questions.length}</span> : null}
                  {q.header ? <span>{q.header}</span> : null}
                </p> : null}
                <p className="askcard__q">{q.question}</p>
                {q.detail ? <p className="small muted askq__detail">{q.detail}</p> : null}
                {q.options.length > 0 ? <p className="xs muted askq__hint">{q.multiSelect ? tr("Select all that apply") : tr("Choose one")}</p> : null}
              </div>
            </div>
            {q.options.length ? <div className="askq__opts" role={q.multiSelect ? 'group' : 'radiogroup'} aria-label={q.question}>{q.options.map((o, oi) => {
              const on = cur.selected.includes(o.label);
              return <button type="button" key={`${o.label}-${oi}`} role={q.multiSelect ? 'checkbox' : 'radio'} aria-checked={on} className="askq__opt" data-multi={q.multiSelect || undefined} data-on={on || undefined} disabled={busy} onKeyDown={e => onOptionKeyDown(e, e.currentTarget.parentElement as HTMLDivElement)} onClick={() => pick(q, o.label)}>
                <span className={q.multiSelect ? 'askq__indicator askq__indicator--check' : 'askq__indicator askq__indicator--radio'} aria-hidden="true">{on && q.multiSelect ? <Icon name="check" size={12}/> : null}</span>
                <span className="askq__opt-body"><span>{o.label}</span>{o.description ? <small className="muted">{o.description}</small> : null}</span>
              </button>;
            })}</div> : null}
            <input
              ref={el => { inputRefs.current[q.id] = el; }}
              className="st-input askq__own"
              aria-label={`${q.question} — ${tr("Type your own answer")}`}
              placeholder={hasSelection ? tr("Add a note or extra detail (optional)") : tr("Type your own answer")}
              value={cur.custom}
              maxLength={2000}
              disabled={busy}
              onBlur={() => ime.current.reset()}
              onCompositionStart={() => ime.current.compositionStart()}
              onCompositionEnd={e => ime.current.compositionEnd(e.timeStamp)}
              onKeyDown={e => onInputKeyDown(e, qi)}
              onChange={e => type(q, e.target.value)}
            />
          </div>;
        })}
      </div>
      {problem ? <p className="small inline-error askq__problem" role="alert">{problem}</p> : null}
      <div className="askq__acts">
        <Btn size="sm" icon="stop" variant="secondary" disabled={busy} title={tr("Stop the assistant now")} onClick={onStop}>{busyMode === 'stop' ? tr("Stopping…") : tr("Stop")}</Btn>
        <span className="spacer"/>
        {multiQuestion ? <span className="xs muted askq__progress" aria-live="polite">{doneCount} / {ask.questions.length}</span> : null}
        <Btn size="sm" disabled={busy} title={tr("Skip this question and let the assistant continue")} onClick={() => void send(true)}>{busyMode === 'skip' ? tr("Skipping…") : tr("Skip")}</Btn>
        <Btn variant="primary" size="sm" disabled={busy || !complete} onClick={() => void send(false)}>{busyMode === 'send' ? tr("Sending…") : tr("Send answer")}</Btn>
      </div>
    </section>;
}
