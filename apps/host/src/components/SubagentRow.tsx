import { tr } from '../lib/i18n';
import { useEffect, useRef, useState } from 'react';
import { Btn, Dialog, type IconName } from './ui';
import { compactTokens, elapsed } from '../lib/chat-metrics';
import { api } from '../lib/client';
import { useStore } from '../lib/store';
import type { Session } from '../lib/types';
type Agent = NonNullable<Session['agents']>[number];
type TranscriptEntry = {
    role: 'assistant' | 'user' | 'tool';
    text: string;
    at: number;
};
type ChildView = {
    agent?: Agent | null;
    transcript?: TranscriptEntry[];
    available?: boolean;
    live?: boolean;
};
const states: Record<Agent['status'], {
    label: string;
    icon: IconName;
}> = {
    working: { label: 'In progress', icon: 'clock' }, completed: { label: 'Completed', icon: 'check' },
    failed: { label: 'Failed', icon: 'error' }, queued: { label: 'Queued', icon: 'clock' }, cancelled: { label: 'Cancelled', icon: 'x' },
    interrupted: { label: 'Stopped (can continue)', icon: 'pause' },
};
/** The words sent when the person presses Continue on a stopped helper: in the person's language, so the helper answers in it
 *  (a helper answers in the language of the task it is given). The key is the one the chat's own Continue uses. */
const CONTINUE_KEY = 'Continue from where you left off. Do not repeat what you have already written.';
export function SubagentRow({ agent: a, now = Date.now() }: {
    agent: Agent;
    now?: number;
}) {
    const [open, setOpen] = useState(false);
    const [stopping, setStopping] = useState(false);
    const [stopNote, setStopNote] = useState('');
    const [entries, setEntries] = useState<TranscriptEntry[]>([]);
    const [remote, setRemote] = useState<Agent | null>(null);
    const [available, setAvailable] = useState(true);
    const [loading, setLoading] = useState(false);
    const [loadError, setLoadError] = useState('');
    const [draft, setDraft] = useState('');
    const [sending, setSending] = useState(false);
    const [sendError, setSendError] = useState('');
    const scroller = useRef<HTMLDivElement>(null);
    const liveRef = useRef(false);
    const chatId = useStore(s => s.activeSession);
    const observedAt = useRef(Date.now());
    const duration = a.durationMs + (a.status === 'working' ? Math.max(0, now - (a.updatedAt ?? observedAt.current)) : 0);
    const shown = remote ?? a;
    const state = states[a.status] ?? states.queued;
    const description = tr(a.description);
    const live = a.status === 'working' || a.status === 'queued';
    const childId = a.childSessionId ?? a.id;
    /** Show the newest line without the person having to scroll. */
    useEffect(() => {
        const el = scroller.current;
        if (el)
            el.scrollTop = el.scrollHeight;
    }, [entries, open]);
    /** Read the conversation while the dialog is open; a working helper keeps growing it. */
    useEffect(() => {
        if (!open || !chatId)
            return;
        let alive = true;
        let timer: ReturnType<typeof setTimeout> | undefined;
        const read = async () => {
            try {
                const answer = await api.get<ChildView>(`/api/chat/subagents/${encodeURIComponent(childId)}?sessionId=${encodeURIComponent(chatId)}`);
                if (!alive)
                    return;
                setEntries(Array.isArray(answer.transcript) ? answer.transcript : []);
                setRemote(answer.agent ?? null);
                setAvailable(answer.available !== false);
                setLoadError('');
                // A watched helper that just finished: bring its row in the panel up to date too.
                if (answer.live)
                    liveRef.current = true;
                else if (liveRef.current) {
                    liveRef.current = false;
                }
            }
            catch (e) {
                if (!alive)
                    return;
                setLoadError(e instanceof Error ? e.message : String(e));
            }
            finally {
                if (alive) {
                    setLoading(false);
                    timer = setTimeout(() => { void read(); }, 1000);
                }
            }
        };
        setLoading(true);
        void read();
        return () => {
            alive = false;
            if (timer)
                clearTimeout(timer);
        };
    }, [open, chatId, childId]);
    /** Ask the host to stop this helper. Its session stays, so the person can ask it to go on later. */
    async function stop() {
        if (!chatId)
            return;
        setStopping(true);
        setStopNote('');
        try {
            const answer = await api.post<{ stopped: number; pending?: string[] }>('/api/chat/subagent/stop', { sessionId: chatId, childId: a.childSessionId ?? a.id });
            if (answer.pending?.length)
                setStopNote(tr('The helper has not stopped yet. It may finish on its own.'));
            else if (!answer.stopped)
                setStopNote(tr('That helper is no longer running.'));
        }
        catch (e) {
            setStopNote(e instanceof Error ? e.message : String(e));
        }
        finally {
            setStopping(false);
        }
    }
    /** Send the typed message (or the Continue sentence) to this helper. */
    async function send(text?: string) {
        const value = (text ?? draft).trim();
        if (!value || !chatId || sending)
            return;
        const mine: TranscriptEntry = { role: 'user', text: value, at: Date.now() };
        setSending(true);
        setSendError('');
        setEntries(prev => [...prev, mine]);
        if (text == null)
            setDraft('');
        try {
            await api.post('/api/chat/subagent/prompt', { sessionId: chatId, childId, text: value });
        }
        catch (e) {
            setEntries(prev => prev.filter(entry => entry !== mine));
            if (text == null)
                setDraft(value);
            setSendError(e instanceof Error ? e.message : String(e));
        }
        finally {
            setSending(false);
        }
    }
    function openChat(e: { target: EventTarget | null; preventDefault?: () => void }) {
        if ((e.target as HTMLElement | null)?.closest('button, a, input, textarea'))
            return;
        e.preventDefault?.();
        setOpen(true);
    }
    return <div className="subagent-row" data-status={a.status} role="button" tabIndex={0} aria-expanded={open} aria-label={tr("Open conversation")} onClick={openChat} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { if (!(e.target as HTMLElement).closest('button')) { e.preventDefault(); setOpen(true); } } }}>
    <span className="subagent-row__dot" aria-label={tr(state.label)}/>
    <div className="subagent-row__copy"><div className="subagent-row__title"><strong title={a.name}>{a.name}</strong><span className="subagent-row__status">{tr(state.label)}</span></div><p title={description}>{description}</p>{a.warning ? <p className="subagent-row__warning" role="status">{tr(a.warning)}</p> : null}{stopNote ? <p className="subagent-row__warning" role="status">{stopNote}</p> : null}</div>
    <div className="subagent-row__metrics"><span>{tr(compactTokens(a.tokens))}{tr(" tok")}</span><span>{tr(elapsed(duration))}</span>{tr(live ? <Btn size="sm" variant="secondary" icon="stop" disabled={stopping} aria-label={tr("Stop {0}", [a.name])} onClick={() => void stop()}>{tr(stopping ? "Stopping…" : "Stop")}</Btn> : null)}</div>
  <Dialog open={open} title={<span>{a.name}</span>} onClose={() => setOpen(false)} className="subagent-chat-dialog" actions={<Btn onClick={() => setOpen(false)}>{tr("Done")}</Btn>}>
    <div className="subagent-chat">
      <div className="subagent-chat__scroll" ref={scroller} aria-live="polite">
        {loading && !entries.length ? <p className="small muted">{tr("Loading…")}</p> : null}
        {!loading && !loadError && !entries.length ? <p className="small muted">{tr("This helper has not written anything yet.")}</p> : null}
        {entries.map((entry, index) => <div key={`${entry.at}-${index}`} className="subagent-chat__line" data-role={entry.role}>
          {entry.role === 'assistant' ? null : <span className="subagent-chat__who">{tr(entry.role === 'user' ? 'You' : 'Tool')}</span>}
          <span className="subagent-chat__text">{entry.text}</span>
        </div>)}
      </div>
      {loadError ? <p className="small subagent-row__warning" role="status">{tr("The conversation could not be loaded.")}<span className="muted"> {loadError}</span></p> : null}
      {!available && !loadError ? <p className="small muted">{tr("The helper's conversation is not kept after Strata restarts.")}</p> : null}
      {sendError ? <p className="small subagent-row__warning" role="status">{tr("The message could not be sent.")}<span className="muted"> {sendError}</span></p> : null}
      {tr(shown.status === 'interrupted' ? <div className="subagent-chat__continue"><Btn size="sm" variant="secondary" icon="play" title={tr("Ask this helper to go on.")} disabled={sending} onClick={() => void send(tr(CONTINUE_KEY))}>{tr("Continue")}</Btn></div> : null)}
      <div className="subagent-chat__compose">
        <textarea className="st-input" rows={2} value={draft} placeholder={tr("Message this helper…")} aria-label={tr("Message this helper…")} onChange={e => setDraft(e.target.value)} onKeyDown={e => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); void send(); } }}/>
        <Btn variant="primary" icon="send" disabled={!draft.trim() || sending} onClick={() => void send()}>{tr(sending ? "Sending…" : "Send")}</Btn>
      </div>
    </div>
  </Dialog></div>;
}
