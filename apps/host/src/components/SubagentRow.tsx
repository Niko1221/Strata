import { tr } from '../lib/i18n';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Btn, Dialog, Icon, type IconName } from './ui';
import { compactTokens, elapsed } from '../lib/chat-metrics';
import { api } from '../lib/client';
import { scrollBehavior } from '../lib/responsive';
import { applyAgentFrame, useStore } from '../lib/store';
import type { Session } from '../lib/types';
import type { ChildTool } from './SubagentTranscript';
import {
    FOLLOW_SLACK,
    KEYS_UP,
    distanceFromBottom,
    newContentCount,
    wheelLeavesBottom,
} from '../lib/subagent-scroll';
import { markSendFailed, markSendRetrying } from '../lib/subagent-send';
import { SubagentTranscript, type ChildEntry, type LocalChildEntry } from './SubagentTranscript';

type Agent = NonNullable<Session['agents']>[number];
type ChildView = {
    agent?: Agent | null;
    transcript?: ChildEntry[];
    available?: boolean;
    historyOnly?: boolean;
    live?: boolean;
    revision?: number;
};

const states: Record<Agent['status'], { label: string; icon: IconName }> = {
    working: { label: 'In progress', icon: 'clock' },
    completed: { label: 'Completed', icon: 'check' },
    failed: { label: 'Failed', icon: 'error' },
    queued: { label: 'Queued', icon: 'clock' },
    cancelled: { label: 'Cancelled', icon: 'x' },
    interrupted: { label: 'Stopped (can continue)', icon: 'pause' },
};

const modeInfo: Record<'manual' | 'plan' | 'bypass', { label: string; note: string; icon: IconName }> = {
    manual: {
        label: 'Manual',
        note: 'Manual: this helper asks before it changes anything. Approve or deny its requests here.',
        icon: 'shield',
    },
    plan: {
        label: 'Plan',
        note: 'Plan: read-only. This helper cannot change files or run commands that change anything.',
        icon: 'file',
    },
    bypass: {
        label: 'Bypass permissions',
        note: 'Bypass: permission checks are skipped. This helper can change files and run commands without asking.',
        icon: 'bolt',
    },
};

const errorInfo = (error: unknown) => {
    const body = (error as { body?: { code?: string } } | null)?.body;
    return {
        message: error instanceof Error ? error.message : String(error),
        code: typeof body?.code === 'string' ? body.code : null,
    };
};

/** A helper that lost its runtime can leave a prompt pending; never leave the box saying "Sending…" forever. */
const REQUEST_TIMEOUT_MS = 25_000;
const withTimeout = <T,>(promise: Promise<T>, ms = REQUEST_TIMEOUT_MS): Promise<T> => new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('The helper did not answer in time. Open the main chat once, then try again.')), ms);
    promise.then(
        value => { clearTimeout(timer); resolve(value); },
        error => { clearTimeout(timer); reject(error); },
    );
});

export function SubagentRow({ agent: a, now = Date.now() }: {
    agent: Agent;
    now?: number;
}) {
    const [open, setOpen] = useState(false);
    const [stopping, setStopping] = useState(false);
    const [stopNote, setStopNote] = useState('');
    const [entries, setEntries] = useState<ChildEntry[]>([]);
    const [local, setLocal] = useState<LocalChildEntry[]>([]);
    const [remote, setRemote] = useState<Agent | null>(null);
    const [available, setAvailable] = useState(true);
    const [historyOnly, setHistoryOnly] = useState(false);
    const [loading, setLoading] = useState(false);
    const [loadError, setLoadError] = useState('');
    const [draft, setDraft] = useState('');
    const [sending, setSending] = useState(false);
    const [sendError, setSendError] = useState('');
    const [away, setAway] = useState(false);
    const [newCount, setNewCount] = useState(0);
    const [resolving, setResolving] = useState<string | null>(null);
    const scroller = useRef<HTMLDivElement>(null);
    const inner = useRef<HTMLDivElement>(null);
    const composer = useRef<HTMLTextAreaElement>(null);
    const continueClient = useRef<string | null>(null);
    const lastAgentSig = useRef('');
    const pinned = useRef(true);
    const readerInput = useRef(0);
    const lastRevision = useRef(-1);
    const lastSignature = useRef('');
    const lastCount = useRef(0);
    const chatId = useStore(s => s.activeSession);
    const observedAt = useRef(Date.now());
    const shown = remote ?? a;
    const mode = shown.mode ? modeInfo[shown.mode] : null;
    const duration = shown.durationMs + (shown.status === 'working' ? Math.max(0, now - (shown.updatedAt ?? observedAt.current)) : 0);
    const state = states[shown.status] ?? states.queued;
    const description = tr(shown.description);
    const live = shown.status === 'working' || shown.status === 'queued';
    const childId = a.childSessionId ?? a.id;
    const display = [...entries, ...local];

    const scrollToLatest = useCallback((smooth = false) => {
        const el = scroller.current;
        if (!el) return;
        pinned.current = true;
        setAway(false);
        setNewCount(0);
        const top = el.scrollHeight;
        if (smooth) el.scrollTo({ top, behavior: scrollBehavior() });
        else el.scrollTop = top;
    }, []);

    const forceBottom = useCallback(() => {
        pinned.current = true;
        setAway(false);
        setNewCount(0);
        requestAnimationFrame(() => scrollToLatest(false));
    }, [scrollToLatest]);

    /** Opening starts at the newest line; after that the reader's own scrolling decides. */
    useEffect(() => {
        if (!open) return;
        pinned.current = true;
        readerInput.current = Date.now();
        lastRevision.current = -1;
        lastSignature.current = '';
        lastCount.current = entries.length;
        setAway(false);
        setNewCount(0);
        setLoadError('');
        // Deliberately not keyed on entries: a poll must not pull the view back.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [open, childId]);

    /** Read the conversation while the dialog is open. Replacing the array never moves the scroll position. */
    useEffect(() => {
        if (!open || !chatId) return;
        let alive = true;
        let timer: ReturnType<typeof setTimeout> | undefined;
        const read = async () => {
            try {
                const answer = await api.get<ChildView>(`/api/chat/subagents/${encodeURIComponent(childId)}?sessionId=${encodeURIComponent(chatId)}`);
                if (!alive) return;
                const next = Array.isArray(answer.transcript) ? answer.transcript : [];
                const revision = typeof answer.revision === 'number' ? answer.revision : next.length;
                if (revision !== lastRevision.current) {
                    lastRevision.current = revision;
                    setEntries(next);
                }
                if (answer.agent) {
                    // `updatedAt` moves on every host read; status, mode, tokens and the activity
                    // line are the changes worth merging into the chat's row.
                    const signature = `${answer.agent.status}:${answer.agent.mode ?? ''}:${answer.agent.tokens}:${answer.agent.warning ?? ''}:${answer.agent.description ?? ''}`;
                    if (signature !== lastAgentSig.current) {
                        lastAgentSig.current = signature;
                        // The host is the owner of a helper row; the dialog's poll is also the status feed for
                        // a helper prompted after its parent turn ended.
                        applyAgentFrame(chatId, answer.agent);
                    }
                }
                setRemote(answer.agent ?? null);
                setAvailable(answer.available !== false);
                setHistoryOnly(answer.historyOnly === true);
                setLoadError('');
            } catch (e) {
                if (!alive) return;
                const info = errorInfo(e);
                setLoadError(info.message);
                if (info.code === 'child-gone') {
                    setAvailable(false);
                    setHistoryOnly(true);
                }
            } finally {
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
            if (timer) clearTimeout(timer);
        };
    }, [open, chatId, childId]);

    /** Opening lands in the input, so typing works without a second click; Esc still closes the dialog. */
    useEffect(() => {
        if (!open) return;
        const frame = requestAnimationFrame(() => composer.current?.focus());
        return () => cancelAnimationFrame(frame);
    }, [open, childId]);

    /** Keep a pinned view pinned when its content box grows, not on every poll frame. */
    useEffect(() => {
        if (!open) return;
        const el = scroller.current;
        const box = inner.current;
        if (!el || !box || typeof ResizeObserver === 'undefined') return;
        const observer = new ResizeObserver(() => {
            if (pinned.current) el.scrollTop = el.scrollHeight;
        });
        observer.observe(box);
        return () => observer.disconnect();
    }, [open, childId]);

    /**
     * A new entry while the reader is at the bottom follows it; while the reader is away it only
     * updates the "new messages" count. The signature catches a growing streamed answer and a tool
     * result whose length changed.
     */
    useEffect(() => {
        const signature = `${entries.length}:${entries.reduce((total, entry) => total + entry.text.length + (entry.tool?.result?.length ?? 0), 0)}`;
        const changed = signature !== lastSignature.current;
        const previousCount = lastCount.current;
        lastSignature.current = signature;
        lastCount.current = entries.length;
        if (!open || !changed) return;
        if (pinned.current) scrollToLatest(false);
        else setNewCount(count => count + Math.max(1, newContentCount(previousCount, entries.length, false)));
    }, [entries, open, scrollToLatest]);

    /** The host echo replaces the optimistic line once its request id arrives. */
    useEffect(() => {
        if (!local.length || !entries.length) return;
        const sources = new Set(entries.map(entry => entry.sourceId).filter((id): id is string => !!id));
        const hostUsers = entries.filter(entry => entry.role === 'user' && entry.kind !== 'task');
        setLocal(previous => {
            const next = previous.filter(entry => {
                const shownByHost = (entry.requestId && sources.has(entry.requestId))
                    || hostUsers.some(row => row.text === entry.text && Math.abs(row.at - entry.at) < 60_000);
                // A failed line with no echo is the only copy of the text: keep it and its Retry. If the
                // host accepted the request after the browser timed out, its echo proves it: drop the ghost.
                return !shownByHost;
            });
            return next.length === previous.length ? previous : next;
        });
    }, [entries, local.length]);

    /** Ask the host to stop this helper. Its session stays, so the person can ask it to go on later. */
    async function stop() {
        if (!chatId) return;
        setStopping(true);
        setStopNote('');
        try {
            const answer = await api.post<{ stopped: number; pending?: string[] }>('/api/chat/subagent/stop', { sessionId: chatId, childId });
            if (answer.pending?.length) setStopNote(tr('The helper has not stopped yet. It may finish on its own.'));
            else if (!answer.stopped) setStopNote(tr('That helper is no longer running.'));
        } catch (e) {
            setStopNote(errorInfo(e).message);
        } finally {
            setStopping(false);
        }
    }

    /** Approve or deny a helper's waiting tool from its own panel. The host keys the answer by child:call. */
    async function resolveChildTool(tool: ChildTool, approved: boolean) {
        if (!chatId || !tool.id || resolving) return;
        setResolving(tool.id);
        setSendError('');
        setEntries(previous => previous.map(entry => entry.tool?.id === tool.id
            ? { ...entry, tool: { ...entry.tool, state: approved ? 'running' : 'denied', ...(approved ? {} : { result: 'Denied. Nothing ran.' }) } }
            : entry));
        try {
            await api.post('/api/chat/approval', { sessionId: chatId, toolId: tool.id, decision: approved ? 'approve' : 'deny' });
        } catch (e) {
            setSendError(errorInfo(e).message);
        } finally {
            setResolving(null);
        }
    }

    async function deliver(value: string, clientId: string, existingId: string | null = null) {
        if (!chatId) return false;
        if (existingId) {
            setLocal(previous => previous.map(entry => entry.id === existingId ? markSendRetrying(entry) : entry));
        } else {
            const mine: LocalChildEntry = {
                id: clientId,
                clientId,
                role: 'user',
                kind: 'message',
                author: 'user',
                text: value,
                at: Date.now(),
                state: 'sending',
            };
            setLocal(previous => [...previous, mine]);
        }
        setSending(true);
        setSendError('');
        forceBottom();
        try {
            const answer = await withTimeout(api.post<{ messageId?: string; requestId?: string }>('/api/chat/subagent/prompt', {
                sessionId: chatId,
                childId,
                text: value,
                clientId,
            }));
            const queued = shown.status === 'working' || shown.status === 'queued';
            setLocal(previous => previous.map(entry => entry.id === clientId
                ? { ...entry, state: queued ? 'queued' : 'sent', requestId: answer.requestId, error: null }
                : entry));
            if (draft === value) setDraft('');
            requestAnimationFrame(() => composer.current?.focus());
            return true;
        } catch (e) {
            const info = errorInfo(e);
            // The host's cold-start reason is a plain instruction, not a code: keep the locale key so the
            // line follows the page's language (the message renders through tr()).
            const message = info.code === 'child-parent-cold' ? 'fx131' : info.message;
            setLocal(previous => previous.map(entry => entry.id === clientId ? markSendFailed(entry, message) : entry));
            if (info.code === 'child-history-only' || info.code === 'child-gone') {
                setHistoryOnly(true);
                if (info.code === 'child-gone') setAvailable(false);
            }
            return false;
        } finally {
            setSending(false);
        }
    }

    /** Send the typed message. The box is cleared only after acceptance; a failure leaves both the line and the draft. */
    async function send(text?: string) {
        const value = (text ?? draft).trim();
        if (!value || !chatId || sending) return;
        const clientId = `local-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
        await deliver(value, clientId);
    }

    function retry(entry: LocalChildEntry) {
        if (sending) return;
        void deliver(entry.text, entry.clientId, entry.id);
    }

    /** Continue is a new run in the same conversation, never a fake user bubble. */
    async function continueTask() {
        if (!chatId || sending || historyOnly) return;
        setSending(true);
        setSendError('');
        forceBottom();
        const clientId = continueClient.current ?? `continue-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
        continueClient.current = clientId;
        try {
            await withTimeout(api.post('/api/chat/subagent/prompt', { sessionId: chatId, childId, continue: true, clientId }));
            continueClient.current = null;
            requestAnimationFrame(() => composer.current?.focus());
        } catch (e) {
            const info = errorInfo(e);
            setSendError(info.message);
            if (info.code === 'child-history-only' || info.code === 'child-gone') setHistoryOnly(true);
        } finally {
            setSending(false);
        }
    }

    function detach() {
        pinned.current = false;
        setAway(true);
    }

    function openChat(e: { target: EventTarget | null; preventDefault?: () => void }) {
        if ((e.target as HTMLElement | null)?.closest('button, a, input, textarea')) return;
        e.preventDefault?.();
        setOpen(true);
    }

    return <div className="subagent-row" data-status={shown.status} role="button" tabIndex={0} aria-expanded={open} aria-label={tr("Open conversation")} onClick={openChat} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { if (!(e.target as HTMLElement).closest('button')) { e.preventDefault(); setOpen(true); } } }}>
    <span className="subagent-row__dot" aria-label={tr(state.label)}/>
    <div className="subagent-row__copy"><div className="subagent-row__title"><strong title={shown.name}>{shown.name}</strong>{mode ? <span className="subagent-row__mode" data-mode={shown.mode} title={tr(mode.note)}>{tr(mode.label)}</span> : null}<span className="subagent-row__status">{tr(state.label)}</span></div><p title={description}>{description}</p>{shown.warning ? <p className="subagent-row__warning" role="status">{tr(shown.warning)}</p> : null}{stopNote ? <p className="subagent-row__warning" role="status">{stopNote}</p> : null}</div>
    <div className="subagent-row__metrics"><span>{tr(compactTokens(shown.tokens))}{tr(" tok")}</span><span>{tr(elapsed(duration))}</span>{tr(live ? <Btn size="sm" variant="secondary" icon="stop" disabled={stopping} aria-label={tr("Stop {0}", [shown.name])} onClick={() => void stop()}>{tr(stopping ? "Stopping…" : "Stop")}</Btn> : null)}</div>
  <Dialog open={open} title={<span className="subagent-chat__title">{shown.name}{mode ? <span className="subagent-mode-badge" data-mode={shown.mode}>{tr(mode.label)}</span> : null}</span>} onClose={() => setOpen(false)} className="subagent-chat-dialog" actions={<Btn onClick={() => setOpen(false)}>{tr("Done")}</Btn>}>
    <div className="subagent-chat">
      {mode ? <div className="subagent-mode-note" data-mode={shown.mode} role="status"><Icon name={mode.icon} size={14}/><span>{tr(mode.note)}</span></div> : null}
      <div className="subagent-chat__scrollwrap">
        <div className="subagent-chat__scroll" ref={scroller} tabIndex={0} aria-label={shown.name} aria-busy={loading || sending || undefined}
          onScroll={e => {
              const distance = distanceFromBottom(e.currentTarget);
              pinned.current = distance <= FOLLOW_SLACK;
              setAway(distance > FOLLOW_SLACK);
              if (distance <= FOLLOW_SLACK) setNewCount(0);
          }}
          onWheel={e => {
              readerInput.current = Date.now();
              if (wheelLeavesBottom(e.deltaY) && e.currentTarget.scrollTop > 0) detach();
          }}
          onTouchMove={() => { readerInput.current = Date.now(); }}
          onPointerDown={() => { readerInput.current = Date.now(); }}
          onKeyDown={e => {
              readerInput.current = Date.now();
              if (e.key === 'End' && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
                  e.preventDefault();
                  scrollToLatest(true);
                  return;
              }
              if (KEYS_UP.has(e.key) && e.target === e.currentTarget && e.currentTarget.scrollTop > 0) detach();
          }}>
          <div className="subagent-chat__inner" ref={inner}>
            {loading && !display.length ? <p className="small muted">{tr("Loading…")}</p> : null}
            {historyOnly ? <p className="subagent-chat__history" role="status">{tr('Only the history of this helper is available. You can read it, but cannot continue.')}</p> : null}
            {!loading && !loadError && !display.length ? <p className="small muted">{tr("This helper has not written anything yet.")}</p> : null}
            <SubagentTranscript entries={display} name={shown.name} onRetry={retry} onDetach={detach} onResolve={resolveChildTool} resolving={resolving}/>
          </div>
        </div>
      </div>
      {away ? <div className="subagent-chat__jumprow"><button type="button" className="subagent-chat__jump" aria-label={newCount > 0 ? tr('{0} new messages', [newCount]) : tr('Jump to latest')} title={newCount > 0 ? tr('{0} new messages', [newCount]) : tr('Jump to latest')} onClick={() => scrollToLatest(true)}>
        {newCount > 0 ? tr('{0} new messages', [newCount]) : tr('Jump to latest')}<Icon name="chevron" size={13}/>
      </button></div> : null}
      {loadError ? <p className="small subagent-row__warning" role="status">{tr("The conversation could not be loaded.")}<span className="muted"> {loadError}</span></p> : null}
      {!available && !loadError && !historyOnly ? <p className="small muted">{tr('No saved conversation is available for this helper.')}</p> : null}
      {sendError ? <p className="small subagent-row__warning" role="status">{tr('The message could not be sent.')}<span className="muted"> {tr(sendError)}</span></p> : null}
      {tr(shown.status === 'interrupted' && !historyOnly ? <div className="subagent-chat__continue"><Btn size="sm" variant="secondary" icon="play" title={tr("Ask this helper to go on.")} disabled={sending} onClick={() => void continueTask()}>{tr("Continue")}</Btn></div> : null)}
      <div className="subagent-chat__compose">
        <textarea ref={composer} className="st-input" rows={2} value={draft} disabled={historyOnly} placeholder={historyOnly ? tr('This helper is not available. You can read any saved history, but cannot send a message.') : tr("Message this helper…")} aria-label={tr("Message this helper…")} onChange={e => setDraft(e.target.value)} onKeyDown={e => {
            if (e.nativeEvent.isComposing) return;
            // Enter sends; Shift+Enter is a new line. The helper panel is a chat, not a form.
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void send(); }
        }}/>
        <Btn variant="primary" icon="send" disabled={!draft.trim() || sending || historyOnly} onClick={() => void send()}>{tr(sending ? "Sending…" : "Send")}</Btn>
      </div>
    </div>
  </Dialog></div>;
}
