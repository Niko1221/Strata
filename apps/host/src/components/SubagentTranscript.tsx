import { Fragment, useEffect, useRef, type ReactNode } from 'react';
import { tr } from '../lib/i18n';
import { Btn, Icon } from './ui';
import { Markdown } from './Markdown';

export type ChildTool = {
    id?: string;
    name?: string;
    verb?: string;
    target?: string;
    preview?: string;
    state?: 'running' | 'done' | 'error' | 'denied' | 'proposed' | 'interrupted';
    durationMs?: number;
    args?: unknown;
    result?: string;
    effect?: string;
    scope?: string;
};

export type ChildEntry = {
    id: string;
    role: 'assistant' | 'user' | 'tool' | 'system';
    kind?: 'message' | 'task' | 'tool' | 'notice';
    author?: 'assistant' | 'user' | 'parent' | 'system' | 'tool';
    text: string;
    at: number;
    open?: boolean;
    partial?: boolean;
    sourceId?: string;
    tool?: ChildTool;
};

export type LocalChildEntry = ChildEntry & {
    clientId: string;
    state: 'sending' | 'queued' | 'sent' | 'failed';
    error?: string | null;
    requestId?: string;
};

const TASK_MISSING = 'The main assistant did not include a task description.';

const stateWords: Record<string, string> = {
    running: 'Running…',
    done: 'Done',
    error: 'Failed',
    denied: 'Denied',
    proposed: 'Needs your approval',
    interrupted: 'Stopped',
};

const pretty = (value: unknown) => {
    if (value === null || value === undefined) return '';
    if (typeof value === 'string') return value;
    try { return JSON.stringify(value, null, 2); } catch { return String(value); }
};

const timeText = (ms: number) => {
    if (!Number.isFinite(ms) || ms <= 0) return '';
    return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
};

function TaskCard({ entry }: { entry: ChildEntry }) {
    const text = entry.text === TASK_MISSING ? tr(entry.text) : entry.text;
    return <section className="subagent-task" aria-label={tr('Task from the main assistant')}>
        <header className="subagent-task__head"><Icon name="experts" size={14}/><span>{tr('Task from the main assistant')}</span></header>
        <div className="st-bubble subagent-bubble"><Markdown text={text}/></div>
    </section>;
}

function ToolRow({ tool, onDetach, onResolve, resolving }: { tool: ChildTool; onDetach?: () => void; onResolve?: (tool: ChildTool, approved: boolean) => void; resolving?: boolean }) {
    const state = tool.state ?? 'running';
    const args = pretty(tool.args);
    const hasArgs = !!args && !(typeof tool.args === 'object' && tool.args !== null && !Array.isArray(tool.args) && Object.keys(tool.args).length === 0);
    const waiting = state === 'proposed' && !!tool.id && !!onResolve;
    // A request that needs the person opens itself: the Approve/Deny buttons must not stay behind a
    // collapsed row. The person can still close it after answering.
    const row = useRef<HTMLDetailsElement>(null);
    const auto = useRef(false);
    useEffect(() => {
        if (!waiting || !row.current) return;
        auto.current = true;
        row.current.open = true;
    }, [waiting]);
    return <details ref={row} className="subagent-tool" data-state={state} aria-label={tool.verb || tool.name || tr('Tool')} onToggle={e => {
        if (!e.currentTarget.open) return;
        // Opening a waiting request is the panel's own doing; it must not detach a following reader.
        if (auto.current) { auto.current = false; return; }
        onDetach?.();
    }}>
        <summary>
            <Icon name={tool.scope === 'shell' ? 'bolt' : 'tool'} size={15}/>
            <span className="subagent-tool__verb">{tr(tool.verb || 'Used a tool')}</span>
            {tool.target ? <code className="subagent-tool__target" title={tool.target}>{tool.target}</code> : null}
            <span className="subagent-tool__state">{tr(stateWords[state] ?? '')}</span>
            {tool.durationMs ? <span className="subagent-tool__time">{timeText(tool.durationMs)}</span> : null}
            <Icon name="chevron" className="subagent-tool__chev" size={13}/>
        </summary>
        <div className="subagent-tool__body">
            {waiting ? <div className="subagent-tool__approval" role="group" aria-label={tr('Waiting for your approval')}>
                <span role="status">{tr('Waiting for your approval')}</span>
                <Btn size="sm" variant="secondary" disabled={resolving} onClick={() => onResolve?.(tool, false)}>{tr('Deny')}</Btn>
                <Btn size="sm" variant="primary" icon="check" disabled={resolving} onClick={() => onResolve?.(tool, true)}>{tr('Approve')}</Btn>
            </div> : null}
            {tool.effect ? <p className="small muted">{tr(tool.effect)}</p> : null}
            {hasArgs ? <><h4>{tr('Arguments')}</h4><pre>{args}</pre></> : null}
            {tool.result ? <><h4>{tr('Result')}</h4><pre>{tool.result}</pre></> : null}
            <details className="subagent-tool__raw"><summary>{tr('Raw details')}</summary><pre>{pretty(tool)}</pre></details>
        </div>
    </details>;
}

/** A new run of the same helper: a direct message is a separate turn, not a continuation of the last bubble. */
function RunDivider({ at }: { at: number }) {
    const time = Number.isFinite(at) ? new Date(at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
    return <div className="subagent-run" role="separator"><span>{tr('New run')}</span>{time ? <time>{time}</time> : null}</div>;
}

function Notice({ entry }: { entry: ChildEntry }) {
    return <div className="subagent-notice" role="status">{tr(entry.text)}</div>;
}

function Message({ entry, name, onRetry }: { entry: LocalChildEntry | ChildEntry; name: string; onRetry?: (entry: LocalChildEntry) => void }) {
    const local = 'state' in entry ? entry as LocalChildEntry : null;
    const parent = entry.author === 'parent';
    const user = entry.author === 'user' || (entry.role === 'user' && !parent);
    const role = user ? 'user' : parent ? 'parent' : 'assistant';
    const who = user ? `${tr('You')} · ${tr('Direct message')}` : parent ? tr('Main assistant') : name;
    return <article className={`subagent-msg subagent-msg--${role}`} data-role={role}>
        <div className="subagent-msg__who">{who}</div>
        <div className="st-bubble subagent-bubble">
            {entry.text ? <Markdown text={entry.text}/> : null}
            {entry.open ? <span className="caret" aria-hidden="true"/> : null}
        </div>
        {entry.partial ? <p className="subagent-msg__partial">{tr('Interrupted before it finished.')}</p> : null}
        {local?.state === 'sending' ? <span className="subagent-msg__state">{tr('Sending…')}</span> : null}
        {local?.state === 'queued' ? <span className="subagent-msg__state">{tr('Queued')}</span> : null}
        {local?.state === 'failed' ? <div className="subagent-msg__error" role="alert">
            <span>{tr('The message could not be sent.')}{local.error ? <span className="muted"> {tr(local.error)}</span> : null}</span>
            {onRetry ? <Btn size="sm" variant="secondary" onClick={() => onRetry(local)}>{tr('Retry')}</Btn> : null}
        </div> : null}
    </article>;
}

/** The readable transcript: task card, author labels, Markdown answers, and one line per tool. */
export function SubagentTranscript({ entries, name, onRetry, onDetach, onResolve, resolving }: {
    entries: Array<ChildEntry | LocalChildEntry>;
    name: string;
    onRetry?: (entry: LocalChildEntry) => void;
    onDetach?: () => void;
    onResolve?: (tool: ChildTool, approved: boolean) => void;
    resolving?: string | null;
}) {
    const isTool = (entry: ChildEntry | LocalChildEntry) => entry.kind === 'tool' || entry.role === 'tool';
    const nodes: ReactNode[] = [];
    let previous: ChildEntry | LocalChildEntry | null = null;
    for (let index = 0; index < entries.length; index += 1) {
        const entry = entries[index];
        const user = entry.kind !== 'task' && (entry.author === 'user' || entry.role === 'user');
        if (user && previous) nodes.push(<RunDivider key={`${entry.id}-run`} at={entry.at}/>);
        if (isTool(entry)) {
            // Consecutive tool calls are one block of steps: a tighter list with a rail, so they do
            // not read as separate messages between the task and the answer.
            const group: Array<ChildEntry | LocalChildEntry> = [];
            let next = index;
            while (next < entries.length && isTool(entries[next])) { group.push(entries[next]); next += 1; }
            index = next - 1;
            nodes.push(<section className="subagent-tools" key={`tools-${group[0].id}`}>
                {group.map(row => <ToolRow key={row.id} tool={row.tool ?? {}} onDetach={onDetach} onResolve={onResolve} resolving={resolving === row.tool?.id}/>)}
            </section>);
            previous = group[group.length - 1];
            continue;
        }
        if (entry.kind === 'task') nodes.push(<Fragment key={entry.id}><TaskCard entry={entry}/></Fragment>);
        else if (entry.kind === 'notice' || entry.role === 'system') nodes.push(<Fragment key={entry.id}><Notice entry={entry}/></Fragment>);
        else nodes.push(<Fragment key={entry.id}><Message entry={entry} name={name} onRetry={onRetry}/></Fragment>);
        previous = entry;
    }
    return <div className="subagent-chat__entries">{nodes}</div>;
}
