import {sessionTitle} from '../lib/session-title';
import { tr } from '../lib/i18n';
import { useEffect, useRef, useState } from 'react';
import { ArtifactTabs, type ExtraPreviewTab } from './ArtifactPanel';
import { FloatingMenu } from './FloatingMenu';
import { Popover } from './Popover';
import { Icon, IconBtn } from './ui';
import { SamplingDrawer } from '../app/SamplingDrawer';
import { downloadSessionLog } from '../lib/session-log';
import { SubagentRow } from './SubagentRow';
import { ChatModelPicker } from './ChatModelPicker';
import { runningAgents } from '../lib/agents';
import { api } from '../lib/client';
import { applyAgentRows, useStore } from '../lib/store';
import type { Session } from '../lib/types';
import type { PreviewArtifact } from '../lib/artifacts';
export function ChatToolbar({ session, items, selected, onSelect, onClose, onCloseTab, inline, expanded, onExpand, extraTabs = [], previewWidth }: {
    previewWidth?: number;
    extraTabs?: ExtraPreviewTab[];
    session: Session | null;
    items: PreviewArtifact[];
    selected: string | null;
    onSelect: (id: string) => void;
    onClose: () => void;
    onCloseTab: (id: string) => void;
    inline: boolean;
    expanded: boolean;
    onExpand: () => void;
}) {
    const remote = useStore(s => s.remoteModel);
    const [agents, setAgents] = useState(false), [menu, setMenu] = useState(false), [settings, setSettings] = useState(false);
    const agentsRef = useRef<HTMLButtonElement>(null), menuRef = useRef<HTMLButtonElement>(null);
    const count = runningAgents(session?.agents ?? []);
    const liveAgents = (session?.agents ?? []).some(a => a.status === 'working' || a.status === 'queued');
    // A helper's tool asks for approval under a colon-prefixed id (child:call) in the parent chat's own
    // message list. A helper prompted directly never puts a card there, so the host's own rows also carry
    // a waiting count. The dot says so while the panel is closed, instead of waiting for a person to open it.
    const waitingInMessages = (session?.messages ?? []).reduce((n, m) => n + (m.tools ?? []).filter(t => t.state === 'proposed' && String(t.id).includes(':')).length, 0);
    const waitingInRows = (session?.agents ?? []).reduce((n, a) => n + (a.waiting ?? 0), 0);
    const waiting = Math.max(waitingInMessages, waitingInRows);
    const [now, setNow] = useState(Date.now);
    useEffect(() => { if (!agents || !count)
        return; setNow(Date.now()); const timer = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(timer); }, [agents, count, session?.id]);
    // A helper prompted outside a parent turn saves its own row on the host; the page's chat stream
    // knows nothing about it. While any row is live, merge the small host row list (no transcripts).
    useEffect(() => {
        const id = session?.id;
        if (!id || !liveAgents) return;
        let alive = true;
        let timer: ReturnType<typeof setTimeout> | undefined;
        const tick = async () => {
            try {
                const answer = await api.get<{ agents?: unknown[] }>(`/api/chat/subagents?sessionId=${encodeURIComponent(id)}`);
                if (alive && Array.isArray(answer.agents)) applyAgentRows(id, answer.agents);
            } catch { /* a transient host gap: the next tick retries */ }
            if (alive) timer = setTimeout(() => { void tick(); }, 2000);
        };
        void tick();
        return () => { alive = false; if (timer) clearTimeout(timer); };
    }, [session?.id, liveAgents]);
    useEffect(() => setAgents(false), [session?.id]);
    const previews = !!selected && inline;
    const title = session ? sessionTitle(session) : tr('New chat');
    return <header className="chat-toolbar" style={previews && !expanded && previewWidth ? { gridTemplateColumns: `minmax(0,1fr) ${previewWidth}px` } : undefined} data-timeskip={selected?.startsWith('time-skip:') || undefined} data-preview={previews || undefined} data-expanded={expanded && previews || undefined}>
    <div className="chat-toolbar__conversation"><h1 title={title}>{title}</h1><ChatModelPicker /><button ref={agentsRef} className="chat-agents" title={tr(waiting ? '{0} helper approvals waiting · View overview' : count ? '{0} subagents running · View overview' : 'View subagent overview', waiting ? [waiting] : count ? [count] : undefined)} aria-label={tr(waiting ? '{0} helper approvals waiting — show overview' : count ? '{0} subagents running — show overview' : 'Subagents — show overview', waiting ? [waiting] : count ? [count] : undefined)} aria-haspopup="dialog" aria-expanded={agents} data-working={count > 0 || undefined} data-waiting={waiting > 0 || undefined} onClick={() => setAgents(v => !v)}><Icon name="experts" size={17}/>{tr(waiting > 0 ? <span className="agent-waiting-dot" title={tr('A helper is waiting for your approval')}/> : count > 0 ? <span className="agent-live-dot"/> : null)}<span className="chat-agents__label">{tr("Subagents")}</span>{tr(session?.agents?.length ? <span className="chat-agents__count">{tr(count || session.agents.length)}</span> : null)}<Icon name="chevron" className={agents ? 'is-open' : ''} size={13}/></button><span className="spacer"/><button ref={menuRef} className="st-btn st-btn--icon" title={tr("Chat options")} aria-label={tr("Chat options")} aria-haspopup="menu" aria-expanded={menu} onClick={() => setMenu(v => !v)}><Icon name="ellipsis"/></button>{tr(!selected && (items.length || extraTabs.length) ? <IconBtn name="panel" title={tr("Open previews")} onClick={() => onSelect(extraTabs.at(-1)?.id ?? items[0].id)}/> : null)}</div>
    {tr(previews ? <div className="chat-toolbar__previews"><ArtifactTabs extraTabs={extraTabs} onCloseTab={onCloseTab} items={items} selected={selected} onSelect={onSelect}/><div className="preview-window-actions"><div id="preview-file-actions" className="preview-file-actions"/><IconBtn name={expanded ? 'contract' : 'expand'} title={tr(expanded ? 'Exit full screen preview' : 'Full screen preview')} onClick={onExpand}/><IconBtn name="panel" title={tr("Collapse preview sidebar")} onClick={onClose}/></div></div> : null)}
    {tr(agents ? <Popover anchor={agentsRef.current} title={tr("Subagents")} detail={session?.agents?.length ? <>{tr(count)}{tr(" running \u00B7 ")}{tr(session.agents.filter(a => a.status === 'completed').length)}{tr(" completed \u00B7 ")}{tr(session.agents.length)}{tr(" total")}</> : undefined} width={460} onClose={() => setAgents(false)}>{tr(session?.agents?.length ? <>{tr(session.agents.map(a => <SubagentRow key={a.id} agent={a} now={now}/>))}</> : <div className="subagents-empty"><Icon name="experts" size={24}/><strong>{tr("No subagents recorded")}</strong><p className="small muted">{tr("Subagents that run in this conversation appear here, with their progress, token usage and elapsed time. Completed runs stay visible.")}</p></div>)}</Popover> : null)}
    {tr(menu ? <FloatingMenu anchor={menuRef.current} title={tr("Chat options")} onClose={() => setMenu(false)}><button role="menuitem" className="menu__item" disabled={!!remote} title={tr(remote ? "API models use their provider’s response settings." : undefined)} onClick={() => { setMenu(false); setSettings(true); }}><Icon name="sliders"/>{tr("Response settings")}</button><button role="menuitem" className="menu__item" disabled={!session} onClick={() => { if (session)
        downloadSessionLog(session); setMenu(false); }}><Icon name="download"/>{tr("Download session log")}</button></FloatingMenu> : null)}
    <SamplingDrawer open={settings} onClose={() => setSettings(false)}/>
  </header>;
}
