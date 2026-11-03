/**
 * Strata Dev (docs/DEV_BUILD.md): the Dev panel. Opened by the DEV badge or Ctrl+Shift+D; loaded only in Dev (dev.tsx, lazy import).
 * Tabs: Extensions (UI extensions and backend hooks, on/off, errors, Reload), Inspector (element ids on hover), Logs, Requests.
 * Dev tooling, so its words are not translated.
 */
import { Fragment, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../lib/client';
import { registry, setExtensionEnabled, useDev } from './dev';
import { SLOTS } from './registry';

type Tab = 'extensions' | 'inspector' | 'logs' | 'requests';
interface Hook { file: string; name: string; kind: 'js' | 'py'; events: string[]; routes: string[]; enabled: boolean; error: string | null; stats: Record<string, { calls: number; totalMs: number; maxMs: number }> }
interface LogEntry { seq: number; at: string; source: string; level: string; line: string }
interface RequestRow { id: string; at: string; client: string; endpoint: string; model: string | null; state: string; durationMs: number | null; tokensIn: number | null; tokensOut: number | null; tps: number | null; error: string | null; hooks: { file: string; event: string; calls: number; ms: number; error?: string }[]; hookMs: number }

const SOURCES = ['host', 'engine', 'harness', 'hooks', 'ui'];
const ms = (n: number | null | undefined) => (n == null ? '–' : n >= 1000 ? `${(n / 1000).toFixed(2)} s` : `${Math.round(n * 10) / 10} ms`);

export default function DevPanel({ onClose }: { onClose: () => void }) {
    const dev = useDev();
    const [tab, setTab] = useState<Tab>('extensions');
    return <section className="devpanel" role="dialog" aria-label="Strata Dev panel">
        <header className="devpanel__head">
            <strong>Strata Dev</strong>
            <nav className="devpanel__tabs" role="tablist">
                {(['extensions', 'inspector', 'logs', 'requests'] as const).map(t => <button key={t} role="tab" aria-selected={tab === t} className="devpanel__tab" onClick={() => setTab(t)}>{t[0].toUpperCase() + t.slice(1)}</button>)}
            </nav>
            <button className="devpanel__x" aria-label="Close Dev panel" onClick={onClose}>×</button>
        </header>
        <div className="devpanel__body">
            {tab === 'extensions' ? <Extensions /> : tab === 'inspector' ? <Inspector /> : tab === 'logs' ? <Logs /> : <Requests />}
        </div>
        <footer className="devpanel__foot" title={dev?.configFile}>{dev?.configFile}{dev?.warnings.length ? ` · ${dev.warnings.length} warning(s)` : ''}</footer>
    </section>;
}

function Extensions() {
    useDev();
    const [hooks, setHooks] = useState<Hook[]>([]);
    const [hooksDir, setHooksDir] = useState('');
    const [busy, setBusy] = useState(false);
    const refresh = () => api.get<{ hooks: Hook[]; hooksDir: string }>('/api/dev/state').then(s => { setHooks(s.hooks); setHooksDir(s.hooksDir); }).catch(() => undefined);
    useEffect(() => { void refresh(); }, []);
    const reload = async () => {
        setBusy(true);
        // The host reads dev.config.json and the hooks again; the page reloads for the UI extensions and the tokens.
        try { await api.post('/api/dev/reload'); } catch { /* the page reload still helps */ }
        location.reload();
    };
    const extensions = registry.extensions();
    return <div className="devpanel__stack">
        <div className="devpanel__row"><h3>UI extensions</h3><span className="devpanel__muted">dev/extensions/ui</span><span className="devpanel__spacer" /><button className="devpanel__btn" disabled={busy} onClick={() => void reload()}>Reload</button></div>
        {extensions.length ? <ul className="devpanel__list">{extensions.map(e => <li key={e.name}>
            <label className="devpanel__row"><input type="checkbox" checked={e.enabled} onChange={ev => void setExtensionEnabled(e.name, ev.target.checked)} /><strong>{e.name}</strong><span className="devpanel__muted">{e.enabled ? e.loaded ? 'loaded' : e.error ? 'failed' : 'loading' : 'off'}</span></label>
            {e.contributions.length ? <div className="devpanel__muted">{e.contributions.join(', ')}</div> : null}
            {e.error ? <pre className="devpanel__err">{e.error}{e.errorStack ? `\n${e.errorStack}` : ''}</pre> : null}
        </li>)}</ul> : <p className="devpanel__muted">No extension found. Put one in dev/extensions/ui/&lt;name&gt;/index.tsx (see info-button).</p>}
        <div className="devpanel__row"><h3>Backend hooks</h3><span className="devpanel__muted" title={hooksDir}>dev/extensions/backend</span></div>
        {hooks.length ? <ul className="devpanel__list">{hooks.map(h => <li key={h.file}>
            <label className="devpanel__row"><input type="checkbox" checked={h.enabled} onChange={ev => void api.post<{ hooks: Hook[] }>('/api/dev/hooks', { file: h.file, enabled: ev.target.checked }).then(r => setHooks(r.hooks)).catch(() => undefined)} /><strong>{h.file}</strong><span className="devpanel__muted">{h.events.join(', ') || 'no events'}</span></label>
            {h.routes.length ? <div className="devpanel__muted">{h.routes.join(', ')}</div> : null}
            {Object.keys(h.stats).length ? <div className="devpanel__muted">{Object.entries(h.stats).map(([ev, s]) => `${ev}: ${s.calls}× avg ${ms(s.totalMs / s.calls)} max ${ms(s.maxMs)}`).join(' · ')}</div> : null}
            {h.error ? <pre className="devpanel__err">{h.error}</pre> : null}
        </li>)}</ul> : <p className="devpanel__muted">No hooks.</p>}
    </div>;
}

/** Hover marks the element under the pointer with its id; a click copies the id (and does nothing else while the inspector is on). */
function Inspector() {
    useDev();
    const [on, setOn] = useState(false);
    const [hit, setHit] = useState<{ id: string; rect: DOMRect } | null>(null);
    const [copied, setCopied] = useState('');
    useEffect(() => {
        if (!on) { setHit(null); return; }
        const find = (target: EventTarget | null) => {
            const el = target instanceof Element ? target.closest('[data-dev-id]') : null;
            if (!el || el.closest('.devpanel')) return null;
            // A marker draws no box of its own (display: contents): the area is that of what it holds.
            const boxes = [...el.children].map(c => c.getBoundingClientRect()).filter(r => r.width || r.height);
            const rect = boxes.length ? boxes.reduce((a, b) => new DOMRect(Math.min(a.left, b.left), Math.min(a.top, b.top), Math.max(a.right, b.right) - Math.min(a.left, b.left), Math.max(a.bottom, b.bottom) - Math.min(a.top, b.top))) : el.getBoundingClientRect();
            return { id: el.getAttribute('data-dev-id') ?? '', rect };
        };
        const move = (e: MouseEvent) => setHit(find(e.target));
        const click = (e: MouseEvent) => {
            const h = find(e.target);
            if (!h) return;
            e.preventDefault(); e.stopPropagation();
            void navigator.clipboard?.writeText(h.id).catch(() => undefined);
            setCopied(h.id);
        };
        document.addEventListener('mousemove', move, true);
        document.addEventListener('click', click, true);
        return () => { document.removeEventListener('mousemove', move, true); document.removeEventListener('click', click, true); };
    }, [on]);
    const ids = registry.seen();
    return <div className="devpanel__stack">
        <label className="devpanel__row"><input type="checkbox" checked={on} onChange={e => setOn(e.target.checked)} /><strong>Show element ids on hover</strong><span className="devpanel__muted">click copies the id</span></label>
        {copied ? <p className="devpanel__muted">Copied: <code>{copied}</code></p> : null}
        <p className="devpanel__muted">Use an id in dev.config.json → ui.elements, e.g. {'{ "sidebar.monitor": { "hidden": true } }'}, or in an extension: strata.hide(id), rename, move, replace. Slots: {SLOTS.join(', ')}.</p>
        <h3>Ids seen on this page</h3>
        <ul className="devpanel__ids">{ids.map(id => <li key={id}><code>{id}</code></li>)}</ul>
        {hit ? <div className="devinspect" style={{ left: hit.rect.left, top: hit.rect.top, width: hit.rect.width, height: hit.rect.height }}><span className="devinspect__label">{hit.id}</span></div> : null}
    </div>;
}

function Logs() {
    const [entries, setEntries] = useState<LogEntry[]>([]);
    const [sources, setSources] = useState<string[]>(SOURCES);
    const [q, setQ] = useState('');
    const [paused, setPaused] = useState(false);
    const [error, setError] = useState('');
    const end = useRef<HTMLDivElement>(null);
    useEffect(() => {
        if (paused) return;
        let live = true;
        const poll = () => api.get<{ entries: LogEntry[] }>(`/api/dev/logs?sources=${sources.join(',')}&limit=1500`).then(r => { if (live) { setEntries(r.entries); setError(''); } }).catch(e => { if (live) setError(String(e?.message ?? e)); });
        void poll();
        const timer = setInterval(() => void poll(), 2000);
        return () => { live = false; clearInterval(timer); };
    }, [sources, paused]);
    const shown = useMemo(() => { const n = q.trim().toLowerCase(); return n ? entries.filter(e => e.line.toLowerCase().includes(n) || e.level === n || e.source === n) : entries; }, [entries, q]);
    useEffect(() => { if (!paused) end.current?.scrollIntoView({ block: 'end' }); }, [shown, paused]);
    const text = () => shown.map(e => `${e.at} [${e.source}/${e.level}] ${e.line}`).join('\n');
    const save = () => {
        const url = URL.createObjectURL(new Blob([`${text()}\n`], { type: 'text/plain' }));
        const a = Object.assign(document.createElement('a'), { href: url, download: `strata-dev-logs-${new Date().toISOString().replace(/[:.]/g, '-')}.txt` });
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    };
    return <div className="devpanel__stack devpanel__stack--fill">
        <div className="devpanel__row devpanel__wrap">
            {SOURCES.map(s => <label key={s} className="devpanel__chip"><input type="checkbox" checked={sources.includes(s)} onChange={e => setSources(v => e.target.checked ? [...v, s] : v.filter(x => x !== s))} />{s}</label>)}
            <input className="devpanel__input" type="search" placeholder="Filter (text, error, warn)" value={q} onChange={e => setQ(e.target.value)} />
            <button className="devpanel__btn" onClick={() => setPaused(p => !p)}>{paused ? 'Resume' : 'Pause'}</button>
            <button className="devpanel__btn" onClick={() => void navigator.clipboard?.writeText(text())}>Copy</button>
            <button className="devpanel__btn" onClick={save}>Save as file</button>
        </div>
        {error ? <p className="devpanel__err">{error}</p> : null}
        <div className="devpanel__log" role="log">{shown.map((e, i) => <div key={`${e.seq}-${i}`} className="devpanel__line" data-level={e.level}><span className="devpanel__muted">{e.at.slice(11, 23)}</span> <span className="devpanel__src">{e.source}</span> {e.line}</div>)}<div ref={end} /></div>
    </div>;
}

function Requests() {
    const [rows, setRows] = useState<RequestRow[]>([]);
    const [open, setOpen] = useState<string | null>(null);
    useEffect(() => {
        let live = true;
        const poll = () => api.get<{ requests: RequestRow[] }>('/api/dev/requests').then(r => { if (live) setRows(r.requests); }).catch(() => undefined);
        void poll();
        const timer = setInterval(() => void poll(), 2000);
        return () => { live = false; clearInterval(timer); };
    }, []);
    if (!rows.length) return <p className="devpanel__muted">No requests yet. Chat turns and API calls (Serving) show here with their hook times.</p>;
    return <table className="devpanel__table"><thead><tr><th>Time</th><th>Endpoint</th><th>Model</th><th>State</th><th>Duration</th><th>Tokens in/out</th><th>tok/s</th><th>Hooks</th></tr></thead><tbody>
        {rows.map(r => <Fragment key={r.id}>
            <tr data-error={r.error ? true : undefined} onClick={() => setOpen(o => o === r.id ? null : r.id)}>
                <td>{r.at.slice(11, 19)}</td><td>{r.endpoint}</td><td>{r.model ?? '–'}</td><td>{r.state}</td><td>{ms(r.durationMs)}</td><td>{r.tokensIn ?? '–'} / {r.tokensOut ?? '–'}</td><td>{r.tps ?? '–'}</td><td>{r.hooks.length ? ms(r.hookMs) : '–'}</td>
            </tr>
            {open === r.id ? <tr><td colSpan={8}>
                {r.error ? <pre className="devpanel__err">{r.error}</pre> : null}
                {r.hooks.length ? <ul className="devpanel__list">{r.hooks.map(h => <li key={`${h.file}-${h.event}`}>{h.file} · {h.event} · {h.calls}× · {ms(h.ms)}{h.error ? ` · ${h.error}` : ''}</li>)}</ul> : <span className="devpanel__muted">No hook ran for this request.</span>}
                <div className="devpanel__muted">{r.id} · {r.client}</div>
            </td></tr> : null}
        </Fragment>)}
    </tbody></table>;
}
