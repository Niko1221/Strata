/**
 * Strata Dev (docs/DEV_BUILD.md): skills tools for the Skills page and the skill choice per chat. Both draw nothing outside Dev.
 *  - DevSkillTools: import (folder or .zip into dev/skills), per skill its source, folder, validator findings, tools, versions with
 *    restore, export as .zip.
 *  - DevChatSkills: a button in the chat toolbar; the chat's own list of skills (none chosen: all of them).
 * Dev tooling: its words are not translated.
 */
import { useEffect, useRef, useState } from 'react';
import { api } from '../lib/client';
import { loadSkills, setChatPreferences, useStore } from '../lib/store';
import { Btn, Dialog, ErrorNotice, Icon } from '../components/ui';
import { Popover } from '../components/Popover';
import { useDev } from './dev-state';

interface DevSkill { id: string; name: string; source: string; folder: string | null; dir: string | null; validation: { errors: string[]; warnings: string[]; lines: number } | null; files: string[]; tools: string[]; compatibility: string | null; versions: number }
interface Version { n: number; at: string; reason: string; files: number; truncated: boolean; edited: boolean }

export function DevSkillTools() {
    const dev = useDev();
    const skills = useStore(s => s.skills);
    const [rows, setRows] = useState<DevSkill[]>([]);
    const [dir, setDir] = useState('');
    const [path, setPath] = useState(''), [replace, setReplace] = useState(false);
    const [error, setError] = useState(''), [note, setNote] = useState('');
    const [versionsOf, setVersionsOf] = useState<DevSkill | null>(null);
    const refresh = () => api.get<{ dir: string; skills: DevSkill[] }>('/api/dev/skills').then(r => { setRows(r.skills); setDir(r.dir); }).catch(e => setError(String(e?.message ?? e)));
    useEffect(() => { if (dev) void refresh(); }, [dev, skills]);
    if (!dev) return null;
    async function importNow() {
        setError(''); setNote('');
        try {
            const r = await api.post<{ id: string; validation: DevSkill['validation'] }>('/api/dev/skills/import', { path: path.trim(), replace });
            setNote(`Imported as ${r.id}.${r.validation?.errors.length ? ` Problems: ${r.validation.errors.join(' ')}` : ''}`);
            setPath('');
            await loadSkills();
        } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    }
    return <section className="st-card devskills" aria-label="Strata Dev skills">
        <div className="devskills__head"><strong>Strata Dev</strong><span className="small muted">dev/skills: {dir}</span></div>
        <div className="devskills__import">
            <input className="st-input" placeholder="Path of a skill folder or a .zip" value={path} onChange={e => setPath(e.target.value)} />
            <label className="row small"><input type="checkbox" checked={replace} onChange={e => setReplace(e.target.checked)} />replace</label>
            <Btn size="sm" icon="folder-plus" disabled={!path.trim()} onClick={() => void importNow()}>Import</Btn>
        </div>
        {error ? <ErrorNotice message={error} /> : null}
        {note ? <p className="small">{note}</p> : null}
        <table className="devskills__table"><thead><tr><th>Skill</th><th>Source</th><th>Check</th><th>Files</th><th>Tools</th><th>Versions</th><th /></tr></thead><tbody>
            {rows.map(r => <tr key={r.id}>
                <td title={r.dir ?? ''}><strong>{r.name}</strong><div className="xs muted">{r.id}{r.compatibility ? ` · needs: ${r.compatibility}` : ''}</div></td>
                <td>{r.source}</td>
                <td>{r.validation ? r.validation.errors.length ? <span className="devskills__bad" title={r.validation.errors.join('\n')}>{r.validation.errors.length} error(s)</span> : r.validation.warnings.length ? <span className="devskills__warn" title={r.validation.warnings.join('\n')}>{r.validation.warnings.length} warning(s)</span> : <span className="devskills__ok">ok · {r.validation.lines} lines</span> : <span className="xs muted">built in</span>}</td>
                <td title={r.files.join('\n')}>{r.files.length || '–'}</td>
                <td>{r.tools.join(', ') || '–'}</td>
                <td><Btn size="sm" disabled={!r.versions} onClick={() => setVersionsOf(r)}>{r.versions}</Btn></td>
                <td><a className="st-btn st-btn--sm" href={`/api/dev/skills/${encodeURIComponent(r.id)}/export`} download>Export .zip</a></td>
            </tr>)}
        </tbody></table>
        <p className="xs muted">Scripts in a skill run through the assistant's shell (Manual asks, Plan does not run them); tools.json tools are command tools. Every change is kept as a version.</p>
        {versionsOf ? <SkillVersions skill={versionsOf} onClose={() => { setVersionsOf(null); void refresh(); }} /> : null}
    </section>;
}

function SkillVersions({ skill, onClose }: { skill: DevSkill; onClose: () => void }) {
    const [versions, setVersions] = useState<Version[]>([]);
    const [error, setError] = useState(''), [busy, setBusy] = useState(false);
    useEffect(() => { void api.get<{ versions: Version[] }>(`/api/dev/skills/${encodeURIComponent(skill.id)}/versions`).then(r => setVersions(r.versions)).catch(e => setError(String(e?.message ?? e))); }, [skill.id]);
    async function restore(n: number) {
        setBusy(true); setError('');
        try { const r = await api.post<{ versions: Version[] }>(`/api/dev/skills/${encodeURIComponent(skill.id)}/rollback`, { n }); setVersions(r.versions); await loadSkills(); }
        catch (e) { setError(e instanceof Error ? e.message : String(e)); }
        finally { setBusy(false); }
    }
    return <Dialog open title={`Versions of ${skill.name}`} onClose={onClose} actions={<Btn onClick={onClose}>Close</Btn>}>
        {error ? <ErrorNotice message={error} /> : null}
        <ul className="devskills__versions">{versions.map((v, i) => <li key={v.n}>
            <span><strong>#{v.n}</strong> {new Date(v.at).toLocaleString()} · {v.files} file(s){v.edited ? ' · app edit' : ''}{v.truncated ? ' · cut at 5 MB' : ''}{i === 0 ? ' · current' : ''}</span>
            {i > 0 ? <Btn size="sm" disabled={busy} onClick={() => void restore(v.n)}>Restore</Btn> : null}
        </li>)}</ul>
    </Dialog>;
}

/** The chat toolbar's skill choice (Dev only). Nothing chosen: every enabled skill, as outside Dev. */
export function DevChatSkills({ sessionId }: { sessionId: string | null }) {
    const dev = useDev();
    const skills = useStore(s => s.skills);
    const chosen = useStore(s => s.sessions.find(x => x.id === sessionId)?.preferences?.skills ?? null);
    const [open, setOpen] = useState(false);
    const ref = useRef<HTMLButtonElement>(null);
    if (!dev) return null;
    const enabled = skills.filter(s => s.enabled);
    const set = (next: string[] | null) => setChatPreferences(sessionId ?? undefined, { skills: next });
    const toggle = (id: string) => { const base = chosen ?? enabled.map(s => s.id); set(base.includes(id) ? base.filter(x => x !== id) : [...base, id]); };
    return <>
        <button ref={ref} type="button" className="st-btn st-btn--icon" title={chosen ? `Skills in this chat: ${chosen.length}` : 'Skills in this chat: all'} aria-label="Skills in this chat" aria-expanded={open} onClick={() => setOpen(v => !v)}><Icon name="skills" size={18} />{chosen ? <span className="devskills__count">{chosen.length}</span> : null}</button>
        {open ? <Popover anchor={ref.current} title="Skills in this chat" width={340} onClose={() => setOpen(false)}>
            <div className="devskills__chat">
                <label className="row small"><input type="checkbox" checked={!chosen} onChange={e => set(e.target.checked ? null : enabled.map(s => s.id))} />All enabled skills</label>
                {enabled.map(s => <label key={s.id} className="row small"><input type="checkbox" disabled={!chosen} checked={!chosen || chosen.includes(s.id)} onChange={() => toggle(s.id)} />{s.name}<span className="xs muted">{s.source}</span></label>)}
                <p className="xs muted">The direct chat uses only these. In an agent turn the message names them, so the assistant loads only these.</p>
            </div>
        </Popover> : null}
    </>;
}
