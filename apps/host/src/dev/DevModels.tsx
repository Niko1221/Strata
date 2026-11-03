/**
 * Strata Dev (docs/DEV_BUILD.md): the Dev part of the model menu. Draws nothing outside Dev.
 *  - Overlays from dev/models/*.json on the run configs (Control applies them at every start, strata_control/dev_models.py): add,
 *    duplicate, edit, delete; the profile per overlay ("Schnell", "Coder", "Max Qualität" or the file's own) is switched here.
 *  - Adding a model: a Hugging Face repo, a local .gguf (checked: size, and whether the Strata engine knows it), or a running
 *    OpenAI-compatible server (saved as an API connection, server/model-providers.mjs, so it is in the chat's model picker at once).
 *  - A RAM/VRAM warning from the detected hardware: a warning, never a refusal.
 * A .gguf the Strata engine does not know "läuft über llama.cpp": the Dev runner (server/dev-llama.mjs) starts the person's llama-server
 * for it and saves it as an API connection; its state, Start and Stop are here.
 * Dev tooling: its words are not translated.
 */
import { useEffect, useState } from 'react';
import { api } from '../lib/client';
import { Btn, Dialog, ErrorNotice, Select } from '../components/ui';
import { useDev } from './dev-state';

interface Overlay { id: string; data: Record<string, unknown> | null; problems: string[] }
interface State { dir: string; overlays: Overlay[]; profiles: { id: string; label: string }[]; choices: Record<string, string>; installed: { model: string; name: string; ready: boolean }[]; hardware: { ramGb: number; vramGb: number } | null }
interface Inspect { path: string; files: string[]; sizeGb: number; engine: 'strata' | 'llama.cpp' | 'unknown'; selection: { family: string; model: string } | null; warning: { level: string; text: string } | null }
type Source = 'hf' | 'gguf' | 'server';
interface Llama { status: 'stopped' | 'starting' | 'running' | 'stopping' | 'error'; model: string | null; gguf: string | null; port: number; error: string | null; available: boolean; problem: string | null; lastLines: string[] }
const isLlama = (data: Record<string, unknown> | null) => typeof data?.runner === 'string' && data.runner.startsWith('llama.cpp');

/** The same rule as the host's memoryWarning (dev-catalog.mjs): fits the graphics memory, needs RAM beside it, or reads from the disk. */
export function memoryWarning(sizeGb: number, hw: State['hardware']) {
    if (!Number.isFinite(sizeGb) || sizeGb <= 0 || !hw) return null;
    if (sizeGb <= hw.vramGb * 0.9) return null;
    if (sizeGb <= hw.vramGb + hw.ramGb * 0.75) return { level: 'warn', text: `About ${sizeGb.toFixed(1)} GB: more than the ${hw.vramGb.toFixed(0)} GB of graphics memory; the rest runs from RAM and is slower. It can still be added.` };
    return { level: 'danger', text: `About ${sizeGb.toFixed(1)} GB: more than graphics memory and RAM together (${(hw.vramGb + hw.ramGb).toFixed(0)} GB); very slow. It can still be added.` };
}

export function DevModels() {
    const dev = useDev();
    const [state, setState] = useState<State | null>(null);
    const [error, setError] = useState('');
    const [editing, setEditing] = useState<{ id: string; text: string; create: boolean } | null>(null);
    const [adding, setAdding] = useState(false);
    const [llama, setLlama] = useState<Llama | null>(null);
    const refresh = () => Promise.all([api.get<State>('/api/dev/models').then(setState), api.get<Llama>('/api/dev/llama').then(setLlama)]).catch(e => setError(String(e?.message ?? e)));
    useEffect(() => { if (dev) void refresh(); }, [dev]);
    // While llama-server starts (loading a model can take a while) its state is asked again every two seconds.
    useEffect(() => { if (llama?.status !== 'starting' && llama?.status !== 'stopping') return; const t = setInterval(() => void api.get<Llama>('/api/dev/llama').then(setLlama).catch(() => undefined), 2000); return () => clearInterval(t); }, [llama?.status]);
    if (!dev) return null;
    async function run(fn: () => Promise<State | unknown>) {
        setError('');
        try { const r = await fn(); if (r && typeof r === 'object' && 'overlays' in r) setState(r as State); else await refresh(); return true; }
        catch (e) { setError(e instanceof Error ? e.message : String(e)); return false; }
    }
    async function save() {
        if (!editing) return;
        let data: unknown;
        try { data = JSON.parse(editing.text); } catch (e) { setError(`Not valid JSON: ${(e as Error).message}`); return; }
        if (await run(() => api.post<State>('/api/dev/models', { id: editing.id, data, create: editing.create }))) setEditing(null);
    }
    async function startLlama(o: Overlay) {
        const src = (o.data?.source ?? {}) as { gguf?: string };
        setError('');
        setLlama(l => l ? { ...l, status: 'starting', error: null } : l);
        try { setLlama(await api.post<Llama>('/api/dev/llama/start', { gguf: src.gguf, label: String(o.data?.label ?? o.id), context: typeof o.data?.context === 'number' ? o.data.context : undefined })); }
        catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    }
    const profileOptions = (o: Overlay) => [{ value: '', label: `File default${typeof o.data?.profile === 'string' ? ` (${o.data.profile})` : ''}` }, ...(state?.profiles ?? []).map(p => ({ value: p.id, label: p.label })), ...Object.keys((o.data?.profiles as Record<string, unknown>) ?? {}).filter(k => !(state?.profiles ?? []).some(p => p.id === k)).map(k => ({ value: k, label: k }))];
    return <section className="st-card devmodels" aria-label="Strata Dev models">
        <div className="devskills__head"><strong>Strata Dev · models</strong><span className="small muted">dev/models: {state?.dir}</span><span className="spacer" /><Btn size="sm" icon="plus" onClick={() => setAdding(true)}>Add model</Btn><Btn size="sm" onClick={() => setEditing({ id: 'new-overlay', text: JSON.stringify({ model: 'qwen3.8-flash-next-*', label: 'My overlay', context: 65536, thinking: 'medium', sampling: { temperature: 0.8 } }, null, 2), create: true })}>New overlay</Btn></div>
        {error ? <ErrorNotice message={error} /> : null}
        {llama ? <div className="devmodels__llama" data-status={llama.status}><strong>llama.cpp runner</strong> <span className="small">{llama.status === 'running' ? `serves ${llama.model} on port ${llama.port} (API connection "llama.cpp (Strata Dev)" in the chat's model picker)` : llama.status === 'starting' ? `starting ${llama.model ?? ''} on port ${llama.port}…` : llama.status === 'error' ? llama.error : llama.available ? `ready (port ${llama.port})` : llama.problem}</span>{llama.status === 'running' || llama.status === 'starting' ? <Btn size="sm" onClick={() => void api.post<Llama>('/api/dev/llama/stop').then(setLlama).catch(e => setError(String(e?.message ?? e)))}>Stop</Btn> : null}{llama.status === 'error' && llama.lastLines.length ? <pre className="devmodels__lines">{llama.lastLines.join('\n')}</pre> : null}</div> : null}
        {state?.overlays.length ? <table className="devskills__table"><thead><tr><th>Overlay</th><th>Applies to</th><th>Profile</th><th>Source</th><th /></tr></thead><tbody>
            {state.overlays.map(o => <tr key={o.id}>
                <td><strong>{String(o.data?.label ?? o.id)}</strong><div className="xs muted">{o.id}.json</div>{o.problems.length ? <div className="devskills__bad xs">{o.problems.join('; ')}</div> : null}</td>
                <td>{String(o.data?.model ?? '–')}<div className="xs muted">{state.installed.filter(m => globMatch(String(o.data?.model ?? ''), m.name) || globMatch(String(o.data?.model ?? ''), m.model)).map(m => m.model).join(', ') || 'no installed model matches'}</div></td>
                <td><Select ariaLabel={`Profile of ${o.id}`} value={state.choices[o.id] ?? ''} options={profileOptions(o)} onChange={v => void run(() => api.post<State>(`/api/dev/models/${o.id}/profile`, { profile: v || null }))} /></td>
                <td className="xs">{sourceText(o.data?.source)}{isLlama(o.data) ? <div><span className="devmodels__badge">läuft über llama.cpp</span></div> : null}</td>
                <td className="devmodels__actions">
                    {isLlama(o.data) ? <Btn size="sm" variant="primary" disabled={!llama?.available || llama.status === 'starting'} title={llama?.available ? 'Start llama-server with this .gguf' : llama?.problem ?? ''} onClick={() => void startLlama(o)}>{llama?.status === 'running' && llama.gguf === (o.data?.source as { gguf?: string })?.gguf ? 'Restart' : 'Start'}</Btn> : null}
                    <Btn size="sm" onClick={() => setEditing({ id: o.id, text: JSON.stringify(o.data ?? {}, null, 2), create: false })}>Edit</Btn>
                    <Btn size="sm" onClick={() => setEditing({ id: `${o.id}-copy`, text: JSON.stringify(o.data ?? {}, null, 2), create: true })}>Duplicate</Btn>
                    <Btn size="sm" variant="danger" onClick={() => void run(() => api.del<State>(`/api/dev/models/${o.id}`))}>Delete</Btn>
                </td>
            </tr>)}
        </tbody></table> : <p className="small muted">No overlays yet. An overlay changes a run config at every start (context, engine options, thinking, sampling, chat template) without touching the file; a profile switches between named sets of those.</p>}
        <p className="xs muted">Applied by Strata Control at the next start of the model (a tuned preset still goes on top).</p>
        {editing ? <Dialog open title={editing.create ? 'New overlay' : `Edit ${editing.id}.json`} onClose={() => setEditing(null)} actions={<><Btn onClick={() => setEditing(null)}>Cancel</Btn><Btn variant="primary" onClick={() => void save()}>Save</Btn></>}>
            <label className="input-field"><span>File name (id)</span><input className="st-input" value={editing.id} disabled={!editing.create} onChange={e => setEditing({ ...editing, id: e.target.value })} /></label>
            <label className="input-field"><span>Overlay (JSON: model, label, source, quant, context, vision, args, env, thinking, sampling, chat_template, profiles, profile)</span><textarea className="st-input devmodels__json" rows={16} value={editing.text} onChange={e => setEditing({ ...editing, text: e.target.value })} /></label>
            {error ? <ErrorNotice message={error} /> : null}
        </Dialog> : null}
        {adding ? <AddModel hardware={state?.hardware ?? null} onClose={() => setAdding(false)} onSaved={() => { setAdding(false); void refresh(); }} /> : null}
    </section>;
}

function sourceText(source: unknown) {
    const s = (source ?? {}) as { hf?: string; file?: string; gguf?: string; server?: string };
    return s.hf ? `Hugging Face: ${s.hf}${s.file ? ` / ${s.file}` : ''}` : s.gguf ? `Local: ${s.gguf}` : s.server ? `Server: ${s.server}` : 'the run config';
}

function globMatch(pattern: string, name: string) {
    if (!pattern || !name) return false;
    const re = new RegExp(`^${pattern.toLowerCase().replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*').replace(/\?/g, '.')}$`);
    return re.test(name.toLowerCase());
}

function AddModel({ hardware, onClose, onSaved }: { hardware: State['hardware']; onClose: () => void; onSaved: () => void }) {
    const [source, setSource] = useState<Source>('gguf');
    const [id, setId] = useState(''), [label, setLabel] = useState(''), [pattern, setPattern] = useState('');
    const [repo, setRepo] = useState(''), [file, setFile] = useState(''), [sizeGb, setSizeGb] = useState('');
    const [path, setPath] = useState(''), [checked, setChecked] = useState<Inspect | null>(null);
    const [url, setUrl] = useState('http://127.0.0.1:8080/v1'), [modelId, setModelId] = useState(''), [key, setKey] = useState('');
    const [context, setContext] = useState('32768');
    const [error, setError] = useState(''), [busy, setBusy] = useState(false);
    const warning = source === 'gguf' ? checked?.warning ?? null : source === 'hf' ? memoryWarning(Number(sizeGb), hardware) : null;
    async function inspect() {
        setBusy(true); setError(''); setChecked(null);
        try { setChecked(await api.post<Inspect>('/api/dev/models/inspect', { path: path.trim() })); } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); }
    }
    async function save() {
        setBusy(true); setError('');
        try {
            if (source === 'server') {
                // A running OpenAI-compatible server is an API connection (Settings -> Models has the same): it shows in the chat's model picker.
                await api.post('/api/model-providers', { kind: 'custom', protocol: 'openai-chat', name: label || url, baseUrl: url.trim(), apiKey: key.trim(), models: [{ id: modelId.trim(), name: label || modelId.trim(), contextMax: Number(context) || null }] });
            } else {
                const name = id.trim().toLowerCase();
                const data = {
                    model: pattern.trim() || (checked?.selection ? `*${checked.selection.model.toLowerCase()}*` : name),
                    label: label || name,
                    source: source === 'hf' ? { hf: repo.trim(), ...(file.trim() ? { file: file.trim() } : {}) } : { gguf: path.trim() },
                    ...(Number(context) ? { context: Number(context) } : {}),
                    runner: source === 'gguf' && checked?.engine !== 'strata' ? 'llama.cpp' : 'strata',
                    ...(source === 'hf' && Number(sizeGb) ? { sizeGb: Number(sizeGb) } : checked ? { sizeGb: Math.round(checked.sizeGb * 10) / 10 } : {}),
                };
                await api.post('/api/dev/models', { id: name, data, create: true });
            }
            onSaved();
        } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); }
    }
    const ready = source === 'server' ? !!url.trim() && !!modelId.trim() : !!id.trim() && (source === 'hf' ? !!repo.trim() : !!checked);
    return <Dialog open title="Add a model (Strata Dev)" onClose={onClose} actions={<><Btn onClick={onClose}>Cancel</Btn><Btn variant="primary" disabled={!ready || busy} onClick={() => void save()}>Add</Btn></>}>
        <div className="form-stack">
            <Select ariaLabel="Where the model comes from" value={source} options={[{ value: 'gguf', label: 'A local .gguf file' }, { value: 'hf', label: 'A Hugging Face repository' }, { value: 'server', label: 'A running OpenAI-compatible server or API' }]} onChange={v => setSource(v as Source)} />
            {source !== 'server' ? <label className="input-field"><span>File name (id)</span><input className="st-input" value={id} placeholder="my-model" onChange={e => setId(e.target.value)} /></label> : null}
            <label className="input-field"><span>Display name</span><input className="st-input" value={label} onChange={e => setLabel(e.target.value)} /></label>
            {source === 'gguf' ? <>
                <div className="row"><input className="st-input" placeholder="C:\models\model.gguf (or its folder)" value={path} onChange={e => { setPath(e.target.value); setChecked(null); }} /><Btn size="sm" disabled={!path.trim() || busy} onClick={() => void inspect()}>Check</Btn></div>
                {checked ? <p className="small">{checked.files.length} file(s), {checked.sizeGb.toFixed(1)} GB · {checked.engine === 'strata' ? `the Strata engine knows it (${checked.selection?.family} ${checked.selection?.model}): load it with "Use a local GGUF"` : checked.engine === 'llama.cpp' ? 'not a model the Strata engine is tuned for: läuft über llama.cpp (Start in the list after adding)' : 'Strata Control could not check it: it is added to run via llama.cpp'}</p> : null}
            </> : null}
            {source === 'hf' ? <>
                <label className="input-field"><span>Repository</span><input className="st-input" placeholder="org/name-GGUF" value={repo} onChange={e => setRepo(e.target.value)} /></label>
                <label className="input-field"><span>File (optional)</span><input className="st-input" placeholder="model-Q4_K_M.gguf" value={file} onChange={e => setFile(e.target.value)} /></label>
                <label className="input-field"><span>Size in GB (for the memory check)</span><input className="st-input" inputMode="decimal" value={sizeGb} onChange={e => setSizeGb(e.target.value)} /></label>
            </> : null}
            {source === 'server' ? <>
                <label className="input-field"><span>Base URL</span><input className="st-input" value={url} onChange={e => setUrl(e.target.value)} /></label>
                <label className="input-field"><span>Model id</span><input className="st-input" value={modelId} onChange={e => setModelId(e.target.value)} /></label>
                <label className="input-field"><span>API key (optional)</span><input className="st-input" type="password" value={key} onChange={e => setKey(e.target.value)} /></label>
            </> : null}
            {source !== 'server' ? <label className="input-field"><span>Applies to run configs (model_name glob, optional)</span><input className="st-input" placeholder="qwen3.8-flash-next-*" value={pattern} onChange={e => setPattern(e.target.value)} /></label> : null}
            <label className="input-field"><span>Context (tokens)</span><input className="st-input" inputMode="numeric" value={context} onChange={e => setContext(e.target.value)} /></label>
            {warning ? <p className={warning.level === 'danger' ? 'devskills__bad small' : 'devskills__warn small'} role="status">{warning.text}</p> : null}
            {error ? <ErrorNotice message={error} /> : null}
        </div>
    </Dialog>;
}
