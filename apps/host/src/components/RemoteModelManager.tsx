import { useEffect, useRef, useState } from 'react';
import { tr } from '../lib/i18n';
import { api } from '../lib/client';
import { sizeGb } from '../lib/format';
import { Btn, Dialog, ErrorNotice, Icon } from './ui';
import './RemoteModelManager.css';

interface Target { id: string; name: string; kind: string; connectionId?: string }
interface ManagedModel { id: string; name: string; sizeGb?: number | null; state: string; detail?: string; readPictures?: boolean }
interface DownloadJob { id: string; modelId?: string; name: string; kind: string; state: string; summary: string; percent?: number; doneGb?: number; totalGb?: number; error?: string }
interface Snapshot {
    supported: boolean; reason?: string; targetName: string;
    capabilities?: { downloads?: boolean; imports?: boolean };
    models: ManagedModel[]; jobs: DownloadJob[];
}
interface DownloadPlan {
    planId: string; targetName: string; modelName: string; downloadGb?: number; diskFreeGb?: number; diskNeededGb?: number;
    diskShort: boolean | string | null; readPictures: boolean; warnings?: string[]; expiresAt: number;
}
interface ImportPlan { planId: string; targetName: string; modelName: string; sizeGb?: number; files: string[]; warnings?: string[]; expiresAt: number }
interface RemoteFolder { directory: string; parent: string | null; entries: { name: string; path: string; kind: 'folder' | 'model'; bytes?: number }[] }
type Review = { kind: 'download'; plan: DownloadPlan } | { kind: 'import'; plan: ImportPlan };
const running = (job: DownloadJob) => job.state === 'running' || job.state === 'queued';
const failure = (error: unknown) => error instanceof Error ? error.message : 'The server could not complete this request.';
const gigabytes = (value: number | null | undefined) => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? sizeGb(value) : 'Not reported';

/** Manage only the saved connection selected by the person; URLs and credentials stay on the host. */
export function RemoteModelManager({ connectionId, name, onClose }: { connectionId: string; name: string; onClose: () => void }) {
    const [target, setTarget] = useState<Target | null>(null), [snapshot, setSnapshot] = useState<Snapshot | null>(null);
    const [loading, setLoading] = useState(true), [pending, setPending] = useState(''), [error, setError] = useState(''), [notice, setNotice] = useState('');
    const [attempt, setAttempt] = useState(0);
    const [tab, setTab] = useState<'download' | 'import'>('download'), [filter, setFilter] = useState('');
    const [pictures, setPictures] = useState<Record<string, boolean>>({}), [review, setReview] = useState<Review | null>(null);
    const [folder, setFolder] = useState<RemoteFolder | null>(null), [directory, setDirectory] = useState('');
    const generation = useRef(0), targetId = useRef<string | null>(null), reading = useRef(false), readVersion = useRef(0);
    const endpoint = (id: string) => `/api/model-management/targets/${encodeURIComponent(id)}`;
    const isCurrent = (version: number, id?: string) => generation.current === version && (id === undefined || targetId.current === id);
    const targetName = snapshot?.targetName || target?.name || name;
    const hasRunningJob = snapshot?.jobs?.some(running) ?? false;

    async function refresh(id: string, showError = true) {
        if (reading.current) return;
        const version = generation.current, request = ++readVersion.current;
        reading.current = true;
        try {
            const value = await api.get<Snapshot>(endpoint(id));
            if (isCurrent(version, id) && readVersion.current === request) setSnapshot(value);
        } catch (err) { if (isCurrent(version, id) && readVersion.current === request && showError) setError(failure(err)); }
        finally { if (isCurrent(version, id)) reading.current = false; }
    }

    useEffect(() => {
        const version = ++generation.current;
        targetId.current = null; reading.current = false;
        setTarget(null); setSnapshot(null); setLoading(true); setError(''); setReview(null);
        void (async () => {
            try {
                const result = await api.get<{ targets: Target[] }>('/api/model-management/targets');
                if (!isCurrent(version)) return;
                const selected = result.targets.find(item => item.connectionId === connectionId);
                if (!selected) throw new Error('This connection does not offer model management. Enable it on the other Strata computer, then try again.');
                targetId.current = selected.id; setTarget(selected);
                const value = await api.get<Snapshot>(endpoint(selected.id));
                if (isCurrent(version, selected.id)) setSnapshot(value);
            } catch (err) { if (isCurrent(version)) setError(failure(err)); }
            finally { if (isCurrent(version)) setLoading(false); }
        })();
        return () => { generation.current += 1; targetId.current = null; };
    }, [connectionId, attempt]);

    useEffect(() => {
        if (!target || snapshot?.supported !== true) return;
        let cancelled = false;
        let timer: ReturnType<typeof setTimeout>;
        const poll = () => {
            timer = setTimeout(async () => {
                if (cancelled) return;
                if (document.visibilityState !== 'hidden') await refresh(target.id, false);
                if (!cancelled) poll();
            }, 2500);
        };
        poll();
        return () => { cancelled = true; clearTimeout(timer); };
    }, [target?.id, snapshot?.supported]);

    async function perform(label: string, work: (id: string, version: number) => Promise<void>) {
        if (!target || pending) return;
        const version = generation.current, id = target.id;
        // A poll begun before this action must not replace its newer job state.
        readVersion.current += 1;
        setPending(label); setError(''); setNotice('');
        try { await work(id, version); }
        catch (err) { if (isCurrent(version, id)) setError(failure(err)); }
        finally { if (isCurrent(version, id)) setPending(''); }
    }

    async function planDownload(model: ManagedModel) {
        await perform('plan', async (id, version) => {
            const plan = await api.post<DownloadPlan>(endpoint(id) + '/plan', { modelId: model.id, readPictures: model.readPictures === true && (pictures[model.id] ?? true) });
            if (isCurrent(version, id)) setReview({ kind: 'download', plan });
        });
    }
    async function browse(path?: string) {
        await perform('browse', async (id, version) => {
            const result = await api.post<RemoteFolder>(endpoint(id) + '/imports/browse', path?.trim() ? { directory: path.trim() } : {});
            if (isCurrent(version, id)) { setFolder(result); setDirectory(result.directory); }
        });
    }
    async function inspect(path: string) {
        await perform('inspect', async (id, version) => {
            const plan = await api.post<ImportPlan>(endpoint(id) + '/imports/inspect', { path });
            if (isCurrent(version, id)) setReview({ kind: 'import', plan });
        });
    }
    async function confirm() {
        const selected = review;
        if (!selected) return;
        await perform('confirm', async (id, version) => {
            if (selected.kind === 'download') {
                const result = await api.post<{ job: DownloadJob }>(endpoint(id) + '/downloads', { planId: selected.plan.planId });
                if (isCurrent(version, id)) {
                    setReview(null); setNotice(tr('Download started on {0}. You can close this window and check back later.', [selected.plan.targetName]));
                    setSnapshot(value => value ? { ...value, jobs: [...value.jobs.filter(job => job.id !== result.job.id), result.job] } : value);
                }
            } else {
                const result = await api.post<{ model: { id: string; name: string } }>(endpoint(id) + '/imports', { planId: selected.plan.planId });
                if (isCurrent(version, id)) {
                    setReview(null); setNotice(tr('{0} was imported on {1}. Load it there to make it available through the API.', [result.model.name, selected.plan.targetName]));
                }
            }
            if (isCurrent(version, id)) await refresh(id);
        });
    }
    async function cancel(job: DownloadJob) {
        await perform('cancel-' + job.id, async (id, version) => {
            const result = await api.post<{ job: DownloadJob }>(endpoint(id) + `/downloads/${encodeURIComponent(job.id)}/cancel`, {});
            if (isCurrent(version, id)) {
                setSnapshot(value => value ? { ...value, jobs: value.jobs.map(item => item.id === job.id ? result.job : item) } : value);
                setNotice(tr('Download stopped on {0}. Completed files are kept.', [targetName]));
            }
        });
    }

    const expired = review ? !Number.isFinite(review.plan.expiresAt) || review.plan.expiresAt <= Date.now() : false;
    const diskShort = review?.kind === 'download' && !!review.plan.diskShort;
    const diskShortMessage = review?.kind === 'download' && typeof review.plan.diskShort === 'string'
        ? review.plan.diskShort : tr('There is not enough free disk space on this computer.');
    const visibleModels = snapshot?.models?.filter(model => `${model.name} ${model.detail ?? ''}`.toLowerCase().includes(filter.toLowerCase())) ?? [];
    const warnings = review?.plan.warnings ?? [];
    return <Dialog open title={tr('Models on {0}', [targetName])} wide className="remote-model-manager" onClose={onClose}
        actions={review ? <><Btn disabled={!!pending} onClick={() => { setReview(null); setError(''); }}>{tr('Back')}</Btn><Btn variant="primary" disabled={!!pending || expired || diskShort} onClick={() => void confirm()}>{tr(pending === 'confirm' ? 'Working…' : review.kind === 'download' ? 'Download on {0}' : 'Import on {0}', pending === 'confirm' ? [] : [review.plan.targetName])}</Btn></> : <Btn onClick={onClose}>{tr('Close')}</Btn>}>
        <div className="remote-model-target"><Icon name="computer" size={20}/><div><strong>{targetName}</strong><p className="small muted">{tr('Downloads and imports happen on {0}.', [targetName])}</p></div></div>
        {loading ? <p className="small muted" role="status">{tr('Checking model management…')}</p> : null}
        {!loading && snapshot?.supported === false ? <div className="remote-model-empty"><Icon name="info" size={25}/><strong>{tr('Model management is unavailable')}</strong><p className="small muted">{tr(snapshot.reason || 'Update Strata and enable remote model management under Serving on that computer, then try again.')}</p><Btn icon="refresh" onClick={() => setAttempt(value => value + 1)}>{tr('Try again')}</Btn></div> : null}
        {!loading && snapshot?.supported && !review ? <>
            <div className="remote-model-toolbar"><div className="remote-model-tabs" aria-label={tr('Model management')}>
                <button type="button" aria-pressed={tab === 'download'} onClick={() => setTab('download')}>{tr('Download models')}</button>
                <button type="button" aria-pressed={tab === 'import'} disabled={!snapshot.capabilities?.imports} onClick={() => setTab('import')}>{tr('Import GGUF')}</button>
            </div><Btn size="sm" icon="refresh" disabled={!!pending} onClick={() => { setError(''); void refresh(target!.id); }}>{tr('Refresh')}</Btn></div>
            {tab === 'download' ? <div className="remote-model-catalog">
                {!snapshot.capabilities?.downloads ? <p className="small muted">{tr('This server does not support model downloads.')}</p> : <>
                    <label className="sr-only" htmlFor="remote-model-filter">{tr('Find a model')}</label><input id="remote-model-filter" className="st-input" type="search" placeholder={tr('Find a model…')} value={filter} onChange={event => setFilter(event.target.value)}/>
                    {visibleModels.length ? visibleModels.map(model => {
                        const installed = ['installed', 'loaded', 'ready', 'imported'].includes(model.state);
                        return <div className="remote-model-row" key={model.id}><Icon name="models" size={21}/><div className="remote-model-row__body"><strong>{model.name}</strong><p className="small muted">{tr(gigabytes(model.sizeGb))}{model.detail ? ' · ' + tr(model.detail) : ''}</p>
                            {model.readPictures && !installed ? <label className="remote-model-pictures small"><input type="checkbox" checked={pictures[model.id] ?? true} disabled={!!pending || hasRunningJob} onChange={event => setPictures(value => ({ ...value, [model.id]: event.target.checked }))}/>{tr('Include picture reading')}</label> : null}</div>
                            <Btn size="sm" icon={installed ? 'check' : 'download'} disabled={!!pending || hasRunningJob || installed} onClick={() => void planDownload(model)}>{tr(installed ? model.state === 'loaded' ? 'In use' : model.state === 'imported' ? 'Imported' : 'Downloaded' : 'Review download')}</Btn></div>;
                    }) : <p className="small muted">{tr(filter ? 'No matching models.' : 'This server has no downloadable models.')}</p>}
                </>}
            </div> : <div className="remote-model-import">
                <p className="small muted">{tr('Choose GGUF files already stored on {0}. Files stay on that computer.', [targetName])}</p>
                <form className="remote-model-path" onSubmit={event => { event.preventDefault(); if (!pending) void browse(directory); }}><input className="st-input" aria-label={tr('Folder on {0}', [targetName])} placeholder={tr('Folder on this server')} value={directory} spellCheck={false} disabled={!!pending} onChange={event => setDirectory(event.target.value)}/><Btn type="submit" icon="folder" disabled={!!pending}>{tr(pending === 'browse' ? 'Reading…' : 'Browse')}</Btn></form>
                {folder ? <><div className="remote-model-folder"><code title={folder.directory}>{folder.directory}</code><Btn size="sm" disabled={!!pending || !folder.parent} onClick={() => folder.parent && void browse(folder.parent)}>{tr('Up a folder')}</Btn></div>
                    <div className="remote-model-files" aria-label={tr('Model files on {0}', [targetName])} aria-busy={pending === 'browse'}>{folder.entries.length ? folder.entries.map(entry => <button type="button" className="remote-model-file" key={entry.path} disabled={!!pending || hasRunningJob && entry.kind === 'model'} onClick={() => void (entry.kind === 'folder' ? browse(entry.path) : inspect(entry.path))}><Icon name={entry.kind === 'folder' ? 'folder' : 'model-file'} size={19}/><span>{entry.name}</span>{entry.kind === 'folder' ? <Icon name="chevron" size={15}/> : <small className="muted">{typeof entry.bytes === 'number' ? gigabytes(entry.bytes / 1e9) : ''}</small>}</button>) : <p className="small muted">{tr('No GGUF files in this folder. Choose another folder.')}</p>}</div></> : <Btn icon="folder" disabled={!!pending} onClick={() => void browse()}>{tr('Open server model folder')}</Btn>}
            </div>}
            {snapshot.jobs?.length ? <section className="remote-model-jobs" aria-label={tr('Downloads on {0}', [targetName])}><h3 className="h4">{tr('Downloads on {0}', [targetName])}</h3>{snapshot.jobs.map(job => <div className="remote-model-job" key={job.id}><div className="remote-model-job__head"><strong>{job.name}</strong><span className="small muted">{tr(job.state === 'done' ? 'Complete' : job.state === 'cancelled' ? 'Stopped' : job.state === 'error' ? 'Failed' : 'Working')}</span></div><p className="small muted" role={running(job) ? 'status' : undefined}>{tr(job.summary)}</p>{running(job) ? <><progress max={100} value={Number.isFinite(job.percent) ? Math.max(0, Math.min(100, job.percent!)) : undefined} aria-label={tr('Download progress for {0}', [job.name])}/><div className="remote-model-job__footer"><span className="xs muted">{Number.isFinite(job.doneGb) && Number.isFinite(job.totalGb) ? `${gigabytes(job.doneGb)} / ${gigabytes(job.totalGb)}` : tr('Preparing files…')}</span><Btn size="sm" disabled={!!pending} onClick={() => void cancel(job)}>{tr(pending === 'cancel-' + job.id ? 'Stopping…' : 'Stop download')}</Btn></div></> : null}{job.error ? <ErrorNotice message={job.error}/> : null}</div>)}</section> : null}
        </> : null}
        {review ? <div className="remote-model-review"><h3 className="h4">{review.plan.modelName}</h3><p className="small muted">{tr(review.kind === 'download' ? 'Download this model on {0}.' : 'Import these existing files on {0}.', [review.plan.targetName])}</p>
            <dl className="remote-model-facts"><div><dt>{tr(review.kind === 'download' ? 'Download size' : 'Model files')}</dt><dd>{tr(gigabytes(review.kind === 'download' ? review.plan.downloadGb : review.plan.sizeGb))}</dd></div>{review.kind === 'download' ? <><div><dt>{tr('Disk space needed')}</dt><dd>{tr(gigabytes(review.plan.diskNeededGb))}</dd></div><div><dt>{tr('Free on this computer')}</dt><dd>{tr(gigabytes(review.plan.diskFreeGb))}</dd></div><div><dt>{tr('Picture reading')}</dt><dd>{tr(review.plan.readPictures ? 'Included' : 'Not included')}</dd></div></> : null}</dl>
            {review.kind === 'import' && review.plan.files.length ? <details className="remote-model-import-files"><summary>{tr('Files on {0}', [review.plan.targetName])}</summary><ul>{review.plan.files.map((file, index) => <li key={index}><code>{file}</code></li>)}</ul></details> : null}
            {warnings.length ? <ul className="remote-model-warnings">{warnings.map((warning, index) => <li key={index}>{tr(warning)}</li>)}</ul> : null}
            {diskShort ? <ErrorNotice message={diskShortMessage}/> : null}
            {expired ? <p className="small muted" role="status">{tr('This review has expired. Go back and review the model again.')}</p> : null}
        </div> : null}
        {notice ? <p className="small remote-model-notice" role="status"><Icon name="check" size={16}/>{tr(notice)}</p> : null}
        {error ? <ErrorNotice message={error}/> : null}
        {error && !snapshot && !loading ? <Btn icon="refresh" onClick={() => setAttempt(value => value + 1)}>{tr('Try again')}</Btn> : null}
    </Dialog>;
}
