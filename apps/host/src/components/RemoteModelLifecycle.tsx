import { useEffect, useRef, useState } from 'react';
import { tr } from '../lib/i18n';
import { sizeGb } from '../lib/format';
import { servingHostRequest } from './RemoteServingPanel';
import { Btn, Dialog, ErrorNotice, Icon, Select } from './ui';
import './RemoteModelLifecycle.css';

interface Model { id: string; name: string; state: string; preparable: boolean; reason?: string }
interface Engine { id: string; name: string; backend?: string }
interface Profile { id: string; name: string; modelId: string; context: number; state: string; executionCompatibility?: string }
interface Job { id: string; kind: 'prepare' | 'serve'; state: string; requestId: string; summary: string; canCancel: boolean; percent?: number; profileId?: string; receiptId?: string }
interface Snapshot {
    supported: boolean; targetName?: string; reason?: string; capabilities?: { prepare?: boolean; serve?: boolean };
    models: Model[]; engines: Engine[]; profiles: Profile[]; jobs: Job[];
    runtime: { state: string; profileId?: string; modelId?: string; loaded?: boolean | null; busy?: boolean | null; host?: string; port?: number };
}
interface Plan {
    planId: string; expiresAt: number; targetName?: string; modelName: string; warnings?: string[];
    engineName?: string; context?: number; diskNeededGb?: number; diskFreeGb?: number; diskShort?: boolean | string;
    profileId?: string; host?: string; port?: number; action?: 'start' | 'switch';
}
type Review = { kind: 'prepare' | 'serve'; plan: Plan };
interface Session { targetId: string; disposed: boolean; pending: boolean; serial: number; reading: AbortController | null; controllers: Set<AbortController> }
const active = (job: Job) => ['queued', 'running'].includes(job.state);
const failure = (value: unknown) => value instanceof Error ? value.message : 'This model action could not finish.';
const gb = (value: number | undefined) => Number.isFinite(value) && value! >= 0 ? sizeGb(value!) : tr('Not reported');

/** This panel talks only to the Mac host. SSH configuration and managed API keys stay there. */
export function RemoteModelLifecycle({ targetId, name, onClose }: { targetId: string; name: string; onClose: () => void }) {
    const [snapshot, setSnapshot] = useState<Snapshot | null>(null), [loading, setLoading] = useState(true);
    const [pending, setPending] = useState(false), [error, setError] = useState(''), [notice, setNotice] = useState('');
    const [modelId, setModelId] = useState(''), [engineId, setEngineId] = useState(''), [context, setContext] = useState('32768');
    const [review, setReview] = useState<Review | null>(null), [clock, setClock] = useState(Date.now());
    const sessionRef = useRef<Session | null>(null), renderedTarget = useRef(targetId);
    renderedTarget.current = targetId;
    const current = (session: Session) => !session.disposed && sessionRef.current === session && renderedTarget.current === session.targetId;
    async function request<T>(session: Session, suffix: string, body?: unknown, reading = false) {
        const controller = new AbortController(); session.controllers.add(controller);
        if (reading) session.reading = controller;
        try { return await servingHostRequest<T>(`/api/remote-lifecycle/targets/${encodeURIComponent(session.targetId)}` + suffix, controller.signal, body); }
        finally { session.controllers.delete(controller); if (session.reading === controller) session.reading = null; }
    }
    async function refresh(session: Session, clearError = false) {
        if (!current(session) || session.pending || session.reading) return;
        const serial = ++session.serial;
        try {
            const value = await request<Snapshot>(session, '', undefined, true);
            if (!current(session) || serial !== session.serial) return;
            setSnapshot(value); if (clearError) setError('');
            setModelId(selected => value.models.some(model => model.id === selected) ? selected : value.models.find(model => model.preparable)?.id ?? value.models[0]?.id ?? '');
            setEngineId(selected => value.engines.some(engine => engine.id === selected) ? selected : value.engines[0]?.id ?? '');
        } catch (err) { if (current(session) && serial === session.serial) { setSnapshot(null); setError(failure(err)); } }
        finally { if (current(session) && serial === session.serial) setLoading(false); }
    }
    useEffect(() => {
        const session: Session = { targetId, disposed: false, pending: false, serial: 0, reading: null, controllers: new Set() };
        sessionRef.current = session;
        setSnapshot(null); setLoading(true); setPending(false); setError(''); setNotice(''); setReview(null); setModelId(''); setEngineId('');
        let timer: ReturnType<typeof setTimeout>;
        const poll = async () => {
            if (!current(session)) return;
            if (document.visibilityState !== 'hidden') await refresh(session);
            if (current(session)) timer = setTimeout(poll, 2500);
        };
        void poll();
        return () => { session.disposed = true; clearTimeout(timer); for (const controller of session.controllers) controller.abort(); session.controllers.clear(); };
    }, [targetId]);
    useEffect(() => {
        if (!review || !Number.isFinite(review.plan.expiresAt)) return;
        const timer = setTimeout(() => setClock(Date.now()), Math.max(0, Math.min(2_147_483_647, review.plan.expiresAt - Date.now() + 5)));
        return () => clearTimeout(timer);
    }, [review]);
    async function perform(work: (session: Session) => Promise<void>) {
        const session = sessionRef.current;
        if (!session || !current(session) || session.pending) return;
        session.pending = true; session.serial++; session.reading?.abort(); setPending(true); setError(''); setNotice('');
        try { await work(session); }
        catch (err) { if (current(session)) setError(failure(err)); }
        finally { if (current(session)) { session.pending = false; setPending(false); } }
    }
    const selectedModel = snapshot?.models.find(model => model.id === modelId);
    const busy = snapshot?.jobs.some(active) ?? false;
    const canChangeRuntime = !!snapshot && (snapshot.runtime.state === 'stopped' || snapshot.runtime.state === 'ready' && snapshot.runtime.busy === false);
    const destination = snapshot?.targetName || name;
    async function preparePlan() {
        if (!selectedModel?.preparable || !engineId || busy || !snapshot?.capabilities?.prepare) return;
        await perform(async session => {
            const plan = await request<Plan>(session, '/prepare-plan', { modelId, engineId, context: Number(context) });
            if (current(session)) { setClock(Date.now()); setReview({ kind: 'prepare', plan }); }
        });
    }
    async function servePlan(profile: Profile) {
        if (busy || !snapshot?.capabilities?.serve || !canChangeRuntime) return;
        await perform(async session => {
            const plan = await request<Plan>(session, '/serve-plan', { profileId: profile.id });
            if (current(session)) {
                if (plan.profileId !== profile.id) throw new Error('The prepared model changed. Refresh and review again.');
                setClock(Date.now()); setReview({ kind: 'serve', plan });
            }
        });
    }
    const expired = !!review && (!Number.isFinite(review.plan.expiresAt) || review.plan.expiresAt <= Math.max(clock, Date.now()));
    const changed = !!review && (!snapshot || busy || review.kind === 'serve' && (!canChangeRuntime || !snapshot.profiles.some(profile => profile.id === review.plan.profileId)));
    async function confirm() {
        const selected = review;
        if (!selected || expired || changed || selected.plan.diskShort || selected.plan.expiresAt <= Date.now()) return;
        await perform(async session => {
            setReview(null);
            const result = await request<{ job: Job }>(session, '/' + selected.kind, { planId: selected.plan.planId });
            if (current(session)) {
                setSnapshot(value => value ? { ...value, jobs: [...value.jobs.filter(job => job.id !== result.job.id), result.job] } : value);
                setNotice(tr('Started on {0}. You can close this window and check progress later.', [destination]));
            }
        });
    }
    async function cancel(job: Job) {
        await perform(async session => {
            const result = await request<{ job: Job }>(session, `/jobs/${encodeURIComponent(job.id)}/cancel`, {});
            if (current(session)) setSnapshot(value => value ? { ...value, jobs: value.jobs.map(item => item.id === result.job.id ? result.job : item) } : value);
        });
    }
    async function connect(job: Job) {
        if (!job.receiptId || job.state !== 'done') return;
        await perform(async session => {
            const result = await request<{ connectionId: string; name: string }>(session, '/connect', { receiptId: job.receiptId });
            if (current(session)) setNotice(tr('{0} is saved as a separate connection. Select it in the chat model picker or Serving → Destination.', [result.name]));
        });
    }
    const runtime = snapshot?.runtime;
    return <Dialog open wide title={tr('Prepare and serve on {0}', [destination])} className="remote-lifecycle" onClose={onClose}
        actions={review ? <><Btn disabled={pending} onClick={() => { setReview(null); setError(''); }}>{tr('Back')}</Btn><Btn variant="primary" disabled={pending || expired || changed || !!review.plan.diskShort} onClick={() => void confirm()}>{tr(review.kind === 'prepare' ? 'Prepare on {0}' : review.plan.action === 'switch' ? 'Switch on {0}' : 'Start API on {0}', [destination])}</Btn></> : <Btn onClick={onClose}>{tr('Back to model files')}</Btn>}>
        <div className="remote-lifecycle-target"><Icon name="computer" size={24}/><div><strong>{destination}</strong><p className="small muted">{tr('Prepare models here and switch your managed Strata API. Existing services, including Maya, stay separate.')}</p></div><Btn size="sm" icon="refresh" disabled={pending || loading} onClick={() => { const session = sessionRef.current; if (session) void refresh(session, true); }}>{tr('Refresh')}</Btn></div>
        {loading ? <p role="status" className="small muted">{tr('Checking preparation and serving…')}</p> : null}
        {snapshot?.reason ? <p role="status" className="small muted">{tr(snapshot.reason)}</p> : null}
        {snapshot?.supported && !review ? <>
            <section className="remote-lifecycle-card stack" aria-labelledby="remote-prepare-title"><h3 id="remote-prepare-title" className="h4">{tr('Prepare downloaded or imported files')}</h3>
                {snapshot.models.length ? <><label className="input-field"><span>{tr('Model files')}</span><Select ariaLabel={tr('Model files to prepare')} translateLabels={false} value={modelId} options={snapshot.models.map(model => ({ value: model.id, label: model.name }))} onChange={setModelId} disabled={pending || busy}/></label>
                {selectedModel?.reason ? <p className="small muted">{tr(selectedModel.reason)}</p> : null}
                <div className="remote-lifecycle-choices"><label className="input-field"><span>{tr('Strata engine')}</span><Select ariaLabel={tr('Strata engine for preparation')} translateLabels={false} value={engineId} options={snapshot.engines.map(engine => ({ value: engine.id, label: engine.name }))} onChange={setEngineId} disabled={pending || busy || !snapshot.engines.length}/></label><label className="input-field"><span>{tr('Context length')}</span><Select ariaLabel={tr('Prepared context length')} value={context} options={[{ value: '32768', label: '32K' }, { value: '65536', label: '64K' }, { value: '131072', label: '128K' }]} onChange={setContext} disabled={pending || busy}/></label></div>
                {!snapshot.engines.length ? <p className="small muted">{tr('No existing Strata engine was found. An installed Strata engine is needed for this workflow.')}</p> : null}
                <div><Btn disabled={pending || busy || !snapshot.capabilities?.prepare || !selectedModel?.preparable || !engineId} onClick={() => void preparePlan()}>{tr('Review preparation')}</Btn></div></> : <p className="small muted">{tr('Download or import model files first, then return here to prepare them.')}</p>}
            </section>
            <section className="remote-lifecycle-card stack" aria-labelledby="remote-profiles-title"><h3 id="remote-profiles-title" className="h4">{tr('Prepared models')}</h3>
                {snapshot.profiles.length ? snapshot.profiles.map(profile => <div className="remote-lifecycle-profile" key={profile.id}><div><strong>{profile.name}</strong><p className="small muted">{profile.context / 1024}K · {tr(runtime?.profileId === profile.id && runtime.state === 'ready' ? runtime.loaded ? 'Model loaded' : 'API listening · model unloaded' : 'Prepared · engine execution not verified')}</p></div><Btn size="sm" disabled={pending || busy || !snapshot.capabilities?.serve || !canChangeRuntime || runtime?.profileId === profile.id && runtime.state === 'ready'} onClick={() => void servePlan(profile)}>{tr(runtime?.profileId === profile.id && runtime.state === 'ready' ? 'Current model' : runtime?.state === 'ready' ? 'Review switch' : 'Review start API')}</Btn></div>) : <p className="small muted">{tr('Prepared models appear here after preparation finishes.')}</p>}
                <p className="small muted">{tr('A new managed API starts with the model unloaded. Save its connection, then use Serving to load it when enough memory is free. Switching refuses active requests. Other API listeners keep running.')}</p>
                {runtime && runtime.state !== 'stopped' && runtime.busy === null ? <p role="status" className="small muted">{tr('The managed API has not reported reliable idle status. Switching stays unavailable until it can be verified.')}</p> : null}
                {runtime?.busy === true ? <p role="status" className="small muted">{tr('The managed API is busy. Wait for its requests to finish before switching.')}</p> : null}
            </section>
            {snapshot.jobs.length ? <section className="stack" aria-label={tr('Preparation and serving jobs')}><h3 className="h4">{tr('Progress on {0}', [destination])}</h3>{snapshot.jobs.map(job => <div className="remote-lifecycle-card stack" key={job.id}><div className="remote-lifecycle-job-heading"><strong>{tr(job.kind === 'prepare' ? 'Model preparation' : 'Managed API')}</strong><span className="small muted">{tr(job.state === 'done' ? 'Complete' : job.state === 'error' ? 'Failed' : job.state === 'cancelled' ? 'Stopped' : 'Working')}</span></div><p className="small muted" role={active(job) ? 'status' : undefined}>{tr(job.summary)}</p>{active(job) ? <><progress max={100} value={Number.isFinite(job.percent) ? Math.max(0, Math.min(100, job.percent!)) : undefined} aria-label={tr('Model preparation progress')}/>{job.canCancel ? <div><Btn size="sm" disabled={pending} onClick={() => void cancel(job)}>{tr('Stop preparation')}</Btn></div> : null}</> : job.state === 'done' && job.receiptId ? <div><Btn size="sm" disabled={pending} onClick={() => void connect(job)}>{tr('Save managed connection')}</Btn></div> : null}</div>)}</section> : null}
        </> : null}
        {review ? <div className="remote-lifecycle-review stack"><h3 className="h4">{review.plan.modelName}</h3><p>{tr(review.kind === 'prepare' ? 'Prepare these model files on {0}?' : review.plan.action === 'switch' ? 'Switch the managed Strata API on {0} to this model?' : 'Start a separate managed Strata API on {0}?', [destination])}</p><dl className="remote-model-facts"><div><dt>{tr('Destination')}</dt><dd>{destination}</dd></div>{review.kind === 'prepare' ? <><div><dt>{tr('Engine')}</dt><dd>{review.plan.engineName}</dd></div><div><dt>{tr('Context')}</dt><dd>{(review.plan.context ?? 0) / 1024}K</dd></div><div><dt>{tr('Disk space needed')}</dt><dd>{gb(review.plan.diskNeededGb)}</dd></div><div><dt>{tr('Disk space free')}</dt><dd>{gb(review.plan.diskFreeGb)}</dd></div></> : <><div><dt>{tr('Managed API')}</dt><dd><code>{review.plan.host}:{review.plan.port}</code></dd></div><div><dt>{tr('Initial model state')}</dt><dd>{tr('Unloaded')}</dd></div></>}</dl>{review.plan.warnings?.length ? <ul className="remote-model-warnings">{review.plan.warnings.map((warning, index) => <li key={index}>{tr(warning)}</li>)}</ul> : null}{review.plan.diskShort ? <ErrorNotice message={typeof review.plan.diskShort === 'string' ? review.plan.diskShort : tr('There is not enough free disk space. Free space and review again.')}/> : null}{expired ? <p role="status" className="small muted">{tr('This review expired. Go back and review again.')}</p> : changed ? <p role="status" className="small muted">{tr('The server state changed. Go back and review again.')}</p> : null}</div> : null}
        {pending ? <p role="status" className="small muted">{tr('Waiting for {0}…', [destination])}</p> : null}
        {notice ? <p role="status" className="small remote-model-notice"><Icon name="check" size={16}/>{notice}</p> : null}
        {error ? <ErrorNotice message={error}/> : null}
    </Dialog>;
}
