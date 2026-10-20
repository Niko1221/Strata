import { useEffect, useRef, useState } from 'react';
import { ApiError } from '../lib/client';
import { tr } from '../lib/i18n';
import { Btn, Dialog, ErrorNotice, Icon } from './ui';
import './RemoteServingPanel.css';

export interface ServingTarget { id: string; name: string; kind: string; connectionId?: string }
interface Snapshot {
    supported: boolean; targetName: string; modelId?: string; loaded: boolean | null; busy: boolean | null;
    reason?: string; capabilities: { load: boolean; unload: boolean }; warnings?: string[];
}
type Action = 'load' | 'unload';
interface Plan { planId: string; targetName: string; modelId: string; action: Action; warnings?: string[]; expiresAt: number }
interface Session { targetId: string; disposed: boolean; pending: boolean; readVersion: number; reading: AbortController | null; controllers: Set<AbortController> }
const failure = (error: unknown) => error instanceof Error ? error.message : 'The server could not complete this request.';

/** Requests go through the local host; saved server addresses and credentials never enter this panel. */
export async function servingHostRequest<T>(path: string, signal: AbortSignal, body?: unknown): Promise<T> {
    const base = (globalThis as { __STRATA_HOST__?: string }).__STRATA_HOST__ ?? '';
    const response = await fetch(base + path, { method: body === undefined ? 'GET' : 'POST', signal, cache: 'no-store',
        ...(body === undefined ? {} : { headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }) });
    const data = await response.json();
    if (!response.ok) throw new ApiError(typeof data?.error === 'string' ? data.error : 'The server could not complete this request.', response.status);
    return data as T;
}

export function RemoteServingPanel({ targetId, name }: { targetId: string; name: string }) {
    const [snapshot, setSnapshot] = useState<Snapshot | null>(null), [loading, setLoading] = useState(true);
    const [pending, setPending] = useState(false), [error, setError] = useState(''), [notice, setNotice] = useState('');
    const [review, setReview] = useState<Plan | null>(null), [clock, setClock] = useState(Date.now());
    const sessionRef = useRef<Session | null>(null), renderedTarget = useRef(targetId);
    renderedTarget.current = targetId;
    const endpoint = `/api/remote-serving/targets/${encodeURIComponent(targetId)}`;
    const current = (session: Session) => !session.disposed && sessionRef.current === session && renderedTarget.current === session.targetId;

    async function request<T>(session: Session, path: string, body?: unknown, reading = false) {
        const controller = new AbortController();
        session.controllers.add(controller);
        if (reading) session.reading = controller;
        try { return await servingHostRequest<T>(path, controller.signal, body); }
        finally { session.controllers.delete(controller); if (session.reading === controller) session.reading = null; }
    }
    async function refresh(session: Session, clearError = false) {
        if (!current(session) || session.pending || session.reading) return;
        const version = ++session.readVersion;
        try {
            const value = await request<Snapshot>(session, `/api/remote-serving/targets/${encodeURIComponent(session.targetId)}`, undefined, true);
            if (current(session) && version === session.readVersion) { setSnapshot(value); if (clearError) setError(''); }
        } catch (err) {
            if (current(session) && version === session.readVersion) { setSnapshot(null); setError(failure(err)); }
        } finally { if (current(session) && version === session.readVersion) setLoading(false); }
    }
    useEffect(() => {
        const session: Session = { targetId, disposed: false, pending: false, readVersion: 0, reading: null, controllers: new Set() };
        sessionRef.current = session;
        setSnapshot(null); setLoading(true); setPending(false); setError(''); setNotice(''); setReview(null);
        let timer: ReturnType<typeof setTimeout>;
        const poll = async () => {
            if (!current(session)) return;
            if (document.visibilityState !== 'hidden') await refresh(session);
            if (current(session)) timer = setTimeout(poll, 2500);
        };
        void poll();
        return () => {
            session.disposed = true; clearTimeout(timer);
            for (const controller of session.controllers) controller.abort();
            session.controllers.clear();
        };
    }, [targetId]);
    useEffect(() => {
        if (!review || !Number.isFinite(review.expiresAt)) return;
        const timer = setTimeout(() => setClock(Date.now()), Math.max(0, Math.min(2_147_483_647, review.expiresAt - Date.now() + 5)));
        return () => clearTimeout(timer);
    }, [review]);

    const actionable = (action: Action) => snapshot?.supported === true && snapshot.busy === false && typeof snapshot.modelId === 'string'
        && snapshot.capabilities?.[action] === true && snapshot.loaded === (action === 'unload');
    async function perform(work: (session: Session) => Promise<void>) {
        const session = sessionRef.current;
        if (!session || !current(session) || session.pending) return;
        session.pending = true; session.readVersion += 1; session.reading?.abort();
        setPending(true); setError(''); setNotice('');
        try { await work(session); }
        catch (err) { if (current(session)) setError(failure(err)); }
        finally { if (current(session)) { session.pending = false; setPending(false); } }
    }
    async function plan(action: Action) {
        if (!actionable(action)) return;
        const modelId = snapshot!.modelId;
        await perform(async session => {
            const value = await request<Plan>(session, endpoint + '/plan', { action });
            if (!current(session)) return;
            if (value.action !== action || value.modelId !== modelId || typeof value.planId !== 'string' || !value.planId)
                throw new Error('The server model changed. Refresh its status and review again.');
            setClock(Date.now()); setReview(value);
        });
    }
    const expired = !!review && (!Number.isFinite(review.expiresAt) || review.expiresAt <= Math.max(clock, Date.now()));
    const changed = !!review && (!actionable(review.action) || snapshot?.modelId !== review.modelId);
    async function confirm() {
        const selected = review;
        if (!selected || expired || selected.expiresAt <= Date.now() || changed) return;
        await perform(async session => {
            // A preview is single-use, including when the connection drops after the server accepts it.
            setReview(null);
            try {
                const value = await request<Snapshot>(session, endpoint, { planId: selected.planId });
                if (current(session)) { setSnapshot(value); setNotice(tr('Serving status updated on {0}.', [selected.targetName])); }
            } catch (err) { if (current(session)) setSnapshot(null); throw err; }
        });
        const session = sessionRef.current;
        if (session && current(session)) await refresh(session);
    }
    const targetName = snapshot?.targetName || name;
    return <div className="remote-serving stack stack--lg">
        <section className="st-card remote-serving-overview" aria-labelledby="remote-serving-title">
            <div className="remote-serving-heading"><Icon name="computer" size={24}/><div><h2 id="remote-serving-title" className="h3">{tr('Serving on {0}', [targetName])}</h2><p className="small muted">{tr('Control only the model already configured on this server.')}</p></div><Btn size="sm" icon="refresh" disabled={pending || loading} onClick={() => { const session = sessionRef.current; if (session) void refresh(session, true); }}>{tr('Refresh')}</Btn></div>
            {loading ? <p role="status" className="small muted">{tr('Checking serving status…')}</p> : null}
            {snapshot ? <dl className="summary-list"><div><dt>{tr('Current configured model')}</dt><dd><code>{snapshot.modelId || tr('Not reported')}</code></dd></div><div><dt>{tr('Model state')}</dt><dd>{tr(snapshot.loaded === true ? 'Model loaded' : snapshot.loaded === false ? 'Model unloaded' : 'Not reported')}</dd></div><div><dt>{tr('Requests')}</dt><dd>{tr(snapshot.busy === true ? 'Busy' : snapshot.busy === false ? 'Idle' : 'Not reported')}</dd></div></dl> : null}
            {!loading && snapshot?.supported === false ? <p className="small muted">{tr('Serving controls are unavailable for this connection.')}</p> : null}
            {snapshot?.reason ? <p className="small muted" role="status">{tr(snapshot.reason)}</p> : null}
            {snapshot?.warnings?.length ? <ul className="remote-serving-warnings">{snapshot.warnings.map((warning, index) => <li key={index}>{tr(warning)}</li>)}</ul> : null}
            <div className="remote-serving-actions"><Btn disabled={pending || !actionable('load')} onClick={() => void plan('load')}>{tr('Review load')}</Btn><Btn disabled={pending || !actionable('unload')} onClick={() => void plan('unload')}>{tr('Review unload')}</Btn></div>
        </section>
        {pending ? <p className="small muted" role="status">{tr('Waiting for {0}…', [targetName])}</p> : null}
        <p className="small muted remote-serving-explanation">{tr('Unloading releases the current model from memory. The API listener remains available, and the next chat request may load the model again.')}</p>
        {notice ? <p className="small" role="status">{notice}</p> : null}
        {error ? <ErrorNotice message={error}/> : null}
        <Dialog open={!!review} title={tr(review?.action === 'unload' ? 'Review unload on {0}' : 'Review load on {0}', [review?.targetName || targetName])} onClose={() => { if (!pending) setReview(null); }} actions={<><Btn disabled={pending} onClick={() => setReview(null)}>{tr('Back')}</Btn><Btn variant="primary" disabled={pending || expired || changed} onClick={() => void confirm()}>{tr(review?.action === 'unload' ? 'Unload on {0}' : 'Load on {0}', [review?.targetName || targetName])}</Btn></>}>
            {review ? <div className="remote-serving-review"><p>{tr(review.action === 'unload' ? 'Unload the current model on {0}?' : 'Load the current model on {0}?', [review.targetName])}</p><dl className="summary-list"><div><dt>{tr('Destination')}</dt><dd>{review.targetName}</dd></div><div><dt>{tr('Current configured model')}</dt><dd><code>{review.modelId}</code></dd></div></dl>{review.warnings?.length ? <ul className="remote-serving-warnings">{review.warnings.map((warning, index) => <li key={index}>{tr(warning)}</li>)}</ul> : null}{expired ? <p role="status" className="small muted">{tr('This review has expired. Go back and review the action again.')}</p> : changed ? <p role="status" className="small muted">{tr('The server state changed. Go back and review the action again.')}</p> : null}</div> : null}
        </Dialog>
    </div>;
}
