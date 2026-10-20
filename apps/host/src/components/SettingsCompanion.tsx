import { useEffect, useRef, useState } from 'react';
import { api } from '../lib/client';
import { tr } from '../lib/i18n';
import { Btn, Dialog, ErrorNotice, Icon, IconBtn, Pill } from './ui';
import './SettingsCompanion.css';

interface PairedDevice {
  id: string;
  deviceName: string;
  createdAt: string;
}

interface CompanionState {
  enabled: boolean;
  invitation: {
    url: string;
    code: string;
    fingerprint: string;
    address: string;
    expiresAt: number;
    available: boolean;
    qrAvailable: boolean;
  } | null;
  devices: PairedDevice[];
  error?: string | null;
}

/** The desktop host owns pairing. Opening this panel never starts a listener. */
export function SettingsCompanion() {
  const [state, setState] = useState<CompanionState | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [now, setNow] = useState(Date.now);
  const [remove, setRemove] = useState<PairedDevice | null>(null);
  const [copied, setCopied] = useState('');
  const [qrFailed, setQrFailed] = useState(false);
  const mounted = useRef(false);
  const request = useRef(0);
  const polling = useRef(false);
  const changing = useRef(false);
  const copyTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  async function load() {
    if (!mounted.current || polling.current || changing.current || document.hidden) return;
    polling.current = true;
    const revision = ++request.current;
    try {
      const next = await api.get<CompanionState>('/api/companion');
      if (mounted.current && revision === request.current) {
        setState(next);
        setError('');
        setNow(Date.now());
      }
    } catch {
      if (mounted.current && revision === request.current) setError('Pairing status could not be loaded. Try again.');
    } finally {
      polling.current = false;
    }
  }

  useEffect(() => {
    mounted.current = true;
    void load();
    const refresh = () => { if (!document.hidden) void load(); };
    const timer = setInterval(refresh, 3000);
    document.addEventListener('visibilitychange', refresh);
    return () => {
      mounted.current = false;
      ++request.current;
      clearInterval(timer);
      clearTimeout(copyTimer.current);
      document.removeEventListener('visibilitychange', refresh);
    };
  }, []);

  const invitation = state?.invitation;
  useEffect(() => { setQrFailed(false); setCopied(''); }, [invitation?.url]);

  async function change(path: string, method: 'post' | 'del' = 'post') {
    if (changing.current) return;
    changing.current = true;
    setBusy(true);
    setError('');
    ++request.current; // An older status request cannot undo this operation.
    try {
      const next = await api[method]<CompanionState>(path);
      if (mounted.current) {
        setState(next);
        setNow(Date.now());
        setRemove(null);
      }
    } catch (cause) {
      if (mounted.current) setError(cause instanceof Error ? cause.message : 'Pairing could not be updated. Check that Strata is running and try again.');
    } finally {
      changing.current = false;
      if (mounted.current) setBusy(false);
    }
  }

  async function copy(value: string, field: string) {
    try {
      await navigator.clipboard.writeText(value);
      if (!mounted.current) return;
      setCopied(field);
      clearTimeout(copyTimer.current);
      copyTimer.current = setTimeout(() => { if (mounted.current) setCopied(''); }, 2000);
    } catch {
      if (mounted.current) setError('The clipboard is unavailable. Select the connection details and copy them manually.');
    }
  }

  const seconds = invitation ? Math.max(0, Math.ceil((invitation.expiresAt - now) / 1000)) : 0;
  const available = state?.enabled && invitation?.available && seconds > 0;
  const expiration = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
  const host = (globalThis as { __STRATA_HOST__?: string }).__STRATA_HOST__ ?? '';

  return <div className="stack stack--lg settings-companion">
    <div className="companion-heading">
      <div><h2 className="h3">{tr('Pair your iPhone')}</h2><p className="small muted settings-models__intro">{tr('Your Mac does the work. Your iPhone keeps you connected.')}</p></div>
      {state ? <Pill tone={state.enabled ? 'accent' : 'plain'} dot={state.enabled}>{tr(state.enabled ? 'Pairing enabled' : 'Pairing off')}</Pill> : null}
    </div>

    {error || state?.error ? <ErrorNotice message={tr(error || state!.error!)} onRetry={() => void load()} /> : null}
    {!state ? !error ? <p className="small muted" role="status">{tr('Loading pairing settings…')}</p> : null : <>
      {!state.enabled ? <section className="st-card companion-enable" aria-labelledby="companion-enable-title">
        <span className="companion-mark"><Icon name="link" size={28}/></span>
        <div><h3 className="h4" id="companion-enable-title">{tr('Take Strata with you')}</h3><p className="small muted">{tr('Read your Mac’s chat history, continue its conversations, follow browser and tool activity, and answer tool approvals from your iPhone.')}</p></div>
        <p className="small muted">{tr('Keep your Mac and iPhone on the same trusted Wi-Fi network. Enable pairing when you are ready to connect.')}</p>
        <div><Btn variant="primary" icon="link" disabled={busy} onClick={() => void change('/api/companion/start')}>{tr(busy ? 'Starting pairing…' : 'Enable iPhone pairing')}</Btn></div>
      </section> : <section className="st-card companion-invitation" aria-labelledby="companion-invitation-title">
        <div className="companion-invitation__head"><h3 className="h4" id="companion-invitation-title">{tr(available ? 'Scan to connect' : 'Ready for another iPhone?')}</h3>{available ? <span className="xs muted companion-expiration"><Icon name="clock" size={14}/>{tr('Expires in {0}', [expiration])}</span> : null}</div>
        {available && invitation ? <>
          <div className="companion-pairing-grid">
            <div className="companion-qr-frame">
              {invitation.qrAvailable && !qrFailed ? <img className="companion-qr" src={`${host}/api/companion/qr?v=${invitation.expiresAt}`} alt={tr('Scan this QR code in Strata on your iPhone')} width={224} height={224} onError={() => setQrFailed(true)}/> : <div className="companion-qr-fallback"><Icon name="link" size={32}/><p className="small muted">{tr('Use the connection details below to pair manually.')}</p></div>}
            </div>
            <div className="companion-instructions">
              <ol><li>{tr('Open Strata on your iPhone.')}</li><li>{tr('Choose Pair your Mac.')}</li><li>{tr('Scan this code to connect.')}</li></ol>
              <div><span className="xs muted">{tr('One-time pairing code')}</span><div className="companion-code-row"><code className="companion-code" aria-label={tr('Pairing code')}>{invitation.code}</code><IconBtn name={copied === 'code' ? 'check' : 'copy'} title={tr(copied === 'code' ? 'Pairing code copied' : 'Copy pairing code')} onClick={() => void copy(invitation.code, 'code')}/></div></div>
              <p className="xs muted">{tr('This invitation works once. Keep both devices on the same trusted Wi-Fi network and leave Strata open on your Mac.')}</p>
            </div>
          </div>
          <details className="companion-manual">
            <summary>{tr('Connect manually')}</summary>
            <div className="stack">
              <p className="small muted">{tr('On your iPhone, paste the invitation or enter these details in Pair your Mac.')}</p>
              <CopyDetail label={tr('Pairing invitation')} value={invitation.url} copied={copied === 'invitation'} onCopy={() => void copy(invitation.url, 'invitation')} />
              <CopyDetail label={tr('Mac address')} value={invitation.address} copied={copied === 'address'} onCopy={() => void copy(invitation.address, 'address')} />
              <CopyDetail label={tr('Certificate fingerprint')} value={invitation.fingerprint} copied={copied === 'fingerprint'} onCopy={() => void copy(invitation.fingerprint, 'fingerprint')} />
            </div>
          </details>
        </> : <p className="small muted">{tr('The previous invitation was used or expired. Create a new code to connect another iPhone. Your paired devices keep their access.')}</p>}
        <div className="companion-invitation__actions"><Btn size="sm" icon="refresh" disabled={busy} onClick={() => void change('/api/companion/invitation')}>{tr(busy ? 'Updating…' : 'Create new code')}</Btn><span className="xs muted"><Icon name="shield" size={14}/>{tr('Encrypted connection to this Mac')}</span></div>
      </section>}

      <section className="settings-section" aria-labelledby="companion-devices-title">
        <h3 className="h4 companion-section-title" id="companion-devices-title">{tr('Paired devices')}</h3>
        <div className="st-card st-card--flush">
          {state.devices.length ? <ul className="companion-devices">{state.devices.map(device => <li key={device.id} className="companion-device"><span className="companion-device__icon"><Icon name="link" size={20}/></span><div className="companion-device__name"><strong>{device.deviceName}</strong><span className="xs muted">{tr('Paired {0}', [new Date(device.createdAt).toLocaleDateString()])}</span></div><Btn size="sm" variant="ghost" disabled={busy || !state.enabled} title={!state.enabled ? tr('Enable pairing to manage paired devices') : undefined} aria-label={tr('Remove {0}', [device.deviceName])} onClick={() => setRemove(device)}>{tr('Remove')}</Btn></li>)}</ul> : <p className="companion-empty small muted">{tr(state.enabled ? 'No paired devices yet. Your iPhone will appear here after you connect.' : 'Enable pairing to view and manage paired devices.')}</p>}
        </div>
        {!state.enabled && state.devices.length ? <p className="xs muted">{tr('Enable pairing to manage paired devices.')}</p> : null}
      </section>

      {state.enabled ? <div className="companion-disable"><div><strong className="small">{tr('iPhone access')}</strong><p className="xs muted">{tr('Turning pairing off disconnects phones. Paired devices are remembered for next time.')}</p></div><Btn size="sm" disabled={busy} onClick={() => void change('/api/companion/stop')}>{tr('Disable pairing')}</Btn></div> : null}
    </>}
    <span className="sr-only" role="status">{copied ? tr('Copied to clipboard') : ''}</span>
    <Dialog open={remove !== null} compact title={tr('Remove this iPhone?')} onClose={() => { if (!busy) setRemove(null); }} actions={<><Btn disabled={busy} onClick={() => setRemove(null)}>{tr('Keep device')}</Btn><Btn variant="danger" disabled={busy} onClick={() => { if (remove) void change(`/api/companion/devices/${encodeURIComponent(remove.id)}`, 'del'); }}>{tr(busy ? 'Removing…' : 'Remove device')}</Btn></>}>
      <p>{remove ? tr('“{0}” will lose access to this Mac immediately. Pair it again to reconnect.', [remove.deviceName]) : null}</p>
    </Dialog>
  </div>;
}

function CopyDetail({ label, value, copied, onCopy }: { label: string; value: string; copied: boolean; onCopy: () => void }) {
  return <div className="companion-detail"><span className="xs muted">{label}</span><div className="companion-detail__value"><code tabIndex={0}>{value}</code><IconBtn name={copied ? 'check' : 'copy'} title={copied ? tr('Copied') : tr('Copy {0}', [label])} onClick={onCopy}/></div></div>;
}
