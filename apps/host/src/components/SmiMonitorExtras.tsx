/** SMI2000: the Monitor's extras - clear the recent requests, the sampling card, the KV format and the VRAM bar. */
import { useState } from 'react';
import { api } from '../lib/client';
import { localeTag, tr } from '../lib/i18n';
import { hasReading } from '../lib/monitor-hardware';
import { applyNativeEvent, type NativeState } from '../lib/native';
import { kvDetails, vramSegments } from '../lib/smi-expert-tiers';
import type { MonitorState } from '../lib/types';
import { Btn } from './ui';

const number = (value: number | null | undefined, digits = 0) => hasReading(value) ? value.toLocaleString(localeTag(), { maximumFractionDigits: digits, minimumFractionDigits: digits }) : '—';

/** "Clear" in Recent requests: the first click asks, the second clears the list and the totals on the server (the model keeps running). */
export function SmiClearRequests({ ids }: { ids: string[] }) {
    const [armed, setArmed] = useState(false), [busy, setBusy] = useState(false), [error, setError] = useState('');
    if (!ids.length)
        return null;
    async function clear() {
        if (!armed) {
            setArmed(true);
            setTimeout(() => setArmed(false), 4000);
            return;
        }
        setBusy(true);
        setError('');
        try {
            for (const id of ids)
                applyNativeEvent(await api.post<NativeState>(`/api/native/instances/${encodeURIComponent(id)}/metrics/reset`, {}));
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
        finally {
            setBusy(false);
            setArmed(false);
        }
    }
    return <span className="smi-clear">{error ? <span className="xs inline-error">{tr(error)}</span> : null}<Btn size="sm" disabled={busy} onClick={() => void clear()}>{tr(busy ? 'Clearing…' : armed ? 'Click again to clear' : 'Clear')}</Btn></span>;
}

const SAMPLING_ROWS: [string, string, number][] = [
    ['temperature', 'Temperature', 2], ['top_p', 'Top-p', 2], ['top_k', 'Top-k', 0], ['min_p', 'Min-p', 2],
    ['repetition_penalty', 'Repetition penalty', 2], ['presence_penalty', 'Presence penalty', 2], ['frequency_penalty', 'Frequency penalty', 2],
    ['penalty_last_n', 'Penalty window', 0], ['seed', 'Seed', 0],
];

/** The sampling a request gets for the fields it leaves out: the model card, then the model's config, then the Chat settings. */
export function SmiSamplingCard({ sampling }: { sampling: MonitorState['sampling'] }) {
    const thinking = sampling?.thinking ?? {}, instruct = sampling?.instruct ?? {}, config = sampling?.config ?? {};
    const rows = SAMPLING_ROWS.filter(([key]) => hasReading(thinking[key]) || hasReading(instruct[key]));
    const split = rows.some(([key]) => thinking[key] !== instruct[key]);
    const cell = (v: unknown, digits: number) => typeof v === 'number' ? number(v, digits) : '—';
    return <section className="st-card smi-sampling" aria-label={tr('Sampling')}>
      <h2 className="h4">{tr('Sampling')}</h2>
      {rows.length ? <table className="smi-sampling__table"><thead><tr><th/>{split ? <><th className="num">{tr('Thinking')}</th><th className="num">{tr('No thinking')}</th></> : <th className="num">{tr('Value')}</th>}</tr></thead>
        <tbody>{rows.map(([key, label, digits]) => <tr key={key}><td>{tr(label)}{key in config ? <span className="smi-sampling__mark" title={tr('Set in the model settings')}>●</span> : null}</td><td className="num">{cell(thinking[key], digits)}</td>{split ? <td className="num">{cell(instruct[key], digits)}</td> : null}</tr>)}</tbody></table>
        : <p className="small muted">{tr(sampling ? 'Greedy: no sampling values are set.' : 'Restart the model to see its sampling.')}</p>}
      <p className="xs muted">{tr('Used for the fields a request leaves out. ● set in the model settings; the others come from the model card or the Chat settings.')}</p>
    </section>;
}

/** The KV cache format under "Context fill": what it is and how much memory it takes. */
export function SmiKvInfo({ engine }: { engine: Record<string, any> | null | undefined }) {
    const kv = kvDetails(engine);
    if (!kv)
        return null;
    return <div className="smi-kv">
      <div className="monitor-bar-reading"><span>{tr('KV cache')}</span><span className="muted num">{kv.mode}{hasReading(kv.gb) ? ` · ≈${number(kv.gb, 2)} GB` : ''}</span></div>
      <p className="xs muted">{tr(kv.format)}{kv.note ? <> {tr(kv.note)}</> : null}{hasReading(kv.kbPerToken) ? <> {tr('About {0} KB per context token.', [number(kv.kbPerToken, 1)])}</> : null}{kv.streaming ? <> {tr('Kept in RAM; only the part attention reads is in VRAM.')}</> : null}</p>
    </div>;
}

/** One coloured bar of what fills the card: the engine's own start figures, the KV estimate, the rest and the free memory. */
export function SmiVramBar({ engine, usedGb, totalGb }: { engine: Record<string, any> | null | undefined; usedGb: number | null; totalGb: number | null | undefined }) {
    const segments = vramSegments(engine, usedGb, totalGb);
    if (!segments)
        return null;
    return <div className="smi-vram">
      <div className="smi-stack smi-stack--thin" role="img" aria-label={tr('VRAM use')}>{segments.map(s => <span key={s.id} data-part={s.id} style={{ width: `${s.pct}%` }} title={`${tr(s.label)}: ${number(s.gb, 2)} GB`}/>)}</div>
      <dl className="smi-vram__legend">{segments.map(s => <div key={s.id} data-part={s.id} title={tr(s.label)}><dt><i/>{tr(s.short)}</dt><dd className="num">{s.approx ? '≈' : ''}{number(s.gb, s.gb < 10 ? 2 : 1)}</dd></div>)}</dl>
    </div>;
}
