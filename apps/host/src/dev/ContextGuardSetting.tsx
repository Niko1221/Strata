/**
 * Strata Dev (docs/DEV_BUILD.md, "contextGuard"): the switch and the three shares of the window, saved into dev.config.json by the host
 * (POST /api/dev/context-guard). Shown in Settings -> General in Dev only. The values apply to the next message; "While a task runs"
 * reaches the agent when it starts again.
 */
import { useState } from 'react';
import { api } from '../lib/client';
import { pushToast } from '../lib/store';
import type { ContextGuardValues, DevSummary } from '../lib/app-mode';
import { devSummaryNow, setDevSummary } from './dev-state';

/** Dev text is English only, like the Dev panel (no key in the language files). */
export const CONTEXT_GUARD_LABEL = 'Summarise long chats early';
export const CONTEXT_GUARD_HINT = 'Strata Dev only. Condenses older messages once the chat holds this share of the model’s window, so a long chat never runs full. Saved in dev.config.json.';
const DEFAULTS: ContextGuardValues = { enabled: false, midTaskAt: 0.55, compactAfterTurnAt: 0.55, compactBeforeSendAt: 0.6 };
const ROWS: { key: 'midTaskAt' | 'compactAfterTurnAt' | 'compactBeforeSendAt'; label: string }[] = [
    { key: 'compactBeforeSendAt', label: 'Before sending' },
    { key: 'compactAfterTurnAt', label: 'After an answer' },
    { key: 'midTaskAt', label: 'While a task runs' },
];

export function ContextGuardSetting() {
    const [value, setValue] = useState<ContextGuardValues>(() => devSummaryNow()?.contextGuard ?? DEFAULTS);
    const [busy, setBusy] = useState(false);
    async function save(next: ContextGuardValues) {
        const before = value;
        setValue(next);
        setBusy(true);
        try {
            const reply = await api.post<{ dev: DevSummary }>('/api/dev/context-guard', next);
            setDevSummary(reply.dev);
            if (reply.dev.contextGuard) setValue(reply.dev.contextGuard);
        } catch (error) {
            setValue(before);
            pushToast('warn', 'Could not save the setting', error instanceof Error ? error.message : String(error));
        } finally { setBusy(false); }
    }
    return <div className="stack" style={{ gap: 10 }}>
        <button className="st-toggle" role="switch" aria-checked={value.enabled} aria-label={'Summarise long chats early'} disabled={busy} onClick={() => void save({ ...value, enabled: !value.enabled })}/>
        {value.enabled ? ROWS.map(row => <label className="row" key={row.key} style={{ gap: 10 }}>
            <span style={{ minWidth: 150 }}>{row.label}</span>
            <input type="range" className="st-slider" aria-label={row.label} min={30} max={90} step={5} value={Math.round(value[row.key] * 100)} disabled={busy}
                onChange={e => setValue({ ...value, [row.key]: Number(e.target.value) / 100 })}
                onPointerUp={() => void save(value)} onKeyUp={() => void save(value)}/>
            <output>{Math.round(value[row.key] * 100)}{' %'}</output>
        </label>) : null}
    </div>;
}
