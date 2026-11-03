/**
 * Strata Dev (docs/DEV_BUILD.md, "planGuidance"): the switch, the thinking level the assistant uses while it writes a plan, and the most
 * steps a plan keeps. Saved into dev.config.json by the host (POST /api/dev/plan-guidance). Settings -> General, Dev only. A change applies to
 * the next plan. Dev text is English only, like the Dev panel (no key in the language files).
 */
import { useState } from 'react';
import { api } from '../lib/client';
import { pushToast } from '../lib/store';
import type { DevSummary, PlanGuidanceValues } from '../lib/app-mode';
import { Select } from '../components/ui';
import { devSummaryNow, setDevSummary } from './dev-state';

export const PLAN_GUIDANCE_LABEL = 'Better plans';
export const PLAN_GUIDANCE_HINT = 'Strata Dev only. When the assistant writes a plan it is told how to build a good one, how many steps it may use, and it thinks at the level chosen here for that one turn. Saved in dev.config.json.';

const DEFAULTS: PlanGuidanceValues = { enabled: false, thinking: null, maxSteps: 20 };
const CHAT = 'Chat';
const LEVELS = [{ value: CHAT, label: 'Same as the chat' }, ...(['Off', 'Low', 'Medium', 'High', 'Max'] as const).map(v => ({ value: v as string, label: v }))];
const STEPS = [12, 16, 20, 30, 40].map(n => ({ value: String(n), label: `${n} steps` }));

export function PlanGuidanceSetting() {
    const [value, setValue] = useState<PlanGuidanceValues>(() => devSummaryNow()?.planGuidance ?? DEFAULTS);
    const [busy, setBusy] = useState(false);
    async function save(next: PlanGuidanceValues) {
        const before = value;
        setValue(next);
        setBusy(true);
        try {
            const reply = await api.post<{ dev: DevSummary }>('/api/dev/plan-guidance', next);
            setDevSummary(reply.dev);
            if (reply.dev.planGuidance) setValue(reply.dev.planGuidance);
        } catch (error) {
            setValue(before);
            pushToast('warn', 'Could not save the setting', error instanceof Error ? error.message : String(error));
        } finally { setBusy(false); }
    }
    return <div className="stack" style={{ gap: 10 }}>
        <button className="st-toggle" role="switch" aria-checked={value.enabled} aria-label={PLAN_GUIDANCE_LABEL} disabled={busy} onClick={() => void save({ ...value, enabled: !value.enabled })}/>
        {value.enabled ? <>
            <label className="row" style={{ gap: 10 }}>
                <span style={{ minWidth: 150 }}>Thinking while planning</span>
                <Select translateLabels={false} ariaLabel="Thinking while planning" disabled={busy} value={value.thinking ?? CHAT} options={LEVELS}
                    onChange={v => void save({ ...value, thinking: v === CHAT ? null : v as PlanGuidanceValues['thinking'] })}/>
            </label>
            <label className="row" style={{ gap: 10 }}>
                <span style={{ minWidth: 150 }}>Most steps in a plan</span>
                <Select translateLabels={false} ariaLabel="Most steps in a plan" disabled={busy} value={String(value.maxSteps)} options={STEPS.some(o => o.value === String(value.maxSteps)) ? STEPS : [...STEPS, { value: String(value.maxSteps), label: `${value.maxSteps} steps` }]}
                    onChange={v => void save({ ...value, maxSteps: Number(v) })}/>
            </label>
        </> : null}
    </div>;
}
