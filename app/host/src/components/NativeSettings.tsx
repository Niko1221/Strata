import { DecimalInput } from './DecimalInput';
import { tr } from '../lib/i18n';
import {nativeSettingsResponse,nativeSettingsValues} from '../lib/native-settings-response';
import { describeChange, type TunedChange } from '../lib/tune-values';
import { useEffect, useRef, useState } from 'react';
import { useRequestOwner } from '../lib/use-request-owner';
import { api } from '../lib/client';
import { refreshNative, useNative, type NativeField, type Q36MemoryReceipt } from '../lib/native';
import { Btn, ErrorNotice, Select } from './ui';
import { speedOptions } from '../lib/deepseek-speed';
import { draftingChoices } from '../lib/drafting';
import type { RunCard } from '../lib/native-fit';
import { NativeMcpEditor } from './NativeMcpEditor';
import { RefusalProjectionStatus } from './RefusalProjectionStatus';
import type { RefusalProjectionSettings } from '../lib/refusal-projection';
/** One of the speed features: what it is called, what it does, whether it is on by default, and the engine environment it writes outside its default (server/native-settings.mjs). */
interface SpeedSwitch {
    id: string;
    label: string;
    hint: string;
    default: 'on' | 'off';
    env: Record<string, string>;
    /** What an earlier test build wrote for the same state: it reads as set, and goes with it. */
    legacy?: Record<string, string>;
    /** The Expert settings group the row is in (Speed when not given). */
    group?: string;
    /** The two option names instead of On / Off (the picture reader: Strata's own / llama.cpp's). */
    words?: { on: string; off: string };
}
const switchGroup = (sw: SpeedSwitch) => sw.group ?? 'Speed';
const switchHas = (entries: Record<string, string> | undefined, env: Record<string, string> | undefined) => !!entries && Object.keys(entries).length > 0 && Object.entries(entries).every(([name, value]) => env?.[name] === value);
const switchIsSet = (sw: SpeedSwitch, env: Record<string, string> | undefined) => switchHas(sw.env, env) || switchHas(sw.legacy, env);
const switchIsOn = (sw: SpeedSwitch, env: Record<string, string> | undefined) => switchIsSet(sw, env) !== (sw.default === 'on');
export function NativeSettings({ model }: {
    model: string;
}) {
    const native = useNative();
    const details = useRef<HTMLDetailsElement>(null);
    const [revision,setRevision]=useState(0);
    const owner = useRequestOwner(JSON.stringify([model, revision]));
    const saveRequest = useRef(0);
    const [modelFields, setModelFields] = useState<NativeField[] | null>([]);
    const [settingsIssues, setSettingsIssues] = useState<{ field: string; message: string }[]>([]);
    const [projection, setProjection] = useState<RefusalProjectionSettings | null>(null);
    const [hints, setHints] = useState<Record<string, string>>({});   // a field's hint worded for this model (the Drafting size)
    const [switches, setSwitches] = useState<SpeedSwitch[]>([]);   // the speed switches that apply to this model on this PC
    const [dsExperts, setDsExperts] = useState(false);      // the installed DeepSeek engine knows --experts (0.1.50 or newer)
    const [tuned, setTuned] = useState<{ preset: { name: string } | null; changed: TunedChange[] } | null>(null);
    const [values, setValues] = useState<Record<string, any>>({}), [mcp, setMcp] = useState('{}'), [original, setOriginal] = useState(''), [text, setText] = useState(''), [tab, setTab] = useState('fields'), [pending, setPending] = useState(false), [error, setError] = useState(''), [notice, setNotice] = useState(''), [query, setQuery] = useState('');
    // Keep native user toggles across loading/editor renders; only a new error opens the panel.
    useEffect(() => {
        if (error && details.current) details.current.open = true;
    }, [error]);
    useEffect(() => {
        let active = true;
        const visit = owner.current;
        setPending(false);
        setOriginal('');
        setModelFields([]);
        setSettingsIssues([]);
        setProjection(null);
        setValues({});
        setText('');
        setMcp('{}');
        setError('');
        setNotice('');
        void api.get<{
            values: Record<string, any>;
            mcp: Record<string, any>;
            tuned?: { preset: { name: string } | null; changed: TunedChange[] } | null;
            hints?: Record<string, string>;
            capabilities?: { dsExperts?: boolean };
            speedSwitches?: SpeedSwitch[];
            fields?: NativeField[];
            settingsIssues?: { field: string; message: string }[];
            refusalProjection?: RefusalProjectionSettings;
        }>(`/api/native/models/${model}/settings`).then(input => {
            if (!active || owner.current !== visit)
                return;
            setModelFields(input.fields ?? null);
            setSettingsIssues(input.settingsIssues ?? []);
            setProjection(input.refusalProjection ?? null);
            setDsExperts(!!input.capabilities?.dsExperts);
            setSwitches(input.speedSwitches ?? []);
            setTuned(input.tuned ?? null);
            setHints(input.hints ?? {});
            const value = nativeSettingsResponse(input);
            setValues(value.values);
            setText(JSON.stringify(value.values, null, 2));
            setMcp(JSON.stringify(value.mcp ?? {}, null, 2));
            setOriginal(JSON.stringify({ values: value.values, mcp: value.mcp ?? {} }));
        }).catch(e => {
            if (active && owner.current === visit)
                setError(e.message);
        });
        return () => { active = false; };
    }, [model,revision]);
    const change = (id: string, value: any) => { const next = { ...values, [id]: value }; setValues(next); setText(JSON.stringify(next, null, 2)); setNotice(''); };
    /** The default removes the switch's environment entries, the other state writes them: the model's file then carries nothing for a default launch. */
    const changeSwitch = (sw: SpeedSwitch, on: boolean) => { const env = { ...(values.env ?? {}) }; const write = on !== (sw.default === 'on'); for (const [name, value] of Object.entries(sw.env)) { if (write) env[name] = value; else delete env[name]; } if (!write) for (const name of Object.keys(sw.legacy ?? {})) delete env[name]; change('env', env); };
    async function save() {
        const visit = owner.current, request = ++saveRequest.current;
        const current = () => owner.current === visit && saveRequest.current === request;
        setPending(true);
        setError('');
        try {
            const parsed = nativeSettingsValues(tab === 'json' ? JSON.parse(text) : values);
            const tools = JSON.parse(mcp);
            const input = await api.patch<{
                values: Record<string, any>;
                mcp: Record<string, any>;
                fields?: NativeField[];
                settingsIssues?: { field: string; message: string }[];
                refusalProjection?: RefusalProjectionSettings;
            }>(`/api/native/models/${model}/settings`, { values: parsed, mcp: tools });
            if (!current()) return;
            const result = nativeSettingsResponse(input);
            if (input.fields) setModelFields(input.fields);
            setSettingsIssues(input.settingsIssues ?? []);
            setProjection(input.refusalProjection ?? null);
            setValues(result.values);
            setMcp(JSON.stringify(result.mcp, null, 2));
            setText(JSON.stringify(result.values, null, 2));
            setOriginal(JSON.stringify({ values: result.values, mcp: result.mcp }));
            setNotice('Saved. Load the model again to apply these settings. A backup of the previous configuration was kept.');
            await refreshNative(true);
        }
        catch (e) {
            if (current()) setError(e instanceof Error ? e.message : String(e));
        }
        finally {
            if (current()) setPending(false);
        }
    }
    function chooseTab(next: string) {
        if (tab === 'json') {
            try {
                setValues(nativeSettingsValues(JSON.parse(text)));
            }
            catch {
                setError('Fix the JSON before changing tabs.');
                return;
            }
        }
        setTab(next);
    }
    const dirty = (() => {
        try {
            return original !== JSON.stringify({ values: tab === 'json' ? JSON.parse(text) : values, mcp: JSON.parse(mcp) });
        }
        catch {
            return true;
        }
    })();
    if (!original) return <details className="native-settings" ref={details}>
      <summary>{tr('Native engine settings')}</summary>
      {error ? <><ErrorNotice message={tr('Couldn’t load model settings.')} next={tr('Check that the engine is connected and this model is installed, then try again.')} onRetry={() => setRevision(value => value + 1)}/><details className="native-settings-diagnostic"><summary className="small">{tr('Details')}</summary><p className="small muted">{error}</p></details></> : <p className="small muted" role="status">{tr('Reading saved settings…')}</p>}
    </details>;
    const family = (native?.catalog?.families ?? []).find(f => f.sizes.some(s => s.id === model))?.family ?? null;
    const draftingOffered = !!draftingChoices(family, native?.capabilities?.dflash2);
    const projectionField = modelFields?.find(f => f.id === 'refusal_direction_removal');
    const rows = (modelFields ?? []).filter(f => f.id !== 'refusal_direction_removal' && (f.id !== 'ds_experts' || dsExperts || values[f.id] != null) && (f.id !== 'drafting' || draftingOffered || values[f.id] != null) && `${tr(f.label)} ${f.id} ${tr(f.group)}`.toLowerCase().includes(query.toLowerCase()));
    const switchRows = switches.filter(sw => `${tr(sw.label)} ${sw.id} ${tr(switchGroup(sw))}`.toLowerCase().includes(query.toLowerCase()));
    const switchRow = (sw: SpeedSwitch) => <label className="native-setting-row" key={sw.id}><span><strong>{tr(sw.label)}</strong><small>{tr(sw.hint)}</small></span><Select ariaLabel={tr(sw.label)} value={switchIsOn(sw, values.env) ? 'on' : 'off'} options={sw.words ? [{ value: sw.default, label: sw.words[sw.default] }, { value: sw.default === 'on' ? 'off' : 'on', label: sw.words[sw.default === 'on' ? 'off' : 'on'] }] : sw.default === 'on' ? [{ value: 'on', label: 'On (recommended)' }, { value: 'off', label: 'Off' }] : [{ value: 'off', label: 'Off (recommended)' }, { value: 'on', label: 'On' }]} onChange={v => changeSwitch(sw, v === 'on')}/></label>;
    const memoryOffered = modelFields?.some(f => f.id === 'q36_memory' && f.supported !== false) || values.q36_memory != null;
    const memory = actualMemoryReceipt(native?.instances.find(i => i.model === model && i.status === 'ready' &&
        ['idle', 'reading', 'generating'].includes(i.phase) && ['idle', 'reading', 'generating'].includes(i.metrics?.live?.state))?.metrics?.engine);
    return <details className="native-settings" ref={details}><summary>{tr("Native engine settings")}</summary>{tr(tuned?.changed?.length ? <div className="native-tuned"><p className="small"><strong>{tr("Your tuning '{0}' sets", [tuned.preset?.name ?? ''])}</strong></p><ul>{tuned.changed.map(c => { const d = describeChange(c); return <li key={c.flag}><span>{d.name}</span><b>{d.now || tr('off')}</b>{d.was !== null ? <small className="muted">{tr("was {0}", [d.was])}</small> : null}</li>; })}</ul></div> : null)}<p className="small muted">{tr(!original ? 'Reading saved settings… ' : '')}{tr("Settings for this model. Blank values use Strata\u2019s defaults.")}</p><div className="editor-tabs" role="tablist" aria-label={tr("Native settings editor")}>{tr(['fields', 'json', 'tools'].map(t => <button key={t} role="tab" aria-selected={tab === t} onClick={() => chooseTab(t)}>{tr(t === 'fields' ? 'Parameters' : t === 'json' ? 'JSON' : 'MCP tools')}</button>))}</div>
  {settingsIssues.map(issue => <ErrorNotice key={issue.field} message={issue.message}/>)}
  <section className="native-settings-group" aria-label={tr('Refusal-direction removal')}>
    <label className="native-setting-row"><span><strong>{tr('Refusal-direction removal')}</strong><small>{tr('Experimental, off by default. Removes the bundled refusal direction from the original Qwen3.8-Flash-Next residual stream at layers 4–44. Changes responses; neither behaviour nor speed is guaranteed. Reload to apply. Model weights are unchanged.')}</small></span>{projectionField ? <NativeFieldInput field={projectionField} value={values.refusal_direction_removal} onChange={v => change('refusal_direction_removal', v)}/> : null}</label>
    <p className="small muted">{tr(projection?.requested ? 'Saved load setting: on. Reload to apply.' : 'Saved load setting: off (default).')}</p>
    {projection?.reason ? <p className="small muted">{tr(projection.reason)}</p> : null}
    <RefusalProjectionStatus instance={native?.instances.find(i => i.model === model)}/>
  </section>
  {memoryOffered ? <p className="small muted" data-q36-memory-observed={memory?.mode}>{memory
    ? tr('Actual expert storage: {0}. CPU staging {1} bytes; H2D staging {2} bytes; I/O scratch {3} bytes. Expert working storage only; total process memory is not bounded.', [memory.mode, String(memory.cpuStageBytes), String(memory.h2dStageBytes), String(memory.ioScratchBytes)])
    : tr('Actual expert storage is not reported yet.')}</p> : null}
  {modelFields === null ? <p className="small muted">{tr('Serving capabilities are unavailable. Read the settings again before choosing cache formats.')}</p> : null}
  {tr(tab === 'fields' ? <><input className="st-input" type="search" aria-label={tr("Search native parameters")} placeholder={tr("Search parameters…")} value={query} onChange={e => setQuery(e.target.value)}/>{tr(['Responses', 'GPU & memory', 'Serving', 'Image reading', 'Tools', 'Recovery'].map(group => rows.some(f => f.group === group) || switchRows.some(sw => switchGroup(sw) === group) ? <section className="native-settings-group" key={group}><h3 className="h4">{tr(group)}</h3>{tr(rows.filter(f => f.group === group).map(f => <label className="native-setting-row" key={f.id}><span><strong>{tr(f.label)}</strong>{tr(f.supported === false ? <small>{f.unavailableReason ?? tr("Requires a newer Strata installation.")}</small> : f.hint ? <small>{tr(hints[f.id] ?? f.hint)}</small> : null)}</span><NativeFieldInput field={f} model={model} value={values[f.id]} onChange={v => change(f.id, v)}/></label>))}{tr(switchRows.filter(sw => switchGroup(sw) === group).map(switchRow))}</section> : null))}
   {tr(switchRows.some(sw => switchGroup(sw) === 'Speed') ? <section className="native-settings-group"><h3 className="h4">{tr("Speed")}</h3>{tr(switchRows.filter(sw => switchGroup(sw) === 'Speed').map(switchRow))}</section> : null)}<section className="native-settings-group"><h3 className="h4">{tr("Sampling")}</h3>{tr([['temperature', 'Temperature'], ['top_p', 'Top-p'], ['top_k', 'Top-k'], ['min_p', 'Min-p'], ['repetition_penalty', 'Repetition penalty'], ['presence_penalty', 'Presence penalty'], ['frequency_penalty', 'Frequency penalty'], ['penalty_last_n', 'Penalty window'], ['seed', 'Seed']].map(([id, label]) => <label className="native-setting-row" key={id}><span><strong>{tr(label)}</strong></span><DecimalInput ariaLabel={tr("Native {0}", [label])} value={values.sampling?.[id] ?? null} onChange={n => { const sampling = { ...values.sampling }; if (n === null) delete sampling[id]; else sampling[id] = n; change('sampling', sampling); }}/></label>))}</section></> : tab === 'json' ? <><p className="small muted">{tr("Engine arguments and STRATA environment flags are editable here. Program paths and network access remain managed by the host.")}</p><textarea className="st-input native-json" aria-label={tr("Native configuration JSON")} spellCheck={false} value={text} onChange={e => { setText(e.target.value); setNotice(''); }}/></> : <>
   <p className="small muted">{tr("Add local programs or HTTP MCP servers. Connected tools can read or change the resources you give them access to. Enable tools in chat when you want the model to use them.")}</p><NativeMcpEditor text={mcp} onChange={value => { setMcp(value); setNotice(''); }}/><details><summary className="small">{tr("Configuration example")}</summary><pre className="native-example">{'{\n  "files": {\n    "command": "npx",\n    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/your/folder"],\n    "disabled": false\n  }\n}'}</pre><p className="xs muted">{tr("Saved credentials are masked. Keep the mask to preserve an existing value; type a new value to replace it.")}</p></details></>)}
  <div className="native-settings-actions"><Btn disabled={pending || !original || !dirty} variant="primary" onClick={() => void save()}>{tr(pending ? 'Saving…' : 'Save settings')}</Btn><Btn disabled={pending} onClick={() => { const saved = JSON.parse(original || '{"values":{},"mcp":{}}'); setValues(saved.values); setText(JSON.stringify(saved.values, null, 2)); setMcp(JSON.stringify(saved.mcp, null, 2)); setError(''); }}>{tr("Restore saved")}</Btn></div>{tr(notice ? <p role="status" className="small">{tr(notice)}</p> : null)}{tr(error ? <ErrorNotice message={error} onRetry={() => setRevision(value => value + 1)}/> : null)}
 </details>;
}
function NativeFieldInput({ field, value, onChange, model }: {
    field: NativeField;
    model?: string;
    value: any;
    onChange: (v: any) => void;
}) {
    if (field.supported === false)
        return <span>{value != null && value !== '' ? <><code>{String(value)}</code> <button className="text-btn" aria-label={`Clear ${field.label}`} onClick={() => onChange(null)}>{tr('Clear')}</button></> : <span className="small muted">{tr('Unavailable')}</span>}</span>;
    if (field.type === 'array')
        return <NativeAliases field={field} value={value} onChange={onChange}/>;
    if (field.type === 'boolean')
        return <Select ariaLabel={tr(field.label)} value={value === null || value === undefined ? '' : String(value)} options={[{ value: '', label: 'Default' }, { value: 'true', label: 'On' }, { value: 'false', label: 'Off' }]} onChange={v => onChange(v === '' ? null : v === 'true')}/>;
    if (field.id === 'ds_experts')
        return <DeepSeekSpeed field={field} model={model} value={value} onChange={onChange}/>;
    if (field.type === 'enum')
        return <Select ariaLabel={tr(field.label)} value={value ?? ''} options={[{ value: '', label: tr(field.defaultLabel ?? 'Default') }, ...(value != null && value !== '' && !field.choices?.includes(value) ? [{ value, label: `${value} (stored; not supported)` }] : []), ...(field.choices ?? []).map(v => ({ value: v, label: tr(field.choiceLabels?.[v] ?? v) }))]} onChange={v => onChange(v || null)}/>;
    if (field.type === 'number')
        return <DecimalInput ariaLabel={tr(field.label)} placeholder={tr("Default")} value={value === null || value === undefined || value === '' || !Number.isFinite(Number(value)) ? null : Number(value)} onChange={onChange}/>;
    return <input className="st-input" aria-label={tr(field.label)} placeholder={tr("Default")} value={Array.isArray(value) ? value.join(', ') : value ?? ''} onChange={e => onChange(e.target.value === '' ? null : field.type === 'array' ? e.target.value.split(',').map(s => s.trim()).filter(Boolean) : field.type === 'number' ? Number(e.target.value) : e.target.value)}/>;
}
/** The service may stay ready after its child exits. Require its current admitted child binding. */
function actualMemoryReceipt(info: Record<string, unknown> | undefined): Q36MemoryReceipt | null {
    if (!info || !['legacy', 'bounded-experts-v1'].includes(String(info.q36_memory_mode)) ||
        info.q36_memory_scope !== 'expert-working-storage' || info.q36_memory_total_process_bound !== 0) return null;
    const child = info.native_child as Record<string, unknown> | undefined;
    if (!child || child.admitted !== true || child.alive !== true || info.q36_memory_admitted !== true ||
        typeof child.generation !== 'string' || !/^[a-zA-Z0-9-]{1,80}$/.test(child.generation) ||
        typeof child.process_id !== 'number' || !Number.isSafeInteger(child.process_id) || child.process_id <= 0 ||
        info.q36_memory_generation !== child.generation || info.q36_memory_process_id !== child.process_id) return null;
    const bytes = [info.q36_memory_cpu_stage_bytes, info.q36_memory_h2d_stage_bytes, info.q36_memory_io_scratch_bytes];
    if (!bytes.every(n => typeof n === 'number' && Number.isSafeInteger(n) && n >= (info.q36_memory_mode === 'legacy' ? 0 : 1))) return null;
    return {mode: info.q36_memory_mode as Q36MemoryReceipt['mode'], cpuStageBytes: bytes[0] as number, h2dStageBytes: bytes[1] as number, ioScratchBytes: bytes[2] as number};
}
/** DeepSeek's Speed choice. The engine decides the default itself (--experts auto); the labels name the same default (lib/deepseek-speed.ts). */
function DeepSeekSpeed({ field, value, onChange, model }: { field: NativeField; model?: string; value: any; onChange: (v: any) => void }) {
    const native = useNative();
    // the catalog row of this model says whether the engine gives 5 on RDNA4 cards here (one card, or several holding half of the experts)
    const size = (native?.catalog?.families ?? []).flatMap(f => f.sizes ?? []).find(s => s.model.toLowerCase() === (model ?? '').toLowerCase());
    const options = speedOptions(native?.hardware?.gpus as RunCard[] | undefined, size?.rdna4_faster).map(o => ({ ...o, label: tr(o.label) }));
    return <Select disabled={field.supported === false} ariaLabel={tr(field.label)} value={value === null || value === undefined ? '' : String(value)}
      options={options} onChange={v => onChange(v === '' ? null : v)}/>;
}
function NativeAliases({ field, value, onChange }: {
    field: NativeField;
    value: any;
    onChange: (value: any) => void;
}) { const [draft, setDraft] = useState(Array.isArray(value) ? value.join(', ') : ''); useEffect(() => setDraft(Array.isArray(value) ? value.join(', ') : ''), [JSON.stringify(value)]); return <input className="st-input" aria-label={tr(field.label)} disabled={field.supported === false} placeholder={tr("chat, coding")} value={draft} onChange={e => setDraft(e.target.value)} onBlur={() => onChange(draft.trim() ? draft.split(',').map(v => v.trim()).filter(Boolean) : null)}/>; }
