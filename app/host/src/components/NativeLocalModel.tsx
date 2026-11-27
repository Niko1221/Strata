import { tr } from '../lib/i18n';
import { useEffect, useState } from 'react';
import { Screen } from './layout';
import { Btn, ErrorNotice, Icon } from './ui';
import { api } from '../lib/client';
import { useImeKeys } from '../lib/ime-keys';
import { parseLocation } from '../lib/router';
import { nativeAction, useNative, type NativeJob } from '../lib/native';
import { nativeCatalogModels, catalogVersionLabel } from '../lib/native-catalog';
import { sizeGb } from '../lib/format';
import { NativeProgress } from './NativeModels';
import { ownFilesVision } from '../lib/own-files-vision';

import { cleanDisplayName, customTarget, suggestedDisplayName } from '../lib/custom-models';

/** `custom`: the files are not a build Strata publishes (a fine-tune, a merge, another maker's quantization) but have the layout of a model its engines run: they are used in place under a name the person chooses. */
interface Preview { path: string; preview: { model_dir?: string; file_kind?: 'gguf' | 'strata'; plain_issues?: string[]; already_set_up?: { data_dir?: string | null; pack?: string | null } | null; file_count: number; total_bytes: number; file_set_status: string; issues: string[]; messages?: string[]; projectors?: { name: string; status: string; pinned: boolean; plain: string }[]; reads_pictures?: boolean; candidate_selection: { family: string; model: string; custom?: boolean; suggested_label?: string; family_title?: string; base_family?: string; base_model?: string } | null } }

/** The model families Strata's own engines run, in the words of the unsupported-variant message. */
export const SUPPORTED_FAMILIES_TEXT = 'Strata runs Qwen3.8-Flash-Next, Qwen3.6-35B-A3B, HY-4 Flash and Inkling-2-Mini, and compatible fine-tunes and quantizations of them. Download one from Models, or choose its .strata or .gguf files.';

/** The words for why a folder of model files cannot be used: a sentence and what to do, the checker's own codes under Details. */
export function problem(issues: string[]): { message: string; next: string } {
    const has = (re: RegExp) => issues.some(i => re.test(i));
    if (has(/multiple_model_groups/))
        return { message: 'This folder holds more than one model.', next: 'Type the full path of the .gguf file of the one you want, or choose the folder of that model.' };
    if (has(/incomplete_or_invalid_shard_set/))
        return { message: 'Some parts of this model are missing.', next: 'Put every part of one model (all the -0000N-of-0000M files) in one folder, then check again.' };
    if (has(/unsupported_architecture|unsupported_or_missing_architecture/))
        return { message: 'This model type isn’t supported by the Strata engine.', next: 'Add it as a Custom model API instead: start it with llama-server, then open Settings > Models > Add model provider > Custom model API.' };
    if (has(/unrecognized_or_unsupported_model_variant|expert_quantization|tokenizer/))
        return { message: 'Strata can’t run these files yet.', next: SUPPORTED_FAMILIES_TEXT };
    return { message: 'These files could not be checked.', next: 'Check that the folder holds complete .strata or .gguf files, then try again. The file named under Details says what is wrong with it.' };
}

/** "Use local model files": choose the folder of a model you already have, or one .strata or .gguf file, Strata checks it, and prepares and loads it without copying
 *  or downloading the model files. The same page layout and cards as the optimized models page. */
export function NativeLocalModel({ onBack }: { onBack: () => void }) {
    const native = useNative(), ime = useImeKeys();
    const [typed, setTyped] = useState(''), [checked, setChecked] = useState<Preview | null>(null), [pending, setPending] = useState(false), [error, setError] = useState(''), [jobId, setJobId] = useState('');
    const job = native?.jobs.find(j => j.id === jobId);
    const active = native?.jobs.find(j => j.state === 'running');
    const [label, setLabel] = useState('');
    const selection = checked?.preview.candidate_selection;
    const isCustom = selection?.custom === true;
    const usable = !!checked && checked.preview.file_set_status === 'plausible' && !!selection;
    const version = selection ? nativeCatalogModels(native?.catalog?.families ?? []).flatMap(f => f.sizes).find(s => s.sourceFamily === selection.family && s.model === selection.model) : undefined;
    const family = selection ? nativeCatalogModels(native?.catalog?.families ?? []).find(f => f.sizes.some(s => s.sourceFamily === selection.family && s.model === selection.model)) : undefined;
    async function check(path: string) {
        setPending(true);
        setError('');
        setChecked(null);
        try {
            const result = await api.post<Preview & { cancelled?: boolean }>('/api/native/local/preview', { path });
            if (!result.cancelled) {
                setChecked(result);
                setTyped(result.path);
                setLabel(suggestedDisplayName(result.preview.candidate_selection?.suggested_label, result.path));
            }
        }
        catch (e) { setError(e instanceof Error ? e.message : String(e)); }
        finally { setPending(false); }
    }
    /* The first-run page hands over the folder the person chose (#3507): it is checked at once, as if they had pressed Check files. */
    useEffect(() => { const path = parseLocation(window.location.hash).query.get('path'); if (path) { setTyped(path); void check(path); } }, []);
    /* Files that sit in a Strata data folder (<data>/models/<tag>) with their prepared model data beside them: that folder is used as the data
     * folder, so the prepared data is not made a second time. A folder the installer would not accept as a data folder is not passed. */
    const ownDataFolder = (() => { const d = checked?.preview.already_set_up?.pack ? checked.preview.already_set_up.data_dir : null; return d && /strata/i.test(d.split(/[\\/]/).filter(Boolean).pop() ?? '') ? d : null; })();
    async function prepare() {
        if (!usable || !selection || !checked)
            return;
        setPending(true);
        setError('');
        try {
            const isInkling = selection.family === 'inkling-2-mini';
            // A custom build is used from the very file that was picked (the folder can hold the official files too); it is saved under the name the person chose, never over the official build of the same size.
            const localTarget = isCustom ? customTarget(checked.path, checked.preview.model_dir, true) : isInkling && /\.(gguf|strata)$/i.test(checked.path) ? checked.path : (checked.preview.model_dir ?? checked.path);
            const customName = isCustom ? cleanDisplayName(label) : '';
            if (isCustom && !customName)
                throw new Error('Give this model a name.');
            const result = await nativeAction<NativeJob>('operations', { kind: 'install', args: { family: selection.family, model: selection.model, gguf_dir: localTarget, ...(isCustom ? { custom_label: customName, drafting: 'auto' } : {}), ...(ownDataFolder && !isCustom ? { data_dir: ownDataFolder } : {}), context: isInkling ? 32768 : (native?.catalog?.recommendation?.context ?? 32768), vision: ownFilesVision(checked.preview.projectors, native?.visionReader, native?.installed?.models, selection.family), kv: 'int8', loadAfter: true } });
            setJobId(result.id);
        }
        catch (e) { setError(e instanceof Error ? e.message : String(e)); }
        finally { setPending(false); }
    }
    const found = checked ? problem(checked.preview.issues) : null;
    return <Screen plainHeader title={tr("Model")} actions={<Btn onClick={onBack}>{tr("Back")}</Btn>}>
  <div className="native-model-layout">
   <div className="stack stack--lg">
    <section className="st-card native-choice"><h2 className="h3">{tr("Use local model files (.strata or .gguf)…")}</h2>
     <p className="small muted">{tr("Already have a supported model on this computer, such as a compatible fine-tune or quantization? Choose the folder that holds its files, or one of its .strata or .gguf files. Strata checks them, prepares the model and loads it. Nothing is copied or downloaded, and your files are never changed.")}</p>
     <label className="input-field"><span>{tr("Folder with the model files, or one .strata or .gguf file")}</span><input className="st-input" value={typed} placeholder={tr("Full path to the folder or the file")} spellCheck={false} onChange={e => setTyped(e.target.value)} {...ime.events} onKeyDown={e => { if (ime.owns(e)) return; if (e.key === 'Enter' && typed.trim()) void check(typed); }}/></label>
     <div className="native-settings-actions"><Btn icon="folder" disabled={pending || !!active} onClick={() => void check('')}>{tr("Choose folder")}</Btn><Btn disabled={pending || !typed.trim() || !!active} onClick={() => void check(typed)}>{tr(pending && !jobId ? 'Checking…' : 'Check files')}</Btn></div>
    </section>
    {tr(checked ? <section className="st-card native-choice">{tr(usable ? <>
      <h2 className="h3">{tr(selection?.family_title ?? family?.title ?? 'Model found')}{tr(isCustom ? <> <span className="version-tag">{tr("Custom")}</span></> : null)}</h2>
      <p><strong>{tr(isCustom ? 'Custom build' : version ? catalogVersionLabel(version) : selection?.model ?? '')}</strong> · {tr(isCustom ? (version ? `${tr('Closest build:')} ${catalogVersionLabel(version)}` : selection?.model ?? '') : selection?.model ?? '')}</p>
      {tr(isCustom ? <label className="input-field local-model-name"><span>{tr("Display name")}</span><input className="st-input" aria-label={tr("Display name")} value={label} maxLength={80} spellCheck={false} disabled={pending || !!active} onChange={e => setLabel(e.target.value)}/><small className="muted">{tr("This is how the model is listed in Strata. Your files stay where they are.")}</small></label> : null)}
      <p className="small muted">{tr("{0} files · {1}", [checked.preview.file_count, sizeGb(checked.preview.total_bytes / 1024 ** 3)])}</p>
      {tr(checked.preview.messages?.map(m => <p className="small muted" key={m}>{tr(m)}</p>))}
      <div className="native-model-actions"><Btn variant="primary" icon="models" disabled={pending || !!active || (isCustom && !cleanDisplayName(label))} onClick={() => void prepare()}>{tr(pending ? 'Checking…' : 'Prepare and load')}</Btn></div>
     </> : <>
      <h2 className="h3">{tr(found!.message)}</h2>
      <p className="small muted">{tr(found!.next)}</p>
      {tr(checked.preview.plain_issues?.map(m => <p className="small muted" key={m}>{tr(m)}</p>))}
      {tr(checked.preview.messages?.map(m => <p className="small muted" key={m}>{tr(m)}</p>))}
      {tr(checked.preview.issues.length ? <details><summary className="small">{tr("Details")}</summary><pre className="small muted error-detail-text">{checked.preview.issues.join(String.fromCharCode(10))}</pre></details> : null)}
     </>)}</section> : null)}
    {tr(active ? <NativeProgress job={active}/> : job?.state === 'error' ? <ErrorNotice message={job.error ?? job.summary} detail={job.details}/> : job?.state === 'done' ? <p className="small" role="status"><Icon name="check" size={15}/> {tr(job.summary)}</p> : null)}
    {tr(error ? <ErrorNotice message={error}/> : null)}
   </div>
   <aside className="stack stack--lg"><section className="st-card"><h2 className="h4">{tr("This computer")}</h2><p>{tr(native?.hardware?.ram_gb || '—')}{tr(" GB RAM")}</p>{tr(native?.hardware?.gpus.length ? native.hardware.gpus.map(g => <p className="small" key={g.index}>{tr(g.name)} · {tr(g.vram_gb)}{tr(" GB")}</p>) : null)}</section></aside>
  </div>
 </Screen>;
}
