import { useEffect, useState } from 'react';
import { tr } from '../lib/i18n';
import { useFolderPrompt } from '../lib/folder-pick';
import { Btn, Dialog } from './ui';

/** The fallback for a system without a folder chooser: say so in plain words and take the folder's path as text. */
export function FolderPathDialog() {
  const prompt = useFolderPrompt();
  const [path, setPath] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => { if (prompt) { setPath(''); setError(''); } }, [prompt]);
  const browse = async () => {
    if (!prompt?.browse) return;
    setBusy(true);
    setError('');
    try { const picked = await prompt.browse(); if (picked) setPath(picked); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  };
  const submit = () => { const text = path.trim(); if (text && !busy) prompt?.resolve(text); };
  return <Dialog open={!!prompt} compact title={tr("Type the folder path")} onClose={() => { if (!busy) prompt?.resolve(null); }}
    actions={<><Btn disabled={busy} onClick={() => prompt?.resolve(null)}>{tr("Cancel")}</Btn>{prompt?.browse ? <Btn icon="folder" disabled={busy} onClick={() => void browse()}>{tr("Choose folder")}</Btn> : null}<Btn variant="primary" disabled={busy || !path.trim()} onClick={submit}>{tr("Use this folder")}</Btn></>}>
    <form className="stack" onSubmit={event => { event.preventDefault(); submit(); }}>
      <p className="small" role="status">{tr(prompt?.message ?? '')}</p>
      {error ? <p className="small" role="alert">{tr(error)}</p> : null}
      <label className="input-field"><span>{tr("Folder path")}</span>
        <input className="st-input" autoFocus disabled={busy} value={path} onChange={event => setPath(event.target.value)} placeholder={tr("/home/you/Documents")} spellCheck={false} aria-label={tr("Folder path")}/>
      </label>
    </form>
  </Dialog>;
}
