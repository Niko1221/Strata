import { useEffect, useRef, useState } from 'react';
import { api } from '../lib/client';
import { tr } from '../lib/i18n';
import { setSkills, useStore } from '../lib/store';
import type { Skill } from '../lib/types';
import { Btn, Dialog, ErrorNotice, Select } from './ui';

type Preview = Pick<Skill,'name'|'description'|'body'|'triggers'> & {markdown:string};
/** File/paste preview has no write side effects. Only Import submits to the existing create route. */
export function SkillImportDialog({open,onClose,onImported}:{open:boolean;onClose:()=>void;onImported:(skill:Skill)=>void}) {
  const workspaces=useStore(s=>s.workspaces);
  const [markdown,setMarkdown]=useState(''),[preview,setPreview]=useState<Preview|null>(null);
  const [triggers,setTriggers]=useState(''),[onlyIn,setOnlyIn]=useState<string|null>(null),[enabled,setEnabled]=useState(false);
  const [busy,setBusy]=useState(false),[error,setError]=useState('');
  const generation=useRef(0);
  useEffect(()=>{generation.current++;if(open){setMarkdown('');setPreview(null);setTriggers('');setOnlyIn(null);setEnabled(false);setError('');setBusy(false);}return()=>{generation.current++;};},[open]);
  async function readFile(file?:File){
    const ticket=++generation.current;
    setError('');setPreview(null);
    if(!file)return;
    if(!/\.md$/i.test(file.name)||file.size>64*1024){setError('Choose a SKILL.md file within 64 KiB.');return;}
    try{const text=await file.text();if(ticket===generation.current)setMarkdown(text);}catch{if(ticket===generation.current)setError('The skill file could not be read.');}
  }
  async function inspect(){
    const ticket=++generation.current;
    setBusy(true);setError('');
    try{const result=await api.post<Omit<Preview,'markdown'>>('/api/skills/preview',{markdown});if(ticket===generation.current){setPreview({...result,markdown});setTriggers(result.triggers.join(', '));}}
    catch(e){if(ticket===generation.current)setError(e instanceof Error?e.message:'The skill could not be previewed.');}
    finally{if(ticket===generation.current)setBusy(false);}
  }
  async function confirm(){
    if(!preview||busy)return;
    const ticket=++generation.current;
    setBusy(true);setError('');
    try{
      const result=await api.post<{id:string;skills:Skill[]}>('/api/skills',{markdown:preview.markdown,importConfirmed:true,triggers:triggers.split(',').map(t=>t.trim()).filter(Boolean),onlyIn,enabled});
      // The successful write already returns the authoritative list. A failed follow-up GET must not make a completed import look failed.
      await setSkills(result.skills);
      if(ticket===generation.current){const created=result.skills.find(s=>s.id===result.id);if(created)onImported(created);onClose();}
    }catch(e){if(ticket===generation.current)setError(e instanceof Error?e.message:'The skill could not be imported.');}
    finally{if(ticket===generation.current)setBusy(false);}
  }
  return <Dialog open={open} title={tr('Import SKILL.md')} onClose={()=>{if(!busy)onClose();}} actions={<><Btn disabled={busy} onClick={onClose}>{tr('Cancel')}</Btn>{preview?<Btn variant="primary" disabled={busy||!triggers.split(',').some(t=>t.trim())} onClick={()=>void confirm()}>{tr(busy?'Importing…':'Import')}</Btn>:<Btn variant="primary" disabled={busy||!markdown.trim()} onClick={()=>void inspect()}>{tr(busy?'Loading…':'Preview skill')}</Btn>}</>}>
    <div className="form-stack"><p className="small muted">{tr('Imports instructions only. Companion scripts, references and dependencies are not installed or executed. Review the instructions before importing.')}</p>
    {!preview?<><label className="input-field"><span>{tr('Choose SKILL.md')}</span><input className="st-input" type="file" accept=".md,text/markdown" disabled={busy} onChange={e=>void readFile(e.target.files?.[0])}/></label><label className="input-field"><span>{tr('Or paste SKILL.md')}</span><textarea className="st-input skill-instructions" rows={8} maxLength={65536} disabled={busy} value={markdown} onChange={e=>{generation.current++;setMarkdown(e.target.value);}}/></label><p className="xs muted">{tr('Up to 64 KiB. Requires name and description in frontmatter, and up to 40000 instruction characters.')}</p></>:<>
      <h3 className="h4">{preview.name}</h3><p className="small">{preview.description}</p>
      <label className="input-field"><span>{tr('Instructions')}</span><textarea className="st-input skill-instructions" rows={8} readOnly value={preview.body}/></label>
      <label className="input-field"><span>{tr('Trigger phrases')}</span><input className="st-input" value={triggers} disabled={busy} onChange={e=>setTriggers(e.target.value)}/><small className="muted">{tr('Separate phrases with commas. In ordinary chat, enabled skills apply when your question includes a trigger phrase.')}</small></label>
      <p className="small">{tr('Example question: {0}',[`${triggers.split(',').find(t=>t.trim())?.trim()??preview.name}: ${tr('please help with this task.')}`])}</p>
      <div className="input-field"><span>{tr('Available in')}</span><Select ariaLabel={tr('Skill workspace scope')} value={onlyIn??'all'} options={[{value:'all',label:'All workspaces'},...workspaces.map(w=>({value:w.id,label:w.name}))]} onChange={value=>setOnlyIn(value==='all'?null:value)}/></div>
      <div className="row"><span>{tr('Skill enabled')}</span><button className="st-toggle" role="switch" aria-label={tr('Skill enabled')} aria-checked={enabled} disabled={busy} onClick={()=>setEnabled(!enabled)}/></div>
      <Btn disabled={busy} onClick={()=>setPreview(null)}>{tr('Choose different instructions')}</Btn>
    </>}{error?<ErrorNotice message={tr(error)}/>:null}</div>
  </Dialog>;
}
