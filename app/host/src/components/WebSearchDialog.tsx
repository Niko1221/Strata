import { useEffect, useRef, useState } from 'react';
import { setSettingsSection } from '../lib/store';
import { api } from '../lib/client';
import { tr } from '../lib/i18n';
import { navigate } from '../lib/router';
import type { WebSearchResult } from '../lib/types';
import { Btn, Dialog, ErrorNotice } from './ui';

interface Availability {available:boolean;destination:string;reason:string|null}
/** Opening the dialog reads capability only; the query starts empty and never comes from chat. */
export function WebSearchDialog({open,sessionId,onClose,onUse}:{open:boolean;sessionId:string|null;onClose:()=>void;onUse:(result:WebSearchResult)=>void}) {
  const [query,setQuery]=useState(''),[capability,setCapability]=useState<Availability|null>(null);
  const [result,setResult]=useState<WebSearchResult|null>(null),[error,setError]=useState(''),[busy,setBusy]=useState(false);
  const controller=useRef<AbortController|null>(null),generation=useRef(0);
  function cancel() {generation.current++;controller.current?.abort();controller.current=null;setBusy(false);}
  useEffect(()=>{
    const ticket=++generation.current;
    setQuery('');setResult(null);setError('');setCapability(null);setBusy(false);
    if(open)void api.get<Availability>('/api/chat/web-search').then(value=>{if(ticket===generation.current)setCapability(value);},()=>{if(ticket===generation.current)setError('Web search settings could not be loaded.');});
    return ()=>{generation.current++;controller.current?.abort();};
  },[open,sessionId]);
  function close(){cancel();onClose();}
  async function search(){
    if(!sessionId||busy||!capability?.available)return;
    cancel();const ticket=++generation.current,abort=new AbortController();controller.current=abort;setBusy(true);setResult(null);setError('');
    try {
      const value=await api.post<WebSearchResult>('/api/chat/web-search',{sessionId,query,confirmed:true},abort.signal);
      if(ticket===generation.current&&!abort.signal.aborted)setResult(value);
    } catch(e){if(ticket===generation.current&&!abort.signal.aborted)setError(e instanceof Error?e.message:'The search could not reach the provider. Check the connection and try again.');}
    finally{if(ticket===generation.current){controller.current=null;setBusy(false);}}
  }
  return <Dialog open={open} title={tr('Search web')} onClose={close} actions={<>
    <Btn onClick={close}>{tr('Cancel')}</Btn>
    {busy?<Btn icon="stop" onClick={cancel}>{tr('Stop')}</Btn>:<Btn disabled={!capability?.available||!query.trim()} onClick={()=>void search()}>{tr('Search')}</Btn>}
    {result?.sources.length?<Btn variant="primary" onClick={()=>{if(Date.now()>=result.expiresAt){setError('Search references expired or belong to another chat. Search again.');return;}onUse(result);close();}}>{tr('Use results in question')}</Btn>:null}
  </>}><div className="form-stack">
    <p className="small muted">{tr('Only the query you enter is sent to the selected search provider. Your conversation and private memory are not used to make the query.')}</p>
    <label className="input-field"><span>{tr('Search query')}</span><input className="st-input" value={query} maxLength={400} disabled={busy} onChange={e=>{setQuery(e.target.value);setResult(null);}} onKeyDown={e=>{if(e.key==='Enter'){e.preventDefault();void search();}}}/></label>
    <p className="xs muted">{tr('Destination')}: {capability?.destination??'DuckDuckGo / Bing'}</p>
    {capability&&!capability.available?<><p role="status">{tr(capability.reason??'Web search is disabled. Enable it in Settings → Tools.')}</p><Btn onClick={()=>{close();setSettingsSection('Tools');navigate('settings');}}>{tr('Open tool settings')}</Btn></>:null}
    {error?<ErrorNotice message={tr(error)}/>:null}
    {busy?<p role="status">{tr('Searching…')}</p>:null}
    {result?<div className="stack web-reference-sources" role="status"><p className="small muted">{tr('Snippets only. Full pages were not read. References apply to your next answer with autonomous tools off.')}</p>{result.sources.length?result.sources.map((source,i)=><article className="st-card" key={source.url}><div className="st-card__body stack"><a href={source.url} target="_blank" rel="noopener noreferrer">{i+1}. {source.title}</a><span className="xs muted">{source.url}</span><p className="small">{source.snippet}</p></div></article>):<p>{tr('No results. Try another search query.')}</p>}</div>:null}
  </div></Dialog>;
}
