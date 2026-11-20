import {useEffect,useRef,useState} from 'react';
import {tr} from '../lib/i18n';
import {Btn,Dialog,ErrorNotice} from './ui';
import './HakuaLlamaChat.css';

export interface LlamaMemoryAccess {profileKey:string;profileRoot?:string;runtimeId:string}
type Message={role:'user'|'assistant';content:string};

/** Memory conversation and credentials stay in this dialog's RAM, outside chat history. */
export function HakuaLlamaChat({access,onClose}:{access:LlamaMemoryAccess;onClose:()=>void}) {
  const [messages,setMessages]=useState<Message[]>([]);
  const [input,setInput]=useState(''),[problem,setProblem]=useState(''),[status,setStatus]=useState('');
  const [read,setRead]=useState(false),[save,setSave]=useState(false),[busy,setBusy]=useState(false);
  const controller=useRef<AbortController|null>(null);
  const log=useRef<HTMLDivElement>(null);
  useEffect(()=>()=>controller.current?.abort(),[]);
  useEffect(()=>{log.current?.scrollTo({top:log.current.scrollHeight});},[messages,status]);
  const close=()=>{controller.current?.abort();onClose();};
  async function send(){
    if(busy||!input.trim())return;
    const abort=new AbortController();controller.current=abort;
    const history=[...messages,{role:'user' as const,content:input.trim()}].slice(-31);
    const consent={read,store_summary:save,turn_id:crypto.randomUUID()};
    setRead(false);setSave(false);setInput('');setProblem('');setStatus(tr('Generating…'));setBusy(true);
    setMessages([...history,{role:'assistant',content:''}]);
    let answer='',done=false;
    try{
      const response=await fetch('/api/hakua/chat',{method:'POST',signal:abort.signal,headers:{
        'content-type':'application/json',authorization:'Bearer '+access.profileKey,
      },body:JSON.stringify({runtimeId:access.runtimeId,messages:history,strata_memory:consent,stream:true,max_tokens:512})});
      if(!response.ok){const data=await response.json();throw new Error(typeof data.error==='string'?data.error:'Memory chat could not start.');}
      if(!response.body)throw new Error('The model returned no response.');
      const reader=response.body.getReader(),decoder=new TextDecoder();let pending='';
      const apply=(line:string)=>{
        if(!line.trim())return;
        const value=JSON.parse(line);
        if(value.error)throw new Error(value.error);
        const text=value.event?.choices?.[0]?.delta?.content;
        if(typeof text==='string'){
          answer+=text;
          setMessages([...history,{role:'assistant',content:answer}]);
        }
        if(value.response){answer=value.response.choices?.[0]?.message?.content??'';setMessages([...history,{role:'assistant',content:answer}]);}
        if(value.done){
          done=true;
          setStatus(tr(value.memory==='stored'?'Summary saved to Hakua-memory.':
            value.memory==='no-summary'?'The model did not return a useful summary. No memory was saved.':
            value.memory==='uncertain'?'Memory write is uncertain. This turn will not be saved again.':
            value.memory==='write-failed'?'The answer is complete, but its summary could not be saved.':
            'This turn did not save a memory.'));
        }
      };
      try{
        while(true){
          const chunk=await reader.read();
          pending+=decoder.decode(chunk.value,{stream:!chunk.done});
          if(pending.length>2*1024*1024)throw new Error('The memory response was too large.');
          let end:number;
          while((end=pending.indexOf('\n'))>=0){apply(pending.slice(0,end));pending=pending.slice(end+1);}
          if(chunk.done){apply(pending);break;}
        }
        if(!done)throw new Error('The memory turn ended before completion.');
      }finally{await reader.cancel().catch(()=>{});reader.releaseLock();}
    }catch(error){
      if(abort.signal.aborted)setStatus(tr('Stopped. A completed summary may already have been saved.'));
      else{setProblem(error instanceof Error?error.message:String(error));setStatus('');}
    }finally{if(controller.current===abort)controller.current=null;setBusy(false);}
  }
  return <Dialog open title={tr('Hakua memory chat')} onClose={close} wide className="hakua-memory-dialog"
    actions={<><Btn onClick={close}>{tr('Close')}</Btn>{busy?<Btn onClick={()=>controller.current?.abort()}>{tr('Stop')}</Btn>:
      <Btn variant="primary" disabled={!input.trim()} onClick={()=>void send()}>{tr('Send')}</Btn>}</>}>
    <p>{tr('Choose read and save consent separately for each turn. This conversation stays in this window; only a consented summary enters Hakua-memory.')}</p>
    <div ref={log} className="hakua-memory-log" role="log" aria-label={tr('Memory conversation')} aria-busy={busy}>
      {messages.map((message,index)=><section key={index} className="hakua-memory-turn">
        <strong>{tr(message.role==='user'?'You':'Hakua')}</strong><p>{message.content}</p>
      </section>)}
    </div>
    {problem?<ErrorNotice message={problem}/>:null}
    <p className="small muted" role="status">{status}</p>
    <label className="input-field"><span>{tr('Message')}</span><textarea className="st-input" rows={4} maxLength={16384}
      value={input} disabled={busy} onChange={event=>setInput(event.target.value)}/></label>
    <div className="hakua-memory-consent">
      <label><input type="checkbox" checked={read} disabled={busy} onChange={event=>setRead(event.target.checked)}/>{tr('Read my saved memories for this turn')}</label>
      <label><input type="checkbox" checked={save} disabled={busy} onChange={event=>setSave(event.target.checked)}/>{tr('Save a summary after this answer completes')}</label>
    </div>
  </Dialog>;
}
