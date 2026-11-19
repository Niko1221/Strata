import {spawn} from 'node:child_process';
import {mkdtemp,mkdir,readFile,readdir,rm} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {createServer} from 'node:net';
const root=resolve(import.meta.dirname,'../..');
export async function host(home,{preloads=[],env={}}={}){
  const socket=createServer();await new Promise(r=>socket.listen(0,'127.0.0.1',r));const port=socket.address().port;await new Promise(r=>socket.close(r));
  const base=`http://127.0.0.1:${port}`;
  const child=spawn(process.execPath,['--import','./tests/fixtures/manual-search-network.mjs',...preloads.flatMap(path=>['--import',path]),'server/index.mjs'],{cwd:root,windowsHide:true,stdio:'pipe',env:{...process.env,STRATA_HOME:home,STRATA_MODELS_DIR:join(home,'models'),STRATA_WORKSPACE:join(home,'work'),STRATA_PORT:String(port),STRATA_REAL_ENGINE:'0',STRATA_FORCE_DEMO:'1',STRATA_NO_ACQUIRE:'1',STRATA_HARNESS_DIR:join(home,'no-harness'),...env}});
  let output='';child.stdout.on('data',b=>output+=b);child.stderr.on('data',b=>output+=b);
  const call=async(path,body,headers={})=>{const res=await fetch(base+path,{method:body===undefined?'GET':'POST',headers:{origin:base,...(body===undefined?{}:{'content-type':'application/json'}),...headers},body:body===undefined?undefined:JSON.stringify(body)});return {status:res.status,data:await res.json()};};
  let ready=false;for(let i=0;i<300;i++){try{if((await call('/api/chat/web-search')).status===200){ready=true;break;}}catch{}await new Promise(r=>setTimeout(r,100));}
  if(!ready){child.kill();throw new Error('Fixture host did not start: '+output.slice(-2000));}
  return {call,base,stop:async()=>{if(child.exitCode!==null||child.signalCode!==null)return;const done=new Promise(r=>child.once('exit',r));child.kill();await done;}};
}
