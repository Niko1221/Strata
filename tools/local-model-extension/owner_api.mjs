/** Optional companion to the managed Tray's explicit, single-owner file protocol. */
import {readFile,lstat,open,link,unlink} from 'node:fs/promises';
import {isAbsolute,join} from 'node:path';
import {randomUUID} from 'node:crypto';

const object=value=>value!==null&&typeof value==='object'&&!Array.isArray(value);
const count=value=>Number.isSafeInteger(value)&&value>=0?value:null;
const tokens=value=>count(value)>0?value:null;
const field=value=>typeof value==='string'&&value.length>0&&value.length<=256&&!/[\x00-\x1f\x7f]/.test(value);
const IDLES=[0,300,600,1800];
class ApiError extends Error{constructor(status,message){super(message);this.status=status;}}

export function localApiUrl(value){
 const url=new URL(value);
 if(url.protocol!=='http:'||!['127.0.0.1','localhost','[::1]'].includes(url.hostname)
  ||url.username||url.password||url.search||url.hash||url.pathname.replace(/\/$/,'')!=='/v1')
  throw new TypeError('Use A Loopback HTTP Model API At /v1.');
 return url.href.replace(/\/$/,'');
}
function originOf(req){
 const peer=req.socket?.remoteAddress??'';
 if(!/^(?:127\.|::ffff:127\.|::1$)/.test(peer))throw new ApiError(403,'A Loopback App Request Is Required.');
 const host=req.headers.host;
 if(typeof host!=='string'||!/^(?:127\.0\.0\.1|localhost)(?::[1-9][0-9]{0,4})?$/.test(host))
  throw new ApiError(403,'A Loopback App Host Is Required.');
 const origin=new URL('http://'+host).origin;
 if(req.headers.origin!==undefined&&req.headers.origin!==origin)throw new ApiError(403,'A Same-Origin App Request Is Required.');
 return origin;
}
async function jsonFile(path){
 const info=await lstat(path);
 if(!info.isFile()||info.isSymbolicLink()||info.size>65536)throw new Error('Invalid Owner Metadata');
 return JSON.parse((await readFile(path,'utf8')).replace(/^\uFEFF/,''));
}
async function input(req){
 if(req.headers['x-strata-desktop']!=='local-model')throw new ApiError(403,'The Local Model Request Marker Is Required.');
 if(!/^application\/json(?:\s*;\s*charset=utf-8)?$/i.test(req.headers['content-type']??'')
  ||(req.headers['content-encoding']??'identity')!=='identity')throw new ApiError(415,'Use Uncompressed JSON.');
 let size=0;const chunks=[];
 for await(const chunk of req){size+=chunk.length;if(size>4096)throw new ApiError(413,'Request Exceeds 4 KiB.');chunks.push(chunk);}
 try{const value=JSON.parse(Buffer.concat(chunks).toString('utf8'));if(!object(value))throw 0;return value;}
 catch{throw new ApiError(400,'Use A JSON Object.');}
}

export function createOwnerApi({ownerDirectory,baseUrl='http://127.0.0.1:8080/v1',modelLabels={},apiKeyEnv,
 fetchImpl=fetch,now=Date.now}={}){
 if(typeof ownerDirectory!=='string'||!isAbsolute(ownerDirectory))throw new TypeError('Use An Absolute Owner Directory.');
 const base=localApiUrl(baseUrl),serverOrigin=new URL(base).origin;
 if(!object(modelLabels)||Object.keys(modelLabels).length>32||Object.entries(modelLabels).some(([id,v])=>
  !field(id)||!object(v)||Object.keys(v).some(k=>!['name','variant'].includes(k))||Object.values(v).some(x=>!field(x))))
  throw new TypeError('Invalid Model Display Labels.');
 if(apiKeyEnv!==undefined&&!/^[A-Z_][A-Z0-9_]{0,63}$/.test(apiKeyEnv))throw new TypeError('Use An API Key Environment Variable Name.');
 async function request(url,options={},modelServer=false){
  try{
   const headers={...options.headers};
   if(modelServer&&apiKeyEnv&&process.env[apiKeyEnv])headers.authorization='Bearer '+process.env[apiKeyEnv];
   const response=await fetchImpl(url,{...options,headers,redirect:'error',signal:AbortSignal.timeout(5000)});
   if(!response.ok)throw new Error('Rejected');
   return await response.json();
  }catch{throw new ApiError(503,modelServer?'The Local Model Server Is Unavailable.':'The App Connection Could Not Be Confirmed.');}
 }
 async function snapshot(origin){
  const results=await Promise.allSettled([jsonFile(join(ownerDirectory,'extension-status.json')),
   request(serverOrigin+'/health',{},true),request(base+'/status',{},true),request(base+'/models',{},true),request(origin+'/api/native')]);
  const [file,health,status,models,native]=results.map(r=>r.status==='fulfilled'?r.value:null);
  const stamp=Date.parse(file?.captured??''),fresh=Number.isFinite(stamp)&&stamp<=now()+2000&&now()-stamp<=10000;
  const id=field(status?.model)?status.model:null;
  const row=Array.isArray(models?.data)?models.data.find(m=>object(m)&&m.id===id&&!m.alias_of):null;
  let sameBase=false;try{sameBase=localApiUrl(file?.apiBaseUrl)===base;}catch{}
  const owned=!!(file?.schema===1&&fresh&&sameBase&&file.api==='available'&&id&&file.modelId===id
   &&health?.status==='ok'&&health.model===id&&status?.service==='strata'&&row&&object(file.selected)&&object(file.effective)
   &&file.selected.profile===file.effective.profile);
  const capacity=tokens(status?.context?.native);
  const effective=owned&&capacity!==null&&file.effective.context===capacity&&status.cache_max_tokens===capacity
   &&health.max_context===capacity?capacity:null;
  const loaded=owned&&typeof status.loaded==='boolean'&&status.loaded===health.loaded&&file.loaded===status.loaded?status.loaded:null;
  const running=file?.request?.state==='running';
  const choices=Array.isArray(native?.catalog?.contexts)?[...new Set(native.catalog.contexts.filter(n=>tokens(n)!==null))]:[];
  const labels=id&&Object.hasOwn(modelLabels,id)?modelLabels[id]:{};
  const name=labels.name??(field(row?.display_name)?row.display_name:field(row?.name)?row.name:field(file?.selected?.profile)?file.selected.profile:id);
  const failed=file?.request?.state==='error';
  const error=!owned?'The Managed Tray Connection Could Not Be Confirmed.':effective===null||loaded===null?'Effective Model State Has Not Been Confirmed.':
   failed?'The Owner Could Not Apply The Last Request. Check The Tray Logs.':null;
  return {file,dto:{available:owned,model:{id,name,variant:labels.variant??'Local Strata Model',loaded,
   loading:owned?file.model==='loading':null,status:file?.model??'unknown',apiBaseUrl:base},
   context:{choices,selected:tokens(file?.selected?.context),effective,
    phase:owned?(running||file.busy?'draining':file.phase):'unknown',activeRequests:count(file?.active),inFlight:count(status?.activity?.in_flight),error},
   autoUnload:{choices:IDLES,selected:owned&&IDLES.includes(file.selected.idle)?file.selected.idle:null,
    effective:owned&&IDLES.includes(file.effective.idle)?file.effective.idle:null},
   request:object(file?.request)?{id:file.request.id,state:file.request.state}:null,error}};
 }
 function confirmed(current){
  const data=current.dto;
  if(!data.available||data.context.effective===null||data.model.loaded===null)throw new ApiError(503,'Confirm The Model Owner And Effective State First.');
  if(data.context.phase!=='ready'||data.context.selected!==data.context.effective||data.context.activeRequests!==0||data.context.inFlight!==0)
   throw new ApiError(409,'Wait For The Model Owner To Be Ready And Idle.');
 }
 async function queue(current,action,value){
  for(const name of ['extension-request.json','extension-running.json']){
   try{await lstat(join(ownerDirectory,name));throw new ApiError(409,'A Model Owner Request Is Already Pending.');}
   catch(e){if(e.code!=='ENOENT')throw e;}
  }
  const id=randomUUID(),temporary=join(ownerDirectory,'extension-'+id+'.tmp');
  const command={schema:1,id,expectedModel:current.dto.model.id,action,...(value===undefined?{}:{value})};
  try{
   const stream=await open(temporary,'wx',0o600);
   try{await stream.writeFile(JSON.stringify(command)+'\n','utf8');await stream.sync();}finally{await stream.close();}
   // Hard linking a complete file publishes one command atomically without overwriting a pending one.
   try{await link(temporary,join(ownerDirectory,'extension-request.json'));}
   catch(e){if(e.code==='EEXIST')throw new ApiError(409,'A Model Owner Request Is Already Pending.');throw e;}
  }finally{await unlink(temporary).catch(e=>{if(e.code!=='ENOENT')throw e;});}
  return {status:'requested',requestId:id};
 }
 async function connection(current,origin){
  const data=current.dto,providers=await request(origin+'/api/model-providers');
  if(!Array.isArray(providers?.connections))throw new ApiError(502,'Invalid App Connection Metadata.');
  const matches=providers.connections.filter(c=>{
   try{return c.protocol==='openai-chat'&&localApiUrl(c.baseUrl)===base&&Array.isArray(c.models)&&c.models.some(m=>m.id===data.model.id);}catch{return false;}
  });
  if(matches.length!==1)throw new ApiError(409,'Add One Matching Local API Connection In Settings → Models, Then Try Again.');
  const saved=matches[0],rows=saved.models.filter(m=>m.id===data.model.id);
  if(rows.length!==1)throw new ApiError(409,'The Physical Model Must Appear Once In The Connection.');
  const changed=rows[0].name!==data.model.name||rows[0].contextMax!==data.context.effective;
  if(changed){
   const draft={id:saved.id,kind:saved.kind,provider:saved.provider,name:saved.name,baseUrl:saved.baseUrl,protocol:saved.protocol,apiKey:'',
    models:saved.models.map(m=>m.id===data.model.id?{...m,name:data.model.name,contextMax:data.context.effective}:{...m})};
   const result=await request(origin+'/api/model-providers',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(draft)});
   const updated=result?.connections?.find(c=>c.id===saved.id)?.models?.find(m=>m.id===data.model.id);
   if(updated?.name!==data.model.name||updated?.contextMax!==data.context.effective)throw new ApiError(502,'The App Did Not Confirm The Model Metadata.');
  }
  return {selection:{connectionId:saved.id,modelId:data.model.id},label:data.model.name,connectionName:saved.name,
   contextMax:data.context.effective,changed};
 }
 async function handle(req,res,url){
  const actions=new Map([['/api/local-model','model'],...['context','idle','load','unload','connection'].map(a=>['/api/local-model/'+a,a])]);
  const action=actions.get(url.pathname);if(!action)return false;
  const send=(code,data)=>{res.writeHead(code,{'content-type':'application/json; charset=utf-8','cache-control':'no-store'});res.end(JSON.stringify(data));};
  try{
   const origin=originOf(req),read=action==='model';
   if(req.method!==(read?'GET':'POST'))throw new ApiError(405,'Unsupported Local Model Method.');
   if(read){send(200,(await snapshot(origin)).dto);return true;}
   const body=await input(req),allowed=action==='context'?['expectedModel','tokens']:action==='idle'?['expectedModel','seconds']:['expectedModel'];
   if(Object.keys(body).some(k=>!allowed.includes(k))||!field(body.expectedModel))throw new ApiError(400,'Use The Announced Physical Model Identity.');
   const current=await snapshot(origin);confirmed(current);
   if(body.expectedModel!==current.dto.model.id)throw new ApiError(409,'The Selected Model Changed. Refresh Before Acting.');
   if(action==='connection'){send(200,await connection(current,origin));return true;}
   const value=action==='context'?body.tokens:action==='idle'?body.seconds:undefined;
   if(action==='context'&&!current.dto.context.choices.includes(value))throw new ApiError(400,'Choose A Context From The Official Catalog.');
   if(action==='idle'&&!IDLES.includes(value))throw new ApiError(400,'Choose Never, 5, 10 Or 30 Minutes.');
   send(202,await queue(current,action,value));
  }catch(e){send(e instanceof ApiError?e.status:500,{error:e instanceof ApiError?e.message:'The Local Model Operation Could Not Be Completed.'});}
  return true;
 }
 return {handle};
}
