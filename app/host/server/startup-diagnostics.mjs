/** Keep the first useful cause independently of the UI's rolling log tail. */
import {StringDecoder} from 'node:string_decoder';

export function createStartupDiagnostics(secret='') {
 const decoder=new StringDecoder('utf8');
 const limit=4096;let pending='',discarding=false,first='',cause='',tail=[];
 const clean=line=>{
  // Redact before bounding the display, including keys split across pipe chunks.
  const value=secret?line.replaceAll(secret,'[redacted]'):line;
  return value.replace(/\x1b\[[0-9;]*m/g,'').replace(/[\x00-\x1f\x7f]/g,' ').trim().slice(0,600);
 };
 const accept=line=>{
  const text=clean(line);
  if(!/error|failed|invalid|unsupported|cannot|bad option|block size|not found/i.test(text))return;
  first ||= text;
  // Fit/retry wrappers can precede the actual model-loader diagnostic.
  if(!cause&&/check_tensor_dims:|error loading model hyperparameters:|unknown model architecture|unsupported model architecture|CUDA error:|out of memory/i.test(text))cause=text;
  tail.push(text);if(tail.length>2)tail.shift();
 };
 return {
  consume(stream,chunk){
   if(stream!=='err')return;
   const text=typeof chunk==='string'?chunk:decoder.write(chunk);
   let offset=0;
   while(offset<text.length){
    const end=text.indexOf('\n',offset),part=text.slice(offset,end<0?text.length:end);
    if(!discarding){
     pending+=part.slice(0,limit-pending.length);
     if(pending.length===limit)discarding=true;
    }
    if(end<0)break;
    accept(pending);pending='';discarding=false;offset=end+1;
   }
  },
  flush(){pending+=decoder.end();if(pending)accept(pending);pending='';discarding=false;},
  summary(){return [...new Set([cause||first,...tail].filter(Boolean))].join(' | ');},
 };
}
