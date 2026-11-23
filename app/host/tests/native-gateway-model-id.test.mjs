import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import {nativeGateway} from '../server/native-gateway.mjs';
import {looksLikeStrataModelId, sameModelName} from '../server/model-ids.mjs';
const listen=server=>new Promise(r=>server.listen(0,'127.0.0.1',()=>r(server.address().port)));
const close=server=>new Promise(r=>{server.closeAllConnections();server.close(r);});

test('which names are Strata catalog ids (#6778)',()=>{
  for(const n of ['iq3_s','IQ2_XS','q2_0','ud-q4_k_xl','UD-IQ4_XS-v31','sq-15','gyro-s','inkling-2-mini-sq-10','hy4-flash-sq-64','qwen3.8-flash-next-iq3_s'])assert.equal(looksLikeStrataModelId(n),true,n);
  for(const n of ['gpt-4o','claude-5-5-sonnet','x','local-model','llama-3.1-8b','', undefined, null, 5])assert.equal(looksLikeStrataModelId(n),false,String(n));
  assert.equal(sameModelName('Qwen3.8-Flash-Next-IQ3_S','iq3_s'),true);
  assert.equal(sameModelName('ud-iq3_xxs','iq3_xxs'),false);
  assert.equal(sameModelName('iq3_s','iq2_xs'),false);
});

test('a request for another Strata size is a 404 naming the loaded model, on all three routes; placeholders and the loaded name still work (#6778)',{timeout:10000},async()=>{
  const received=[];const engine=http.createServer(async(req,res)=>{let text='';for await(const chunk of req)text+=chunk;received.push(JSON.parse(text));res.writeHead(200,{'content-type':'application/json'});res.end(JSON.stringify({type:'message',content:[{type:'text',text:'Hi'}],usage:{input_tokens:1,output_tokens:1}}));});
  const port=await listen(engine);const records=[];
  const serving={open:value=>{const r={...value,id:String(records.length)};records.push(r);return r;},mark:(id,value)=>Object.assign(records[Number(id)],value),finish:(id,value)=>Object.assign(records[Number(id)],value)};
  const loaded={id:'only',apiModelId:'iq2_xs',aliases:['iq2_xs'],status:'ready',port,key:'k',gpus:[0]};
  const control={readyInstances:()=>[loaded],resolveInstance:id=>{const found=id===loaded.id||id===loaded.apiModelId||loaded.aliases.includes(id)||(!id)?loaded:undefined;if(!found)throw new Error('Choose a loaded model ID.');return found;}};
  const gateway=http.createServer(async(req,res)=>{let text='';for await(const chunk of req)text+=chunk;await nativeGateway({req,res,path:req.url,body:JSON.parse(text),control,serving});});
  const p=await listen(gateway);
  const call=(path,body)=>fetch(`http://127.0.0.1:${p}${path}`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});
  try{
    for(const [path,extra] of [['/v1/chat/completions',{messages:[]}],['/v1/messages',{messages:[],max_tokens:5}],['/v1/responses',{input:'Hi'}]]){
      const before=received.length;
      const r=await call(path,{model:'iq3_s',...extra});assert.equal(r.status,404,path);
      const e=(await r.json()).error;assert.equal(e.code,'model_not_found');assert.equal(e.param,'model');assert.match(e.message,/"iq3_s" is not loaded/);assert.match(e.message,/"iq2_xs"/);
      assert.equal(received.length,before,'nothing reached the engine');
    }
    let r=await call('/v1/chat/completions',{model:'gpt-4o',messages:[]});assert.equal(r.status,200);await r.text();assert.equal(received.at(-1).model,'iq2_xs');
    r=await call('/v1/chat/completions',{model:'Qwen3.8-Flash-Next-IQ2_XS',messages:[]});assert.equal(r.status,200);await r.text();assert.equal(received.at(-1).model,'iq2_xs');
    r=await call('/v1/chat/completions',{model:'iq2_xs',messages:[]});assert.equal(r.status,200);await r.text();
  }finally{await close(gateway);await close(engine);}
});
