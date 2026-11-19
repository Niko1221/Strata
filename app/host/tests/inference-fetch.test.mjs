import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import {setTimeout as delay} from 'node:timers/promises';
import {inferenceFetch} from '../server/inference-fetch.mjs';

async function server(t,handler){
 const server=http.createServer(handler);await new Promise(r=>server.listen(0,'127.0.0.1',r));
 t.after(()=>new Promise(r=>{server.closeAllConnections();server.close(r)}));
 return 'http://127.0.0.1:'+server.address().port;
}
test('slow prompt headers and delayed SSE are read without changing request bytes',async t=>{
 let observed;
 const url=await server(t,async(req,res)=>{
  let body='';for await(const chunk of req)body+=chunk;
  observed={body,authorization:req.headers.authorization,length:req.headers['content-length']};
  await delay(60);res.writeHead(200,{'content-type':'text/event-stream'});res.write('data: first\n\n');
  await delay(60);res.end('data: [DONE]\n\n');
 });
 const body=JSON.stringify({messages:[{role:'user',content:'長い入力'}]});
 const response=await inferenceFetch(url,{method:'POST',headers:{authorization:'Bearer fixture'},body},{timeoutMs:1000});
 assert.equal(response.status,200);assert.match(response.headers.get('content-type'),/event-stream/);
 assert.equal(await response.text(),'data: first\n\ndata: [DONE]\n\n');
 assert.deepEqual(observed,{body,authorization:'Bearer fixture',length:String(Buffer.byteLength(body))});
});
test('the deadline applies while waiting for headers and body',async t=>{
 const url=await server(t,(req,res)=>{if(req.url==='/body'){res.writeHead(200);res.write('partial');}});
 await assert.rejects(inferenceFetch(url,{},{timeoutMs:50}),{name:'TimeoutError'});
 const response=await inferenceFetch(url+'/body',{},{timeoutMs:50});
 await assert.rejects(response.text());
});
test('user cancellation closes the upstream connection before and after headers',async t=>{
 let closed=0;
 const url=await server(t,(req,res)=>{res.on('close',()=>closed++);if(req.url==='/body'){res.writeHead(200);res.write('partial');}});
 const before=new AbortController();
 const pending=inferenceFetch(url,{signal:before.signal});await delay(30);before.abort();
 await assert.rejects(pending,{name:'AbortError'});
 const after=new AbortController();const response=await inferenceFetch(url+'/body',{signal:after.signal});after.abort();
 await assert.rejects(response.text());await delay(30);assert.equal(closed,2);
});
test('local redirects are returned without forwarding credentials',async t=>{
 let follow=0;
 const url=await server(t,(req,res)=>{if(req.url==='/target')follow++;res.writeHead(302,{location:'/target'});res.end();});
 const response=await inferenceFetch(url,{headers:{authorization:'Bearer fixture'}});
 assert.equal(response.status,302);await response.text();assert.equal(follow,0);
});
