// Independent HTTP contract check. A barrier starts both service calls together;
// either returned errors or thrown domain errors are valid implementation choices.
import assert from 'node:assert/strict';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
const fixture=process.argv[2]||'typescript-express';
const root=pathToFileURL(resolve(fixture)+'/').href;
const {OrderService}=await import(root+'service.ts');
const {app}=await import(root+'server.ts');
const original=OrderService.prototype.create;
const repositories=new Set();
const queued=[];
OrderService.prototype.create=function(...args) {
  repositories.add(this.repo);
  return new Promise((resolve,reject)=>{
    const timeout=setTimeout(()=>reject(new Error('Concurrency barrier timed out')),5000);
    queued.push({self:this,args,resolve,reject,timeout});
    if(queued.length===2) {
      for(const job of queued.splice(0)) {
        clearTimeout(job.timeout);
        Promise.resolve().then(()=>original.apply(job.self,job.args)).then(job.resolve,job.reject);
      }
    }
  });
};
const server=app();
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
try {
  const url=`http://127.0.0.1:${server.address().port}/orders`;
  const responses=await Promise.all(['a','b'].map(sku=>fetch(url,{
    method:'POST',headers:{'content-type':'application/json','idempotency-key':'concurrent-conflict'},
    body:JSON.stringify({items:[{sku,quantity:1,unitPrice:100}]})
  })));
  const statuses=responses.map(r=>r.status).sort();
  const orders=[...repositories].reduce((n,repo)=>n+repo.orders.length,0);
  console.log(JSON.stringify({fixture,statuses,orders}));
  assert.deepEqual(statuses,[201,409]);
  assert.equal(orders,1);
} finally {
  server.closeAllConnections();
  await new Promise(resolve=>server.close(resolve));
  OrderService.prototype.create=original;
}
