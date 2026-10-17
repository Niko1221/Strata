export type Request = {id:string;sku:string;quantity:number};
function checkRequest(r:Request):{id:string;sku:string;quantity:number}{
 if(typeof r.id!=='string')throw new RangeError('request id must be a string');
 const id=r.id.trim();
 if(id==='')throw new RangeError('request id must be nonempty after trimming');
 if(typeof r.sku!=='string')throw new RangeError(`request ${id} sku must be a string`);
 const sku=r.sku.trim();
 if(sku==='')throw new RangeError(`request ${id} sku must be nonempty after trimming`);
 if(typeof r.quantity!=='number'||!Number.isInteger(r.quantity)||r.quantity<=0)
  throw new RangeError(`request ${id} quantity must be a positive integer`);
 return {id,sku,quantity:r.quantity};
}
export function validateRequests(requests:Request[]):void{
 for(const r of requests)checkRequest(r);
}
export function uniqueRequests(requests:Request[]):Request[]{
 const seen=new Set<string>();const out:Request[]=[];
 for(const r of requests){const n=checkRequest(r);if(seen.has(n.id))continue;seen.add(n.id);out.push(n);}
 return out;
}
