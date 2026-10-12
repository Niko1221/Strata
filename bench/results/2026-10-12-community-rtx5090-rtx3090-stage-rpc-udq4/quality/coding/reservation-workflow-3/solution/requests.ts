export type Request = {id:string;sku:string;quantity:number};
export function uniqueRequests(requests:Request[]):Request[]{
 const seen=new Set<string>();const out:Request[]=[];
 for(const q of requests){
  if(q===null||typeof q!=='object')throw new RangeError('request record must be an object');
  if(typeof q.id!=='string'||typeof q.sku!=='string')throw new RangeError('request id/sku must be strings');
  if(typeof q.quantity!=='number'||!Number.isInteger(q.quantity)||q.quantity<=0)throw new RangeError('request quantity must be a positive integer');
  const id=q.id.trim();const sku=q.sku.trim();
  if(id==='')throw new RangeError('request id must be nonempty');
  if(sku==='')throw new RangeError('request sku must be nonempty');
  if(seen.has(id))continue;
  seen.add(id);out.push({id,sku,quantity:q.quantity});
 }
 return out;
}
