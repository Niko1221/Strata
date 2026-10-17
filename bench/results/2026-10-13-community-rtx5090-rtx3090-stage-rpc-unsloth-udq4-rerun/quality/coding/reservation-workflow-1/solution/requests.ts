export type Request = {id:string;sku:string;quantity:number};
export function uniqueRequests(requests:Request[]):Request[]{
 const seen=new Set<string>();
 const unique:Request[]=[];
 for(const entry of requests){
  if(typeof entry.id!=='string'||typeof entry.sku!=='string')throw new RangeError('request id and sku must be strings');
  const id=entry.id.trim();
  const sku=entry.sku.trim();
  if(id==='')throw new RangeError('request id must be nonempty after trimming');
  if(sku==='')throw new RangeError('request sku must be nonempty after trimming');
  if(!Number.isInteger(entry.quantity)||entry.quantity<1)throw new RangeError('request quantity must be a positive integer');
  if(seen.has(id))continue;
  seen.add(id);
  unique.push({id,sku,quantity:entry.quantity});
 }
 return unique;
}
