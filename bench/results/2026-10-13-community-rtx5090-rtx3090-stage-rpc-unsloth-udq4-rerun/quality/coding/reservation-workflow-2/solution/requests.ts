export type Request = {id:string;sku:string;quantity:number};
export function uniqueRequests(requests:Request[]):Request[]{
 const seen=new Set<string>();const unique:Request[]=[];
 for(const q of requests){
  const id=q.id.trim();const sku=q.sku.trim();
  if(id==='')throw new RangeError('request id must be nonempty after trimming');
  if(sku==='')throw new RangeError('request sku must be nonempty after trimming');
  if(!Number.isInteger(q.quantity)||q.quantity<=0)throw new RangeError('request quantity must be a positive integer');
  if(seen.has(id))continue;
  seen.add(id);
  unique.push({id,sku,quantity:q.quantity});
 }
 return unique;
}
