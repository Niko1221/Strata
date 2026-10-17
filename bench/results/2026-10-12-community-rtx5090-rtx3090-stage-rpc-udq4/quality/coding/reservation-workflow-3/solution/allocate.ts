import {buildStock} from './stock.ts';import type {Stock} from './stock.ts';
import {uniqueRequests} from './requests.ts';import type {Request} from './requests.ts';
export function reserve(stock:Stock[],requests:Request[]){
 const remaining=buildStock(stock);
 const unique=uniqueRequests(requests);
 const accepted:string[]=[];const rejected:{id:string;reason:string}[]=[];
 for(const q of unique){
  const n=remaining.get(q.sku);
  if(n===undefined){rejected.push({id:q.id,reason:'unknown-sku'});continue;}
  if(n>=q.quantity){remaining.set(q.sku,n-q.quantity);accepted.push(q.id);}
  else rejected.push({id:q.id,reason:'insufficient-stock'});
 }
 const sorted=[...remaining.keys()].sort();
 return {accepted,rejected,remaining:Object.fromEntries(sorted.map(k=>[k,remaining.get(k)!]))};
}
