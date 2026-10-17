import {buildStock} from './stock.ts';import type {Stock} from './stock.ts';
import {uniqueRequests} from './requests.ts';import type {Request} from './requests.ts';
export function reserve(stock:Stock[],requests:Request[]){
 const remaining=buildStock(stock);
 const unique=uniqueRequests(requests);
 const accepted:string[]=[];const rejected:{id:string;reason:string}[]=[];
 for(const q of unique){
  const available=remaining.get(q.sku);
  if(available===undefined){rejected.push({id:q.id,reason:'unknown-sku'});continue;}
  if(available>=q.quantity){remaining.set(q.sku,available-q.quantity);accepted.push(q.id);}
  else rejected.push({id:q.id,reason:'insufficient-stock'});
 }
 const sortedKeys=[...remaining.keys()].sort((a,b)=>(a<b?-1:a>b?1:0));
 return {accepted,rejected,remaining:Object.fromEntries(sortedKeys.map(k=>[k,remaining.get(k)!]))};
}
