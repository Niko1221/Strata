import {buildStock} from './stock.ts';import type {Stock} from './stock.ts';
import {validateStock} from './stock.ts';
import {uniqueRequests,validateRequests} from './requests.ts';import type {Request} from './requests.ts';
export type Rejection = {id:string;reason:'unknown-sku'|'insufficient-stock'};
export type ReservationResult = {accepted:string[];rejected:Rejection[];remaining:Record<string,number>};
export function reserve(stock:Stock[],requests:Request[]):ReservationResult{
 validateStock(stock);
 validateRequests(requests);
 const remaining=buildStock(stock);
 const accepted:string[]=[];const rejected:Rejection[]=[];
 for(const r of uniqueRequests(requests)){
  const have=remaining.get(r.sku);
  if(have===undefined){rejected.push({id:r.id,reason:'unknown-sku'});continue;}
  if(have>=r.quantity){remaining.set(r.sku,have-r.quantity);accepted.push(r.id);}
  else rejected.push({id:r.id,reason:'insufficient-stock'});
 }
 const sorted=Object.fromEntries([...remaining].sort((a,b)=>a[0]<b[0]?-1:a[0]>b[0]?1:0));
 return {accepted,rejected,remaining:sorted};
}
