import {buildStock} from './stock.ts';import type {Stock} from './stock.ts';
import {uniqueRequests} from './requests.ts';import type {Request} from './requests.ts';

export type RejectionReason = 'unknown-sku' | 'insufficient-stock';
export type Reservation = {id:string;reason:RejectionReason};
export type ReserveResult = {accepted:string[];rejected:Reservation[];remaining:Record<string,number>};

export function reserve(stock:Stock[],requests:Request[]):ReserveResult{
  // Both calls validate every record (duplicates included) before any stock is consumed.
  const remaining=buildStock(stock);
  const accepted:string[]=[];
  const rejected:Reservation[]=[];
  for(const r of uniqueRequests(requests)){
    const available=remaining.get(r.sku);
    if(available===undefined){rejected.push({id:r.id,reason:'unknown-sku'});continue;}
    if(available>=r.quantity){remaining.set(r.sku,available-r.quantity);accepted.push(r.id);}
    else rejected.push({id:r.id,reason:'insufficient-stock'});
  }
  const sorted=[...remaining.entries()].sort((a,b)=>a[0]<b[0]?-1:a[0]>b[0]?1:0);
  return {accepted,rejected,remaining:Object.fromEntries(sorted)};
}
