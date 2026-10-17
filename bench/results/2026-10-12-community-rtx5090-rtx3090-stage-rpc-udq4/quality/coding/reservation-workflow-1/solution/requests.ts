export type Request = {id:string;sku:string;quantity:number};
import {requireNonnegativeInteger, requireTrimmedNonemptyString} from './stock.ts';

export function requirePositiveInteger(value:number, what:string):number{
  if(typeof value!=='number'||!Number.isInteger(value)||value<=0)
    throw new RangeError(`${what} must be a positive integer`);
  return value;
}

function normalize(record:Request):Request{
  if(record===null||typeof record!=='object')throw new RangeError('request record must be an object');
  return {
    id:requireTrimmedNonemptyString(record.id,'request id'),
    sku:requireTrimmedNonemptyString(record.sku,'request sku'),
    quantity:requirePositiveInteger(record.quantity,`request quantity for ${record.id}`),
  };
}

// Validates every record first (including later duplicates), then keeps the first occurrence per trimmed ID.
export function uniqueRequests(requests:Request[]):Request[]{
  if(!Array.isArray(requests))throw new RangeError('requests must be an array');
  const normalized=requests.map(normalize);
  const seen=new Set<string>();
  const unique:Request[]=[];
  for(const request of normalized){
    if(seen.has(request.id))continue;
    seen.add(request.id);
    unique.push(request);
  }
  return unique;
}
