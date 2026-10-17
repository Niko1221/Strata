export type Stock = {sku:string;units:number};

export function requireTrimmedNonemptyString(value:string, what:string):string{
  if(typeof value!=='string')throw new RangeError(`${what} must be a string`);
  const trimmed=value.trim();
  if(trimmed.length===0)throw new RangeError(`${what} must be nonempty after trimming`);
  return trimmed;
}

export function requireNonnegativeInteger(value:number, what:string):number{
  if(typeof value!=='number'||!Number.isInteger(value)||value<0)
    throw new RangeError(`${what} must be a nonnegative integer`);
  return value;
}

// Validates every record (including duplicates) and combines duplicate SKUs by adding units.
export function buildStock(stock:Stock[]):Map<string,number>{
  if(!Array.isArray(stock))throw new RangeError('stock must be an array');
  const combined=new Map<string,number>();
  for(const entry of stock){
    if(entry===null||typeof entry!=='object')throw new RangeError('stock record must be an object');
    const sku=requireTrimmedNonemptyString(entry.sku,'stock sku');
    const units=requireNonnegativeInteger(entry.units,`stock units for ${sku}`);
    combined.set(sku,(combined.get(sku)??0)+units);
  }
  return combined;
}
