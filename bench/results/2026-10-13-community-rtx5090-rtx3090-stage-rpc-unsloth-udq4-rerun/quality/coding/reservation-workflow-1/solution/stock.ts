export type Stock = {sku:string;units:number};
export function buildStock(stock:Stock[]):Map<string,number>{
 const merged=new Map<string,number>();
 for(const entry of stock){
  if(typeof entry.sku!=='string')throw new RangeError('stock sku must be a string');
  const sku=entry.sku.trim();
  if(sku==='')throw new RangeError('stock sku must be nonempty after trimming');
  if(!Number.isInteger(entry.units)||entry.units<0)throw new RangeError('stock units must be a nonnegative integer');
  merged.set(sku,(merged.get(sku)??0)+entry.units);
 }
 return merged;
}
