export type Stock = {sku:string;units:number};
export function buildStock(stock:Stock[]):Map<string,number>{
 const merged=new Map<string,number>();
 for(const s of stock){
  const sku=s.sku.trim();
  if(sku==='')throw new RangeError('stock sku must be nonempty after trimming');
  if(!Number.isInteger(s.units)||s.units<0)throw new RangeError('stock units must be a nonnegative integer');
  merged.set(sku,(merged.get(sku)??0)+s.units);
 }
 return merged;
}
