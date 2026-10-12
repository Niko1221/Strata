export type Stock = {sku:string;units:number};
function trimmedSku(sku:unknown):string{
 if(typeof sku!=='string')throw new RangeError('stock sku must be a string');
 const t=sku.trim();
 if(t==='')throw new RangeError('stock sku must be nonempty after trimming');
 return t;
}
export function validateStock(stock:Stock[]):void{
 for(const s of stock){
  const sku=trimmedSku(s.sku);
  if(typeof s.units!=='number'||!Number.isInteger(s.units)||s.units<0)
   throw new RangeError(`stock units for ${sku} must be a nonnegative integer`);
 }
}
export function buildStock(stock:Stock[]):Map<string,number>{
 const map=new Map<string,number>();
 for(const s of stock){const sku=trimmedSku(s.sku);map.set(sku,(map.get(sku)??0)+s.units);}
 return map;
}
