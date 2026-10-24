export type Item = {sku:string; quantity:number; unitPrice:number};
export function parseItems(value: unknown): Item[] { return value as Item[]; }
