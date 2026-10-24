export type Order = {id:number; items:unknown[]; total:number};
export class Repository {
  orders:Order[]=[];
  async insert(items:unknown[],total:number) { await Promise.resolve(); const order={id:this.orders.length+1,items,total}; this.orders.push(order); return order; }
}
