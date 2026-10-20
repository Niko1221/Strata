export class Database { stock=3; orders=[]; fail=false;
 async insert(order){await new Promise(r=>setTimeout(r,2));if(this.fail)throw Error('database unavailable');this.orders.push(order);return order;}
}
