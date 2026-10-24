import {Repository} from './repository.ts';
import {parseItems} from './validation.ts';
export class OrderService {
  repo:Repository; constructor(repo:Repository) { this.repo=repo; }
  async create(key:string, body:any) { const items=parseItems(body.items); return {status:201,order:await this.repo.insert(items,0)}; }
}
