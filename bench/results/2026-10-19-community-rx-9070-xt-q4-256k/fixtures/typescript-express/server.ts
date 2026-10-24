import http from 'node:http';
import express from 'express';
import {Repository} from './repository.ts';
import {OrderService} from './service.ts';
export function app() {
 const api=express();
 const service=new OrderService(new Repository());
 api.use(express.json());
 api.post('/orders',async(req,res)=>{
  try { const result=await service.create(String(req.headers['idempotency-key']||''),req.body); res.status(result.status).json(result.order); }
  catch(error) { res.status(500).json({error:'failed'}); }
 });
 return http.createServer(api);
}
