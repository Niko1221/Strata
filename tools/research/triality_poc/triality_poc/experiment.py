"""Reproducible tiny-model demonstration with no network or real-model claims."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,platform,time
from dataclasses import asdict
from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.nn import functional as F
from .core import resolve_device
from .tiny_lm import TinyLM,LMConfig,DataSplit,make_splits


def write_json(path:Path,obj:dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def weights_hash(model:TinyLM,exclude_adapter:bool=True) -> str:
    h=hashlib.sha256()
    for n,p in model.state_dict().items():
        if exclude_adapter and '.adapter.' in n:
            continue
        h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def evaluate(model:TinyLM,data:DataSplit,teacher:TinyLM|None,device:torch.device,batch:int=16) -> dict:
    model.eval();values=[]
    for i in range(0,len(data.x),batch):
        x,y=data.x[i:i+batch].to(device),data.y[i:i+batch].to(device)
        z,h=model(x,return_hidden=True);logp=z.float().log_softmax(-1)
        nll=F.nll_loss(logp.flatten(0,1),y.flatten(),reduction='none').view_as(y).mean(-1)
        row={'nll':nll,'accuracy_suffix':(z.argmax(-1)[:,7:]==y[:,7:]).float().mean(-1)}
        if teacher is not None:
            tz,th=teacher(x,return_hidden=True);tl=tz.float().log_softmax(-1)
            row['teacher_kl']=(tl.exp()*(tl-logp)).sum(-1).mean(-1)
            row['teacher_argmax_agreement']=(z.argmax(-1)==tz.argmax(-1)).float().mean(-1)
            row['hidden_relative_mse']=torch.stack([((a-b).float().square().mean((1,2))/b.float().square().mean((1,2)).clamp_min(1e-12)) for a,b in zip(h,th)],-1)
        values.append({n:v.cpu() for n,v in row.items()})
    result={}
    for n in values[0]:
        arr=torch.cat([v[n] for v in values],0)
        result[n]=arr.mean(0).tolist()
    result['perplexity']=math.exp(result['nll'])
    result['sequences']=len(data.x)
    return result


def fit_adapter(base:TinyLM,kind:str,splits:dict,device:torch.device,steps:int,seed:int,batch:int) -> tuple[TinyLM,dict]:
    student=copy.deepcopy(base);student.set_codec(kind,2,None);student.freeze_except_adapter();student.ste=True
    opt=torch.optim.Adam([p for p in student.parameters() if p.requires_grad],lr=.008)
    g=torch.Generator().manual_seed(80000+seed)
    initial=evaluate(student,splits['validation'],base,device,batch)['teacher_kl']
    best=initial;best_state=copy.deepcopy(student.state_dict());best_step=0;curve=[]
    student.train()
    for step in range(1,steps+1):
        idx=torch.randint(len(splits['train'].x),(batch,),generator=g)
        x=splits['train'].x[idx].to(device)
        with torch.no_grad():
            tz,th=base(x,return_hidden=True);tl=tz.float().log_softmax(-1)
        z,hh=student(x,return_hidden=True)
        kl=(tl.exp()*(tl-z.float().log_softmax(-1))).sum(-1).mean()
        hm=sum(((a-b).float().square().mean()/b.float().square().mean().clamp_min(1e-8)) for a,b in zip(hh,th))/len(hh)
        reg=sum(b.adapter.w_out.square().mean() for b in student.blocks)
        loss=kl+.05*hm+.001*reg
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],1.)
        opt.step()
        if step%20==0 or step==steps:
            metric=evaluate(student,splits['validation'],base,device,batch)['teacher_kl']
            curve.append({'step':step,'train_loss':float(loss.detach()),'validation_teacher_kl':metric})
            if metric<best:
                best,best_step=metric,step;best_state=copy.deepcopy(student.state_dict())
            student.train()
    student.load_state_dict(best_state);student.ste=False;student.eval()
    if weights_hash(student)!=weights_hash(base):
        raise RuntimeError('Frozen base weights were modified.')
    return student,dict(initial_validation_kl=initial,best_validation_kl=best,best_step=best_step,curve=curve,
                        base_weights_unchanged=True,training_quantizer='explicit straight-through estimator')


def run_seed(args,seed:int,device:torch.device) -> dict:
    torch.manual_seed(seed);splits=make_splits(seed,args.sequence)
    base=TinyLM(LMConfig()).to(device);base.train()
    # Unused adapters are excluded from teacher pretraining.
    params=[p for n,p in base.named_parameters() if '.adapter.' not in n]
    opt=torch.optim.AdamW(params,lr=.002,weight_decay=.01)
    g=torch.Generator().manual_seed(9000+seed);losses=[];start=time.perf_counter()
    for step in range(args.teacher_steps):
        idx=torch.randint(len(splits['pretrain'].x),(args.batch,),generator=g)
        x,y=splits['pretrain'].x[idx].to(device),splits['pretrain'].y[idx].to(device)
        z=base(x);loss=F.cross_entropy(z.flatten(0,1),y.flatten())
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
        if (step+1)%50==0:
            losses.append({'step':step+1,'train_nll':float(loss.detach())})
    base.eval()
    for p in base.parameters():p.requires_grad_(False)
    arms={'teacher_exact':evaluate(base,splits['test'],None,device,args.batch)}
    for name,kind,kbits,vbits in [('single2','single',2,None),('single6','single',6,None),
                                 ('triality3x2','triality',2,None),('generic3x2','generic',2,None),
                                 ('triality3x2_v3','triality',2,3)]:
        m=copy.deepcopy(base);m.set_codec(kind,kbits,vbits)
        arms[name]=evaluate(m,splits['test'],base,device,args.batch)
    training={}
    for kind in ('triality','generic'):
        student,train=fit_adapter(base,kind,splits,device,args.adapter_steps,seed,args.batch)
        training[kind]=train
        arms[kind+'_trained']=evaluate(student,splits['test'],base,device,args.batch)
        if kind=='triality':
            student.set_codec('exact',None,None)
            arms['triality_trained_exact_kv']=evaluate(student,splits['test'],base,device,args.batch)
            student.set_codec('triality',2,None)
        torch.save({'scope':'tiny synthetic model only','config':asdict(base.cfg),'seed':seed,
                    'kind':kind,'base_hash':weights_hash(base),'adapter':student.adapter_state()},
                   Path(args.out)/f'adapter_{kind}_seed{seed}.pt')
    torch.save({'config':asdict(base.cfg),'state_dict':base.state_dict(),'seed':seed},Path(args.out)/f'teacher_seed{seed}.pt')
    if device.type=='cuda':torch.cuda.synchronize(device)
    return {'seed':seed,'config':asdict(base.cfg),'total_parameters':sum(p.numel() for p in base.parameters()),
            'adapter_parameters':sum(p.numel() for n,p in base.named_parameters() if '.adapter.' in n),
            'data_hashes':{n:d.sha256 for n,d in splits.items()},'arms':arms,'training':training,
            'teacher_training_curve':losses,'elapsed_seconds':time.perf_counter()-start}


def main() -> int:
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');p.add_argument('--out',default='results/tiny_lm')
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);p.add_argument('--teacher-steps',type=int,default=350)
    p.add_argument('--adapter-steps',type=int,default=160);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--sequence',type=int,default=32);args=p.parse_args()
    device=resolve_device(args.device);torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    Path(args.out).mkdir(parents=True,exist_ok=True)
    out={'scope':'trained tiny synthetic autoregressive Transformer; NOT 27B/Strata/NLP quality evidence',
         'environment':{'torch':torch.__version__,'python':platform.python_version(),'device':str(device),
                        'cuda_build':torch.version.cuda,'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None},
         'args':vars(args),'runs':[]}
    for seed in args.seeds:
        r=run_seed(args,seed,device);out['runs'].append(r)
        write_json(Path(args.out)/'results.json',out)
        print(json.dumps({'seed':seed,'seconds':r['elapsed_seconds'],'arms':r['arms']},ensure_ascii=False),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
